import config
from langgraph.graph import END, StateGraph

from agents.generator import generate_sql
from agents.intent import parse_intent
from agents.planner import plan_query
from agents.schema_retriever import retrieve_schema
from agents.scope_guard import check_module_scope_fit
from agents.selector import select_tables
from agents.validator import validate_sql
from cache.semantic_query_cache import lookup_similar_sql, remember_successful_sql
from db.utils import extract_referenced_tables
from state import AgentState
from tools.modules import resolve_scope


def _route_after_scope(state: AgentState) -> str:
    if state.get("scope_blocked"):
        return "blocked"
    return "continue"


def _route_after_validation(state: AgentState) -> str:
    if state.get("validation_passed"):
        return "end"
    if state.get("retry_count", 0) >= config.MAX_RETRY_ATTEMPTS:
        return "end"
    return "retry"


def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("parse_intent", parse_intent)
    graph.add_node("check_scope", check_module_scope_fit)
    graph.add_node("select_tables", select_tables)
    graph.add_node("retrieve_schema", retrieve_schema)
    graph.add_node("plan_query", plan_query)
    graph.add_node("generate_sql", generate_sql)
    graph.add_node("validate_sql", validate_sql)

    graph.set_entry_point("parse_intent")
    graph.add_edge("parse_intent", "check_scope")
    graph.add_conditional_edges(
        "check_scope",
        _route_after_scope,
        {"blocked": END, "continue": "select_tables"},
    )
    graph.add_edge("select_tables", "retrieve_schema")
    graph.add_edge("retrieve_schema", "plan_query")
    graph.add_edge("plan_query", "generate_sql")
    graph.add_edge("generate_sql", "validate_sql")

    graph.add_conditional_edges(
        "validate_sql",
        _route_after_validation,
        {"end": END, "retry": "generate_sql"},
    )

    return graph.compile()


def _cache_hit_state(
    *,
    user_query: str,
    scope,
    hit: dict,
    logs: list[str],
) -> AgentState:
    sql = hit["sql"]
    tables = hit.get("selected_tables") or extract_referenced_tables(sql)
    sim = float(hit.get("similarity") or 0)
    match_type = hit.get("match_type") or "semantic"
    logs.append(
        f"[Query Cache] HIT ({match_type}, similarity={sim:.2f}) — "
        f"reusing SQL from: {hit.get('question', '')!r}. "
        "LLM pipeline skipped."
    )
    from llm import get_last_token_usage, snapshot_token_usage

    session = snapshot_token_usage()
    logs.append(
        f"[Peek tokens] Cache hit — 0 tokens this question | "
        f"session total={session.get('total_tokens', 0)} "
        f"({session.get('calls', 0)} call(s))"
    )
    return {
        "user_query": user_query,
        "active_modules": list(scope.module_ids),
        "allowed_tables": sorted(scope.allowed_tables),
        "allowed_tools": sorted(scope.allowed_tool_names),
        "retry_count": 0,
        "scope_blocked": False,
        "cache_hit": True,
        "cache_similarity": sim,
        "selected_tables": tables,
        "selected_tools": [],
        "schema_metadata": {},
        "query_plan": (
            f"Reused cached SQL (similar to: {hit.get('question', '')}). "
            "No new LLM generation."
        ),
        "generated_sql": sql,
        "validation_passed": True,
        "validation_errors": [],
        "final_sql": sql,
        "token_usage": {
            "query": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "calls": 0,
            },
            "session": session,
            "last_call": get_last_token_usage(),
        },
        "agent_logs": logs,
    }


def run_pipeline(user_query: str, module_ids: list[str] | None = None) -> AgentState:
    if not module_ids:
        raise ValueError(
            "At least one module must be selected before generating SQL."
        )
    scope = resolve_scope(module_ids)
    if not scope.allowed_tables:
        raise ValueError("Selected modules resolve to no tables.")
    if not scope.allowed_tool_names:
        raise ValueError("Selected modules resolve to no tools.")

    logs = [
        f"[Module Scope] modules={', '.join(scope.module_ids)} | "
        f"tables={', '.join(sorted(scope.allowed_tables))} | "
        f"tools={len(scope.allowed_tool_names)}"
    ]

    hit = lookup_similar_sql(user_query, sorted(scope.allowed_tables))
    if hit:
        return _cache_hit_state(
            user_query=user_query,
            scope=scope,
            hit=hit,
            logs=logs,
        )

    logs.append("[Query Cache] MISS — running LLM pipeline.")
    from llm import format_token_usage_log, get_last_token_usage, snapshot_token_usage, usage_since

    usage_before = snapshot_token_usage()
    app = build_graph()
    initial_state: AgentState = {
        "user_query": user_query,
        "active_modules": list(scope.module_ids),
        "allowed_tables": sorted(scope.allowed_tables),
        "allowed_tools": sorted(scope.allowed_tool_names),
        "retry_count": 0,
        "scope_blocked": False,
        "cache_hit": False,
        "agent_logs": logs,
        "validation_errors": [],
    }
    result = app.invoke(initial_state)
    if result.get("validation_passed") and result.get("final_sql"):
        remember_successful_sql(
            question=user_query,
            sql=result["final_sql"],
            module_ids=list(scope.module_ids),
            allowed_tables=sorted(scope.allowed_tables),
            selected_tables=result.get("selected_tables") or [],
        )
        logs_out = list(result.get("agent_logs") or [])
        logs_out.append(
            "[Query Cache] Stored validated SQL for similar future questions."
        )
        result = {**result, "agent_logs": logs_out, "cache_hit": False}

    query_usage = usage_since(usage_before)
    session_usage = snapshot_token_usage()
    last_call = get_last_token_usage()
    logs_out = list(result.get("agent_logs") or [])
    if query_usage.get("calls"):
        logs_out.append(
            format_token_usage_log(
                {
                    "prompt_tokens": query_usage["prompt_tokens"],
                    "completion_tokens": query_usage["completion_tokens"],
                    "total_tokens": query_usage["total_tokens"],
                    "reported": bool(query_usage["total_tokens"]),
                },
                session_usage,
            )
        )
    else:
        logs_out.append("[Peek tokens] No Peek LiteLLM calls in this run.")
    return {
        **result,
        "agent_logs": logs_out,
        "token_usage": {
            "query": query_usage,
            "session": session_usage,
            "last_call": last_call,
        },
    }
