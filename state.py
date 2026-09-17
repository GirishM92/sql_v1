from typing import Any, TypedDict


class AgentState(TypedDict, total=False):
    user_query: str
    active_modules: list[str]
    allowed_tables: list[str]
    allowed_tools: list[str]
    intent: dict[str, Any]
    scope_blocked: bool
    selected_tables: list[str]
    selected_tools: list[str]
    schema_metadata: dict[str, Any]
    query_plan: str
    generated_sql: str
    validation_errors: list[str]
    validation_passed: bool
    retry_count: int
    final_sql: str
    cache_hit: bool
    cache_similarity: float
    token_usage: dict[str, Any]
    agent_logs: list[str]
