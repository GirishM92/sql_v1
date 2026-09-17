import json
from pathlib import Path

import pandas as pd
import streamlit as st

import config
from db.utils import (
    ensure_database,
    execute_select,
    fetch_table,
    get_dashboard_metrics,
    get_table_names,
    get_table_stats,
)
from graph import run_pipeline

DEMO_QUERIES = [
    {
        "text": "List all active home loans with total outstanding over 1000000",
        "modules": ["lending"],
    },
    {
        "text": "Show customers with high risk band in Mumbai",
        "modules": ["customers"],
    },
    {
        "text": "Which branches are in the West region?",
        "modules": ["branch"],
    },
    {
        "text": "Show unsecured loan products and their ROI range",
        "modules": ["product"],
    },
    {
        "text": "Find pending EMI payments",
        "modules": ["lending"],
    },
    {
        "text": "Show collateral valued above 5000000",
        "modules": ["lending"],
    },
    {
        "text": "List loans from Mumbai Andheri branch with customer names",
        "modules": ["lending", "customers", "branch"],
    },
    {
        "text": "Show facility name, account number and total outstanding for pan ASFPD1325C",
        "modules": ["lending", "customers"],
    },
    {
        "text": "Show TOTAL_OS and FB_LIMIT from base report for Mumbai Andheri branch",
        "modules": ["base_report"],
    },
    {
        "text": "List NPA accounts with DPD and CUSTOMER_NAME from base report",
        "modules": ["base_report"],
    },
    {
        "text": "Show CAM_EXPIRY_DATE and INTERNAL_RATING for customers in base report",
        "modules": ["base_report"],
    },
    {
        "text": "Show FD_AMOUNT and COLLATERAL_ID where FD backed exposure is present",
        "modules": ["base_report"],
    },
]


def _active_module_ids() -> list[str]:
    return list(st.session_state.get("active_modules") or [])


def apply_sidebar_config() -> None:
    from tools.modules import list_modules, resolve_scope

    st.sidebar.header("LLM Provider")
    providers = config.list_providers()
    provider_ids = [pid for pid, _ in providers]
    provider_labels = [label for _, label in providers]
    current = config.get_provider()
    try:
        current_index = provider_ids.index(current)
    except ValueError:
        current_index = 0

    selected_label = st.sidebar.selectbox(
        "Select provider",
        options=provider_labels,
        index=current_index,
        help="Any OpenAI-compatible endpoint works via 'OpenAI-compatible API'.",
    )
    selected_id = provider_ids[provider_labels.index(selected_label)]
    config.set_provider(selected_id)
    spec = config.get_provider_spec()

    model = st.sidebar.text_input(
        "Model",
        value=config.get_active_model(),
        key=f"llm_model_{selected_id}",
    )
    if model:
        config.set_model(model)

    show_url = bool(
        spec["requires_base_url"]
        or spec["default_base_url"]
        or selected_id in ("openai_compatible", "azure", "ollama", "peek")
    )
    if show_url:
        base_url = st.sidebar.text_input(
            "Base URL",
            value=config.get_base_url(),
            placeholder="https://api.example.com/v1",
            key=f"llm_base_url_{selected_id}",
        )
        config.set_base_url(base_url)

    if spec["requires_api_key"] or selected_id == "openai_compatible":
        api_key = st.sidebar.text_input(
            "API key",
            type="password",
            placeholder="Uses .env if left blank",
            key=f"llm_api_key_{selected_id}",
        )
        if api_key:
            config.set_api_key(api_key)

    if not spec.get("supports_embeddings"):
        st.sidebar.caption(
            "This provider has no embeddings API. Tool matching uses TF-IDF "
            "unless you set EMBEDDING_PROVIDER=openai or ollama in .env."
        )

    if selected_id == "peek":
        st.sidebar.markdown("### 🔵 Peek Token Usage")

        from llm import get_token_usage_summary, reset_token_usage

        summary = get_token_usage_summary()
        session = summary.get("session") or {}

        with st.sidebar.container():
            st.markdown(
                """
        <style>
        .token-card {
            padding: 12px 15px;
            background-color: #f7f9fc;
            border-radius: 8px;
            border: 1px solid #dce3ec;
            margin-bottom: 10px;
        }
        .token-title {
            font-size: 15px;
            font-weight: 600;
            color: #1f4e79;
            margin-bottom: 8px;
        }
        .token-row {
            display: flex;
            justify-content: space-between;
            font-size: 14px;
            padding: 3px 0px;
        }
        .token-label {
            font-weight: 500;
            color: #444;
        }
        .token-value {
            font-weight: 600;
            color: #000;
        }
        </style>
        """,
                unsafe_allow_html=True,
            )

            st.markdown('<div class="token-card">', unsafe_allow_html=True)
            st.markdown(
                '<div class="token-title">Session Token Summary</div>',
                unsafe_allow_html=True,
            )

            st.markdown(
                f"""
            <div class="token-row"><span class="token-label">Prompt tokens:</span>
            <span class="token-value">{session.get('prompt_tokens', 0):,}</span></div>

            <div class="token-row"><span class="token-label">Completion tokens:</span>
            <span class="token-value">{session.get('completion_tokens', 0):,}</span></div>

            <div class="token-row"><span class="token-label">Total tokens:</span>
            <span class="token-value">{session.get('total_tokens', 0):,}</span></div>

            <div class="token-row"><span class="token-label">LLM calls:</span>
            <span class="token-value">{session.get('calls', 0):,}</span></div>
            """,
                unsafe_allow_html=True,
            )
            st.markdown("</div>", unsafe_allow_html=True)

        if st.sidebar.button("Reset token counters"):
            reset_token_usage()
            st.rerun()

    st.sidebar.divider()
    st.sidebar.header("Active modules")
    modules = list_modules()
    options = {f"{m['name']} ({m['id']})": m["id"] for m in modules}
    labels = list(options.keys())
    default_labels = [
        lab
        for lab, mid in options.items()
        if mid in (st.session_state.get("active_modules") or [])
    ]
    selected_labels = st.sidebar.multiselect(
        "Modules for this session",
        options=labels,
        default=default_labels,
        help="Only tables/tools in these modules are available to the agent.",
    )
    st.session_state["active_modules"] = [options[lab] for lab in selected_labels]

    if st.session_state["active_modules"]:
        try:
            scope = resolve_scope(st.session_state["active_modules"])
            st.sidebar.caption(
                f"{len(scope.allowed_tables)} table(s), "
                f"{len(scope.allowed_tool_names)} tool(s) in scope"
            )
            st.sidebar.write(", ".join(sorted(scope.allowed_tables)))
        except Exception as exc:
            st.sidebar.warning(str(exc))
    else:
        st.sidebar.warning("Select at least one module to generate SQL.")

    st.sidebar.divider()
    st.sidebar.header("Database")
    ensure_database()
    st.sidebar.caption(f"`{config.DB_PATH}`")

    if st.sidebar.button("Reseed demo database"):
        from db.schema_seed import seed_database
        from tools.table_tools import clear_tool_cache

        seed_database()
        clear_tool_cache()
        st.sidebar.success("Database reseeded (demo + Base Report table).")
        st.rerun()

    scope_tables = set()
    if st.session_state.get("active_modules"):
        try:
            scope_tables = resolve_scope(st.session_state["active_modules"]).allowed_tables
        except Exception:
            scope_tables = set()
    for stat in get_table_stats():
        if scope_tables and stat["table"] not in scope_tables:
            continue
        st.sidebar.metric(label=stat["table"], value=stat["rows"])


def render_header() -> None:
    st.title("HDFC Bank")
    st.caption("Agentic Text-to-SQL — natural language queries with schema guardrails")
    metrics = get_dashboard_metrics()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Customers", metrics["customers"])
    c2.metric("Active Loans", metrics["active_loans"])
    c3.metric("Total Outstanding", f"{metrics['total_outstanding']:,.0f}")
    c4.metric("Branches", metrics["branches"])


def render_query_tab() -> None:
    from tools.modules import resolve_scope

    if "last_result" not in st.session_state:
        st.session_state["last_result"] = None

    st.subheader("Ask a Question")
    st.write(f"Provider: **{config.get_mode_label()}**")

    module_ids = _active_module_ids()
    if not module_ids:
        st.warning("Select one or more **Active modules** in the sidebar first.")
        return

    try:
        scope = resolve_scope(module_ids)
    except Exception as exc:
        st.error(str(exc))
        return

    st.info(
        f"Module scope: `{', '.join(module_ids)}` → "
        f"tables `{', '.join(sorted(scope.allowed_tables))}` "
        f"({len(scope.allowed_tool_names)} tools)"
    )

    scoped_examples = [
        q["text"]
        for q in DEMO_QUERIES
        if set(q["modules"]).issubset(set(module_ids))
        or set(module_ids).intersection(q["modules"])
    ]
    # Prefer examples fully covered by selected modules
    preferred = [
        q["text"]
        for q in DEMO_QUERIES
        if set(q["modules"]).issubset(set(module_ids))
    ]
    example_list = preferred or scoped_examples

    selected_example = st.selectbox(
        "Example questions (filtered by modules)",
        ["— choose an example —"] + example_list,
    )

    default_query = "" if selected_example.startswith("—") else selected_example
    user_query = st.text_area(
        "Your question",
        value=default_query,
        height=100,
        placeholder="e.g. Show home loans with total outstanding over 1000000",
    )

    can_run = bool(user_query.strip()) and bool(module_ids)
    if st.button("Generate SQL", type="primary", disabled=not can_run):
        with st.spinner("Running agent pipeline..."):
            try:
                result = run_pipeline(user_query.strip(), module_ids=module_ids)
                st.session_state["last_result"] = result
                if "last_results_df" in st.session_state:
                    del st.session_state["last_results_df"]
            except Exception as exc:
                st.error(f"Pipeline error: {exc}")
                return

    result = st.session_state.get("last_result")
    if not result:
        return

    with st.expander("Agent logs", expanded=True):
        for log in result.get("agent_logs", []):
            if "FAILED" in log or "Blocked" in log:
                st.error(log)
            elif (
                "PASSED" in log
                or "Selected tables" in log
                or "Module Scope" in log
                or "Query Cache] HIT" in log
            ):
                st.success(log)
            else:
                st.info(log)

    usage = result.get("token_usage") or {}
    query_usage = usage.get("query") or {}
    if query_usage:
        st.subheader("Peek token usage")
        u1, u2, u3, u4 = st.columns(4)
        u1.metric("Prompt tokens", query_usage.get("prompt_tokens", 0))
        u2.metric("Completion tokens", query_usage.get("completion_tokens", 0))
        u3.metric("Total tokens", query_usage.get("total_tokens", 0))
        u4.metric("LLM calls", query_usage.get("calls", 0))
        session_usage = usage.get("session") or {}
        if session_usage:
            st.caption(
                f"Session total: {session_usage.get('total_tokens', 0)} tokens "
                f"across {session_usage.get('calls', 0)} call(s)"
            )

    if result.get("active_modules"):
        st.write("**Modules:**", ", ".join(result["active_modules"]))
    if result.get("allowed_tables"):
        st.write("**Allowed tables:**", ", ".join(result["allowed_tables"]))

    if result.get("intent"):
        with st.expander("Parsed intent"):
            st.json(result["intent"])

    if result.get("selected_tables"):
        st.write("**Selected tables:**", ", ".join(result["selected_tables"]))

    if result.get("query_plan"):
        with st.expander("Query plan"):
            st.markdown(result["query_plan"])

    if result.get("cache_hit"):
        st.success(
            f"Answered from query cache (similarity {result.get('cache_similarity', 0):.0%}) "
            "— LLM generation skipped."
        )

    if result.get("validation_passed") and result.get("final_sql"):
        st.success("SQL validated successfully")
        st.code(result["final_sql"], language="sql")

        st.divider()
        st.subheader("Human-in-the-loop execution")
        approved = st.checkbox("I approve executing this read-only SELECT query")
        if st.button("Execute Query", disabled=not approved):
            try:
                df = execute_select(
                    result["final_sql"],
                    allowed_tables=result.get("allowed_tables")
                    or list(scope.allowed_tables),
                )
                st.session_state["last_results_df"] = df
                st.success(f"Query returned {len(df)} row(s)")
            except Exception as exc:
                st.error(f"Execution error: {exc}")

        if "last_results_df" in st.session_state:
            st.dataframe(st.session_state["last_results_df"], width="stretch")
    else:
        if result.get("scope_blocked"):
            st.error("Out of module scope — enable the required module(s) and retry.")
        else:
            st.error("Could not generate a valid SQL query")
        for err in result.get("validation_errors", []):
            st.warning(err)


def render_database_tab() -> None:
    from tools.modules import resolve_scope

    st.subheader("Database Explorer")
    tables = get_table_names()
    module_ids = _active_module_ids()
    if module_ids:
        try:
            allowed = resolve_scope(module_ids).allowed_tables
            tables = [t for t in tables if t in allowed]
            st.caption("Filtered to active module tables.")
        except Exception as exc:
            st.warning(str(exc))

    if not tables:
        st.info("No tables in the current module scope.")
        return

    selected = st.selectbox("Table", tables)
    limit = st.slider("Max rows", min_value=10, max_value=500, value=100, step=10)

    if selected:
        df = fetch_table(selected, limit=limit)
        st.dataframe(df, width="stretch")
        st.caption(f"Showing up to {limit} rows from `{selected}`")


def render_tools_tab() -> None:
    st.subheader("Registered Tools")
    st.caption(
        "Built-in and custom tools. Discovery matches your sentence to descriptions; "
        "calling a tool returns column metadata for SQL generation."
    )
    from tools.dynamic_tools import list_dynamic_tool_defs
    from tools.modules import resolve_scope
    from tools.table_tools import clear_tool_cache, list_available_tools

    clear_tool_cache()

    allowed_tools = None
    module_ids = _active_module_ids()
    if module_ids:
        try:
            allowed_tools = resolve_scope(module_ids).allowed_tool_names
            st.caption(
                f"Showing tools for modules: {', '.join(module_ids)} "
                f"({len(allowed_tools)} tools)"
            )
        except Exception as exc:
            st.warning(str(exc))

    dyn_list = list_dynamic_tool_defs()
    dyn_defs = {t["name"]: t for t in dyn_list}
    dyn_names = set(dyn_defs)

    registry = list_available_tools(allowed_tools=allowed_tools)
    for entry in registry:
        entry["source"] = "custom" if entry["name"] in dyn_names else "builtin"

    builtin = [e for e in registry if e["name"] not in dyn_names]
    custom_entries = []
    for tdef in dyn_list:
        if allowed_tools is not None and tdef["name"] not in allowed_tools:
            continue
        match = next((e for e in registry if e["name"] == tdef["name"]), None)
        custom_entries.append(
            {
                "name": tdef["name"],
                "description": tdef.get("description")
                or (match["description"] if match else ""),
                "table": tdef.get("table_name")
                or (match["table"] if match else ""),
                "source": "custom",
                "enabled": tdef.get("enabled", True),
                "column_names": tdef.get("column_names") or [],
                "sample_intents": tdef.get("sample_intents") or [],
            }
        )

    c1, c2 = st.columns(2)
    c1.metric("Built-in tools", len(builtin))
    c2.metric("Custom tools", len(custom_entries))

    st.markdown("### Custom tools")
    if not custom_entries:
        st.info("No custom tools yet (or none in the active module scope).")
    else:
        for entry in custom_entries:
            status = "enabled" if entry.get("enabled", True) else "disabled"
            with st.expander(
                f"[CUSTOM] {entry['name']}  →  `{entry['table']}`  ({status})",
                expanded=True,
            ):
                st.markdown("**Type:** Custom (user-created)")
                st.write(entry["description"])
                cols = entry.get("column_names") or []
                if cols:
                    st.markdown("**Columns:** " + ", ".join(f"`{c}`" for c in cols))
                else:
                    st.markdown("**Columns:** all columns from table")
                intents = entry.get("sample_intents") or []
                if intents:
                    st.markdown("**Example questions:**")
                    for intent in intents:
                        st.write(f"- {intent}")

    st.markdown("### Built-in tools")
    if not builtin:
        st.info("No built-in tools in the active module scope.")
    for entry in builtin:
        with st.expander(
            f"{entry['name']}  →  table `{entry['table']}`",
            expanded=False,
        ):
            st.markdown("**Type:** Built-in")
            st.write(entry["description"])
            st.caption(
                "Description is used for matching. "
                "Full columns are returned only when this tool is called."
            )


def render_create_tool_tab() -> None:
    """No-code dynamic tool builder."""
    st.subheader("Create Tool (No-Code)")
    st.caption(
        "Define a new get_*_details tool: write a description, pick a table and columns. "
        "The agent will discover it from the description and use those columns for SQL."
    )

    from tools.dynamic_tools import (
        create_dynamic_tool,
        delete_dynamic_tool,
        list_dynamic_tool_defs,
        list_table_columns,
        normalize_tool_name,
        update_dynamic_tool,
    )
    from tools.modules import modules_for_table
    from tools.table_tools import clear_tool_cache
    from db.utils import ensure_database, get_table_names

    ensure_database()
    tables = get_table_names()

    st.markdown("### New tool")

    table_name = st.selectbox("Source table", tables, key="create_tool_table")
    owning = modules_for_table(table_name) if table_name else []
    if owning:
        st.caption(
            "Included in modules: "
            + ", ".join(f"`{m['name']}` ({m['id']})" for m in owning)
        )
    else:
        st.caption("No module currently includes this table — add it under **Modules**.")

    all_cols = list_table_columns(table_name) if table_name else []
    col_options = [c["name"] for c in all_cols]
    if col_options:
        st.caption(f"{len(col_options)} columns available in `{table_name}`")
    else:
        st.warning(f"No columns found for `{table_name}`.")

    selected_cols = st.multiselect(
        "Columns (leave empty = all columns)",
        options=col_options,
        default=[],
        key=f"create_tool_cols_{table_name}",
    )

    with st.form("create_tool_form", clear_on_submit=True):
        raw_name = st.text_input(
            "Tool name",
            placeholder="my_exposure_slice",
            help="Will be normalized to get_*_details",
        )
        description = st.text_area(
            "Description (used for matching user questions)",
            height=120,
            placeholder=(
                "Fetch customer exposure and outstanding columns for questions about "
                "TOTAL_OS, FB_LIMIT, NPA, or branch-level exposure reporting."
            ),
        )
        sample_intents = st.text_area(
            "Example questions (one per line, optional)",
            height=80,
            placeholder="Show TOTAL_OS by branch\nList NPA accounts",
        )
        st.caption(
            f"Will use table `{table_name}` with "
            f"{len(selected_cols) if selected_cols else 'all'} column(s)."
        )
        submitted = st.form_submit_button("Create tool", type="primary")

    if submitted:
        intents = [line.strip() for line in sample_intents.splitlines() if line.strip()]
        try:
            preview = normalize_tool_name(raw_name)
            created = create_dynamic_tool(
                name=raw_name,
                description=description,
                table_name=table_name,
                column_names=selected_cols,
                sample_intents=intents,
            )
            clear_tool_cache()
            st.success(f"Created tool `{created['name']}` (from `{preview}`)")
            st.info("Open the **Tools** tab to see it under **Custom tools**.")
            st.json(created)
        except Exception as exc:
            st.error(str(exc))

    st.divider()
    st.markdown("### Your tools")
    user_tools = list_dynamic_tool_defs()
    if not user_tools:
        st.info("No user-created tools yet.")
        return

    for tool in user_tools:
        with st.expander(f"{tool['name']}  →  `{tool['table_name']}`", expanded=False):
            st.write(tool.get("description", ""))
            cols = tool.get("column_names") or []
            st.caption(
                f"Columns: {', '.join(cols) if cols else '(all columns)'} | "
                f"Enabled: {tool.get('enabled', True)}"
            )
            c1, c2 = st.columns(2)
            with c1:
                if st.button(
                    "Disable" if tool.get("enabled", True) else "Enable",
                    key=f"toggle_{tool['name']}",
                ):
                    update_dynamic_tool(
                        tool["name"],
                        enabled=not tool.get("enabled", True),
                    )
                    clear_tool_cache()
                    st.rerun()
            with c2:
                if st.button("Delete", key=f"del_{tool['name']}"):
                    delete_dynamic_tool(tool["name"])
                    clear_tool_cache()
                    st.rerun()


def render_create_table_tab() -> None:
    """Create a SQLite table from pasted/uploaded schema JSON (demo)."""
    st.subheader("Create Table from Schema")
    st.caption(
        "Paste or upload a JSON schema → preview DDL → create the table, "
        "register it in the catalog, and optionally attach a tool / module. "
        "Set **`dialect`** in JSON to `sql` (SQLite), `oracle`, or `tsql` — "
        "this controls generated SQL language. Local demo storage remains SQLite."
    )

    from db.custom_tables import (
        SAMPLE_SCHEMA,
        create_table_from_schema,
        delete_custom_table,
        list_custom_table_defs,
        preview_ddl,
        validate_schema_payload,
    )
    from db.utils import ensure_database, get_allowed_tables
    from tools.modules import list_modules
    from tools.table_tools import clear_tool_cache

    ensure_database()

    st.markdown("### Schema JSON")
    _, c_sample = st.columns([3, 1])
    with c_sample:
        if st.button("Load sample watchlist schema"):
            st.session_state["create_table_json"] = json.dumps(SAMPLE_SCHEMA, indent=2)
            st.rerun()

    uploaded = st.file_uploader(
        "Or upload a .json schema file",
        type=["json"],
        key="create_table_upload",
    )
    if uploaded is not None:
        try:
            st.session_state["create_table_json"] = uploaded.read().decode("utf-8")
            st.rerun()
        except Exception as exc:
            st.error(f"Could not read upload: {exc}")

    if "create_table_json" not in st.session_state:
        st.session_state["create_table_json"] = json.dumps(
            {
                "table_name": "tbl_My_Custom_Table",
                "dialect": "sql",
                "description": "Short description of this table.",
                "columns": [
                    {"name": "CUST_ID", "type": "TEXT", "description": "Customer id"},
                    {"name": "AMOUNT", "type": "REAL", "description": "Amount"},
                ],
                "sample_rows": [{"CUST_ID": "1001", "AMOUNT": 1000.0}],
                "create_tool": True,
                "create_module": {
                    "id": "my_custom",
                    "name": "My Custom",
                    "description": "Module for the custom table.",
                },
            },
            indent=2,
        )

    raw = st.text_area(
        "Schema",
        height=360,
        key="create_table_json",
    )

    replace = st.checkbox("Replace if table already exists", value=False)
    modules = list_modules()
    module_ids = [""] + [m["id"] for m in modules]
    add_to = st.selectbox(
        "Or add table to existing module (optional)",
        options=module_ids,
        format_func=lambda x: "(none)" if not x else x,
        key="create_table_add_module",
    )

    b1, b2 = st.columns(2)
    with b1:
        preview_clicked = st.button("Preview DDL")
    with b2:
        create_clicked = st.button("Create table", type="primary")

    payload: dict | None = None
    try:
        payload = json.loads(raw)
        if not isinstance(payload, dict):
            raise ValueError("Schema root must be a JSON object.")
    except Exception as exc:
        if preview_clicked or create_clicked:
            st.error(f"Invalid JSON: {exc}")
        payload = None

    if payload is not None and add_to:
        payload = dict(payload)
        payload["add_to_module"] = add_to
        payload.pop("create_module", None)

    if payload is not None and preview_clicked:
        errors = validate_schema_payload(payload)
        if errors:
            for e in errors:
                st.error(e)
        else:
            try:
                st.code(preview_ddl(payload), language="sql")
            except Exception as exc:
                st.error(str(exc))

    if payload is not None and create_clicked:
        errors = validate_schema_payload(payload)
        if errors:
            for e in errors:
                st.error(e)
        else:
            try:
                result = create_table_from_schema(
                    payload, replace_if_exists=replace
                )
                clear_tool_cache()
                st.success(
                    f"Created table `{result['table']['table_name']}` "
                    f"(dialect=`{result['table'].get('dialect', 'sqlite')}`, "
                    f"{result['rows_inserted']} sample row(s))."
                )
                st.code(result["ddl"], language="sql")
                if result.get("tool"):
                    st.info(f"Created tool {result['tool']['name']}")
                if result.get("module"):
                    st.info(
                        f"Module {result['module']['id']} — "
                        "select it under **Active modules** in the sidebar."
                    )
                st.caption(
                    "Allowlist now includes: "
                    + ", ".join(f"{t}" for t in get_allowed_tables()[-5:])
                )
            except Exception as exc:
                st.error(str(exc))

    st.divider()
    st.markdown("### Registered custom tables")
    defs = list_custom_table_defs()
    if not defs:
        st.info("No custom tables yet.")
        return
    for tdef in defs:
        tname = tdef.get("table_name")
        with st.expander(f"{tname}", expanded=False):
            st.write(tdef.get("description") or "")
            st.caption(
                f"dialect=`{tdef.get('dialect') or 'sqlite'}` · "
                f"{len(tdef.get('columns') or [])} columns · "
                f"updated {tdef.get('updated_at') or '—'}"
            )
            if st.button("Delete table", key=f"del_ct_{tname}"):
                try:
                    delete_custom_table(str(tname), drop_from_db=True)
                    clear_tool_cache()
                    st.rerun()
                except Exception as exc:
                    st.error(str(exc))


def render_modules_tab() -> None:
    st.subheader("Modules")
    st.caption(
        "A module is a named bundle of tables. Selecting a module in the sidebar "
        "limits the agent to those tables and their related tools."
    )
    from tools.modules import (
        delete_module,
        list_modules,
        upsert_module,
    )
    from db.utils import ensure_database, get_allowed_tables

    ensure_database()
    available_tables = list(get_allowed_tables())

    st.markdown("### Create / update module")
    with st.form("module_form", clear_on_submit=False):
        mid = st.text_input("Module id", placeholder="branch")
        name = st.text_input("Display name", placeholder="Branch")
        description = st.text_area("Description", height=80)
        tables = st.multiselect("Tables", options=available_tables)
        submitted = st.form_submit_button("Save module", type="primary")

    if submitted:
        try:
            saved = upsert_module(
                module_id=mid,
                name=name,
                description=description,
                tables=tables,
            )
            st.success(f"Saved module `{saved['id']}`")
            st.rerun()
        except Exception as exc:
            st.error(str(exc))

    st.divider()
    st.markdown("### Existing modules")
    for mod in list_modules():
        with st.expander(f"{mod['name']} (`{mod['id']}`)", expanded=False):
            st.write(mod.get("description") or "")
            st.markdown(
                "**Tables:** " + ", ".join(f"`{t}`" for t in (mod.get("tables") or []))
            )
            explicit = mod.get("tool_names") or []
            if explicit:
                st.markdown(
                    "**Tool allowlist:** " + ", ".join(f"`{t}`" for t in explicit)
                )
            if st.button("Delete", key=f"del_mod_{mod['id']}"):
                try:
                    delete_module(mod["id"])
                    st.rerun()
                except Exception as exc:
                    st.error(str(exc))


def render_schema_tab() -> None:
    from tools.modules import resolve_scope
    from tools.table_tools import load_catalog

    st.subheader("Schema Catalog")
    st.caption("Metadata knowledge base used by agents (full catalog never sent to LLM)")

    catalog = load_catalog()
    keys = list(catalog.keys())
    module_ids = _active_module_ids()
    if module_ids:
        try:
            allowed = resolve_scope(module_ids).allowed_tables
            keys = [
                k
                for k in keys
                if k in allowed or catalog[k].get("table_name") in allowed
            ]
            st.caption("Filtered to active module tables / slices.")
        except Exception as exc:
            st.warning(str(exc))

    if not keys:
        st.info("No catalog entries in the current module scope.")
        return

    table = st.selectbox("Table metadata", keys)
    if table:
        meta = catalog[table]
        st.markdown(f"**Description:** {meta.get('description', '')}")

        columns = meta.get("columns", [])
        if columns:
            st.markdown("**Columns**")
            st.dataframe(pd.DataFrame(columns), width="stretch")

        relationships = meta.get("relationships", [])
        if relationships:
            st.markdown("**Relationships**")
            for rel in relationships:
                st.write(f"- {rel.get('description', '')}")

        samples = meta.get("sample_queries", [])
        if samples:
            st.markdown("**Sample queries**")
            for sample in samples:
                st.code(sample.get("sql", ""), language="sql")


def main() -> None:
    st.set_page_config(
        page_title="HDFC Bank | Text-to-SQL",
        page_icon="MTB",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    if "active_modules" not in st.session_state:
        st.session_state["active_modules"] = []

    apply_sidebar_config()
    render_header()

    (
        tab_query,
        tab_db,
        tab_tools,
        tab_create,
        tab_table,
        tab_modules,
        tab_schema,
    ) = st.tabs(
        [
            "Ask Question",
            "Database",
            "Tools",
            "Create Tool",
            "Create Table",
            "Modules",
            "Schema Catalog",
        ]
    )

    with tab_query:
        render_query_tab()
    with tab_db:
        render_database_tab()
    with tab_tools:
        render_tools_tab()
    with tab_create:
        render_create_tool_tab()
    with tab_table:
        render_create_table_tab()
    with tab_modules:
        render_modules_tab()
    with tab_schema:
        render_schema_tab()


if __name__ == "__main__":
    main()
