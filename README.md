# Agentic Text-to-SQL (Banking)

Enterprise-grade natural language to SQL with module-scoped access, LangGraph orchestration, and human-in-the-loop execution.

**Architecture:** [docs/architecture.md](docs/architecture.md)

## Quick start (Windows)

```powershell
cd C:\Users\acer\Desktop\agent_to_sql
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
# Edit .env — set OPENAI_API_KEY or use Ollama
python db\schema_seed.py
streamlit run app.py
```

**Flutter / HTTP API** (same product capabilities as Streamlit + CLI):

```powershell
uvicorn api.app:app --reload --host 0.0.0.0 --port 8000
```

Open http://127.0.0.1:8000/docs — see [docs/flutter_api.md](docs/flutter_api.md).

Select **Active modules** in the sidebar before asking a question.

## Features

- **Schema isolation**: LLM only sees metadata for semantically selected tables
- **Multi-agent pipeline**: Intent → Tool Selection → Schema → Planner → Generator → Validator
- **Banking guardrails**: Keyword blocklist, schema containment, EXPLAIN dry-run
- **HITL execution**: SQL is never auto-executed; user must approve with `y`
- **Dual mode**: OpenAI (cloud) or Ollama + DeepSeek (local)
- **Embedding-based table search**: Ollama/OpenAI embeddings with TF-IDF fallback
- **SQL generation**: LangChain `create_sql_query_chain` limited to selected tables only

## Setup

```bash
# 1. Create virtual environment (optional)
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac/Linux

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure environment
cp .env.example .env

# 4. Create the mock bank database
python db/schema_seed.py

# 5. (Optional) Preview database contents
python db/show_summary.py
```

### Local mode (Ollama + DeepSeek)

```bash
ollama pull deepseek-coder:6.7b
ollama pull nomic-embed-text
python main.py --ollama
```

### Cloud mode (OpenAI)

```bash
# Set OPENAI_API_KEY in .env, then:
python main.py --openai
```

## Demo Database

Built from **Customer_Loan detail** plus related banking tables for tool-selection demos:

| Table | Tool (fetch details for SQL gen) | Purpose |
|-------|----------------------------------|---------|
| `branches` | `get_branches_details` | Branch master (city/region) |
| `products` | `get_products_details` | Loan product catalog |
| `customers` | `get_customers_details` | Customer names, city, risk |
| `customer_loans` | `get_customer_loans_details` | Main loan portfolio |
| `loan_payments` | `get_loan_payments_details` | EMI / repayment history |
| `collateral` | `get_collateral_details` | Security / property valuation |

Recreate anytime: `python db/schema_seed.py`

## Run

### Web UI (recommended for demos)

```bash
streamlit run app.py
```

Opens a browser UI with:
- **Ask Question** — natural language to SQL with agent logs and HITL execution
- **Database** — browse `customer_loans` and `customers`
- **Schema Catalog** — view metadata the agents use

Use the sidebar to pick a provider (OpenAI, Peek/HDFC LiteLLM, Groq, Ollama, ...).

### CLI

Choose provider via **CLI flag** (overrides `.env`) or set `LLM_PROVIDER` in `.env`.

```bash
# Ollama (local)
python main.py --ollama

# OpenAI
python main.py --openai

# Peek AI / HDFC LiteLLM (Gemini via bank proxy)
python main.py --peek --modules user_master

# Override model name
python main.py --openai --model gpt-4o
python main.py --peek --model gemini-2.5-flash
```

### Provider config (.env)

| Variable | Values | Default |
|----------|--------|---------|
| `LLM_PROVIDER` | `openai`, `peek`, `ollama`, `groq`, ... | `openai` |
| `PEEK_API_KEY` | HDFC LiteLLM key | empty |
| `PEEK_BASE_URL` | LiteLLM root (no `/chat/completions` required) | HDFC UAT proxy |
| `PEEK_MODEL` / `LLM_MODEL` | deployment name | `gemini-2.5-flash` |
| `PEEK_TEMPERATURE` | LiteLLM temperature | `0.7` |
| `PEEK_MAX_TOKENS` | LiteLLM max tokens | `1000` |

Peek calls HDFC LiteLLM as OpenAI `chat/completions`: `{ model, messages, temperature, max_tokens }`. That is **not** MCP JSON-RPC (`method: tools/call`). MCP is a separate tool host protocol.

`LLM_MODE` still works as an alias. `local` = `ollama`, `cloud` = `openai`, `hdfc` = `peek`.

## Test

```bash
python -m unittest tests.test_guardrails -v
```

## Multi-table tools

```
User sentence
  → match against tool DESCRIPTIONS (embeddings)
  → call selected tools (get_customer_loans_details, get_customers_details, ...)
  → each tool FETCHes FULL table metadata (no create / no execute)
  → planner + create_sql_query_chain generate SQL
  → validator + HITL
```

Add a new table: add it to `metadata/catalog.json` and a matching `get_*_details` tool in `tools/tools_factory.py`.

```
User Query
    → Intent Agent
    → Tool Selection Agent (embedding semantic search)
    → Schema Agent (expose only selected tables)
    → Query Planner Agent
    → SQL Generator Agent
    → SQL Validator Agent ──(retry up to 3x)──→ Generator
    → Return SQL → HITL approval → Execute (read-only)
```

## Example Queries

- "List all active home loans with total outstanding over 1000000"
- "Show secured loans with residential security"
- "Find loans with ROI above 10 percent"
- "Update loan outstanding to zero." → blocked by validator
