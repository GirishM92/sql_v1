import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "db" / "bank_enterprise.db"
CATALOG_PATH = BASE_DIR / "metadata" / "catalog.json"
DYNAMIC_TOOLS_PATH = BASE_DIR / "metadata" / "dynamic_tools.json"
MODULES_PATH = BASE_DIR / "metadata" / "modules.json"
CUSTOM_TABLES_PATH = BASE_DIR / "metadata" / "custom_tables.json"
QUERY_CACHE_PATH = BASE_DIR / "metadata" / "query_cache.json"

# Chat backends used by llm.py
CHAT_OLLAMA = "ollama"
CHAT_OPENAI = "openai"
CHAT_ANTHROPIC = "anthropic"
CHAT_GOOGLE = "google"
CHAT_AZURE = "azure"
CHAT_PEEK = "peek"

# Provider registry. chat_backend tells llm.py which LangChain client to build.
# OpenAI-compatible vendors reuse ChatOpenAI + a default base URL (no extra package).
PROVIDER_SPECS: dict[str, dict] = {
    "ollama": {
        "label": "Ollama (Local)",
        "default_model": "llama3:latest",
        "default_embed_model": "nomic-embed-text",
        "chat_backend": CHAT_OLLAMA,
        "default_base_url": "http://localhost:11434",
        "api_key_env": (),
        "aliases": ("local",),
        "requires_api_key": False,
        "requires_base_url": False,
        "supports_embeddings": True,
    },
    "openai": {
        "label": "OpenAI",
        "default_model": "gpt-4o-mini",
        "default_embed_model": "text-embedding-3-small",
        "chat_backend": CHAT_OPENAI,
        "default_base_url": "",
        "api_key_env": ("OPENAI_API_KEY",),
        "aliases": ("cloud",),
        "requires_api_key": True,
        "requires_base_url": False,
        "supports_embeddings": True,
    },
    "anthropic": {
        "label": "Anthropic (Claude)",
        "default_model": "claude-3-5-sonnet-latest",
        "default_embed_model": None,
        "chat_backend": CHAT_ANTHROPIC,
        "default_base_url": "",
        "api_key_env": ("ANTHROPIC_API_KEY",),
        "aliases": ("claude",),
        "requires_api_key": True,
        "requires_base_url": False,
        "supports_embeddings": False,
    },
    "google": {
        "label": "Google (Gemini)",
        "default_model": "gemini-2.0-flash",
        "default_embed_model": "models/text-embedding-004",
        "chat_backend": CHAT_GOOGLE,
        "default_base_url": "",
        "api_key_env": ("GOOGLE_API_KEY", "GEMINI_API_KEY"),
        "aliases": ("gemini", "google_genai"),
        "requires_api_key": True,
        "requires_base_url": False,
        "supports_embeddings": True,
    },
    "azure": {
        "label": "Azure OpenAI",
        "default_model": "gpt-4o-mini",
        "default_embed_model": "text-embedding-3-small",
        "chat_backend": CHAT_AZURE,
        "default_base_url": "",
        "api_key_env": ("AZURE_OPENAI_API_KEY", "OPENAI_API_KEY"),
        "aliases": ("azure_openai",),
        "requires_api_key": True,
        "requires_base_url": True,
        "supports_embeddings": True,
    },
    "groq": {
        "label": "Groq",
        "default_model": "llama-3.3-70b-versatile",
        "default_embed_model": None,
        "chat_backend": CHAT_OPENAI,
        "default_base_url": "https://api.groq.com/openai/v1",
        "api_key_env": ("GROQ_API_KEY",),
        "aliases": (),
        "requires_api_key": True,
        "requires_base_url": False,
        "supports_embeddings": False,
    },
    "openrouter": {
        "label": "OpenRouter",
        "default_model": "openai/gpt-4o-mini",
        "default_embed_model": "openai/text-embedding-3-small",
        "chat_backend": CHAT_OPENAI,
        "default_base_url": "https://openrouter.ai/api/v1",
        "api_key_env": ("OPENROUTER_API_KEY", "LLM_API_KEY"),
        "aliases": (),
        "requires_api_key": True,
        "requires_base_url": False,
        "supports_embeddings": True,
    },
    "together": {
        "label": "Together AI",
        "default_model": "meta-llama/Llama-3.3-70B-Instruct-Turbo",
        "default_embed_model": "togethercomputer/m2-bert-80M-8k-retrieval",
        "chat_backend": CHAT_OPENAI,
        "default_base_url": "https://api.together.xyz/v1",
        "api_key_env": ("TOGETHER_API_KEY",),
        "aliases": (),
        "requires_api_key": True,
        "requires_base_url": False,
        "supports_embeddings": True,
    },
    "deepseek": {
        "label": "DeepSeek",
        "default_model": "deepseek-chat",
        "default_embed_model": None,
        "chat_backend": CHAT_OPENAI,
        "default_base_url": "https://api.deepseek.com/v1",
        "api_key_env": ("DEEPSEEK_API_KEY",),
        "aliases": (),
        "requires_api_key": True,
        "requires_base_url": False,
        "supports_embeddings": False,
    },
    "mistral": {
        "label": "Mistral AI",
        "default_model": "mistral-large-latest",
        "default_embed_model": "mistral-embed",
        "chat_backend": CHAT_OPENAI,
        "default_base_url": "https://api.mistral.ai/v1",
        "api_key_env": ("MISTRAL_API_KEY",),
        "aliases": (),
        "requires_api_key": True,
        "requires_base_url": False,
        "supports_embeddings": True,
    },
    "xai": {
        "label": "xAI (Grok)",
        "default_model": "grok-2-latest",
        "default_embed_model": None,
        "chat_backend": CHAT_OPENAI,
        "default_base_url": "https://api.x.ai/v1",
        "api_key_env": ("XAI_API_KEY",),
        "aliases": ("grok",),
        "requires_api_key": True,
        "requires_base_url": False,
        "supports_embeddings": False,
    },
    "fireworks": {
        "label": "Fireworks",
        "default_model": "accounts/fireworks/models/llama-v3p3-70b-instruct",
        "default_embed_model": None,
        "chat_backend": CHAT_OPENAI,
        "default_base_url": "https://api.fireworks.ai/inference/v1",
        "api_key_env": ("FIREWORKS_API_KEY",),
        "aliases": (),
        "requires_api_key": True,
        "requires_base_url": False,
        "supports_embeddings": False,
    },
    "openai_compatible": {
        "label": "OpenAI-compatible API",
        "default_model": "gpt-4o-mini",
        "default_embed_model": "text-embedding-3-small",
        "chat_backend": CHAT_OPENAI,
        "default_base_url": "",
        "api_key_env": ("LLM_API_KEY", "OPENAI_API_KEY"),
        "aliases": ("compatible", "vllm", "lmstudio", "llamacpp"),
        "requires_api_key": False,
        "requires_base_url": True,
        "supports_embeddings": True,
    },
    "peek": {
        "label": "Peek AI (HDFC LiteLLM)",
        "default_model": "gemini-2.5-flash",
        "default_embed_model": None,
        "chat_backend": CHAT_PEEK,
        "default_base_url": "https://genai-proxy-uat.hdfcbankuat.com/UAT/litellm",
        "api_key_env": ("PEEK_API_KEY", "LLM_API_KEY"),
        "aliases": ("hdfc", "litellm"),
        "requires_api_key": True,
        "requires_base_url": False,
        "supports_embeddings": False,
    },
}

_ALIAS_TO_PROVIDER: dict[str, str] = {}
for _pid, _spec in PROVIDER_SPECS.items():
    _ALIAS_TO_PROVIDER[_pid] = _pid
    for _alias in _spec["aliases"]:
        _ALIAS_TO_PROVIDER[_alias] = _pid

# Backward-compatible env names still work; LLM_* is the canonical override.
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "deepseek-coder:6.7b")
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")

AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-08-01-preview")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "")

PEEK_VERIFY_SSL = os.getenv("PEEK_VERIFY_SSL", "false").lower() in ("1", "true", "yes")

# Use LangChain create_sql_query_chain for SQL generation (with selected tables only)
USE_SQL_QUERY_CHAIN = os.getenv("USE_SQL_QUERY_CHAIN", "true").lower() in (
    "1",
    "true",
    "yes",
)

MAX_RETRY_ATTEMPTS = 3
SEMANTIC_SEARCH_TOP_K = 3
EMBEDDING_SIMILARITY_THRESHOLD = float(os.getenv("EMBEDDING_SIMILARITY_THRESHOLD", "0.25"))

# Semantic SQL cache: skip LLM when a similar in-scope question already succeeded
QUERY_CACHE_ENABLED = os.getenv("QUERY_CACHE_ENABLED", "true").lower() in (
    "1",
    "true",
    "yes",
)
QUERY_CACHE_SIMILARITY_THRESHOLD = float(
    os.getenv("QUERY_CACHE_SIMILARITY_THRESHOLD", "0.90")
)

BLOCKED_SQL_KEYWORDS = frozenset(
    {
        "DROP",
        "DELETE",
        "INSERT",
        "UPDATE",
        "ALTER",
        "TRUNCATE",
        "GRANT",
        "REVOKE",
        "REPLACE",
        "CREATE",
        "ATTACH",
        "DETACH",
        "PRAGMA",
        "VACUUM",
    }
)

ALLOWED_SQL_KEYWORDS = frozenset(
    {
        "SELECT",
        "FROM",
        "WHERE",
        "JOIN",
        "INNER",
        "LEFT",
        "RIGHT",
        "OUTER",
        "ON",
        "GROUP",
        "BY",
        "ORDER",
        "HAVING",
        "LIMIT",
        "OFFSET",
        "AS",
        "AND",
        "OR",
        "NOT",
        "IN",
        "LIKE",
        "BETWEEN",
        "IS",
        "NULL",
        "DISTINCT",
        "ASC",
        "DESC",
        "CASE",
        "WHEN",
        "THEN",
        "ELSE",
        "END",
        "COUNT",
        "SUM",
        "AVG",
        "MIN",
        "MAX",
        "CAST",
        "COALESCE",
    }
)

# Runtime provider state (CLI / UI can override env)
PROVIDER = "openai"
ACTIVE_MODEL: str | None = None
_API_KEY_OVERRIDE: str | None = None
_BASE_URL_OVERRIDE: str | None = None
# Alias kept so older code that reads MODE still works
MODE = PROVIDER


def list_providers() -> list[tuple[str, str]]:
    """Return (id, label) pairs in UI order."""
    return [(pid, spec["label"]) for pid, spec in PROVIDER_SPECS.items()]


def known_provider_ids() -> tuple[str, ...]:
    return tuple(PROVIDER_SPECS.keys())


def normalize_provider(value: str) -> str:
    key = value.lower().strip()
    provider = _ALIAS_TO_PROVIDER.get(key)
    if not provider:
        known = ", ".join(PROVIDER_SPECS)
        raise ValueError(
            f"Invalid LLM provider '{value}'. Use one of: {known} "
            "(aliases: local=ollama, cloud=openai, claude=anthropic, "
            "gemini=google, compatible=openai_compatible, hdfc=peek)."
        )
    return provider


def normalize_mode(value: str) -> str:
    """Backward-compatible alias for normalize_provider."""
    return normalize_provider(value)


def get_provider_spec(provider: str | None = None) -> dict:
    return PROVIDER_SPECS[provider or PROVIDER]


def set_provider(provider: str) -> None:
    """Set the active LLM provider at runtime."""
    global PROVIDER, MODE, ACTIVE_MODEL, _API_KEY_OVERRIDE, _BASE_URL_OVERRIDE
    new = normalize_provider(provider)
    if new != PROVIDER:
        ACTIVE_MODEL = None
        _API_KEY_OVERRIDE = None
        _BASE_URL_OVERRIDE = None
    PROVIDER = new
    MODE = new


def set_mode(mode: str) -> None:
    """Backward-compatible alias for set_provider."""
    set_provider(mode)


def get_provider() -> str:
    return PROVIDER


def is_ollama() -> bool:
    return PROVIDER == "ollama"


def is_openai() -> bool:
    return PROVIDER == "openai"


def set_model(model: str) -> None:
    """Override the active provider's model name."""
    global ACTIVE_MODEL, OLLAMA_MODEL, OPENAI_MODEL
    name = (model or "").strip()
    ACTIVE_MODEL = name or None
    if not name:
        return
    if is_ollama():
        OLLAMA_MODEL = name
    elif is_openai():
        OPENAI_MODEL = name


def set_api_key(api_key: str) -> None:
    global _API_KEY_OVERRIDE, OPENAI_API_KEY
    key = (api_key or "").strip()
    _API_KEY_OVERRIDE = key or None
    if key and is_openai():
        OPENAI_API_KEY = key


def set_base_url(base_url: str) -> None:
    global _BASE_URL_OVERRIDE, OLLAMA_BASE_URL
    url = (base_url or "").strip().rstrip("/")
    _BASE_URL_OVERRIDE = url or None
    if url and is_ollama():
        OLLAMA_BASE_URL = url


def get_active_model() -> str:
    if ACTIVE_MODEL:
        return ACTIVE_MODEL
    unified = os.getenv("LLM_MODEL", "").strip()
    if unified:
        return unified
    spec = get_provider_spec()
    if PROVIDER == "ollama":
        return OLLAMA_MODEL
    if PROVIDER == "openai":
        return OPENAI_MODEL
    if PROVIDER == "peek":
        peek_model = os.getenv("PEEK_MODEL", "").strip()
        if peek_model:
            return peek_model
    return spec.get("default_model") or spec.get("default_model") or ""


def get_api_key(provider: str | None = None) -> str:
    target = provider or PROVIDER
    if target == PROVIDER and _API_KEY_OVERRIDE:
        return _API_KEY_OVERRIDE
    if target == PROVIDER:
        unified = os.getenv("LLM_API_KEY", "").strip()
        if unified:
            return unified
    spec = get_provider_spec(target)
    for env_name in spec.get("api_key_env") or spec.get("api_key_env") or ():
        value = os.getenv(env_name, "").strip()
        if value:
            return value
    if target == "openai":
        return OPENAI_API_KEY
    return ""


def get_base_url(provider: str | None = None) -> str:
    target = provider or PROVIDER
    if target == PROVIDER and _BASE_URL_OVERRIDE:
        return _BASE_URL_OVERRIDE
    if target == PROVIDER:
        unified = os.getenv("LLM_BASE_URL", "").strip().rstrip("/")
        if unified:
            return unified
    spec = get_provider_spec(target)
    if target == "ollama":
        return OLLAMA_BASE_URL.rstrip("/")
    if target == "azure":
        return (AZURE_OPENAI_ENDPOINT or spec["default_base_url"]).rstrip("/")
    return (spec["default_base_url"] or "").rstrip("/")


def get_embed_model() -> str:
    unified = os.getenv("LLM_EMBED_MODEL", "").strip()
    if unified:
        return unified
    provider = get_embeddings_provider()
    spec = get_provider_spec(provider)
    if provider == "ollama":
        return OLLAMA_EMBED_MODEL
    if provider == "openai":
        return OPENAI_EMBED_MODEL
    return spec.get("default_embed_model") or "text-embedding-3-small"


def get_embeddings_provider() -> str:
    """Chat and embeddings can be different vendors (e.g. Groq chat + Ollama embed)."""
    raw = os.getenv("EMBEDDING_PROVIDER", "auto").strip().lower()
    if raw in ("none", "off", "tfidf"):
        raise ValueError("Embeddings disabled (EMBEDDING_PROVIDER=none); use TF-IDF.")
    if raw and raw not in ("auto",):
        return normalize_provider(raw)
    spec = get_provider_spec()
    if spec.get("supports_embeddings"):
        return PROVIDER
    raise ValueError(
        f"{spec['label']} has no embeddings API. Set EMBEDDING_PROVIDER=openai "
        "or ollama, or rely on TF-IDF fallback."
    )


def get_llm_temperature() -> float:
    raw = os.getenv("LLM_TEMPERATURE") or os.getenv("PEEK_TEMPERATURE")
    if raw and raw.strip():
        return float(raw)
    if PROVIDER == "peek":
        return 0.7
    return 0.0


def get_llm_max_tokens() -> int:
    raw = os.getenv("LLM_MAX_TOKENS") or os.getenv("PEEK_MAX_TOKENS")
    if raw and raw.strip():
        return int(raw)
    if PROVIDER == "peek":
        return 1000
    return 0


def get_provider_label() -> str:
    spec = get_provider_spec()
    return f"{spec['label']} ({get_active_model()})"


def get_mode_label() -> str:
    """Backward-compatible alias for get_provider_label."""
    return get_provider_label()


def get_embeddings_label() -> str:
    try:
        provider = get_embeddings_provider()
    except ValueError:
        return "TF-IDF"
    spec = get_provider_spec(provider)
    return spec["label"]


_initial = os.getenv("LLM_PROVIDER") or os.getenv("LLM_MODE") or "ollama"
set_provider(_initial)
