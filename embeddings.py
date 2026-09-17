import os

from langchain_core.embeddings import Embeddings

import config


def get_embeddings_client() -> Embeddings:
    """Return embeddings for the configured embedding provider.

    Chat and embeddings can differ: e.g. Groq for SQL generation and Ollama
    (or OpenAI) for tool matching. Set EMBEDDING_PROVIDER to override.
    """
    provider = config.get_embeddings_provider()
    spec = config.get_provider_spec(provider)
    model = config.get_embed_model()
    backend = spec["chat_backend"]

    if backend == config.CHAT_OLLAMA:
        return _embed_ollama(model)
    if backend == config.CHAT_GOOGLE:
        return _embed_google(model, provider)
    if backend == config.CHAT_AZURE:
        return _embed_azure(model, provider)
    if backend == config.CHAT_ANTHROPIC:
        raise ValueError(
            "Anthropic has no embeddings API. Set EMBEDDING_PROVIDER=openai or ollama."
        )
    if backend == config.CHAT_PEEK:
        raise ValueError(
            "Peek / HDFC LiteLLM has no embeddings API. "
            "Set EMBEDDING_PROVIDER=openai or ollama, or rely on TF-IDF fallback."
        )
    return _embed_openai_compatible(model, provider)


def _embed_ollama(model: str) -> Embeddings:
    try:
        from langchain_ollama import OllamaEmbeddings
    except ImportError:
        try:
            from langchain_community.embeddings import OllamaEmbeddings
        except ImportError as exc:
            raise ImportError(
                "Ollama embeddings require langchain-ollama. "
                "Install with: pip install langchain-ollama"
            ) from exc
    return OllamaEmbeddings(
        model=model,
        base_url=config.get_base_url("ollama"),
    )


def _embed_google(model: str, provider: str) -> Embeddings:
    try:
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
    except ImportError as exc:
        raise ImportError(
            "Google embeddings require langchain-google-genai. "
            "Install with: pip install langchain-google-genai"
        ) from exc
    key = config.get_api_key(provider) or os.getenv("GOOGLE_API_KEY") or os.getenv(
        "GEMINI_API_KEY", ""
    )
    if not key:
        raise ValueError("GOOGLE_API_KEY is not set.")
    return GoogleGenerativeAIEmbeddings(model=model, google_api_key=key)


def _embed_azure(model: str, provider: str) -> Embeddings:
    from langchain_openai import AzureOpenAIEmbeddings

    endpoint = config.get_base_url(provider)
    if not endpoint:
        raise ValueError("AZURE_OPENAI_ENDPOINT is not set.")
    key = config.get_api_key(provider)
    if not key:
        raise ValueError("AZURE_OPENAI_API_KEY is not set.")
    return AzureOpenAIEmbeddings(
        azure_deployment=model,
        azure_endpoint=endpoint,
        api_key=key,
        api_version=config.AZURE_OPENAI_API_VERSION,
    )


def _embed_openai_compatible(model: str, provider: str) -> Embeddings:
    from langchain_openai import OpenAIEmbeddings

    spec = config.get_provider_spec(provider)
    key = config.get_api_key(provider)
    base_url = config.get_base_url(provider)
    if spec["requires_api_key"] and not key:
        raise ValueError(
            f"{spec['label']} embeddings require an API key. "
            "Set EMBEDDING_PROVIDER and the matching *_API_KEY, or use TF-IDF fallback."
        )
    kwargs: dict = {
        "model": model,
        "api_key": key or "not-needed",
    }
    if base_url:
        kwargs["base_url"] = base_url
    return OpenAIEmbeddings(**kwargs)
