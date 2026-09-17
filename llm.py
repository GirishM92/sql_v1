from __future__ import annotations

import json
from typing import Any, Optional

import requests
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import AliasChoices, ConfigDict, Field

import config

DEFAULT_PEEK_BASE_URL = "https://genai-proxy-uat.hdfcbankuat.com/UAT/litellm"

# Peek / LiteLLM token counters (process-wide session)
_TOKEN_SESSION = {
    "prompt_tokens": 0,
    "completion_tokens": 0,
    "total_tokens": 0,
    "calls": 0,
}
_TOKEN_LAST: dict[str, Any] = {}
_TOKEN_CALLS: list[dict[str, Any]] = []


def get_llm() -> BaseChatModel:
    """Return a LangChain chat model for the configured provider."""
    provider = config.get_provider()
    spec = config.get_provider_spec(provider)
    model = config.get_active_model()
    backend = spec["chat_backend"]

    if backend == config.CHAT_PEEK:
        return _chat_peek(model)
    if backend == config.CHAT_OLLAMA:
        return _chat_ollama(model)
    if backend == config.CHAT_ANTHROPIC:
        return _chat_anthropic(model)
    if backend == config.CHAT_GOOGLE:
        return _chat_google(model)
    if backend == config.CHAT_AZURE:
        return _chat_azure(model)
    return _chat_openai_compatible(model, provider)


def invoke_llm(system_prompt: str, user_prompt: str) -> str:
    llm = get_llm()
    response = llm.invoke(
        [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]
    )
    content = response.content
    if isinstance(content, list):
        content = "".join(
            part.get("text", str(part)) if isinstance(part, dict) else str(part)
            for part in content
        )
    return str(content).strip()


def _missing_package(package: str, provider_label: str) -> ImportError:
    return ImportError(
        f"{provider_label} requires `{package}`. Install with: pip install {package}\n"
        "Alternatively use an OpenAI-compatible gateway, e.g.\n"
        "  LLM_PROVIDER=openrouter  LLM_MODEL=anthropic/claude-3.5-sonnet"
    )


def _require_api_key(provider: str) -> str:
    spec = config.get_provider_spec(provider)
    key = config.get_api_key(provider)
    if spec["requires_api_key"] and not key:
        env_names = spec.get("api_key_env") or spec.get("api_key_env") or ("LLM_API_KEY",)
        hinted = " or ".join(env_names)
        raise ValueError(
            f"{spec['label']} requires an API key. Set {hinted} in .env, "
            "or pass it in the UI / --api-key."
        )
    return key or "not-needed"


def _chat_ollama(model: str) -> BaseChatModel:
    try:
        from langchain_ollama import ChatOllama
    except ImportError:
        try:
            from langchain_community.chat_models import ChatOllama
        except ImportError as exc:
            raise _missing_package("langchain-ollama", "Ollama") from exc
    return ChatOllama(
        base_url=config.get_base_url() or config.OLLAMA_BASE_URL,
        model=model,
        temperature=0,
    )


def _chat_anthropic(model: str) -> BaseChatModel:
    try:
        from langchain_anthropic import ChatAnthropic
    except ImportError as exc:
        raise _missing_package("langchain-anthropic", "Anthropic") from exc
    return ChatAnthropic(
        model=model,
        api_key=_require_api_key("anthropic"),
        temperature=0,
    )


def _chat_google(model: str) -> BaseChatModel:
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
    except ImportError as exc:
        raise _missing_package("langchain-google-genai", "Google Gemini") from exc
    return ChatGoogleGenerativeAI(
        model=model,
        google_api_key=_require_api_key("google"),
        temperature=0,
    )


def _chat_azure(model: str) -> BaseChatModel:
    from langchain_openai import AzureChatOpenAI

    endpoint = config.get_base_url()
    if not endpoint:
        raise ValueError(
            "Azure OpenAI requires a base URL. Set AZURE_OPENAI_ENDPOINT or LLM_BASE_URL."
        )
    return AzureChatOpenAI(
        azure_deployment=model,
        azure_endpoint=endpoint,
        api_key=_require_api_key("azure"),
        api_version=config.AZURE_OPENAI_API_VERSION,
        temperature=0,
    )


def _chat_openai_compatible(model: str, provider: str) -> BaseChatModel:
    from langchain_openai import ChatOpenAI

    spec = config.get_provider_spec(provider)
    base_url = config.get_base_url()
    if spec["requires_base_url"] and not base_url:
        raise ValueError(
            "OpenAI-compatible provider requires LLM_BASE_URL "
            "(e.g. http://localhost:1234/v1 for LM Studio / vLLM)."
        )
    kwargs: dict = {
        "model": model,
        "api_key": _require_api_key(provider),
        "temperature": 0,
    }
    if base_url:
        kwargs["base_url"] = base_url
    return ChatOpenAI(**kwargs)


def resolve_peek_chat_url(url: str) -> str:
    """HDFC C# posts to .../chat/completions. Accept root or full path."""
    cleaned = (url or "").strip().rstrip("/")
    if not cleaned:
        cleaned = DEFAULT_PEEK_BASE_URL
    if cleaned.endswith("/chat/completions"):
        return cleaned
    return f"{cleaned}/chat/completions"


def langchain_messages_to_openai(messages: list[BaseMessage]) -> list[dict[str, Any]]:
    """Match HDFC BuildLiteLLMRequest message mapping."""
    payload: list[dict[str, Any]] = []
    for message in messages:
        if isinstance(message, SystemMessage):
            payload.append({"role": "system", "content": _text(message.content)})
        elif isinstance(message, HumanMessage):
            payload.append({"role": "user", "content": _text(message.content)})
        elif isinstance(message, ToolMessage):
            payload.append(
                {
                    "role": "tool",
                    "tool_call_id": getattr(message, "tool_call_id", "") or "",
                    "name": getattr(message, "name", "") or "",
                    "content": _text(message.content),
                }
            )
        elif isinstance(message, AIMessage):
            item: dict[str, Any] = {
                "role": "assistant",
                "content": _text(message.content),
            }
            tool_calls = getattr(message, "tool_calls", None) or []
            if tool_calls:
                item["tool_calls"] = [
                    {
                        "id": tc.get("id") or "",
                        "type": "function",
                        "function": {
                            "name": tc.get("name") or "",
                            "arguments": _json_args(tc.get("args")),
                        },
                    }
                    for tc in tool_calls
                ]
            payload.append(item)
        else:
            payload.append(
                {"role": "user", "content": _text(getattr(message, "content", message))}
            )
    return payload


def build_litellm_request(
    *,
    model: str,
    messages: list[dict[str, Any]],
    temperature: float,
    max_tokens: int,
    tools: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Same body shape as HDFC C# BuildLiteLLMRequest."""
    request: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if tools:
        request["tools"] = [
            {
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool.get("description") or "",
                    "parameters": tool.get("parameters")
                    or tool.get("input_schema")
                    or {},
                },
            }
            for tool in tools
        ]
    return request


def parse_litellm_usage(data: Any) -> dict[str, int]:
    """Read token counts from LiteLLM / OpenAI or Gemini-style responses."""
    usage = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
    }
    if not isinstance(data, dict):
        return usage

    raw = data.get("usage")
    if isinstance(raw, dict):
        usage["prompt_tokens"] = int(
            raw.get("prompt_tokens")
            or raw.get("input_tokens")
            or raw.get("prompt_token_count")
            or 0
        )
        usage["completion_tokens"] = int(
            raw.get("completion_tokens")
            or raw.get("output_tokens")
            or raw.get("completion_token_count")
            or 0
        )
        usage["total_tokens"] = int(
            raw.get("total_tokens")
            or raw.get("total_token_count")
            or (usage["prompt_tokens"] + usage["completion_tokens"])
        )
        return usage

    meta = data.get("usageMetadata") or data.get("usage_metadata")
    if isinstance(meta, dict):
        usage["prompt_tokens"] = int(meta.get("promptTokenCount") or meta.get("prompt_tokens") or 0)
        usage["completion_tokens"] = int(
            meta.get("candidatesTokenCount")
            or meta.get("completion_tokens")
            or 0
        )
        usage["total_tokens"] = int(
            meta.get("totalTokenCount")
            or meta.get("total_tokens")
            or (usage["prompt_tokens"] + usage["completion_tokens"])
        )
    return usage


def record_token_usage(
    usage: dict[str, int],
    *,
    model: str = "",
    source: str = "peek",
) -> dict[str, Any]:
    """Accumulate Peek token usage for the current process/session."""
    prompt = int(usage.get("prompt_tokens") or 0)
    completion = int(usage.get("completion_tokens") or 0)
    total = int(usage.get("total_tokens") or (prompt + completion))
    entry = {
        "source": source,
        "model": model,
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": total,
        "reported": bool(prompt or completion or total),
    }
    _TOKEN_SESSION["prompt_tokens"] += prompt
    _TOKEN_SESSION["completion_tokens"] += completion
    _TOKEN_SESSION["total_tokens"] += total
    _TOKEN_SESSION["calls"] += 1
    global _TOKEN_LAST
    _TOKEN_LAST = dict(entry)
    _TOKEN_CALLS.append(entry)
    return entry


def snapshot_token_usage() -> dict[str, int]:
    return dict(_TOKEN_SESSION)


def usage_since(snapshot: dict[str, int] | None) -> dict[str, int]:
    current = snapshot_token_usage()
    if not snapshot:
        return current
    return {
        "prompt_tokens": current["prompt_tokens"] - int(snapshot.get("prompt_tokens") or 0),
        "completion_tokens": current["completion_tokens"] - int(snapshot.get("completion_tokens") or 0),
        "total_tokens": current["total_tokens"] - int(snapshot.get("total_tokens") or 0),
        "calls": current["calls"] - int(snapshot.get("calls") or 0),
    }


def get_last_token_usage() -> dict[str, Any]:
    return dict(_TOKEN_LAST)


def get_token_usage_summary() -> dict[str, Any]:
    return {
        "last_call": dict(_TOKEN_LAST),
        "session": snapshot_token_usage(),
        "calls": list(_TOKEN_CALLS),
    }


def reset_token_usage() -> None:
    _TOKEN_SESSION.update(
        {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "calls": 0}
    )
    _TOKEN_LAST.clear()
    _TOKEN_CALLS.clear()


def format_token_usage_log(entry: dict[str, Any], session: dict[str, int] | None = None) -> str:
    session = session or snapshot_token_usage()
    reported = entry.get("reported", True)
    note = "" if reported else " (LiteLLM did not return usage)"
    return (
        f"[Peek tokens] prompt={entry.get('prompt_tokens', 0)} "
        f"completion={entry.get('completion_tokens', 0)} "
        f"total={entry.get('total_tokens', 0)}{note} | "
        f"session total={session.get('total_tokens', 0)} "
        f"({session.get('calls', 0)} call(s))"
    )


def parse_litellm_content(data: Any) -> str:
    if isinstance(data, str):
        return data
    if not isinstance(data, dict):
        return str(data)

    choices = data.get("choices") or []
    if choices:
        message = choices[0].get("message") or {}
        content = message.get("content")
        if content:
            return _text(content)

    candidates = data.get("candidates") or []
    if candidates:
        parts = ((candidates[0].get("content") or {}).get("parts") or [])
        if parts:
            return _text(parts[0].get("text") or "")

    return _text(data.get("text") or data.get("response") or "")


def _text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            part.get("text", str(part)) if isinstance(part, dict) else str(part)
            for part in content
        )
    return str(content)


def _json_args(args: Any) -> str:
    if args is None:
        return "{}"
    if isinstance(args, str):
        return args
    return json.dumps(args)


class PeekChatModel(BaseChatModel):
    """HDFC Peek / LiteLLM chat client.

    Posts OpenAI-style JSON to /chat/completions, matching C# BuildLiteLLMRequest:
      { model, messages, temperature, max_tokens, tools? }

    MCP JSON-RPC (`method: tools/call`) is a different protocol used by their
    C# MCP host. It is not sent to this LiteLLM endpoint.
    """

    model_config = ConfigDict(
        protected_namespaces=(),
        extra="ignore",
        populate_by_name=True,
    )

    # LangChain 1.x / Pydantic v2: do not use a custom __init__.
    # `model` is the LiteLLM deployment name (gemini-2.5-flash).
    model: str = Field(
        default="gemini-2.5-flash",
        validation_alias=AliasChoices("model", "model_name"),
    )
    peek_url: str = Field(default=DEFAULT_PEEK_BASE_URL)
    peek_api_key: str = Field(default="")
    temperature: float = 0.7
    max_tokens: int = 1000
    verify_ssl: bool = False
    request_timeout: float = 60.0
    extra_tools: list = Field(default_factory=list)

    @property
    def model_name(self) -> str:
        return self.model

    @property
    def _llm_type(self) -> str:
        return "peek-hdfc-litellm"

    @property
    def _identifying_params(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "base_url": self.peek_url,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: Optional[list[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        url = resolve_peek_chat_url(self.peek_url)
        payload = build_litellm_request(
            model=self.model,
            messages=langchain_messages_to_openai(messages),
            temperature=float(kwargs.get("temperature", self.temperature)),
            max_tokens=int(kwargs.get("max_tokens", self.max_tokens)),
            tools=kwargs.get("tools") or self.extra_tools or None,
        )
        if stop:
            payload["stop"] = stop

        headers = {"Content-Type": "application/json"}
        if self.peek_api_key:
            headers["Authorization"] = f"Bearer {self.peek_api_key}"

        if not self.verify_ssl:
            requests.packages.urllib3.disable_warnings()  # type: ignore[attr-defined]

        try:
            response = requests.post(
                url,
                json=payload,
                headers=headers,
                timeout=self.request_timeout,
                verify=self.verify_ssl,
            )
        except requests.exceptions.SSLError as exc:
            raise RuntimeError(
                "Peek LiteLLM SSL error. For HDFC UAT set PEEK_VERIFY_SSL=false."
            ) from exc
        except requests.exceptions.RequestException as exc:
            raise RuntimeError(
                f"Peek LiteLLM could not reach {url}. "
                "Check PEEK_BASE_URL and bank VPN/network."
            ) from exc

        try:
            response.raise_for_status()
        except requests.HTTPError as exc:
            raise RuntimeError(
                f"Peek LiteLLM call failed ({response.status_code}) at {url}: "
                f"{response.text[:800]}"
            ) from exc

        try:
            data = response.json()
        except ValueError as exc:
            raise RuntimeError(
                f"Peek LiteLLM returned non-JSON from {url}: {response.text[:800]}"
            ) from exc

        text = parse_litellm_content(data)
        if not text:
            raise RuntimeError(
                f"Peek LiteLLM returned an empty completion from {url}: "
                f"{str(data)[:800]}"
            )
        usage = parse_litellm_usage(data)
        record_token_usage(usage, model=self.model, source="peek")
        message = AIMessage(
            content=text,
            usage_metadata={
                "input_tokens": usage["prompt_tokens"],
                "output_tokens": usage["completion_tokens"],
                "total_tokens": usage["total_tokens"],
            },
            response_metadata={
                "model_name": self.model,
                "token_usage": usage,
                "provider": "peek",
            },
        )
        if run_manager:
            run_manager.on_llm_new_token(text)
        return ChatResult(
            generations=[ChatGeneration(message=message)],
            llm_output={"token_usage": usage, "model_name": self.model},
        )


def _chat_peek(model: str) -> BaseChatModel:
    return PeekChatModel(
        model=model,
        peek_url=config.get_base_url("peek") or DEFAULT_PEEK_BASE_URL,
        peek_api_key=_require_api_key("peek"),
        temperature=config.get_llm_temperature(),
        max_tokens=config.get_llm_max_tokens() or 1000,
        verify_ssl=config.PEEK_VERIFY_SSL,
        disable_streaming=True,
    )
