"""BitNetProvider — a :class:`Provider` backed by a local OpenAI-compatible LLM server.

Speaks ``POST /v1/chat/completions`` over HTTP (e.g. a llama.cpp / BitNet server on
``localhost``). This provider handles ONLY transport + wire translation; it advertises
``supports_tools=False`` by default. For tool calling, compose it with
:class:`~zds_llm_provider.text_tools.TextToolCallingProvider`::

    raw = BitNetProvider(base_url="http://localhost:8081", model="bitnet")
    provider = TextToolCallingProvider(raw, mode="auto")

``httpx`` is an optional dependency (``zds-llm-provider[bitnet]``). The module imports it
under a guard so the package stays importable without it; the constructor raises a helpful
:class:`ImportError` only when no client is supplied AND httpx is unavailable. In tests a
mock client is injected, so neither httpx nor a network is required.
"""

from __future__ import annotations

import json
from typing import Any

from zds_llm_provider.messages import Message, ToolResultBlock
from zds_llm_provider.types import (
    AuthError,
    Capabilities,
    ContextWindowExceeded,
    LLMResult,
    Provider,
    RateLimited,
    RequestFailed,
    ToolCall,
)
from zds_llm_provider.usage import Usage

try:
    import httpx
except ImportError:  # pragma: no cover - exercised via monkeypatch in tests
    httpx = None  # type: ignore[assignment]


def _translate_messages(messages: list[Message], system: str | None) -> list[dict[str, Any]]:
    """Translate canonical :class:`Message`s to OpenAI-shaped wire dicts.

    - ``system`` (the argument) → a leading ``{"role": "system", ...}`` entry
    - user / assistant → ``{"role": ..., "content": <concatenated text>}``; an assistant
      turn that carries tool calls also gets an OpenAI ``tool_calls`` array (arguments
      serialized to a JSON string, as the wire format requires)
    - tool → one ``{"role": "tool", "tool_call_id": ..., "content": ...}`` per result
    """
    out: list[dict[str, Any]] = []
    if system:
        out.append({"role": "system", "content": system})
    for msg in messages:
        if msg.role == "tool":
            for block in msg.blocks:
                if isinstance(block, ToolResultBlock):
                    out.append(
                        {
                            "role": "tool",
                            "tool_call_id": block.tool_use_id,
                            "content": block.output,
                        }
                    )
            continue
        entry: dict[str, Any] = {"role": msg.role, "content": msg.text}
        if msg.role == "assistant" and msg.tool_uses:
            entry["tool_calls"] = [
                {
                    "id": tu.id,
                    "type": "function",
                    "function": {"name": tu.name, "arguments": json.dumps(tu.input)},
                }
                for tu in msg.tool_uses
            ]
        out.append(entry)
    return out


def _parse_tool_calls(raw: Any) -> list[ToolCall]:
    """Parse tool calls from an OpenAI-shaped ``message.tool_calls`` array.

    Argument decoding handles three shapes: a ``dict`` (used as-is), a JSON string
    (parsed), and anything unparseable (wrapped as ``{"_raw": <original>}``) so a
    malformed server response degrades gracefully instead of raising.
    """
    if not raw:
        return []
    calls: list[ToolCall] = []
    for item in raw:
        fn = item.get("function", {}) if isinstance(item, dict) else {}
        name = fn.get("name", "")
        args_raw = fn.get("arguments", {})
        if isinstance(args_raw, dict):
            arguments = args_raw
        elif isinstance(args_raw, str):
            try:
                parsed = json.loads(args_raw)
                arguments = parsed if isinstance(parsed, dict) else {"_raw": args_raw}
            except (json.JSONDecodeError, ValueError):
                arguments = {"_raw": args_raw}
        else:
            arguments = {"_raw": args_raw}
        calls.append(
            ToolCall(id=item.get("id", f"call_{len(calls)}"), name=name, arguments=arguments)
        )
    return calls


class BitNetProvider(Provider):
    """Provider backed by a local OpenAI-compatible LLM server.

    Speaks ``/v1/chat/completions``. The HTTP client is injected (tests pass a mock) or
    created from httpx at construction. ``supports_tools`` defaults to ``False`` — compose
    with :class:`~zds_llm_provider.text_tools.TextToolCallingProvider` for tool support.
    """

    def __init__(
        self,
        *,
        base_url: str = "http://localhost:8081",
        model: str = "bitnet",
        temperature: float = 0.7,
        context_window: int = 8192,
        max_output: int | None = None,
        supports_tools: bool = False,
        client: Any | None = None,
        health_path: str = "/health",
        timeout: float = 120.0,
    ) -> None:
        if client is None:
            if httpx is None:
                raise ImportError(
                    "BitNetProvider requires httpx when no client is injected. Install "
                    "with: pip install 'zds-llm-provider[bitnet]' (or pass client=...)."
                )
            client = httpx.AsyncClient(timeout=timeout)
        self._client = client
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._temperature = temperature
        self._context_window = context_window
        self._max_output = max_output
        self._supports_tools = supports_tools
        self._health_path = health_path

    async def acomplete(
        self,
        messages: list[Message],
        *,
        system: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> LLMResult:
        """Translate messages, POST to ``/v1/chat/completions``, normalize the response.

        Backend transport and HTTP errors are mapped onto the provider error taxonomy
        (see the class docstring); vendor exceptions never leak to the agent loop.
        """
        body: dict[str, Any] = {
            "model": self._model,
            "messages": _translate_messages(messages, system),
            "temperature": self._temperature,
        }
        if self._max_output is not None:
            body["max_tokens"] = self._max_output
        if tools:
            body["tools"] = tools

        url = f"{self._base_url}/v1/chat/completions"
        try:
            response = await self._client.post(url, json=body, headers=None)
        except Exception as exc:  # map transport errors; everything else → RequestFailed
            if httpx is not None and isinstance(exc, (httpx.ConnectError, httpx.ConnectTimeout)):
                raise RequestFailed(f"Connection error: {exc}") from exc
            if httpx is not None and isinstance(exc, (httpx.ReadTimeout, httpx.WriteTimeout)):
                raise RequestFailed(f"Timeout: {exc}") from exc
            raise RequestFailed(f"Request failed: {exc}") from exc

        if response.status_code != 200:
            self._raise_for_status(response.status_code, response.text)

        data = response.json()
        choice = (data.get("choices") or [{}])[0]
        message = choice.get("message", {})
        usage_data = data.get("usage", {}) or {}
        return LLMResult(
            text=message.get("content") or "",
            tool_calls=_parse_tool_calls(message.get("tool_calls")),
            finish_reason=choice.get("finish_reason"),
            usage=Usage(
                prompt_tokens=usage_data.get("prompt_tokens", 0),
                completion_tokens=usage_data.get("completion_tokens", 0),
                total_tokens=usage_data.get("total_tokens", 0),
            ),
            raw=data,
        )

    @staticmethod
    def _raise_for_status(status_code: int, body: str) -> None:
        """Map a non-200 HTTP status onto the provider error taxonomy."""
        if status_code in (401, 403):
            raise AuthError(f"Authentication failed ({status_code}): {body}")
        if status_code == 429:
            raise RateLimited(f"Rate limited: {body}")
        if status_code == 400 and "context" in body.lower():
            raise ContextWindowExceeded(f"Context window exceeded: {body}")
        raise RequestFailed(f"Request failed ({status_code}): {body}")

    async def _check_health(self) -> bool:
        """Probe the health endpoint. Returns ``True`` on HTTP 200, ``False`` otherwise.

        A transport error (server down) returns ``False`` rather than raising — health is
        a best-effort liveness probe, not a hard precondition.
        """
        try:
            response = await self._client.get(f"{self._base_url}{self._health_path}")
        except Exception:
            return False
        return bool(response.status_code == 200)

    def count_tokens(self, messages: list[Message], *, system: str | None = None) -> int:
        """Rough char/4 estimate — no local tokenizer is available."""
        total = len(system or "")
        total += sum(len(msg.text) for msg in messages)
        return total // 4

    def capabilities(self) -> Capabilities:
        """Static capabilities. ``supports_tools`` is False by default (compose for tools)."""
        return Capabilities(
            supports_tools=self._supports_tools,
            supports_vision=False,
            supports_caching=False,
            context_window=self._context_window,
            max_output=self._max_output,
        )


__all__ = ["BitNetProvider"]
