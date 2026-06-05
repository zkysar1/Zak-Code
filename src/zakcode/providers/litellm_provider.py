"""litellm-backed :class:`Provider` implementation.

This is the ONLY module in zakcode that imports litellm. It translates the
provider-neutral :class:`~zakcode.messages.Message` model into the OpenAI-shaped
wire format litellm expects, performs the async completion call, and normalizes
the response back into an :class:`~zakcode.providers.base.LLMResult`.

litellm has no published type stubs, so calls into it are treated as ``Any`` and
their results are validated/narrowed explicitly before use.
"""

from __future__ import annotations

import json
import os
from collections.abc import AsyncIterator
from typing import Any

import litellm

from zakcode.config import Settings
from zakcode.messages import (
    Message,
    TextBlock,
    ThinkingBlock,
    ToolResultBlock,
    ToolUseBlock,
)
from zakcode.providers.base import (
    AuthError,
    Capabilities,
    ContextWindowExceeded,
    LLMResult,
    Provider,
    ProviderError,
    ProviderStreamEvent,
    RateLimited,
    RequestFailed,
    StreamDone,
    StreamTextDelta,
    StreamToolCallDelta,
    StreamUsage,
    ToolCall,
)
from zakcode.providers.registry import get_capabilities
from zakcode.usage import Usage

# Drop kwargs unsupported by a given backend (e.g. temperature on some models)
# rather than raising. Set once at import time. ``setattr`` avoids a spurious
# attr-defined error (litellm ships no type stubs for this module global).
setattr(litellm, "drop_params", True)  # noqa: B010


# Resolve litellm's exception classes defensively. Older/newer versions may not
# expose every type; access via getattr so a missing attribute degrades to None
# (no isinstance match) instead of an import-time failure. litellm ships no type
# stubs, so these attribute reads are untyped regardless. A non-class value (or
# anything not derived from ``BaseException``) is rejected so it can never blow up
# a later ``isinstance`` check.
def _resolve_exc_type(*names: str) -> type[BaseException] | None:
    """Return the first attribute among ``names`` that is an exception class.

    Tries ``litellm`` first, then the bundled ``litellm.exceptions`` /
    ``openai`` namespaces if present. Any miss or non-exception value is skipped
    so resolution never raises at import time.
    """
    namespaces: list[Any] = [litellm]
    exceptions_mod = getattr(litellm, "exceptions", None)
    if exceptions_mod is not None:
        namespaces.append(exceptions_mod)
    for ns in namespaces:
        for name in names:
            candidate = getattr(ns, name, None)
            if isinstance(candidate, type) and issubclass(candidate, BaseException):
                return candidate
    return None


_LiteLLMAuthError = _resolve_exc_type("AuthenticationError")
_LiteLLMPermissionError = _resolve_exc_type("PermissionDeniedError")
_LiteLLMContextWindowError = _resolve_exc_type("ContextWindowExceededError")
_LiteLLMRateLimitError = _resolve_exc_type("RateLimitError")


def _is_ollama_model(model: str) -> bool:
    return model.startswith("ollama/") or model.startswith("ollama_chat/")


def _max_stop_sequences(model: str) -> int | None:
    """The backend's max number of ``stop`` sequences, when known.

    The OpenAI (and Azure OpenAI) Chat Completions API rejects a request with more than 4
    stop strings. The text tool-calling layer can emit up to ~6 sentinels, so it must cap
    to this for those backends (see ``Capabilities.max_stop_sequences``). ``None`` = unknown
    / unbounded (e.g. Ollama accepts many). Returns 4 for the ``openai``/``azure`` prefixes
    and for bare OpenAI model names (``gpt*``/``o1``/``o3``/``o4``/``chatgpt*``).
    """
    prefix = model.split("/", 1)[0].lower() if "/" in model else ""
    if prefix in ("openai", "azure"):
        return 4
    if not prefix and model.lower().startswith(("gpt", "o1", "o3", "o4", "chatgpt")):
        return 4
    return None


#: Cap for Ollama's per-request context window (``num_ctx``). Ollama's default ctx is
#: small and silently truncates the prompt *tail* (the latest user step + tool-protocol
#: rules), which looks like the model "ignoring" the task. We lift num_ctx to the model's
#: real window but clamp here so a huge advertised window (e.g. llama3.1's 128k) cannot
#: OOM a local box. ``capabilities().context_window`` is clamped to the same value so the
#: compactor's threshold matches Ollama's real context (no truncate-before-compact).
_OLLAMA_NUM_CTX_CAP = 16_384


class LiteLLMProvider(Provider):
    """A :class:`Provider` backed by litellm.

    Construct either from a :class:`Settings` instance or from explicit keyword
    arguments. Explicit kwargs win over values read from ``settings``.
    """

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        model: str | None = None,
        temperature: float | None = None,
        api_base: str | None = None,
        api_key: str | None = None,
        num_retries: int | None = None,
        ollama_base_url: str | None = None,
    ) -> None:
        resolved_model = model
        resolved_temperature = temperature
        resolved_api_base = api_base
        resolved_api_key = api_key
        resolved_ollama_base = ollama_base_url

        if settings is not None:
            if resolved_model is None:
                resolved_model = settings.default_model
            if resolved_temperature is None:
                resolved_temperature = settings.temperature
            # Generic endpoint override (any OpenAI-compatible server). Explicit
            # kwargs win; otherwise take the configured values.
            if resolved_api_base is None:
                resolved_api_base = settings.api_base
            if resolved_api_key is None:
                resolved_api_key = settings.api_key
            if resolved_ollama_base is None:
                resolved_ollama_base = settings.ollama_base_url

        if resolved_model is None:
            raise ValueError("a model must be provided via settings or the model kwarg")

        self.model: str = resolved_model
        self.temperature: float = resolved_temperature if resolved_temperature is not None else 0.7
        self.api_base: str | None = resolved_api_base
        self.api_key: str | None = resolved_api_key
        self.num_retries: int = num_retries if num_retries is not None else 0
        self.ollama_base_url: str | None = resolved_ollama_base

        # litellm reads OLLAMA_API_BASE from the environment for ollama_chat/*.
        if _is_ollama_model(self.model) and self.ollama_base_url:
            os.environ["OLLAMA_API_BASE"] = self.ollama_base_url

    # ------------------------------------------------------------------
    # Translation: zakcode Message -> OpenAI-shaped dicts
    # ------------------------------------------------------------------
    @staticmethod
    def _translate_messages(messages: list[Message], system: str | None) -> list[dict[str, Any]]:
        wire: list[dict[str, Any]] = []
        if system:
            wire.append({"role": "system", "content": system})

        for msg in messages:
            if msg.role == "system":
                text = msg.text
                if text:
                    wire.append({"role": "system", "content": text})
                continue

            if msg.role == "tool":
                # Each tool result becomes its own role:"tool" message.
                for block in msg.blocks:
                    if isinstance(block, ToolResultBlock):
                        wire.append(
                            {
                                "role": "tool",
                                "tool_call_id": block.tool_use_id,
                                "content": block.output,
                            }
                        )
                continue

            if msg.role == "user":
                text = "".join(b.text for b in msg.blocks if isinstance(b, TextBlock))
                wire.append({"role": "user", "content": text})
                continue

            # assistant
            text_parts: list[str] = []
            tool_calls: list[dict[str, Any]] = []
            for block in msg.blocks:
                if isinstance(block, TextBlock):
                    text_parts.append(block.text)
                elif isinstance(block, ThinkingBlock):
                    # Thinking is not replayed on the wire.
                    continue
                elif isinstance(block, ToolUseBlock):
                    tool_calls.append(
                        {
                            "id": block.id,
                            "type": "function",
                            "function": {
                                "name": block.name,
                                "arguments": json.dumps(block.input),
                            },
                        }
                    )
            assistant_msg: dict[str, Any] = {"role": "assistant"}
            content = "".join(text_parts)
            # OpenAI requires content to be present (may be None when only tools).
            assistant_msg["content"] = content if content else None
            if tool_calls:
                assistant_msg["tool_calls"] = tool_calls
            wire.append(assistant_msg)

        return wire

    # ------------------------------------------------------------------
    # Normalization: litellm response -> LLMResult
    # ------------------------------------------------------------------
    @staticmethod
    def _parse_tool_calls(raw_tool_calls: Any) -> list[ToolCall]:
        calls: list[ToolCall] = []
        if not raw_tool_calls:
            return calls
        for tc in raw_tool_calls:
            tc_id = _get(tc, "id") or ""
            fn = _get(tc, "function")
            name = _get(fn, "name") or ""
            raw_args = _get(fn, "arguments")
            arguments: dict[str, Any]
            if isinstance(raw_args, dict):
                arguments = raw_args
            elif isinstance(raw_args, str):
                try:
                    parsed = json.loads(raw_args)
                except (json.JSONDecodeError, ValueError):
                    arguments = {"_raw": raw_args}
                else:
                    arguments = parsed if isinstance(parsed, dict) else {"_raw": raw_args}
            elif raw_args is None:
                arguments = {}
            else:
                arguments = {"_raw": str(raw_args)}
            calls.append(ToolCall(id=str(tc_id), name=str(name), arguments=arguments))
        return calls

    @staticmethod
    def _coerce_int(value: Any) -> int:
        """Coerce a usage count to a non-negative int; junk -> 0, never raises."""
        if isinstance(value, bool):  # bool is an int subclass — treat as absent
            return 0
        if isinstance(value, int):
            return max(value, 0)
        if isinstance(value, float):
            return max(int(value), 0)
        if isinstance(value, str):
            try:
                return max(int(float(value)), 0)
            except (TypeError, ValueError):
                return 0
        return 0

    @staticmethod
    def _coerce_cost(value: Any) -> float:
        """Coerce a cost to a non-negative float; junk -> 0.0, never raises."""
        if isinstance(value, bool):
            return 0.0
        if isinstance(value, int | float):
            return max(float(value), 0.0)
        if isinstance(value, str):
            try:
                return max(float(value), 0.0)
            except (TypeError, ValueError):
                return 0.0
        return 0.0

    @classmethod
    def _extract_usage(cls, response: Any) -> Usage:
        usage_obj = _get(response, "usage")
        prompt = cls._coerce_int(_get(usage_obj, "prompt_tokens"))
        completion = cls._coerce_int(_get(usage_obj, "completion_tokens"))
        total = cls._coerce_int(_get(usage_obj, "total_tokens"))
        # If the backend omitted a total but gave the parts, derive it.
        if total == 0 and (prompt or completion):
            total = prompt + completion

        cost = 0.0
        hidden = _get(response, "_hidden_params")
        if isinstance(hidden, dict):
            cost = cls._coerce_cost(hidden.get("response_cost"))

        return Usage(
            prompt_tokens=prompt,
            completion_tokens=completion,
            total_tokens=total,
            cost_usd=cost,
        )

    @classmethod
    def _normalize(cls, response: Any) -> LLMResult:
        choices = _get(response, "choices")
        message = None
        finish_reason: str | None = None
        first = None
        if isinstance(choices, list | tuple) and choices:
            first = choices[0]
        if first is not None:
            message = _get(first, "message")
            fr = _get(first, "finish_reason")
            finish_reason = str(fr) if fr is not None else None

        text = ""
        tool_calls: list[ToolCall] = []
        if message is not None:
            content = _get(message, "content")
            if isinstance(content, str):
                text = content
            tool_calls = cls._parse_tool_calls(_get(message, "tool_calls"))

        raw: dict[str, Any] | None = None
        dump = getattr(response, "model_dump", None)
        if callable(dump):
            try:
                dumped = dump()
            except Exception:
                dumped = None
            if isinstance(dumped, dict):
                raw = dumped

        return LLMResult(
            text=text,
            tool_calls=tool_calls,
            finish_reason=finish_reason,
            usage=cls._extract_usage(response),
            raw=raw,
        )

    # ------------------------------------------------------------------
    # Error mapping
    # ------------------------------------------------------------------
    @staticmethod
    def _extract_retry_after(exc: Exception) -> float | None:
        """Pull a server-suggested retry delay (seconds) off an exception.

        Checks the common ``retry_after`` attribute first, then a numeric
        ``Retry-After`` header on an attached ``response`` (openai/httpx shape).
        Anything non-numeric is ignored so this never raises.
        """
        candidate = getattr(exc, "retry_after", None)
        if isinstance(candidate, bool):  # bool is an int subclass — reject it
            candidate = None
        if isinstance(candidate, int | float):
            return float(candidate)

        response = getattr(exc, "response", None)
        headers = getattr(response, "headers", None)
        if headers is not None:
            try:
                raw = headers.get("retry-after")
            except Exception:
                raw = None
            if raw is not None:
                try:
                    return float(raw)
                except (TypeError, ValueError):
                    return None
        return None

    @classmethod
    def _is_a(cls, exc: Exception, resolved: type[BaseException] | None, *names: str) -> bool:
        """True if ``exc`` matches a resolved class or any fallback class name.

        Class-name matching (over the exception and its MRO) keeps the mapping
        working even when litellm did not expose the type for ``isinstance``.
        """
        if resolved is not None and isinstance(exc, resolved):
            return True
        wanted = set(names)
        return any(base.__name__ in wanted for base in type(exc).__mro__)

    @classmethod
    def _map_error(cls, exc: Exception) -> ProviderError:
        # Prefer isinstance against resolved classes; fall back to class-name
        # matching (across the MRO) so the mapping still works if imports were
        # unavailable. Auth is checked before the generic catch-all; rate-limit
        # carries through any server-suggested retry delay. An unknown exception
        # always degrades to RequestFailed — a raw vendor exception never leaks.
        retry_after = cls._extract_retry_after(exc)

        if cls._is_a(exc, _LiteLLMAuthError, "AuthenticationError") or cls._is_a(
            exc, _LiteLLMPermissionError, "PermissionDeniedError"
        ):
            return AuthError(str(exc))
        if cls._is_a(exc, _LiteLLMContextWindowError, "ContextWindowExceededError"):
            return ContextWindowExceeded(str(exc))
        if cls._is_a(exc, _LiteLLMRateLimitError, "RateLimitError"):
            return RateLimited(str(exc), retry_after=retry_after)
        return RequestFailed(str(exc))

    # ------------------------------------------------------------------
    # Provider interface
    # ------------------------------------------------------------------
    def _build_kwargs(
        self,
        wire_messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        **kw: Any,
    ) -> dict[str, Any]:
        """Assemble the kwargs handed to ``litellm.acompletion``.

        Factored out of :meth:`acomplete` so the request shape (model, endpoint,
        tool wiring) is unit-testable without a network call.
        """
        call_kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": wire_messages,
            "temperature": self.temperature,
            "drop_params": True,
            "num_retries": self.num_retries,
            "stream": False,
        }
        if tools:
            call_kwargs["tools"] = tools
            call_kwargs["tool_choice"] = "auto"
        if self.api_base is not None:
            call_kwargs["api_base"] = self.api_base
        if self.api_key is not None:
            call_kwargs["api_key"] = self.api_key
        if _is_ollama_model(self.model):
            # Lift Ollama's tiny default context window to the model's real one (clamped),
            # so the prompt tail is not silently truncated. Gated on the Ollama prefix —
            # for other backends litellm may wrap an unknown num_ctx into extra_body
            # rather than drop it. setdefault so an explicit caller override wins.
            try:
                window = get_capabilities(self.model).context_window
            except Exception:  # noqa: BLE001 — a capability probe must never fail a call
                window = 0
            if window:
                call_kwargs.setdefault("num_ctx", min(window, _OLLAMA_NUM_CTX_CAP))
        # Allow per-call overrides (e.g. max_tokens) without re-listing them.
        call_kwargs.update(kw)
        return call_kwargs

    async def acomplete(
        self,
        messages: list[Message],
        *,
        system: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        **kw: Any,
    ) -> LLMResult:
        wire_messages = self._translate_messages(messages, system)
        call_kwargs = self._build_kwargs(wire_messages, tools, **kw)

        try:
            response = await litellm.acompletion(**call_kwargs)
        except Exception as exc:  # noqa: BLE001 - mapped to taxonomy below
            raise self._map_error(exc) from exc

        return self._normalize(response)

    # ------------------------------------------------------------------
    # Streaming: litellm chunk stream -> ProviderStreamEvent stream
    # ------------------------------------------------------------------
    @classmethod
    def _parse_chunk(cls, chunk: Any) -> tuple[list[ProviderStreamEvent], str | None]:
        """Translate one raw litellm stream chunk into provider events.

        Returns ``(events, finish_reason)`` where ``events`` is the (possibly
        empty) list of :data:`ProviderStreamEvent` to yield for this chunk and
        ``finish_reason`` is the choice's finish reason if present on this chunk
        (``None`` otherwise). Every field is read via :func:`_get` so the chunk
        may be a pydantic-like object or a plain dict. Nothing here raises: a
        malformed chunk simply yields no events.

        Ordering within a chunk is text -> tool-call fragments -> usage, mirroring
        how a single OpenAI-shaped delta is laid out. ``StreamDone`` is emitted by
        the caller after the loop, never here.
        """
        events: list[ProviderStreamEvent] = []
        finish_reason: str | None = None

        choices = _get(chunk, "choices")
        first = None
        if isinstance(choices, list | tuple) and choices:
            first = choices[0]

        if first is not None:
            delta = _get(first, "delta")

            content = _get(delta, "content")
            if isinstance(content, str) and content:
                events.append(StreamTextDelta(text=content))

            raw_tool_calls = _get(delta, "tool_calls")
            if raw_tool_calls:
                for tc in raw_tool_calls:
                    index = _get(tc, "index")
                    index = index if isinstance(index, int) and not isinstance(index, bool) else 0
                    tc_id = _get(tc, "id")
                    fn = _get(tc, "function")
                    name = _get(fn, "name")
                    raw_args = _get(fn, "arguments")
                    # Fragments pass through verbatim; the loop reassembles them.
                    args_delta = raw_args if isinstance(raw_args, str) else ""
                    events.append(
                        StreamToolCallDelta(
                            index=index,
                            id=tc_id if isinstance(tc_id, str) else None,
                            name=name if isinstance(name, str) else None,
                            arguments_delta=args_delta,
                        )
                    )

            fr = _get(first, "finish_reason")
            if fr is not None:
                finish_reason = str(fr)

        # Usage typically rides the final chunk (stream_options include_usage).
        if _get(chunk, "usage") is not None:
            events.append(StreamUsage(usage=cls._extract_usage(chunk)))

        return events, finish_reason

    async def astream(
        self,
        messages: list[Message],
        *,
        system: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        **kw: Any,
    ) -> AsyncIterator[ProviderStreamEvent]:
        """Stream a completion as true per-token :data:`ProviderStreamEvent`s.

        Overrides the base (acomplete-wrapping) default with real streaming:
        requests ``stream=True`` with ``include_usage`` so the final chunk carries
        token/cost usage, then translates each chunk via :meth:`_parse_chunk`.
        Tool-call argument fragments are passed through verbatim, keyed by index,
        for the loop's streaming accumulator to reassemble. A :class:`StreamDone`
        is always emitted last, even on an empty stream. Any litellm exception —
        whether raised by the setup call or during async iteration — is mapped
        through the error taxonomy; a raw vendor exception never escapes.
        """
        wire_messages = self._translate_messages(messages, system)
        call_kwargs = self._build_kwargs(wire_messages, tools, **kw)
        call_kwargs["stream"] = True
        call_kwargs["stream_options"] = {"include_usage": True}

        finish_reason: str | None = None
        try:
            resp = await litellm.acompletion(**call_kwargs)
            async for chunk in resp:
                events, fr = self._parse_chunk(chunk)
                if fr is not None:
                    finish_reason = fr
                for event in events:
                    yield event
        except ProviderError:
            raise
        except Exception as exc:  # noqa: BLE001 - mapped to taxonomy below
            raise self._map_error(exc) from exc

        yield StreamDone(finish_reason=finish_reason)

    def count_tokens(self, messages: list[Message], *, system: str | None = None) -> int:
        wire_messages = self._translate_messages(messages, system)
        try:
            count = litellm.token_counter(model=self.model, messages=wire_messages)
        except Exception:
            return self._rough_token_estimate(wire_messages)
        # bool is an int subclass; reject it (and any non-numeric) explicitly.
        if isinstance(count, bool):
            return self._rough_token_estimate(wire_messages)
        if isinstance(count, int):
            return count if count >= 0 else self._rough_token_estimate(wire_messages)
        try:
            coerced = int(count)
        except (TypeError, ValueError):
            return self._rough_token_estimate(wire_messages)
        return coerced if coerced >= 0 else self._rough_token_estimate(wire_messages)

    @staticmethod
    def _rough_token_estimate(wire_messages: list[dict[str, Any]]) -> int:
        # ~4 chars per token, plus a small per-message overhead.
        chars = 0
        for m in wire_messages:
            content = m.get("content")
            if isinstance(content, str):
                chars += len(content)
            for tc in m.get("tool_calls", []) or []:
                fn = tc.get("function", {}) if isinstance(tc, dict) else {}
                args = fn.get("arguments", "")
                if isinstance(args, str):
                    chars += len(args)
        return chars // 4 + 4 * len(wire_messages)

    def capabilities(self) -> Capabilities:
        caps = get_capabilities(self.model)
        # Best-effort gate: if litellm is confident the model lacks function
        # calling, reflect that. Never hard-fail on the probe.
        try:
            supports = litellm.supports_function_calling(model=self.model)
        except Exception:
            supports = None
        if supports is False:
            caps = caps.model_copy(update={"supports_tools": False})
        # Keep the reported window in lockstep with the num_ctx we actually request for
        # Ollama (see _OLLAMA_NUM_CTX_CAP / _build_kwargs) so the compactor's threshold
        # matches Ollama's real context and never truncates before compaction fires.
        if _is_ollama_model(self.model) and caps.context_window > _OLLAMA_NUM_CTX_CAP:
            caps = caps.model_copy(update={"context_window": _OLLAMA_NUM_CTX_CAP})
        # Declare the backend's stop-sequence cap so the text layer trims its sentinel list
        # (OpenAI/Azure reject >4); leaves unknown backends unbounded. (audit2 #8)
        limit = _max_stop_sequences(self.model)
        if limit is not None and caps.max_stop_sequences is None:
            caps = caps.model_copy(update={"max_stop_sequences": limit})
        return caps


def _get(obj: Any, key: str) -> Any:
    """Read ``key`` from a dict or an attribute off an object.

    litellm returns pydantic-like response objects whose fields are attributes,
    but tests and some code paths pass plain dicts. This handles both.
    """
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj.get(key)
    return getattr(obj, key, None)
