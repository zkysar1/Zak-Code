"""The vendor-agnostic provider contract.

This module defines the seam between the agent loop and any LLM backend. The loop depends
only on :class:`Provider` (an ABC) and the canonical request/response value objects here;
it never imports ``litellm`` or any vendor SDK. Concrete providers (see
``litellm_provider.py``) translate the canonical shape to a backend and normalize the
response back. Swapping providers is therefore a configuration change, not a code change.

M0 implements the non-streaming path (:meth:`Provider.acomplete`); streaming
(:meth:`Provider.astream`) and its flat event model arrive in M1 (see ``docs/ROADMAP.md``).
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field

from zakcode.messages import Message
from zakcode.usage import Usage

# ── Canonical response value objects ─────────────────────────────────────────


class ToolCall(BaseModel):
    """A normalized tool-call request parsed from a model response.

    ``arguments`` is always a parsed ``dict`` (the provider is responsible for decoding the
    vendor's JSON-string arguments). On a decode failure a provider should fall back to
    ``{"_raw": "<original string>"}`` rather than raising, so the loop never crashes.
    """

    id: str
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class LLMResult(BaseModel):
    """The normalized result of one model call."""

    text: str = ""
    tool_calls: list[ToolCall] = Field(default_factory=list)
    finish_reason: str | None = None
    usage: Usage = Field(default_factory=Usage)
    raw: dict[str, Any] | None = None

    @property
    def has_tool_calls(self) -> bool:
        """Whether the model requested any tool invocations."""
        return bool(self.tool_calls)


class Capabilities(BaseModel):
    """Static facts about a model, used for routing and context budgeting."""

    supports_tools: bool = True
    supports_vision: bool = False
    supports_caching: bool = False
    context_window: int = 8192
    max_output: int | None = None
    #: Maximum number of ``stop`` sequences the backend accepts in one request (e.g. the
    #: OpenAI Chat Completions API rejects more than 4). ``None`` means unbounded / unknown;
    #: the text tool-calling layer caps its sentinel list to this when set. (audit2 #8)
    max_stop_sequences: int | None = None


# ── Streaming events (yielded by Provider.astream) ───────────────────────────
# Added in M1. These describe the *provider*-level token stream; the agent loop
# consumes them and re-emits its own client-facing AgentEvent stream (see
# zakcode.events). Discriminated on ``event`` so a heterogeneous stream is
# trivially (de)serializable for the future HTTP/SSE transport.


class StreamTextDelta(BaseModel):
    """An incremental chunk of assistant text."""

    event: Literal["text_delta"] = "text_delta"
    text: str


class StreamToolCallDelta(BaseModel):
    """An incremental fragment of a tool call, keyed by ``index``.

    Only the first fragment for a given ``index`` carries ``id``/``name``; later
    fragments carry partial ``arguments_delta`` strings to be concatenated and
    JSON-parsed once the call is complete (the streaming-accumulator's job).
    """

    event: Literal["tool_call_delta"] = "tool_call_delta"
    index: int
    id: str | None = None
    name: str | None = None
    arguments_delta: str = ""


class StreamUsage(BaseModel):
    """Token/cost usage for the streamed turn (typically the final chunk)."""

    event: Literal["usage"] = "usage"
    usage: Usage = Field(default_factory=Usage)


class StreamDone(BaseModel):
    """Terminal event: the model finished this streamed response."""

    event: Literal["done"] = "done"
    finish_reason: str | None = None


ProviderStreamEvent = Annotated[
    StreamTextDelta | StreamToolCallDelta | StreamUsage | StreamDone,
    Field(discriminator="event"),
]


# ── Error taxonomy ───────────────────────────────────────────────────────────


class ProviderError(Exception):
    """Base class for all normalized provider errors."""


class AuthError(ProviderError):
    """Authentication/authorization failed (bad or missing credentials)."""


class ContextWindowExceeded(ProviderError):
    """The request exceeded the model's context window.

    Signals the loop to compact rather than retry — this is never retried.
    """


class RateLimited(ProviderError):
    """The provider rate-limited the request.

    ``retry_after`` is the server-suggested delay in seconds, when known.
    """

    def __init__(self, message: str, retry_after: float | None = None) -> None:
        super().__init__(message)
        self.retry_after = retry_after


class RequestFailed(ProviderError):
    """A request failed for a reason not covered by the more specific errors."""


# ── Provider ABC ─────────────────────────────────────────────────────────────


class Provider(ABC):
    """Abstract interface every LLM backend implements.

    Implementations own message/tool translation, response normalization, retry/backoff,
    error-taxonomy mapping, and token/cost accounting. The agent loop holds only this type.
    """

    @abstractmethod
    async def acomplete(
        self,
        messages: list[Message],
        *,
        system: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        response_format: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> LLMResult:
        """Run one non-streaming completion and return a normalized :class:`LLMResult`.

        ``tools`` are OpenAI-shaped JSON-schema tool definitions (the provider translates
        them per-backend). ``response_format`` is an OpenAI-shaped structured-output request
        (``{"type": "json_object"}`` or a ``json_schema`` block — see
        :func:`zakcode.providers.structured.make_response_format`); a backend that cannot honor
        it may ignore it, so callers must still VALIDATE the returned text (the kwarg is a
        request, not a guarantee). Implementations must map backend failures onto the error
        taxonomy above rather than leaking vendor exceptions.
        """
        raise NotImplementedError

    @abstractmethod
    def count_tokens(self, messages: list[Message], *, system: str | None = None) -> int:
        """Best-effort token count for the given context.

        Prefer a real tokenizer when available. Rough estimates (e.g. char/4)
        are acceptable for providers without tokenizer access (local servers,
        bridge-based providers).
        """
        raise NotImplementedError

    @abstractmethod
    def capabilities(self) -> Capabilities:
        """Return static capabilities for the configured model."""
        raise NotImplementedError

    async def astream(
        self,
        messages: list[Message],
        *,
        system: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        response_format: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[ProviderStreamEvent]:
        """Stream a completion as a sequence of :data:`ProviderStreamEvent`.

        The default implementation is **non-streaming**: it awaits
        :meth:`acomplete` and emits the whole result as one text delta (if any),
        one :class:`StreamToolCallDelta` per complete tool call, a usage event,
        and a done event. This makes streaming work for every provider out of the
        box; a concrete provider should override it with true token streaming.
        """
        # Forward structured-output only when set, so this default astream stays a transparent
        # passthrough for a minimal Provider whose acomplete predates response_format.
        if response_format is not None:
            kwargs["response_format"] = response_format
        result = await self.acomplete(messages, system=system, tools=tools, **kwargs)
        if result.text:
            yield StreamTextDelta(text=result.text)
        for index, call in enumerate(result.tool_calls):
            yield StreamToolCallDelta(
                index=index,
                id=call.id,
                name=call.name,
                arguments_delta=json.dumps(call.arguments),
            )
        yield StreamUsage(usage=result.usage)
        yield StreamDone(finish_reason=result.finish_reason)


__all__ = [
    "ToolCall",
    "LLMResult",
    "Capabilities",
    "StreamTextDelta",
    "StreamToolCallDelta",
    "StreamUsage",
    "StreamDone",
    "ProviderStreamEvent",
    "ProviderError",
    "AuthError",
    "ContextWindowExceeded",
    "RateLimited",
    "RequestFailed",
    "Provider",
]
