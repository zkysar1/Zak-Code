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

from abc import ABC, abstractmethod
from typing import Any

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
        **kwargs: Any,
    ) -> LLMResult:
        """Run one non-streaming completion and return a normalized :class:`LLMResult`.

        ``tools`` are OpenAI-shaped JSON-schema tool definitions (the provider translates
        them per-backend). Implementations must map backend failures onto the error
        taxonomy above rather than leaking vendor exceptions.
        """
        raise NotImplementedError

    @abstractmethod
    def count_tokens(self, messages: list[Message], *, system: str | None = None) -> int:
        """Best-effort token count for the given context (real tokenizer, never len/4)."""
        raise NotImplementedError

    @abstractmethod
    def capabilities(self) -> Capabilities:
        """Return static capabilities for the configured model."""
        raise NotImplementedError


__all__ = [
    "ToolCall",
    "LLMResult",
    "Capabilities",
    "ProviderError",
    "AuthError",
    "ContextWindowExceeded",
    "RateLimited",
    "RequestFailed",
    "Provider",
]
