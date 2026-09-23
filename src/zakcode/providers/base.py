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
import re
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Iterable
from dataclasses import dataclass
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
    #: Model-internal reasoning, when the backend surfaces it separately from ``text``
    #: (litellm normalizes Anthropic extended thinking and Groq-hosted reasoning models
    #: to ``message.reasoning_content``). Kept OUT of ``text`` so reasoning never leaks
    #: into the conversation as assistant prose; empty for models that don't reason.
    thinking: str = ""
    tool_calls: list[ToolCall] = Field(default_factory=list)
    finish_reason: str | None = None
    #: The model id the BACKEND echoed on this response (ADR-0214), which is not always
    #: the id the request named: an endpoint may publish several ids as aliases onto one
    #: set of weights, and a router answers with its own canonical name. A run tagged only
    #: by what it ASKED for cannot be reconstructed afterwards -- the server's access log
    #: records the name it resolved to, not the one the caller sent, so the echo is the
    #: only place the two are ever seen side by side. ``None`` when the backend echoed no
    #: model at all. Buffered calls only: a STREAMED call has no such fact to carry, because
    #: litellm stamps the requested model onto every chunk it builds (measured 2026-09-22).
    served_model: str | None = None
    usage: Usage = Field(default_factory=Usage)
    raw: dict[str, Any] | None = None

    @property
    def has_tool_calls(self) -> bool:
        """Whether the model requested any tool invocations."""
        return bool(self.tool_calls)


class UnknownContextWindow(ValueError):
    """A model whose context window nobody knows (ADR-0066).

    Raised at startup (naming every offender) and at provider construction — never
    swallowed into a stand-in number. The message names the model, what was checked, and
    the value the server declares when its ``/v1/models`` listing has one, so the operator
    pastes it into the model's config entry once. A configuration error, not a provider
    failure: it must not be retried, failed over, or reported as ``provider_error``.
    """


@dataclass(frozen=True)
class WindowResolution:
    """Where a model's context window came from (ADR-0066) — and what the server says.

    ``window`` is the number every window-keyed limit runs on; ``source`` names it:
    ``config`` (the model's own entry), ``registry`` (the checked-in table or litellm's
    metadata), ``sentinel`` (a routing name, no model yet), or ``unknown`` (nobody knows
    — the provider refuses to exist). ``served`` is what the server's ``/models`` listing
    declares when it was consulted and had a figure; it is a CHECK, never the source.
    """

    window: int | None
    source: str
    served: int | None = None

    @property
    def mismatch(self) -> bool:
        """The server declares a different window than the one in force."""
        return self.window is not None and self.served is not None and self.served != self.window

    def describe(self) -> str:
        """``131,072 (config)`` / ``128,000 (registry)`` / ``unknown`` — for info panels."""
        if self.window is None:
            return "unknown" if self.source == "unknown" else f"({self.source})"
        text = f"{self.window:,} ({self.source})"
        if self.served is not None:
            text += f" — server declares {self.served:,}" if self.mismatch else ", server agrees"
        return text


class Capabilities(BaseModel):
    """Static facts about a model, used for routing and context budgeting."""

    supports_tools: bool = True
    #: The model nominally supports tools but fails them in practice (e.g. emits a
    #: format its host rejects). Reliability is capability METADATA, not a hardcoded
    #: sort: the auto-resolver skips tools-unreliable models whenever the session
    #: has tools registered, so the rule survives future model additions. (D21)
    tools_unreliable: bool = False
    #: The provider has REMOVED this model from its live catalog. Kept in the
    #: registry so historical receipts and pinned configs still resolve their
    #: capability facts, but nothing may ROUTE to it. Structured rather than a
    #: comment because a comment cannot be asserted: qwen3-32b was decommissioned
    #: 2026-07-19, the note was written in the registry, its successor's
    #: capabilities were registered — and DEFAULT_CATEGORY_MODELS kept routing two
    #: categories to the dead model for ten days because nothing checked. (g-016-83)
    decommissioned: bool = False
    supports_vision: bool = False
    supports_caching: bool = False
    #: The model's context window in tokens. ``None`` means UNKNOWN — a first-class state,
    #: never a stand-in number (ADR-0066). The loop refuses to run on an unknown window, and
    #: the litellm provider refuses to be built on one, because every window-keyed limit
    #: (the seam clamp, the compaction threshold, overflow recovery) inherits it: the old
    #: 8,192 default silently cut every skill body on a 131k pod to 6 KB (coach, 2026-08-28).
    context_window: int | None = None
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


class StreamThinkingDelta(BaseModel):
    """An incremental chunk of model *reasoning* ("thinking"), never assistant text.

    A SEPARATE variant from :class:`StreamTextDelta` on purpose, and the separation
    is the invariant — not a stylistic choice. Reasoning must never be concatenated
    into the assistant's answer: it is the model's scratchpad, it is not addressed
    to the user, and on some providers it is explicitly not for display alongside
    output. Giving it its own event type is what lets a client render it in its own
    region (dim, collapsible, discarded) while making the "just treat it as text"
    mistake impossible to make by accident.

    Reasoning models emit these incrementally throughout a thinking phase that can
    last minutes; before this variant existed the mapper dropped every one of them
    and a streaming client saw a silent zero-byte window for the whole phase.
    """

    event: Literal["thinking_delta"] = "thinking_delta"
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
    StreamTextDelta | StreamThinkingDelta | StreamToolCallDelta | StreamUsage | StreamDone,
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


class TimedOut(RateLimited):
    """The request exceeded the client-side timeout (``ZAKCODE_REQUEST_TIMEOUT``).

    Subclasses :class:`RateLimited` ONLY for its bounded-retry semantics. The
    operator-facing notice must name the timeout, not a rate limit: on an
    uncached local backend a long-context call can genuinely need more than the
    configured timeout, and every retry pays the full prefill again — so the
    remedy is the timeout knob (or a smaller call), and a "rate limited" label
    sends the operator to the wrong one (zc-03 coach boot wedges, 2026-08-25).
    """


class ProviderUnavailable(RateLimited):
    """The provider could not be reached, or answered with a server error: a refused or
    dropped connection, a 5xx.

    Subclasses :class:`RateLimited` ONLY to ride its retry, under the same wall-clock
    backoff horizon as a 429: an inference pod whose router or engine restarts is back
    within a minute or two, and waiting is the remedy. The notice must say what happened,
    as :class:`TimedOut`'s does. Measured 2026-09-23: zakpod1's router restarted at 06:27Z
    and all three bodies on it reported "rate limited; retrying" for an outage no quota or
    throttle caused.
    """


class ModelOutputRejected(RateLimited):
    """The provider rejected the model's own output (e.g. a malformed tool call).

    Groq returns HTTP 400 with ``code: "tool_use_failed"`` when the model emits a
    tool call its validator cannot parse. Like a 429, the documented remedy is to
    retry the call — the model usually produces a valid tool call on the next
    attempt — so this subclasses :class:`RateLimited` to ride the loop's bounded
    retry machinery, with ``retry_after=0`` (nothing to wait for).
    """

    def __init__(self, message: str) -> None:
        super().__init__(message, retry_after=0.0)


class RequestFailed(ProviderError):
    """A request failed for a reason not covered by the more specific errors."""


class QuotaExhausted(ProviderError):
    """The account is out of quota/credit — a PERMANENT refusal, never retried.

    Deliberately NOT a :class:`RateLimited` subclass, and that is the entire fix.
    A quota refusal arrives as HTTP 429 exactly like a transient throttle, so it
    classified as ``RateLimited`` and rode the loop's ~15-minute backoff horizon:
    fifteen minutes of billed compute per turn whose outcome is guaranteed to be
    zero tokens, because the remedy is a billing action outside the fleet's
    control and no retry can reach it.

    Measured 2026-09-14 on a live dev vessel (g-373-80): the first call refused
    at 12:04:25, the loop logged "rate limited; retrying" throughout, and the
    provider's real message surfaced only at 12:19:26 when the budget ran out —
    the turn ended ``stop_reason="provider_error"``, iterations=1, tokens=0,
    transcript 0 B. A run capped or stopped before budget exhaustion never
    reaches the line that carries the truth at all.

    Because this subclasses :class:`ProviderError` directly, neither ``except
    RateLimited`` retry site can catch it, and it reaches the loop's graceful
    ``provider_error`` terminal on the FIRST refusal — which records ``str(exc)``,
    the provider's own text, as the turn error. The fast exit and the operator
    diagnostic are therefore properties of the CLASS, not of a branch someone has
    to remember to keep.
    """


#: Substrings that identify a PERMANENT quota/credit refusal. Lower-cased compare.
#:
#: MEASURED (g-373-80, 2026-09-15): litellm maps HTTP 429 to ``RateLimitError`` on
#: the STATUS CODE ALONE — eight 429 branches in ``exception_mapping_utils.py``,
#: not one of which reads the response body's ``code`` — and ``insufficient_quota``
#: appears ZERO times across its 1,812 files. The permanent-vs-transient
#: distinction is ERASED before the exception reaches us, so no widening of the
#: exception TYPE can recover it. The message text is the only carrier left, which
#: is why this list exists at all rather than a richer type lattice.
#:
#: THIS LIST IS A JUDGEMENT CALL, NOT A MEASUREMENT. It is the vendor phrasing
#: known at the time of writing; no sample of a real error body is recorded in this
#: repo. It is deliberately the ONE place to extend when a provider's wording is
#: observed to differ.
#:
#: Never key a marker on a FIELD NAME such as ``api_key`` / ``token`` / ``secret``
#: / ``password`` / ``bearer``: :func:`~zakcode.secrets.redact_secrets` replaces the
#: value following those names, so such a marker would be destroyed before any
#: reader saw it — silently, and with no error. Key on the literal, as below.
QUOTA_EXHAUSTION_MARKERS: tuple[str, ...] = (
    "insufficient_quota",
    "exceeded your current quota",
    "billing_hard_limit_reached",
    "out of credits",
    "credit balance is too low",
)


def quota_exhaustion_marker(text: str | BaseException) -> str | None:
    """Return the marker proving ``text`` is a permanent quota refusal, else None."""
    lowered = str(text).lower()
    for marker in QUOTA_EXHAUSTION_MARKERS:
        if marker in lowered:
            return marker
    return None


#: How a provider names a request field it refuses, one pattern per vendor phrasing
#: (case-insensitive; group 1 is the field). The list is a JUDGEMENT CALL like the quota
#: markers above — the phrasings known at the time of writing — and the one place to
#: extend when a vendor's wording is observed to differ. The Vertex form is the measured
#: one (2026-09-17, a served Mind on ``vertex_ai_beta``), and it arrives with the JSON
#: body's quotes backslash-escaped inside a bytes repr, which the ``\\*`` runs absorb.
#:
#: A pattern alone NEVER decides anything: :func:`rejected_request_field` returns a name
#: only when it is one the caller SENT. That gate is what keeps the repair honest — a
#: provider complaining about ``tools`` or ``messages`` is a real defect to surface, not
#: a field to strip and retry.
REJECTED_FIELD_PATTERNS: tuple[re.Pattern[str], ...] = (
    # Google (Vertex AI / Gemini): Invalid JSON payload received. Unknown name "x": …
    re.compile(r"unknown name\s+\\*[\"']([\w.\[\]-]+)\\*[\"']", re.IGNORECASE),
    # OpenAI: Unrecognized request argument supplied: x  (or "arguments … x, y")
    re.compile(r"unrecognized request arguments?\s+supplied:\s*([\w.,\s-]+)", re.IGNORECASE),
    # Anthropic (pydantic-style): x: Extra inputs are not permitted
    re.compile(r"([\w.]+):\s*extra inputs are not permitted", re.IGNORECASE),
    # FastAPI / pydantic v2 body validation (vLLM and kin): 'loc': ('body', 'x')
    re.compile(r"[\(\[]\s*[\"']body[\"'],\s*[\"']([\w.]+)[\"']", re.IGNORECASE),
    # Generic server phrasings.
    re.compile(r"unexpected keyword argument\s+[\"']([\w.]+)[\"']", re.IGNORECASE),
    re.compile(r"unknown (?:field|parameter|param|argument)s?[:\s]+[\"']?([\w.]+)", re.IGNORECASE),
    re.compile(r"unsupported parameter:?\s*[\"']([\w.]+)[\"']", re.IGNORECASE),
)


def rejected_request_field(text: str | BaseException, sent: Iterable[str]) -> str | None:
    """The field named in a request refusal ``text``, if it is one of the ``sent`` keys.

    ``sent`` lists the request fields the caller is willing to give up (the keys it
    added to the body itself, plus any wire-level aliases of a rendered kwarg). A dotted
    or bracketed path in the message (``extra_body.x``, ``x[0]``) matches on any of its
    segments, so a provider that reports the path still names the key. None when the
    text names nothing, or nothing the caller sent — the refusal is then a defect to
    surface, never a field to strip.
    """
    candidates = {key for key in sent if key}
    if not candidates:
        return None
    message = str(text)
    for pattern in REJECTED_FIELD_PATTERNS:
        for match in pattern.finditer(message):
            for raw in match.group(1).split(","):
                for segment in re.split(r"[.\[\]]", raw.strip()):
                    if segment in candidates:
                        return segment
    return None


# ── Provider ABC ─────────────────────────────────────────────────────────────


class Provider(ABC):
    """Abstract interface every LLM backend implements.

    Implementations own message/tool translation, response normalization, transport-level
    retry, error-taxonomy mapping, and token/cost accounting. The agent loop holds only
    this type. Rate-limit retry POLICY (whether to wait out a 429, how long, how often)
    deliberately lives one level up, in the agent loop — the harness decides whether
    waiting is worth it, not the transport (a fixed jittered-backoff horizon; see the
    retry-policy constants in ``agent/loop.py``); implementations should surface a
    clean :class:`RateLimited` rather than retrying 429s themselves.
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

    def model_id(self) -> str:
        """The litellm model string this provider runs, for per-model cost attribution.

        A concrete provider that knows its model overrides this (the loop tags each call's usage
        with it for the ``/cost`` breakdown). The default returns ``""`` — a provider without a
        single fixed model (or a test stub) simply isn't attributed, which is harmless.
        """
        return ""

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
        if result.thinking:
            # Reasoning keeps its own event type here too, so a consumer of the default
            # wrapper sees the same shape a true streaming provider emits (ADR-0056).
            yield StreamThinkingDelta(text=result.thinking)
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
    "StreamThinkingDelta",
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
    "UnknownContextWindow",
    "WindowResolution",
]
