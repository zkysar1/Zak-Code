"""The HTTP/WS wire layer — JSON shapes the server and its clients exchange.

This is pure (de)serialization: no FastAPI, no agent logic. It defines

* :func:`event_to_dict` / :func:`event_from_dict` — round-trip a client-facing
  :data:`~zakcode.events.AgentEvent` through a plain ``dict`` (the body of every
  SSE frame and every server→client WebSocket message), and
* the request/response models the REST surface uses
  (:class:`ChatRequest`, :class:`ChatResponse`, :class:`SessionInfo`,
  :class:`ToolInfo`) and the WebSocket control frames
  (:class:`WSClientMessage`, :class:`WSActionRequired`).

Keeping it standalone means the whole wire contract is unit-testable without a
running server, and a future web/IDE client can depend on these shapes alone.
The single source of truth for the *event* shapes remains
:mod:`zakcode.events`; this module only serializes them.
"""

from __future__ import annotations

from typing import Any, Literal, get_args

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

from zakcode.agent.loop import TurnResult
from zakcode.artifacts import ArtifactRef
from zakcode.events import AgentEvent
from zakcode.messages import Message, ToolResultBlock
from zakcode.permissions import PermissionRequest
from zakcode.session.store import Session
from zakcode.tools.base import ToolSpec
from zakcode.usage import Usage

# One adapter for the whole discriminated AgentEvent union, reused for every
# frame. Validation (parse) and dumping (serialize) both go through it so the
# wire form always matches the in-process events exactly.
_EVENT_ADAPTER: TypeAdapter[AgentEvent] = TypeAdapter(AgentEvent)


def event_to_dict(event: AgentEvent) -> dict[str, Any]:
    """Serialize an :data:`AgentEvent` to a JSON-ready ``dict`` (one SSE/WS frame)."""
    return _EVENT_ADAPTER.dump_python(event, mode="json")


def event_from_dict(data: dict[str, Any]) -> AgentEvent:
    """Parse a frame ``dict`` back into the correct :data:`AgentEvent` member.

    Raises ``pydantic.ValidationError`` on an unknown ``event`` or a bad shape —
    a client never silently accepts a malformed frame.
    """
    return _EVENT_ADAPTER.validate_python(data)


def events_schema() -> dict[str, Any]:
    """The JSON Schema of the :data:`AgentEvent` union (the published wire contract).

    Generated from the same :data:`_EVENT_ADAPTER` that serializes/parses every frame,
    so it can never drift from the actual wire form. Served at ``GET /schema/events``
    and used by the web-client contract test to assert the client and server agree on
    the event vocabulary.
    """
    return _EVENT_ADAPTER.json_schema()


def event_type_names() -> list[str]:
    """The sorted set of ``event`` discriminator values (e.g. ``["done", "text", ...]``).

    A tiny, stable contract surface: the web client must handle exactly these, and the
    contract test asserts the renderer and the server share this set.
    """
    names: set[str] = set()
    for member in get_args(get_args(AgentEvent)[0]):
        field = member.model_fields.get("event")
        if field is not None and field.default is not None:
            names.add(str(field.default))
    return sorted(names)


# ── REST request / response models ────────────────────────────────────────────


class ChatRequest(BaseModel):
    """Body of ``POST /chat`` and ``POST /chat/stream``."""

    message: str
    session_id: str | None = None
    model: str | None = Field(
        default=None, description="Optional per-request model override (litellm string)."
    )


class NudgeRequest(BaseModel):
    """Body of ``POST /nudge`` — a viewer suggestion the driver folds into the next
    turn's preamble (never a direct chat message). Sanitization + rate-limiting are
    the caller's (gateway) responsibility; the server only length-caps and queues it."""

    text: str


class RunStopRequest(BaseModel):
    """Body of ``POST /run/stop`` — ask the RUN (not a turn) to end gracefully
    (ADR-0047). ``reason`` is recorded as the run's stop reason: a short lowercase
    token such as ``budget_exhausted``; omitted = ``stopped``."""

    reason: str | None = None


class SayRequest(BaseModel):
    """Body of ``POST /say`` — a user message the driver delivers as the next turn's
    MESSAGE (the watch/talk unification: talking is just the driven session's next
    turn). Distinct from a ``/nudge`` suggestion, which only decorates the preamble.
    Sanitization + rate-limiting + ownership are the caller's (gateway) responsibility;
    the server only length-caps and queues it."""

    text: str


class ObserveRequest(BaseModel):
    """Body of ``POST /observe`` — one perception envelope from this character's VESSEL
    (the environment server's ``PerceptionBridgeVerticle``), carrying world state the mind
    could not otherwise learn.

    The vessel-to-mind direction of the border contract
    (``world/conventions/character-seeding-and-discovery.md``). Distinct from BOTH sibling
    inputs: a ``/say`` IS the next turn's message and a ``/nudge`` decorates its preamble —
    both originate with a PERSON and are precious, hence their single-slot 429. An
    observation originates with the WORLD, is continuous, and is worthless once superseded,
    so this one is latest-wins and never refuses (P4: delivery is lossy; the mind must
    tolerate absence, and must equally tolerate a frame it never saw).

    ``observation`` is an opaque map of perception slices — the vessel's allow-rule is every
    ``privateSelf`` key whose name ends in ``Perception``, so the shape here is deliberately
    NOT enumerated: pinning today's slice names would silently reject tomorrow's. It is
    UNTRUSTED: its text originates in a shared world where other players author content, so
    it is framed as data, never as instruction (P1).
    """

    envelopeVersion: int
    externalClientRef: str
    observedAt: str
    observation: dict[str, Any] = Field(default_factory=dict)
    droppedSlices: list[str] = Field(default_factory=list)


class WatchMarkerRequest(BaseModel):
    """Body of ``POST /watch/{session_id}/marker`` — a server-side meta-event published to a
    session's watch bus so late-joining observers learn of lifecycle changes the ``AgentEvent``
    stream does not carry (e.g. a driver session rotation, a consumed user say). ``event`` is a
    closed allow-list matching the ``SafeEvent`` discriminator, so the body is publishable
    straight to the bus and projects through ``SafeEventProjection`` unchanged; ``reason`` is a
    short human-facing note and ``text`` the user-message body (both secret-redacted at
    projection). Published by the sidecar-driver via
    :meth:`~zakcode.server.client.ServerClient.publish_watch_marker` /
    :meth:`~zakcode.server.client.ServerClient.publish_user_message`."""

    event: Literal["session_rotated", "user_message"] = "session_rotated"
    reason: str = ""
    text: str = ""


class ChatResponse(BaseModel):
    """Buffered result of ``POST /chat`` (the non-streaming turn)."""

    session_id: str
    text: str = ""
    tool_results: list[ToolResultBlock] = Field(default_factory=list)
    usage: Usage = Field(default_factory=Usage)
    cost_usd: float = 0.0
    stop_reason: str = "completed"
    iterations: int = 0
    #: Mirrors ``TurnResult.error`` (secret-redacted provider-failure detail when
    #: ``stop_reason == "provider_error"``; empty otherwise) so a REST consumer can
    #: tell a rate-limit from an auth failure without reading server logs.
    error: str = ""

    @classmethod
    def from_turn(cls, session_id: str, result: TurnResult) -> ChatResponse:
        """Build a response from a session id and the loop's :class:`TurnResult`."""
        text = "\n".join(m.text for m in result.assistant_messages if m.text).strip()
        return cls(
            session_id=session_id,
            text=text,
            tool_results=list(result.tool_results),
            usage=result.usage,
            cost_usd=result.usage.cost_usd,
            stop_reason=result.stop_reason,
            iterations=result.iterations,
            error=result.error,
        )


class CompleteMessage(BaseModel):
    """One message in a ``POST /complete`` request (a raw, tool-less completion)."""

    role: Literal["system", "user", "assistant"]
    content: str


class CompleteRequest(BaseModel):
    """Body of ``POST /complete`` — a raw schema-valid completion (no tools, no agent loop).

    Supply EITHER ``prompt`` (a single user turn) OR ``messages`` (a full exchange), not both
    and not neither. ``schema`` (a JSON Schema) requests schema-valid JSON; ``system`` is an
    optional system prompt; ``model`` overrides the served model for this call.
    """

    model_config = ConfigDict(populate_by_name=True)

    prompt: str | None = None
    messages: list[CompleteMessage] | None = None
    system: str | None = None
    # ``schema`` shadows pydantic's BaseModel.schema(); expose it on the wire as "schema"
    # via the alias while storing it as ``schema_``.
    schema_: dict[str, Any] | None = Field(default=None, alias="schema")
    model: str | None = None

    def to_messages(self) -> list[Message]:
        """Map the request to canonical :class:`~zakcode.messages.Message` objects.

        Raises ``ValueError`` if both ``prompt`` and ``messages`` are set, or neither (the
        route turns that into a 400).
        """
        if self.prompt is not None and self.messages is not None:
            raise ValueError("provide either 'prompt' or 'messages', not both")
        if self.prompt is not None:
            return [Message.user(self.prompt)]
        if self.messages:
            out: list[Message] = []
            for m in self.messages:
                if m.role == "system":
                    out.append(Message.system(m.content))
                elif m.role == "assistant":
                    out.append(Message.assistant_text(m.content))
                else:
                    out.append(Message.user(m.content))
            return out
        raise ValueError("provide either 'prompt' or 'messages'")


class CompleteResponse(BaseModel):
    """Result of ``POST /complete``: schema-valid ``data`` (when a schema was requested) + text."""

    data: Any | None = None
    text: str = ""
    usage: Usage = Field(default_factory=Usage)
    cost_usd: float = 0.0
    repaired: bool = False


class UploadRequest(BaseModel):
    """Body of ``POST /sessions/{id}/uploads``.

    ``data`` accepts raw base64 or a ``data:*/*;base64,...`` URL so browser clients can use
    ``FileReader.readAsDataURL`` without a multipart parser dependency.
    """

    filename: str
    data: str


class UploadResponse(BaseModel):
    """Result of uploading one file into the workspace."""

    path: str
    bytes: int
    artifact: ArtifactRef
    suggested_tool: str = ""
    prompt: str = ""


class SessionInfo(BaseModel):
    """Summary of a stored session (``GET /sessions`` / ``GET /sessions/{id}``)."""

    id: str
    model: str = ""
    cwd: str = ""
    created_at: str = ""
    message_count: int = 0
    usage: Usage = Field(default_factory=Usage)

    @classmethod
    def from_session(cls, session: Session) -> SessionInfo:
        return cls(
            id=session.id,
            model=session.model,
            cwd=session.cwd,
            created_at=session.created_at,
            message_count=len(session.messages),
            usage=session.cumulative_usage(),
        )


class TranscriptMessage(BaseModel):
    """One spoken turn of a session, projected for a viewer (ADR-0041)."""

    role: Literal["user", "assistant"]
    text: str


class SessionTranscript(BaseModel):
    """``GET /sessions/{id}/transcript`` — the conversation as a reader would see it.

    User and assistant TEXT only: tool calls, tool results, thinking and system frames
    are not part of what was said, and the text is passed through the same secret
    redaction the watch stream applies. ``message_count`` is the session's full length
    (every stored message), so a consumer can tell "short transcript" from "short
    session"."""

    session_id: str
    messages: list[TranscriptMessage] = Field(default_factory=list)
    message_count: int = 0


class ToolInfo(BaseModel):
    """A registered tool's public description (``GET /tools``)."""

    name: str
    description: str = ""
    required_permission: str = ""
    concurrency: str = ""
    parameters: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_spec(cls, spec: ToolSpec) -> ToolInfo:
        return cls(
            name=spec.name,
            description=spec.description,
            required_permission=spec.required_permission.name,
            concurrency=spec.concurrency.value,
            parameters=spec.parameters,
        )


# ── WebSocket control frames ──────────────────────────────────────────────────
# Client → server messages are a small discriminated union; server → client
# frames are serialized AgentEvents plus the out-of-band action_required prompt.


class WSActionRequired(BaseModel):
    """Server→client: a tool call needs operator approval before it can run.

    Published by the server's say-inbox permission prompter onto the session's
    watch bus (the ``action_required`` control frame from the architecture's API
    surface — kept separate from :data:`AgentEvent` because it is a request *to*
    the operator, not a turn event). Answered through the say contract: a y/a/n
    written to the inbox (``POST /say``, ``zakcode say``, the cockpit box).
    """

    type: Literal["action_required"] = "action_required"
    tool_name: str
    tier: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    reason: str = ""

    @classmethod
    def from_request(cls, request: PermissionRequest) -> WSActionRequired:
        return cls(
            tool_name=request.tool_name,
            tier=request.tier.name,
            arguments=request.arguments,
            reason=request.reason,
        )


__all__ = [
    "event_to_dict",
    "event_from_dict",
    "ChatRequest",
    "ChatResponse",
    "CompleteMessage",
    "CompleteRequest",
    "CompleteResponse",
    "UploadRequest",
    "UploadResponse",
    "SessionInfo",
    "ToolInfo",
    "WSActionRequired",
]
