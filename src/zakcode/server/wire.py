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

from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field, TypeAdapter

from zakcode.agent.loop import TurnResult
from zakcode.events import AgentEvent
from zakcode.messages import ToolResultBlock
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


# ── REST request / response models ────────────────────────────────────────────


class ChatRequest(BaseModel):
    """Body of ``POST /chat`` and ``POST /chat/stream``."""

    message: str
    session_id: str | None = None
    model: str | None = Field(
        default=None, description="Optional per-request model override (litellm string)."
    )


class ChatResponse(BaseModel):
    """Buffered result of ``POST /chat`` (the non-streaming turn)."""

    session_id: str
    text: str = ""
    tool_results: list[ToolResultBlock] = Field(default_factory=list)
    usage: Usage = Field(default_factory=Usage)
    cost_usd: float = 0.0
    stop_reason: str = "completed"
    iterations: int = 0

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
        )


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


class WSUserInput(BaseModel):
    """Client asks the server to run a turn."""

    type: Literal["input"] = "input"
    message: str


class WSInterrupt(BaseModel):
    """Client asks the server to cancel the in-flight turn."""

    type: Literal["interrupt"] = "interrupt"


class WSApproval(BaseModel):
    """Client answers an :class:`WSActionRequired` permission prompt.

    ``outcome`` is a :class:`~zakcode.permissions.PermissionOutcome` value
    (``allow_once`` / ``allow_session`` / ``deny_once`` / ``deny_session``).
    """

    type: Literal["approval"] = "approval"
    outcome: str


WSClientMessage = Annotated[
    WSUserInput | WSInterrupt | WSApproval,
    Field(discriminator="type"),
]

_CLIENT_MSG_ADAPTER: TypeAdapter[WSClientMessage] = TypeAdapter(WSClientMessage)


def client_message_from_dict(data: dict[str, Any]) -> WSClientMessage:
    """Parse a client→server WebSocket frame into its typed message."""
    return _CLIENT_MSG_ADAPTER.validate_python(data)


class WSActionRequired(BaseModel):
    """Server→client: a tool call needs operator approval before it can run.

    Sent by the WebSocket permission bridge; the client replies with a
    :class:`WSApproval`. This is the ``action_required`` control frame from the
    architecture's API surface (kept separate from :data:`AgentEvent` because it
    is a request *to* the client, not a turn event).
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
    "SessionInfo",
    "ToolInfo",
    "WSUserInput",
    "WSInterrupt",
    "WSApproval",
    "WSClientMessage",
    "WSActionRequired",
    "client_message_from_dict",
]
