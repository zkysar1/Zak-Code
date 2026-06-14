"""Client-facing streaming events emitted by the agent loop (M1).

:meth:`AgentLoop.astream_turn` yields a stream of :data:`AgentEvent` so a client
(CLI now, HTTP/SSE/WebSocket later) can render a turn incrementally — token by
token, tool call by tool call — instead of waiting for the whole turn to finish.

This is the **loop/client** contract. It sits one level above the provider-level
:data:`~zakcode.providers.base.ProviderStreamEvent`: the loop consumes the raw
provider token stream, drives tools and the iteration cycle, and re-emits this
higher-level, transport-friendly stream. Every event is a small pydantic model
discriminated on ``event`` so the whole stream round-trips through JSON unchanged
— the property the future server serializes to SSE/WS frames.

Design note: the non-streaming :class:`~zakcode.agent.loop.TurnResult` remains the
return value of :meth:`AgentLoop.arun_turn`; this event stream is the *incremental*
view of the same turn. ``AgentDone`` carries the same terminal summary fields so a
streaming client never needs the buffered result.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field

from zakcode.usage import Usage


class AgentTextDelta(BaseModel):
    """An incremental chunk of assistant-visible text."""

    event: Literal["text"] = "text"
    text: str


class AgentToolCall(BaseModel):
    """The loop is about to execute a (fully-assembled) tool call."""

    event: Literal["tool_call"] = "tool_call"
    id: str
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class AgentToolResult(BaseModel):
    """A tool finished; its result is being fed back into the conversation."""

    event: Literal["tool_result"] = "tool_result"
    tool_use_id: str
    output: str = ""
    is_error: bool = False


class AgentStatus(BaseModel):
    """A human-facing progress note (e.g. an iteration boundary or a notice).

    Advisory only — clients may render it in a status line or ignore it.
    """

    event: Literal["status"] = "status"
    message: str


class AgentTaskUpdate(BaseModel):
    """The live hierarchical plan changed — a client may render the task list.

    ``plan`` is the human-readable checklist; ``tasks`` is the structured tree (each node:
    ``id, title, status, kind, note, children``) for a rich client that draws its own UI.
    Advisory, like :class:`AgentStatus`: a client may render or ignore it.
    """

    event: Literal["task_update"] = "task_update"
    plan: str = ""
    tasks: list[dict[str, Any]] = Field(default_factory=list)
    finished: int = 0
    total: int = 0
    complete: bool = False


class AgentUsage(BaseModel):
    """Cumulative token/cost usage for the turn so far."""

    event: Literal["usage"] = "usage"
    usage: Usage = Field(default_factory=Usage)


class AgentDone(BaseModel):
    """Terminal event: the turn finished. Mirrors the buffered ``TurnResult``."""

    event: Literal["done"] = "done"
    stop_reason: str
    iterations: int
    usage: Usage = Field(default_factory=Usage)
    #: Thin "this turn struggled" roll-up (mirrors ``TurnResult.degraded``): True when the
    #: turn engaged failure-recovery (a stuck nudge/narrow) or ended non-cleanly
    #: (stuck / doom_loop / recipe_stalled). False on a clean turn.
    degraded: bool = False
    #: Mirrors ``TurnResult.error``: the (already secret-redacted) failure detail when
    #: ``stop_reason == "provider_error"``, so a streaming client consuming only the
    #: terminal event still learns WHY the turn failed; empty on every other stop.
    error: str = ""


AgentEvent = Annotated[
    AgentTextDelta
    | AgentToolCall
    | AgentToolResult
    | AgentStatus
    | AgentTaskUpdate
    | AgentUsage
    | AgentDone,
    Field(discriminator="event"),
]


__all__ = [
    "AgentTextDelta",
    "AgentToolCall",
    "AgentToolResult",
    "AgentStatus",
    "AgentTaskUpdate",
    "AgentUsage",
    "AgentDone",
    "AgentEvent",
]
