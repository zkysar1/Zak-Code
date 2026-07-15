"""SafeEventProjection — the source-side content filter for the public watch surface.

Transforms raw :data:`~zakcode.events.AgentEvent` frames into SAFE public frames BEFORE
they are serialized to a kid-facing watch stream. This is Layer 4 of the watch security
model: filtering happens INSIDE the server's type layer as a **whitelist projection**,
never in a downstream proxy/nginx string filter.

The discipline is subtractive, not cosmetic:

* Tool **arguments** are stripped ENTIRELY (a viewer sees that ``bash`` ran, never the
  command). Tool **output** is stripped ENTIRELY (never the file contents / listing).
* ``usage`` (cost/token) events are DROPPED. So is any unrecognized event type — a
  malformed or future event fails CLOSED (``None``), never leaks.
* Every surviving free-text field (assistant prose, status notes, task titles) is run
  through :func:`~zakcode.secrets.redact_secrets_extended` as defense-in-depth.

``AgentToolResult`` carries no tool name (only ``tool_use_id``), so the projection is
stateful: it records ``id → name`` from each ``AgentToolCall`` and uses it to label the
matching result badge. Instances are therefore per-session (they see a turn's events in
order) and owned by :class:`~zakcode.server.broadcast.SessionBroadcaster`.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from zakcode.events import (
    AgentDone,
    AgentEvent,
    AgentStatus,
    AgentTaskUpdate,
    AgentTextDelta,
    AgentToolCall,
    AgentToolResult,
    AgentUsage,
)
from zakcode.secrets import redact_secrets_extended

#: A soft cap on the in-flight ``id → name`` map so a pathological turn (thousands of
#: unmatched tool calls) cannot grow it without bound. Cleared on every ``AgentDone``.
_MAX_PENDING_TOOL_NAMES = 512


class SafeEventProjection:
    """Project raw agent events into safe, kid-facing watch frames (whitelist)."""

    def __init__(
        self,
        *,
        secret_values: Iterable[str] = (),
        workspace_paths: Iterable[str] = (),
    ) -> None:
        self._secret_values = tuple(secret_values)
        self._workspace_paths = tuple(workspace_paths)
        # tool_use_id -> tool name, so an AgentToolResult (which has no name) can be
        # labeled from the AgentToolCall that preceded it.
        self._tool_names: dict[str, str] = {}

    def _redact(self, text: str) -> str:
        scrubbed, _ = redact_secrets_extended(
            text,
            secret_values=self._secret_values,
            workspace_paths=self._workspace_paths,
        )
        return scrubbed

    def project(self, event: AgentEvent) -> dict[str, Any] | None:
        """Return the safe public frame for ``event``, or ``None`` to drop it.

        Fails CLOSED: any exception (a malformed or unexpected event) yields ``None`` so
        nothing un-projected can ever reach a watcher.
        """
        try:
            return self._project(event)
        except Exception:  # noqa: BLE001 — a projection error must drop the frame, never leak
            return None

    def _project(self, event: AgentEvent) -> dict[str, Any] | None:
        if isinstance(event, AgentTextDelta):
            return {"event": "text", "text": self._redact(event.text)}

        if isinstance(event, AgentStatus):
            # Human-facing progress note; redact defensively (a status could echo a path).
            return {"event": "status", "message": self._redact(event.message)}

        if isinstance(event, AgentTaskUpdate):
            # Keep only the checklist SHAPE: title + status per node. Drop kind/note/children
            # (may carry paths/detail) and the raw ``plan`` string (model-authored free text).
            tasks = [
                {
                    "title": self._redact(str(task.get("title", ""))),
                    "status": str(task.get("status", "")),
                }
                for task in event.tasks
            ]
            return {
                "event": "task_update",
                "tasks": tasks,
                "finished": event.finished,
                "total": event.total,
                "complete": event.complete,
            }

        if isinstance(event, AgentToolCall):
            # STRIP arguments entirely; remember the name to label the result badge.
            if len(self._tool_names) < _MAX_PENDING_TOOL_NAMES:
                self._tool_names[event.id] = event.name
            return {"event": "tool_summary", "name": event.name, "status": "running"}

        if isinstance(event, AgentToolResult):
            # STRIP output/data/artifacts entirely; a viewer sees only that a named tool finished.
            name = self._tool_names.pop(event.tool_use_id, "tool")
            return {
                "event": "tool_summary",
                "name": name,
                "status": "failed" if event.is_error else "completed",
            }

        if isinstance(event, AgentUsage):
            return None  # DROP — cost/token data is operator-internal.

        if isinstance(event, AgentDone):
            self._tool_names.clear()
            # stop_reason is a controlled vocabulary ("completed"/"stuck"/…); strip error,
            # trace, usage, degraded, and routing internals.
            return {"event": "done", "stop_reason": event.stop_reason}

        return None  # unknown/unhandled event → fail closed.


__all__ = ["SafeEventProjection"]
