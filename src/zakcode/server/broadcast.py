"""SessionBroadcaster — per-session fan-out of SAFE watch frames.

A turn's raw events (published from the ``/chat/stream`` and WebSocket turn loops) are
projected through a per-session :class:`~zakcode.server.safe_projection.SafeEventProjection`
into safe frames, buffered in a bounded deque for cursor replay, and fanned out to every
live ``/watch`` subscriber's queue.

The broadcaster only ever holds PROJECTED frames — raw ``AgentEvent``s never enter its
buffers or queues, so a bug on the watch path cannot leak tool arguments/output/usage. It
is pure in-process pub/sub (no network, no persistence): a watch stream is a live view,
not durable state. Each frame gets a per-session monotonic ``seq`` used as the SSE event
id, so a reconnecting client passes ``?since=<seq>`` and receives exactly the frames it
missed from the replay buffer, then live frames.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections import deque
from typing import Any

from zakcode.events import AgentEvent
from zakcode.server.safe_projection import SafeEventProjection

#: Frames retained per session for cursor replay (a reconnecting watcher catches up).
_BUFFER_PER_SESSION = 1000
#: Cap on the number of sessions whose watch state is retained, so a long-lived server that
#: never sees ``DELETE /sessions`` cannot grow unbounded. Oldest un-watched session evicted first.
_MAX_SESSIONS = 64


class SessionBroadcaster:
    """In-process, per-session fan-out of projected watch frames."""

    def __init__(
        self,
        *,
        secret_values: tuple[str, ...] | list[str] = (),
        workspace_paths: tuple[str, ...] | list[str] = (),
        buffer_size: int = _BUFFER_PER_SESSION,
        max_sessions: int = _MAX_SESSIONS,
    ) -> None:
        self._secret_values = tuple(secret_values)
        self._workspace_paths = tuple(workspace_paths)
        self._buffer_size = buffer_size
        self._max_sessions = max_sessions
        self._subscribers: dict[str, set[asyncio.Queue[dict[str, Any]]]] = {}
        self._buffers: dict[str, deque[dict[str, Any]]] = {}
        self._projections: dict[str, SafeEventProjection] = {}
        self._seq: dict[str, int] = {}

    def _projection_for(self, session_id: str) -> SafeEventProjection:
        projection = self._projections.get(session_id)
        if projection is None:
            projection = SafeEventProjection(
                secret_values=self._secret_values,
                workspace_paths=self._workspace_paths,
            )
            self._projections[session_id] = projection
        return projection

    async def publish(self, session_id: str, event: AgentEvent) -> None:
        """Project ``event`` and fan the safe frame out to all subscribers (no-op if dropped)."""
        frame = self._projection_for(session_id).project(event)
        if frame is None:
            return
        seq = self._seq.get(session_id, 0) + 1
        self._seq[session_id] = seq
        record: dict[str, Any] = {"seq": seq, "frame": frame}

        buffer = self._buffers.get(session_id)
        if buffer is None:
            buffer = deque(maxlen=self._buffer_size)
            self._buffers[session_id] = buffer
        buffer.append(record)

        for queue in list(self._subscribers.get(session_id, ())):
            # A slow watcher: drop the frame for it; it resyncs on reconnect via ?since.
            with contextlib.suppress(asyncio.QueueFull):
                queue.put_nowait(record)
        self._evict_if_needed()

    def history(self, session_id: str, since_seq: int = 0) -> list[dict[str, Any]]:
        """Buffered records with ``seq > since_seq`` — replay for a reconnecting watcher."""
        return [record for record in self._buffers.get(session_id, ()) if record["seq"] > since_seq]

    def subscribe(self, session_id: str) -> asyncio.Queue[dict[str, Any]]:
        """Register a live subscriber and return its queue of projected records."""
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=self._buffer_size)
        self._subscribers.setdefault(session_id, set()).add(queue)
        return queue

    def unsubscribe(self, session_id: str, queue: asyncio.Queue[dict[str, Any]]) -> None:
        """Remove a subscriber (its ``/watch`` generator closed)."""
        subscribers = self._subscribers.get(session_id)
        if subscribers is not None:
            subscribers.discard(queue)
            if not subscribers:
                self._subscribers.pop(session_id, None)

    def forget(self, session_id: str) -> None:
        """Drop all replay/projection state for a session (call when the session is deleted).

        Leaves any live subscriber set untouched — their generators unsubscribe on close.
        """
        self._buffers.pop(session_id, None)
        self._projections.pop(session_id, None)
        self._seq.pop(session_id, None)

    def _evict_if_needed(self) -> None:
        """Bound retained-session state: evict the oldest session with no live watcher."""
        while len(self._buffers) > self._max_sessions:
            for session_id in self._buffers:  # dict preserves insertion order → oldest first
                if session_id not in self._subscribers:
                    self._buffers.pop(session_id, None)
                    self._projections.pop(session_id, None)
                    self._seq.pop(session_id, None)
                    break
            else:
                break  # every retained session has a live watcher — keep them all


__all__ = ["SessionBroadcaster"]
