"""Per-session event bus — read-only fan-out for the Pearl watch surface (P0-2).

The core streaming paths (``POST /chat/stream``, ``WS /ws/{session_id}``) deliver each
``AgentEvent`` to the ONE client driving a turn. The watch surface (``GET /watch/{session_id}``,
P0-3) needs a SEPARATE, read-only, late-joining observer to see the same events as they happen —
a parent watching a kid's agent, a dashboard tailing activity. This module is that fan-out.

Guarantees:

* **Bounded memory.** Each session retains at most ``maxlen`` events (default 1000) in a ring
  buffer; the oldest are evicted. A slow or abandoned watcher can never grow memory without
  bound. Because cursors are monotonic, a watcher that fell behind the retained window sees a
  cursor *gap* and knows it missed events.
* **Cursor-addressed replay.** Every published event gets a strictly increasing cursor (starting
  at 1). ``subscribe(since=c)`` replays every retained event with cursor > c, then tails live.
  ``since=None``/``0`` replays the whole retained buffer.
* **Per-subscriber backpressure.** Each subscriber has its own bounded queue; a stuck watcher
  drops ITS OWN oldest queued events (recoverable via the ring buffer + cursor gap) and never
  blocks the publisher or other subscribers.
* **No agent/HTTP coupling.** The bus buffers opaque objects and imports nothing from
  ``zakcode.events`` or the FastAPI app — projection to ``SafeEvent`` happens at the /watch
  endpoint, so the bus is reusable and unit-testable without the FastAPI/pydantic stack.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections import deque
from collections.abc import AsyncGenerator
from typing import Any, Final

_DEFAULT_MAXLEN: Final = 1000
#: Per-subscriber live-queue bound. A watcher that falls this far behind the live tail drops its
#: own oldest queued events (recoverable via the ring buffer + cursor gap); it never blocks the
#: publisher. Sized well above any human-watchable event rate, so only a truly stuck client trims.
_SUBSCRIBER_QUEUE_MAXLEN: Final = 512

#: Sentinel pushed to every subscriber queue on close(), so a live-tailing subscribe() coroutine
#: wakes and returns instead of hanging forever on an idle bus.
_CLOSE: Final = object()


class SessionEventBus:
    """A bounded, cursor-addressed pub/sub buffer for one session's events."""

    def __init__(self, maxlen: int = _DEFAULT_MAXLEN) -> None:
        self._buffer: deque[tuple[int, Any]] = deque(maxlen=maxlen)
        self._cursor = 0
        self._subscribers: set[asyncio.Queue[Any]] = set()
        self._closed = False

    @property
    def latest_cursor(self) -> int:
        """Cursor of the most recently published event (0 before anything is published)."""
        return self._cursor

    @property
    def subscriber_count(self) -> int:
        return len(self._subscribers)

    @property
    def closed(self) -> bool:
        return self._closed

    def publish(self, event: Any) -> int:
        """Append ``event`` to the ring buffer and fan it out to live subscribers.

        Returns the event's cursor. Never blocks: a subscriber whose queue is full drops its own
        oldest queued event to make room (the ring buffer + cursor let it recover the gap).
        Publishing to a closed bus raises ``RuntimeError`` — the caller owns the lifecycle.
        """
        if self._closed:
            raise RuntimeError("publish on a closed SessionEventBus")
        self._cursor += 1
        item = (self._cursor, event)
        self._buffer.append(item)
        for queue in list(self._subscribers):
            _offer(queue, item)
        return self._cursor

    async def subscribe(self, since: int | None = None) -> AsyncGenerator[tuple[int, Any], None]:
        """Yield ``(cursor, event)`` pairs: replay retained events after ``since``, then live.

        Registers a live queue BEFORE snapshotting the buffer so no event is lost in the gap
        between replay and tail; any overlap between the snapshot and the live queue is
        de-duplicated by cursor (each event is yielded exactly once, in cursor order). The
        iterator ends when the bus is closed. Always deregisters its queue on exit (normal
        completion, caller break, or cancellation).
        """
        floor = since or 0
        queue: asyncio.Queue[Any] = asyncio.Queue(maxsize=_SUBSCRIBER_QUEUE_MAXLEN)
        self._subscribers.add(queue)
        try:
            # Snapshot AFTER registering (both are synchronous, so atomic): anything published
            # from here on also lands in `queue`, and the `cursor <= last` guard drops any overlap.
            replay = [pair for pair in list(self._buffer) if pair[0] > floor]
            last = floor
            for cursor, event in replay:
                yield cursor, event
                last = cursor
            if self._closed:
                return
            while True:
                item = await queue.get()
                if item is _CLOSE:
                    return
                cursor, event = item
                if cursor <= last:
                    continue  # already delivered via replay (defensive: snapshot/live overlap)
                yield cursor, event
                last = cursor
        finally:
            self._subscribers.discard(queue)

    def close(self) -> None:
        """Mark the bus closed and wake every live subscriber so it returns."""
        if self._closed:
            return
        self._closed = True
        for queue in list(self._subscribers):
            _offer(queue, _CLOSE)


def _offer(queue: asyncio.Queue[Any], item: Any) -> None:
    """Put ``item`` on ``queue`` without blocking; if full, drop the oldest to make room."""
    try:
        queue.put_nowait(item)
    except asyncio.QueueFull:
        with contextlib.suppress(asyncio.QueueEmpty):  # pragma: no cover
            queue.get_nowait()
        with contextlib.suppress(asyncio.QueueFull):  # pragma: no cover
            queue.put_nowait(item)


class EventBusRegistry:
    """Maps ``session_id`` -> :class:`SessionEventBus`, creating buses on demand."""

    def __init__(self, maxlen: int = _DEFAULT_MAXLEN) -> None:
        self._buses: dict[str, SessionEventBus] = {}
        self._maxlen = maxlen

    def get_or_create(self, session_id: str) -> SessionEventBus:
        bus = self._buses.get(session_id)
        if bus is None or bus.closed:
            bus = SessionEventBus(maxlen=self._maxlen)
            self._buses[session_id] = bus
        return bus

    def get(self, session_id: str) -> SessionEventBus | None:
        bus = self._buses.get(session_id)
        return bus if bus is not None and not bus.closed else None

    def discard(self, session_id: str) -> None:
        """Close and forget a session's bus (call on session delete/end)."""
        bus = self._buses.pop(session_id, None)
        if bus is not None:
            bus.close()

    @property
    def session_ids(self) -> list[str]:
        return [sid for sid, bus in self._buses.items() if not bus.closed]


__all__ = ["EventBusRegistry", "SessionEventBus"]
