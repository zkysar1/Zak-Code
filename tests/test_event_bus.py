"""Tests for the Pearl watch-surface per-session event bus (zakcode.server.event_bus).

The bus is the fan-out that lets a read-only watcher (GET /watch/{session_id}, P0-3) observe a
session's events without disturbing the turn-driver. These tests pin its contract: strictly
monotonic cursors, cursor-addressed replay + live tail with exactly-once delivery, bounded
memory (oldest evicted → cursor gap), per-subscriber backpressure that never blocks the
publisher, clean close (a parked live-tailer is woken), and registry lifecycle.

Async-test note: a subscribe() generator registers its queue and replays the buffer synchronously
up to its first ``yield``; only requesting an item that is not yet buffered parks it at
``await queue.get()``. So replay assertions can pull exactly the retained count with ``_take`` and
then ``aclose()`` cleanly (the generator is parked at a ``yield``, not mid-cancel). Tests that need
a subscriber genuinely PARKED on the live tail (backpressure, close-wakes) drive it from a
background task and wait on ``subscriber_count`` — never via ``wait_for``-cancel, which would run
the generator's ``finally`` and deregister the queue.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from typing import Any

import pytest

from zakcode.server.event_bus import EventBusRegistry, SessionEventBus, _offer

Pair = tuple[int, Any]


async def _take(agen: AsyncGenerator[Pair, None], n: int, timeout: float = 1.0) -> list[Pair]:
    """Pull the next ``n`` items from the iterator, failing fast instead of hanging."""
    return [await asyncio.wait_for(agen.__anext__(), timeout) for _ in range(n)]


async def _await_registered(bus: SessionEventBus, count: int = 1) -> None:
    """Yield the loop until ``count`` subscribers have registered their queues."""
    for _ in range(1000):
        if bus.subscriber_count >= count:
            return
        await asyncio.sleep(0)
    raise AssertionError(f"only {bus.subscriber_count} subscribers registered, wanted {count}")


# ── Cursors + replay ─────────────────────────────────────────────────────────


def test_publish_assigns_strictly_increasing_cursors_from_one() -> None:
    bus = SessionEventBus()
    assert bus.latest_cursor == 0
    assert [bus.publish(x) for x in ("a", "b", "c")] == [1, 2, 3]
    assert bus.latest_cursor == 3


async def test_subscribe_since_none_replays_whole_buffer() -> None:
    bus = SessionEventBus()
    for x in ("a", "b", "c"):
        bus.publish(x)
    agen = bus.subscribe(since=None)
    assert await _take(agen, 3) == [(1, "a"), (2, "b"), (3, "c")]
    await agen.aclose()


async def test_subscribe_since_cursor_replays_only_newer() -> None:
    bus = SessionEventBus()
    for x in ("a", "b", "c", "d"):
        bus.publish(x)
    agen = bus.subscribe(since=2)
    assert await _take(agen, 2) == [(3, "c"), (4, "d")]
    await agen.aclose()


async def test_late_subscriber_sees_events_published_after_it_started() -> None:
    bus = SessionEventBus()
    bus.publish("a")  # cursor 1 (replayed on first take, which also registers the queue)
    agen = bus.subscribe(since=None)
    assert await _take(agen, 1) == [(1, "a")]
    bus.publish("b")  # cursor 2 (live — lands in the now-registered queue)
    bus.publish("c")  # cursor 3 (live)
    assert await _take(agen, 2) == [(2, "b"), (3, "c")]
    await agen.aclose()


# ── Bounded memory / eviction ────────────────────────────────────────────────


async def test_bounded_buffer_evicts_oldest_and_replay_shows_cursor_gap() -> None:
    bus = SessionEventBus(maxlen=3)
    for x in range(1, 6):  # cursors 1..5; only 3..5 retained
        bus.publish(x)
    assert bus.latest_cursor == 5
    agen = bus.subscribe(since=None)  # a since<3 watcher's replay starts at the oldest retained
    got = await _take(agen, 3)
    assert got == [(3, 3), (4, 4), (5, 5)]
    assert got[0][0] == 3  # cursors 1 and 2 were evicted — the gap is detectable
    await agen.aclose()


# ── Multiple subscribers ─────────────────────────────────────────────────────


async def test_two_subscribers_each_receive_all_events_independently() -> None:
    bus = SessionEventBus()
    a = bus.subscribe(since=None)
    b = bus.subscribe(since=None)
    bus.publish("x")  # buffered; both replay it on their first take (they register lazily)
    bus.publish("y")
    assert await _take(a, 2) == [(1, "x"), (2, "y")]
    assert await _take(b, 2) == [(1, "x"), (2, "y")]
    assert bus.subscriber_count == 2
    await a.aclose()
    await b.aclose()


async def test_subscriber_count_returns_to_zero_after_close() -> None:
    bus = SessionEventBus()
    bus.publish("x")
    agen = bus.subscribe(since=None)
    await _take(agen, 1)  # registers the queue, parks at the replay yield
    assert bus.subscriber_count == 1
    await agen.aclose()  # GeneratorExit at the yield → finally deregisters
    assert bus.subscriber_count == 0


# ── Backpressure ─────────────────────────────────────────────────────────────


def test_offer_drops_oldest_when_queue_full() -> None:
    q: asyncio.Queue[int] = asyncio.Queue(maxsize=2)
    _offer(q, 1)
    _offer(q, 2)
    _offer(q, 3)  # full → drop 1, keep {2, 3}
    assert q.get_nowait() == 2
    assert q.get_nowait() == 3
    assert q.empty()


async def test_full_subscriber_never_blocks_publisher() -> None:
    from zakcode.server import event_bus as eb

    bus = SessionEventBus()
    agen = bus.subscribe(since=None)
    parked = asyncio.create_task(agen.__anext__())  # parks at queue.get on the empty bus
    await _await_registered(bus)
    n = eb._SUBSCRIBER_QUEUE_MAXLEN + 50
    for i in range(1, n + 1):  # publish is sync + non-blocking despite the un-drained subscriber
        assert bus.publish(i) == i
    assert bus.latest_cursor == n
    # The synchronous flood never yields, so the parked subscriber cannot drain and its bounded
    # queue keeps only the newest _SUBSCRIBER_QUEUE_MAXLEN events; its first delivered cursor is
    # therefore the oldest still-retained one — direct proof of oldest-drop backpressure.
    assert (await asyncio.wait_for(parked, 1.0))[0] == n - eb._SUBSCRIBER_QUEUE_MAXLEN + 1
    await agen.aclose()


# ── Close semantics ──────────────────────────────────────────────────────────


async def test_close_wakes_a_live_tailing_subscriber() -> None:
    bus = SessionEventBus()
    out: list[Pair] = []

    async def consume() -> None:
        async for item in bus.subscribe(since=None):
            out.append(item)

    task = asyncio.create_task(consume())
    await _await_registered(bus)  # parked at queue.get on the idle bus
    bus.close()  # _CLOSE sentinel wakes it → subscribe returns → async-for ends
    await asyncio.wait_for(task, 1.0)
    assert out == []
    assert bus.closed is True


def test_publish_after_close_raises() -> None:
    bus = SessionEventBus()
    bus.publish("a")
    bus.close()
    with pytest.raises(RuntimeError):
        bus.publish("b")


# ── Registry ─────────────────────────────────────────────────────────────────


def test_registry_get_or_create_is_idempotent_per_session() -> None:
    reg = EventBusRegistry()
    a1 = reg.get_or_create("s1")
    assert reg.get_or_create("s1") is a1
    assert reg.get_or_create("s2") is not a1
    assert sorted(reg.session_ids) == ["s1", "s2"]


def test_registry_get_returns_none_for_missing_or_closed() -> None:
    reg = EventBusRegistry()
    assert reg.get("nope") is None
    bus = reg.get_or_create("s1")
    assert reg.get("s1") is bus
    reg.discard("s1")  # closes the bus and forgets it
    assert bus.closed is True
    assert reg.get("s1") is None
    assert reg.session_ids == []


def test_registry_get_or_create_replaces_a_closed_bus() -> None:
    reg = EventBusRegistry()
    first = reg.get_or_create("s1")
    first.close()
    second = reg.get_or_create("s1")  # a closed bus is replaced, not handed back
    assert second is not first
    assert second.closed is False
