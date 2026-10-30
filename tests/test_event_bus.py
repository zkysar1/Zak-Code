"""Tests for the watch-surface per-session event bus (zakcode.server.event_bus).

The bus is the fan-out that lets a read-only watcher (GET /watch/{session_id}, P0-3) observe a
session's events without disturbing the turn-driver. These tests pin its contract: strictly
monotonic cursors, cursor-addressed replay + live tail with exactly-once delivery, a publish
stamp on every item, bounded memory (oldest evicted → cursor gap), per-subscriber backpressure
that never blocks the publisher, clean close (a parked live-tailer is woken), and registry
lifecycle.

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
Item = tuple[int, Any, float]


async def _take(agen: AsyncGenerator[Item, None], n: int, timeout: float = 1.0) -> list[Pair]:
    """Pull the next ``n`` items as ``(cursor, event)``, failing fast instead of hanging.

    The publish stamp is dropped here; the one test about stamps reads them whole."""
    items = [await asyncio.wait_for(agen.__anext__(), timeout) for _ in range(n)]
    return [(cursor, event) for cursor, event, _at in items]


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


def test_a_bus_keeps_one_epoch_for_its_life_and_no_two_buses_share_one() -> None:
    """Cursors start again from 1 on every bus, so a cursor means something only to the bus that
    issued it. The epoch names the bus: publishing and closing never change it, and buses made
    later (a server start, a session's bus made again) do not repeat one (ADR-0289)."""
    bus = SessionEventBus()
    epoch = bus.epoch
    assert epoch and all(c in "0123456789abcdef" for c in epoch)  # URL-safe, never empty
    bus.publish("a")
    bus.close()
    assert bus.epoch == epoch
    assert len({SessionEventBus().epoch for _ in range(50)}) == 50


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


async def test_every_item_carries_its_publish_stamp_replayed_or_live() -> None:
    """A late or reconnecting watcher replays the buffer in one burst, so the time an item
    ARRIVES says nothing about when it happened. The stamp taken at publish is what lets a
    page time a tool call and stamp a turn truly (ADR-0219) — on replay and live alike."""
    stamps = iter([100.0, 101.5, 107.25])
    bus = SessionEventBus(clock=lambda: next(stamps))
    bus.publish("a")
    bus.publish("b")
    agen = bus.subscribe(since=None)
    replayed = [await asyncio.wait_for(agen.__anext__(), 1.0) for _ in range(2)]
    assert replayed == [(1, "a", 100.0), (2, "b", 101.5)]
    bus.publish("c")  # live: lands in the now-registered queue with its own stamp
    assert await asyncio.wait_for(agen.__anext__(), 1.0) == (3, "c", 107.25)
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


async def test_a_bus_holds_a_long_answer_whole() -> None:
    """A streamed answer is one event per chunk, and a page that leaves a chat while its turn
    runs comes back to what the bus still holds. At 1000 events a long answer (a 400-word reply
    came to ~520) evicted its own start, and the page came back to a turn with no beginning."""
    bus = SessionEventBus()
    for x in range(1, 4001):  # an answer some 3000 words long
        bus.publish(x)
    agen = bus.subscribe(since=None)
    got = await _take(agen, 4000)
    assert got[0] == (1, 1)
    await agen.aclose()


def test_oldest_at_is_the_stamp_of_the_oldest_retained_event() -> None:
    """The line between what a late joiner can replay from the bus and what only the saved
    session still holds: none while the bus is empty (a server restart), then the first
    event's stamp, moving forward as the ring evicts."""
    stamps = iter([10.0, 11.0, 12.0, 13.0])
    bus = SessionEventBus(maxlen=3, clock=lambda: next(stamps))
    assert bus.oldest_at is None
    bus.publish("a")
    bus.publish("b")
    assert bus.oldest_at == 10.0
    bus.publish("c")
    assert bus.oldest_at == 10.0  # full, nothing evicted yet
    bus.publish("d")  # evicts "a"
    assert bus.oldest_at == 11.0


def test_oldest_where_is_the_cursor_and_stamp_of_the_oldest_retained_event_picked() -> None:
    """Where the oldest turn the bus still holds whole begins (ADR-0325): the cursor and stamp of
    the oldest RETAINED event a test picks, and none when no retained event is picked, an
    evicted one included."""
    stamps = iter([10.0, 11.0, 12.0, 13.0, 14.0])
    bus = SessionEventBus(maxlen=3, clock=lambda: next(stamps))
    assert bus.oldest_where(lambda event: event.startswith("start")) is None
    for x in ("start-1", "a", "start-2", "b", "c"):
        bus.publish(x)  # "start-1" and "a" are evicted
    assert bus.oldest_where(lambda event: event.startswith("start")) == (3, 12.0)
    assert bus.oldest_where(lambda event: event == "a") is None


async def test_forget_drops_only_the_events_it_picks_and_hands_out_no_cursor_again() -> None:
    """News that is over must not reach a late joiner, who would read it as happening now: a
    chat's rotation notice, once the chat is current again (ADR-0325). Forgetting drops just
    those events. The rest keep their cursors and replay in order, and the next publish is not
    given a dropped cursor."""
    bus = SessionEventBus()
    for x in ("a", "over", "b", "over", "c"):
        bus.publish(x)
    assert bus.forget(lambda event: event == "over") == 2
    assert bus.forget(lambda event: event == "over") == 0
    assert bus.publish("d") == 6
    agen = bus.subscribe(since=None)
    assert await _take(agen, 4) == [(1, "a"), (3, "b"), (5, "c"), (6, "d")]
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
    out: list[Item] = []

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


def test_a_session_whose_bus_is_made_again_gets_a_new_epoch() -> None:
    """What a client holding a cursor needs to tell: the same bus (keep the cursor) from a new
    one that numbers from 1 again (drop it)."""
    reg = EventBusRegistry()
    first = reg.get_or_create("s1")
    assert reg.get_or_create("s1").epoch == first.epoch
    reg.discard("s1")
    assert reg.get_or_create("s1").epoch != first.epoch


# ── Idle buses let go (ADR-0327) ─────────────────────────────────────────────


@pytest.fixture
def two_idle(monkeypatch: pytest.MonkeyPatch) -> None:
    """The registry keeps two idle buses, so a handful of sessions shows the sweep."""
    from zakcode.server import event_bus as eb

    monkeypatch.setattr(eb, "_IDLE_BUSES", 2)


@pytest.mark.usefixtures("two_idle")
def test_making_a_bus_lets_go_of_the_least_recently_active_idle_buses_past_the_cap() -> None:
    reg = EventBusRegistry()
    reg.in_use = lambda: ()
    a, b, c = (reg.get_or_create(sid) for sid in ("a", "b", "c"))
    a.publish("news")  # published on
    reg.get_or_create("b")  # asked for
    reg.get_or_create("d")  # idle: c, then a, then b; the two most recent are kept
    assert c.closed
    assert reg.get("c") is None
    assert sorted(reg.session_ids) == ["a", "b", "d"]
    assert not (a.closed or b.closed)


@pytest.mark.usefixtures("two_idle")
async def test_a_watched_bus_and_a_bus_in_use_are_never_let_go() -> None:
    """Both are older than every other bus here, so they would be the first to go if idle."""
    reg = EventBusRegistry()
    reg.in_use = lambda: {"used"}
    used = reg.get_or_create("used")
    watched = reg.get_or_create("watched")
    watched.publish("x")
    agen = watched.subscribe(since=None)
    await _take(agen, 1)  # a watcher, parked at its replay
    for sid in ("e1", "e2", "e3", "e4"):
        reg.get_or_create(sid)
    assert not (used.closed or watched.closed)
    assert sorted(reg.session_ids) == ["e2", "e3", "e4", "used", "watched"]
    await agen.aclose()


def test_a_registry_not_told_what_is_in_use_lets_no_bus_go() -> None:
    """A turn holds its bus to publish on, and only the server knows which turns run, so until
    it says, closing any bus could close one under a turn."""
    reg = EventBusRegistry()
    buses = [reg.get_or_create(f"s{i}") for i in range(40)]
    assert not any(bus.closed for bus in buses)
