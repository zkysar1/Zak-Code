"""Unit tests for SessionBroadcaster — per-session fan-out of SAFE watch frames.

Async, pure (no server): asserts projection-at-publish (raw events never enter the bus),
seq monotonicity, cursor replay via history(), multi-subscriber fan-out, drop of usage
events, forget(), and the retained-session eviction cap.
"""

from __future__ import annotations

from zakcode.events import AgentTextDelta, AgentToolCall, AgentUsage
from zakcode.server.broadcast import SessionBroadcaster
from zakcode.usage import Usage


async def test_publish_projects_and_fans_out() -> None:
    bus = SessionBroadcaster()
    queue = bus.subscribe("s1")
    await bus.publish("s1", AgentTextDelta(text="hello"))
    record = queue.get_nowait()
    assert record["seq"] == 1
    assert record["frame"] == {"event": "text", "text": "hello"}


async def test_tool_call_frame_has_no_arguments() -> None:
    bus = SessionBroadcaster()
    queue = bus.subscribe("s1")
    await bus.publish("s1", AgentToolCall(id="c1", name="bash", arguments={"cmd": "rm -rf /"}))
    record = queue.get_nowait()
    assert "arguments" not in record["frame"]
    assert "rm -rf" not in str(record["frame"])


async def test_usage_is_not_published() -> None:
    bus = SessionBroadcaster()
    queue = bus.subscribe("s1")
    await bus.publish("s1", AgentUsage(usage=Usage(cost_usd=5.0)))
    assert queue.empty()


async def test_history_replays_after_cursor() -> None:
    bus = SessionBroadcaster()
    for text in ("a", "b", "c"):
        await bus.publish("s1", AgentTextDelta(text=text))
    assert [r["frame"]["text"] for r in bus.history("s1", 0)] == ["a", "b", "c"]
    assert [r["frame"]["text"] for r in bus.history("s1", 1)] == ["b", "c"]


async def test_multiple_subscribers_each_receive() -> None:
    bus = SessionBroadcaster()
    q1 = bus.subscribe("s1")
    q2 = bus.subscribe("s1")
    await bus.publish("s1", AgentTextDelta(text="x"))
    assert q1.get_nowait()["frame"]["text"] == "x"
    assert q2.get_nowait()["frame"]["text"] == "x"


async def test_forget_clears_state() -> None:
    bus = SessionBroadcaster()
    await bus.publish("s1", AgentTextDelta(text="x"))
    assert bus.history("s1", 0)
    bus.forget("s1")
    assert bus.history("s1", 0) == []


async def test_eviction_bounds_unwatched_sessions() -> None:
    bus = SessionBroadcaster(max_sessions=2)
    await bus.publish("s1", AgentTextDelta(text="1"))
    await bus.publish("s2", AgentTextDelta(text="2"))
    await bus.publish("s3", AgentTextDelta(text="3"))  # exceeds cap → oldest unwatched (s1) evicted
    assert bus.history("s1", 0) == []
    assert bus.history("s2", 0)
    assert bus.history("s3", 0)


async def test_watched_session_survives_eviction() -> None:
    bus = SessionBroadcaster(max_sessions=1)
    bus.subscribe("s1")  # s1 has a live watcher
    await bus.publish("s1", AgentTextDelta(text="1"))
    await bus.publish("s2", AgentTextDelta(text="2"))  # cap exceeded; s1 is watched → evict s2
    assert bus.history("s1", 0)  # watched oldest survives
    assert bus.history("s2", 0) == []
