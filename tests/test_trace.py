"""Tests for the per-turn decision trace (observability) — the TurnTrace model and the loop wiring.

The model tests pin the positional-only ``note`` (so a payload field may be named ``kind``), the
never-raise contract, and JSONL serialization. The loop tests drive the real agent with a scripted
provider (no network) and assert the trace captures the stop, the interventions, the opt-in file
dump, and per-turn reset.
"""

from __future__ import annotations

import json
from pathlib import Path

from zakcode.agent.loop import AgentLoop
from zakcode.agent.trace import TraceEvent, TurnTrace
from zakcode.config import load_settings
from zakcode.evals.harness import ScriptedProvider, call_tool, reply
from zakcode.events import AgentDone
from zakcode.session.store import Session
from zakcode.tools import default_registry
from zakcode.usage import Usage

# ── TurnTrace model ────────────────────────────────────────────────────────────


def test_note_records_event_and_payload_kind_does_not_collide() -> None:
    t = TurnTrace()
    t.note("route", "deep_code", category="deep_code")
    t.note("intervention", "repeated calls", kind="doom_loop")  # 'kind' is a PAYLOAD field here
    assert len(t.events) == 2
    assert isinstance(t.events[0], TraceEvent)
    # The leading positional "intervention" is the EVENT kind; the keyword kind lands in data.
    ev = t.of_kind("intervention")[0]
    assert ev.kind == "intervention" and ev.data["kind"] == "doom_loop"
    assert t.of_kind("route")[0].detail == "deep_code"


def test_note_is_best_effort_never_raises() -> None:
    t = TurnTrace()
    t.note("x", "y", weird=object())  # a non-trivial payload must not raise into the caller
    # The call returns cleanly; whether the event is kept is unimportant — it must not throw.


def test_to_jsonl_is_one_object_per_line() -> None:
    t = TurnTrace()
    t.note("route", "deep_code", category="deep_code")
    t.note("stop", "completed", reason="completed", iterations=3)
    lines = t.to_jsonl().splitlines()
    assert len(lines) == 2
    last = json.loads(lines[1])
    assert last["kind"] == "stop" and last["data"]["reason"] == "completed"


# ── loop wiring ────────────────────────────────────────────────────────────────


def _loop(tmp_path: Path, provider: ScriptedProvider, **over: object) -> AgentLoop:
    settings = load_settings(workspace_root=tmp_path, max_iterations=10, **over)
    return AgentLoop(
        provider, default_registry(), Session(cwd=str(tmp_path), model="t/m"), settings=settings
    )


async def test_completed_turn_traces_a_stop_event(tmp_path: Path) -> None:
    loop = _loop(tmp_path, ScriptedProvider([reply("done")]))
    result = await loop.arun_turn("hi")
    stop = result.trace.of_kind("stop")
    assert len(stop) == 1
    assert stop[0].data["reason"] == "completed" and stop[0].data["iterations"] == 1


async def test_doom_turn_traces_intervention_then_stop(tmp_path: Path) -> None:
    same = call_tool("write_file", {"path": str(tmp_path / "x.txt"), "content": "x"})
    loop = _loop(tmp_path, ScriptedProvider([reply("u")], responder=lambda m, s, i: same))
    result = await loop.arun_turn("loop forever")
    assert result.stop_reason == "doom_loop"
    doom = [e for e in result.trace.of_kind("intervention") if e.data.get("kind") == "doom_loop"]
    assert doom, "the doom_loop intervention should be on the trace"
    assert result.trace.of_kind("stop")[0].data["reason"] == "doom_loop"


async def test_tool_call_is_traced(tmp_path: Path) -> None:
    write = call_tool("write_file", {"path": str(tmp_path / "f.txt"), "content": "hi"})

    def responder(messages: object, system: object, i: int) -> object:
        return write if i == 0 else reply("done")

    loop = _loop(tmp_path, ScriptedProvider([reply("u")], responder=responder))
    result = await loop.arun_turn("write a file")
    tools = result.trace.of_kind("tool")
    assert len(tools) == 1  # one tool call, traced compactly (name + ok)
    assert tools[0].detail == "write_file" and tools[0].data["ok"] is True


async def test_trace_dump_writes_jsonl_when_trace_dir_set(tmp_path: Path) -> None:
    trace_dir = tmp_path / "traces"
    loop = _loop(tmp_path, ScriptedProvider([reply("done")]), trace_dir=str(trace_dir))
    await loop.arun_turn("hi")
    dumped = trace_dir / loop.session.id / "turn_1.jsonl"
    assert dumped.is_file()
    events = [json.loads(line) for line in dumped.read_text(encoding="utf-8").splitlines()]
    assert any(e["kind"] == "stop" for e in events)


async def test_trace_dumps_are_per_session_so_a_restart_keeps_the_previous_turns(
    tmp_path: Path,
) -> None:
    # Turn numbers restart with every session; a flat turn_1.jsonl was overwritten by the
    # next session's first turn (every coach restart erased the boot turn's telemetry).
    trace_dir = tmp_path / "traces"
    first = _loop(tmp_path, ScriptedProvider([reply("one")]), trace_dir=str(trace_dir))
    await first.arun_turn("hi")
    second = _loop(tmp_path, ScriptedProvider([reply("two")]), trace_dir=str(trace_dir))
    await second.arun_turn("hi")
    assert first.session.id != second.session.id
    assert (trace_dir / first.session.id / "turn_1.jsonl").is_file()
    assert (trace_dir / second.session.id / "turn_1.jsonl").is_file()
    assert not (trace_dir / "turn_1.jsonl").exists()


async def test_trace_dump_is_best_effort(tmp_path: Path) -> None:
    # trace_dir pointing at an EXISTING FILE (not a dir) must not raise into the turn.
    clash = tmp_path / "not_a_dir"
    clash.write_text("x", encoding="utf-8")
    loop = _loop(tmp_path, ScriptedProvider([reply("done")]), trace_dir=str(clash))
    result = await loop.arun_turn("hi")  # must complete normally despite the dump failing
    assert result.stop_reason == "completed"


async def test_buffered_call_traces_usage(tmp_path: Path) -> None:
    u = Usage(prompt_tokens=100, completion_tokens=7, total_tokens=107)
    loop = _loop(tmp_path, ScriptedProvider([reply("done", usage=u)]))
    result = await loop.arun_turn("hi")
    events = result.trace.of_kind("usage")
    assert len(events) == 1  # one model call -> one usage event
    data = events[0].data
    assert data["prompt_tokens"] == 100 and data["completion_tokens"] == 7
    assert data["total_tokens"] == 107
    assert data["latency_s"] >= 0.0
    assert "model" in data


async def test_streaming_turn_traces_usage(tmp_path: Path) -> None:
    # ScriptedProvider streams via the base-class astream shim, which emits one
    # StreamUsage from the buffered result — exercising the streaming commit path.
    u = Usage(prompt_tokens=50, completion_tokens=3, total_tokens=53)
    loop = _loop(tmp_path, ScriptedProvider([reply("done", usage=u)]))
    done = [ev async for ev in loop.astream_turn("hi")][-1]
    assert isinstance(done, AgentDone)
    events = done.trace.of_kind("usage")
    assert len(events) == 1
    data = events[0].data
    assert data["prompt_tokens"] == 50 and data["completion_tokens"] == 3
    assert data["streamed"] is True
    assert data["latency_s"] >= 0.0


async def test_trace_resets_each_turn(tmp_path: Path) -> None:
    loop = _loop(tmp_path, ScriptedProvider([reply("a"), reply("b")]))
    r1 = await loop.arun_turn("one")
    r2 = await loop.arun_turn("two")
    assert r1.trace.of_kind("stop") and r2.trace.of_kind("stop")
    # No carryover: turn two's trace is its own (clean turn = route? + one usage
    # per model call + stop, not accumulated).
    assert len(r2.trace.events) <= 3
