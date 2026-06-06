"""Tests for multi-signal stuck detection + the recovery ladder (Bet 2).

Pure-state tests for :class:`~zakcode.agent.stuck.StuckTracker`, plus loop-integration
tests proving that a model stuck in ways the exact-repeat doom guard MISSES (varying-arg
failures, alternating failing calls) is first nudged, then narrowed to read-only tools, and
finally stopped cleanly as ``stop_reason="stuck"`` — while progressing/transient turns are
never affected. Everything is hermetic (scripted providers, in-memory tools).
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from zakcode.agent.loop import AgentLoop
from zakcode.agent.stuck import (
    SIG_ALL_ERRORS,
    SIG_NO_PROGRESS,
    SIG_REPEATED_BATCH,
    SIG_REPEATED_FAILURE,
    StuckAction,
    StuckTracker,
    batch_signature,
    call_signature,
)
from zakcode.config import PermissionTier
from zakcode.events import AgentDone, AgentStatus
from zakcode.messages import Message, ToolResultBlock
from zakcode.providers.base import Capabilities, LLMResult, Provider, ToolCall
from zakcode.session.store import Session
from zakcode.tools.base import (
    ConcurrencyClass,
    Tool,
    ToolContext,
    ToolRegistry,
    ToolResult,
    ToolSpec,
)


def _c(call_id: str, name: str, **args: object) -> ToolCall:
    return ToolCall(id=call_id, name=name, arguments=dict(args))


def _r(tool_use_id: str, *, is_error: bool = False, output: str = "ok") -> ToolResultBlock:
    return ToolResultBlock(tool_use_id=tool_use_id, output=output, is_error=is_error)


# ── canonical signatures ────────────────────────────────────────────────────────


def test_call_signature_ignores_id_and_arg_order() -> None:
    a = _c("id1", "echo", x=1, y=2)
    b = _c("id2", "echo", y=2, x=1)  # different id + key order, same identity
    assert call_signature(a) == call_signature(b)
    assert batch_signature([a]) == batch_signature([b])


def test_batch_signature_distinguishes_args() -> None:
    assert batch_signature([_c("a", "echo", n=1)]) != batch_signature([_c("b", "echo", n=2)])


# ── pure tracker: signals ───────────────────────────────────────────────────────


def test_single_transient_error_is_not_stuck() -> None:
    t = StuckTracker()  # defaults: first action only at streak 3
    t.observe([_c("c1", "boom", n=1)], [_r("c1", is_error=True)], assistant_text="")
    assert t.next_action() is StuckAction.CONTINUE
    assert t.streak == 1
    assert t.took_action is False


def test_all_errors_and_no_progress_fire_together() -> None:
    t = StuckTracker(nudge_at=99, narrow_at=100, stop_at=101)
    t.observe([_c("c1", "boom", n=1)], [_r("c1", is_error=True)], assistant_text="")
    assert set(t.last_signals) == {SIG_ALL_ERRORS, SIG_NO_PROGRESS}


def test_no_progress_requires_no_text() -> None:
    # A failing call WITH reasoning text is only one signal (all-errors) -> not stuck.
    t = StuckTracker(nudge_at=1, narrow_at=2, stop_at=3)
    t.observe([_c("c1", "boom", n=1)], [_r("c1", is_error=True)], assistant_text="let me think")
    assert t.last_signals == [SIG_ALL_ERRORS]
    assert t.next_action() is StuckAction.CONTINUE
    assert t.streak == 0


def test_repeated_batch_signal_in_isolation() -> None:
    # Identical SUCCESSFUL batch twice: repeated-batch fires, the error/progress signals do
    # not (so it is a single vote — the loop's doom guard owns exact repeats).
    t = StuckTracker(nudge_at=99, narrow_at=100, stop_at=101)
    t.observe([_c("c1", "look", q="x")], [_r("c1")], assistant_text="")
    assert SIG_REPEATED_BATCH not in t.last_signals  # nothing to repeat yet
    t.observe([_c("c2", "look", q="x")], [_r("c2")], assistant_text="")
    assert t.last_signals == [SIG_REPEATED_BATCH]


def test_repeated_failure_signal() -> None:
    # The SAME call (identical args) failing twice this turn fires repeated-failure on the
    # second occurrence (text suppresses no-progress so the signal is isolated).
    t = StuckTracker(repeated_failure_at=2, nudge_at=99, narrow_at=100, stop_at=101)
    t.observe([_c("c1", "boom", n=1)], [_r("c1", is_error=True)], assistant_text="ok")
    assert SIG_REPEATED_FAILURE not in t.last_signals
    t.observe([_c("c2", "boom", n=1)], [_r("c2", is_error=True)], assistant_text="ok")
    assert SIG_REPEATED_FAILURE in t.last_signals


def test_error_signatures_ranks_repeated_failures_most_first() -> None:
    # The accessor the lesson writer (R1) reads: call signatures that failed >= repeated_failure_at
    # times, most-failed first; calls below the floor are excluded.
    t = StuckTracker(repeated_failure_at=2)
    for _ in range(3):
        t.observe([_c("x", "boom", k="a")], [_r("x", is_error=True)], assistant_text="t")
    for _ in range(2):
        t.observe([_c("y", "boom", k="b")], [_r("y", is_error=True)], assistant_text="t")
    # 'boom c' fails only once — below the repeated_failure_at floor, so it is excluded.
    t.observe([_c("z", "boom", k="c")], [_r("z", is_error=True)], assistant_text="t")
    sigs = t.error_signatures()
    assert sigs[0] == ("boom", '{"k": "a"}')  # 3 failures rank first
    assert ("boom", '{"k": "b"}') in sigs  # 2 failures included
    assert ("boom", '{"k": "c"}') not in sigs  # 1 failure excluded


def test_successful_iterations_never_stuck() -> None:
    t = StuckTracker(nudge_at=1, narrow_at=2, stop_at=3)
    for i in range(5):
        t.observe([_c(f"c{i}", "look", n=i)], [_r(f"c{i}")], assistant_text="")
        assert t.next_action() is StuckAction.CONTINUE
    assert t.streak == 0
    assert t.took_action is False


# ── pure tracker: ladder ─────────────────────────────────────────────────────────


def _stuck_step(t: StuckTracker, n: int) -> StuckAction:
    # A failing call with no text => all-errors + no-progress (>= 2 signals) every iteration.
    t.observe([_c(f"c{n}", "boom", n=n)], [_r(f"c{n}", is_error=True)], assistant_text="")
    return t.next_action()


def test_sustained_stuck_escalates_nudge_narrow_stop() -> None:
    t = StuckTracker(nudge_at=1, narrow_at=2, stop_at=3)
    assert _stuck_step(t, 1) is StuckAction.NUDGE  # streak 1
    assert _stuck_step(t, 2) is StuckAction.NARROW  # streak 2
    assert _stuck_step(t, 3) is StuckAction.STOP  # streak 3
    assert t.actions == ["nudge", "narrow", "stop"]
    assert t.took_action is True


def test_streak_resets_on_progress_before_escalation() -> None:
    t = StuckTracker(nudge_at=2, narrow_at=3, stop_at=4)
    assert _stuck_step(t, 1) is StuckAction.CONTINUE  # streak 1
    # A successful iteration resets the streak, so the next stuck one starts over.
    t.observe([_c("ok", "look", n=9)], [_r("ok")], assistant_text="")
    assert t.next_action() is StuckAction.CONTINUE
    assert t.streak == 0
    assert _stuck_step(t, 2) is StuckAction.CONTINUE  # streak 1 again, not 2
    assert t.took_action is False


def test_nudge_and_narrow_messages_distinct() -> None:
    t = StuckTracker()
    assert "stuck" in t.nudge_message().lower()
    assert "read-only" in t.narrow_message().lower()
    assert t.nudge_message() != t.narrow_message()


# ── loop integration ────────────────────────────────────────────────────────────


class _BoomTool(Tool):
    spec = ToolSpec(name="boom", description="Always errors.")

    async def execute(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        return ToolResult.error("kaboom")


class _LookTool(Tool):
    spec = ToolSpec(name="look", description="Read-only; always succeeds.")

    async def execute(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        return ToolResult.ok(output="looked")


class _MutateTool(Tool):
    # A write-tier tool (so it is excluded from the read-only NARROW subset) that errors.
    # Records how many times it actually EXECUTED so a test can prove a NARROW step REJECTS
    # it before execution (dispatch gate), not merely withholds it from the schema.
    spec = ToolSpec(
        name="mutate",
        description="Write-tier; always errors.",
        required_permission=PermissionTier.WORKSPACE_WRITE,
        concurrency=ConcurrencyClass.NEVER_PARALLEL,
    )

    def __init__(self) -> None:
        self.executed = 0

    async def execute(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        self.executed += 1
        return ToolResult.error("nope")


def _registry(*tools: Tool) -> ToolRegistry:
    reg = ToolRegistry()
    for tool in tools or (_BoomTool(), _LookTool()):
        reg.register(tool)
    return reg


class _ScriptByCallProvider(Provider):
    """Calls a per-iteration factory to produce each LLMResult; records the tools offered."""

    def __init__(self, factory: Any) -> None:
        self._factory = factory
        self.calls = 0
        self.tools_seen: list[Any] = []

    async def acomplete(
        self, messages: list[Message], *, system: str | None = None, tools: Any = None, **kw: Any
    ) -> LLMResult:
        self.tools_seen.append(tools)
        self.calls += 1
        return self._factory(self.calls)

    def count_tokens(self, messages: list[Message], *, system: str | None = None) -> int:
        return 0

    def capabilities(self) -> Capabilities:
        return Capabilities(supports_tools=True, context_window=8192)


def _loop(provider: Provider, tmp_path: Path, registry: ToolRegistry | None = None) -> AgentLoop:
    return AgentLoop(
        provider,
        registry or _registry(),
        Session(cwd=str(tmp_path), model="test"),
        workspace_root=tmp_path,
        max_iterations=10,
    )


def _tool_names(defs: Sequence[dict[str, Any]] | None) -> set[str]:
    return {d["function"]["name"] for d in (defs or [])}


def test_loop_stops_on_varying_arg_failures(tmp_path: Path) -> None:
    # Each iteration calls a failing tool with a FRESH arg, so the exact-repeat doom guard
    # never fires — only the multi-signal stuck ladder can stop it.
    provider = _ScriptByCallProvider(lambda n: LLMResult(tool_calls=[_c(f"c{n}", "boom", n=n)]))
    result = asyncio.run(_loop(provider, tmp_path).arun_turn("do the thing"))
    assert result.stop_reason == "stuck"  # not doom_loop, not max_iterations
    # nudge@3 -> narrow@4 -> stop@5 (defaults), so it ends at iteration 5, far below the cap.
    assert result.iterations == 5
    assert result.degraded is True
    assert len(result.tool_results) == 5 and all(b.is_error for b in result.tool_results)


def test_loop_alternating_failures_are_caught(tmp_path: Path) -> None:
    # A,B,A,B,A failing calls: never 3 identical in a row, so the doom guard stays silent;
    # the stuck ladder (repeated-failure + all-errors + no-progress) still stops the turn.
    keys = ["a", "b"]
    provider = _ScriptByCallProvider(
        lambda n: LLMResult(tool_calls=[_c(f"c{n}", "boom", k=keys[n % 2])])
    )
    result = asyncio.run(_loop(provider, tmp_path).arun_turn("alternate"))
    assert result.stop_reason == "stuck"
    assert result.degraded is True


def test_loop_recovery_ladder_writes_hints(tmp_path: Path) -> None:
    provider = _ScriptByCallProvider(lambda n: LLMResult(tool_calls=[_c(f"c{n}", "boom", n=n)]))
    loop = _loop(provider, tmp_path)
    asyncio.run(loop.arun_turn("do the thing"))
    transcript = "\n".join(m.text or "" for m in loop.session.messages)
    assert "appear to be stuck" in transcript  # the nudge
    assert "read-only tools are available" in transcript  # the narrow
    # rb-204: loop-injected guidance opens with the SAME control-rail marker as tool-result
    # hints, so the model sees one "next action" vocabulary regardless of the source.
    injected = [m.text for m in loop.session.messages if m.text and "stuck" in m.text]
    assert injected and all(t.startswith("Hint:") for t in injected)


def test_loop_narrow_restricts_offered_tools(tmp_path: Path) -> None:
    # The NARROW step (streak 4) must restrict the NEXT iteration (the 5th call) to read-only
    # tools — the write-tier 'mutate' is dropped from the offered schema, 'look' remains.
    registry = _registry(_LookTool(), _MutateTool())
    provider = _ScriptByCallProvider(lambda n: LLMResult(tool_calls=[_c(f"c{n}", "mutate", n=n)]))
    result = asyncio.run(_loop(provider, tmp_path, registry).arun_turn("keep failing"))
    assert result.stop_reason == "stuck"
    assert "mutate" in _tool_names(provider.tools_seen[0])  # full set early on
    assert _tool_names(provider.tools_seen[4]) == {"look"}  # 5th call: read-only only


def test_loop_narrow_enforces_dispatch_not_just_schema(tmp_path: Path) -> None:
    # review2 #3: NARROW must ENFORCE, not merely advise. A model that ignores the narrowed
    # schema and still emits the write-tier 'mutate' on the narrowed iteration must have it
    # REJECTED (an error result), never executed — on any protocol. mutate executes on the 4
    # un-narrowed iterations (streak 1-4) but is rejected on the 5th (narrowed) iteration.
    mutate = _MutateTool()
    registry = _registry(_LookTool(), mutate)
    provider = _ScriptByCallProvider(lambda n: LLMResult(tool_calls=[_c(f"c{n}", "mutate", n=n)]))
    result = asyncio.run(_loop(provider, tmp_path, registry).arun_turn("keep failing"))
    assert result.stop_reason == "stuck"
    assert result.iterations == 5
    assert mutate.executed == 4  # ran on iters 1-4; the 5th (narrowed) call was rejected
    last = result.tool_results[-1]
    assert last.is_error and isinstance(last.data, dict) and last.data.get("step_restricted")
    # The rejection guides the model toward the read-only tool.
    assert "look" in last.output and "unavailable on this step" in last.output


def test_loop_not_stuck_when_progressing(tmp_path: Path) -> None:
    # Successful varying calls forever: never stuck, so it runs to the iteration cap with a
    # clean (non-degraded) result — no false positive.
    provider = _ScriptByCallProvider(lambda n: LLMResult(tool_calls=[_c(f"c{n}", "look", n=n)]))
    result = asyncio.run(_loop(provider, tmp_path).arun_turn("make progress"))
    assert result.stop_reason == "max_iterations"
    assert result.iterations == 10
    assert result.degraded is False


def test_loop_clean_completion_is_not_degraded(tmp_path: Path) -> None:
    # A normal write-free turn that just answers: degraded stays False.
    provider = _ScriptByCallProvider(lambda n: LLMResult(text="here is the answer"))
    result = asyncio.run(_loop(provider, tmp_path).arun_turn("hello"))
    assert result.stop_reason == "completed"
    assert result.degraded is False


# ── streaming path ───────────────────────────────────────────────────────────────


def _drain(loop: AgentLoop, text: str) -> list[Any]:
    async def run() -> list[Any]:
        return [ev async for ev in loop.astream_turn(text)]

    return asyncio.run(run())


def test_streaming_stuck_stops_and_emits_status(tmp_path: Path) -> None:
    provider = _ScriptByCallProvider(lambda n: LLMResult(tool_calls=[_c(f"c{n}", "boom", n=n)]))
    events = _drain(_loop(provider, tmp_path), "do the thing")
    done = next(e for e in events if isinstance(e, AgentDone))
    assert done.stop_reason == "stuck"
    assert done.degraded is True
    statuses = [e.message for e in events if isinstance(e, AgentStatus)]
    assert any("nudging" in m for m in statuses)
    assert any("read-only" in m for m in statuses)
    assert any("stopping: stuck" in m for m in statuses)


def test_streaming_injected_guidance_carries_rail_marker(tmp_path: Path) -> None:
    # rb-204: the streaming path (the REPL's real path) must wrap loop-injected stuck guidance
    # with the SAME "Hint:" control-rail marker as the buffered path. (review: was untested)
    provider = _ScriptByCallProvider(lambda n: LLMResult(tool_calls=[_c(f"c{n}", "boom", n=n)]))
    loop = _loop(provider, tmp_path)
    _drain(loop, "do the thing")
    injected = [m.text for m in loop.session.messages if m.text and "stuck" in m.text]
    assert injected and all(t.startswith("Hint:") for t in injected)
