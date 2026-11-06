"""The bound on consecutive plan-only completions (ADR-0364).

A completion whose every call is a plan tool, and that neither closed a step nor drew the
harness's advance, cannot move the plan. The third such completion in a row is answered
unexecuted, naming the step to do; any other call ends the run. Tracker-level tests pin the
counting, loop-level tests pin the refusal on BOTH turn paths (a behaviour written at one of
the two is silently absent on the other).
"""

from __future__ import annotations

from typing import Any

import pytest

from tests.test_loop_planning import _done, _judge_ok, _plan_call, _review_ok, _Scripted
from zakcode.agent.loop import _PLAN_TOOLS, AgentLoop
from zakcode.agent.stuck import SIG_PLAN_ONLY, StuckAction, StuckTracker
from zakcode.config import PermissionTier, Settings
from zakcode.messages import ToolResultBlock
from zakcode.providers.base import LLMResult, ToolCall
from zakcode.session.store import Session
from zakcode.tools.base import ConcurrencyClass, Tool, ToolContext, ToolResult, ToolSpec
from zakcode.tools.builtins.default_registry import default_registry
from zakcode.usage import Usage

# ── the tracker ──────────────────────────────────────────────────────────────────


def _plan(n: int) -> list[ToolCall]:
    return [ToolCall(id=f"p{n}", name="update_plan", arguments={"tasks": [{"title": str(n)}]})]


def _receipt(call: ToolCall, **data: Any) -> ToolResultBlock:
    return ToolResultBlock(tool_use_id=call.id, output="Plan updated: 0/1 steps done", data=data)


def _tracker() -> StuckTracker:
    return StuckTracker(plan_tools=_PLAN_TOOLS)


def _observe_plan(tracker: StuckTracker, n: int, **data: Any) -> list[ToolCall]:
    calls = _plan(n)
    tracker.observe(calls, [_receipt(calls[0], **{"finished": 0, **data})])
    return calls


def test_the_third_plan_only_completion_in_a_row_is_the_one_refused() -> None:
    tracker = _tracker()
    assert not tracker.refuses_plan_only(_plan(1))
    _observe_plan(tracker, 1)
    assert not tracker.refuses_plan_only(_plan(2))
    _observe_plan(tracker, 2)
    assert tracker.plan_only_run == 2
    assert tracker.refuses_plan_only(_plan(3))


def test_any_other_call_ends_the_run_even_in_the_same_completion() -> None:
    tracker = _tracker()
    _observe_plan(tracker, 1)
    _observe_plan(tracker, 2)
    mixed = [*_plan(3), ToolCall(id="r1", name="read_file", arguments={"path": "x"})]
    tracker.observe(
        mixed, [_receipt(mixed[0], finished=0), ToolResultBlock(tool_use_id="r1", output="x")]
    )
    assert tracker.plan_only_run == 0
    assert not tracker.refuses_plan_only(_plan(4))


def test_a_plan_call_that_moved_the_plan_is_not_churn() -> None:
    # Lever N walking a worked plan (ADR-0168), and a model closing steps on work it already did.
    walked = _tracker()
    for n in range(5):
        _observe_plan(walked, n, autoadvanced=n > 0)
    assert walked.plan_only_run == 0
    closing = _tracker()
    for n in range(5):
        _observe_plan(closing, n, finished=n)
    assert closing.plan_only_run == 0


def test_an_errored_plan_call_did_not_move_the_plan() -> None:
    tracker = _tracker()
    for n in range(2):
        calls = _plan(n)
        tracker.observe(
            calls, [ToolResultBlock(tool_use_id=calls[0].id, output="bad", is_error=True)]
        )
    assert tracker.refuses_plan_only(_plan(2))


def test_the_bound_is_inert_without_plan_tools_and_cleared_by_a_reset() -> None:
    inert = StuckTracker()
    for n in range(4):
        _observe_plan(inert, n)
    assert not inert.refuses_plan_only(_plan(9))
    tracker = _tracker()
    _observe_plan(tracker, 1)
    _observe_plan(tracker, 2)
    tracker.reset()  # a TURN_END veto sent the model round again
    assert not tracker.refuses_plan_only(_plan(3))


def test_a_refusal_is_its_own_signal_and_no_failure_of_the_plan_tool() -> None:
    tracker = _tracker()
    _observe_plan(tracker, 1)
    _observe_plan(tracker, 2)
    actions = []
    for n in range(3, 8):
        calls = _plan(n)
        refusal = ToolResultBlock(
            tool_use_id=calls[0].id,
            output=tracker.plan_only_message("1 '1'"),
            is_error=True,
            data={"plan_only_bound": True},
        )
        tracker.observe(calls, [refusal])
        assert tracker.last_signals == [SIG_PLAN_ONLY]
        actions.append(tracker.next_action())
    assert tracker.failing_tools() == []  # never "update_plan has failed N times"
    # The first refusal is itself the intervention; then one rung per refusal, to a stop.
    assert actions == [
        StuckAction.CONTINUE,
        StuckAction.NUDGE,
        StuckAction.NARROW,
        StuckAction.STEP_BACK,
        StuckAction.STOP,
    ]


def test_the_refusal_names_the_step_and_says_what_was_counted() -> None:
    tracker = _tracker()
    _observe_plan(tracker, 1)
    _observe_plan(tracker, 2)
    text = tracker.plan_only_message("2 'Run the tests'")
    assert text.startswith("Not executed: this is response 3 in a row that calls only the plan")
    assert "current step is 2 'Run the tests': do that step now with a tool call" in text
    assert "act on it now" in tracker.plan_only_message(None)


# ── the loop, both turn paths ────────────────────────────────────────────────────


class _Probe(Tool):
    spec = ToolSpec(
        name="probe",
        description="a read-only look at the workspace",
        required_permission=PermissionTier.READ_ONLY,
        concurrency=ConcurrencyClass.READ_ONLY_SAFE,
    )

    def __init__(self) -> None:
        self.runs = 0

    async def execute(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        self.runs += 1
        return ToolResult.ok(f"probe {self.runs}: the workspace answered")


def _probe_call(n: int) -> LLMResult:
    call = ToolCall(id=f"probe-{n}", name="probe", arguments={"n": n})
    return LLMResult(text="", tool_calls=[call], usage=Usage(total_tokens=1))


def _steps(*titles: str) -> list[dict[str, Any]]:
    return [{"title": t} for t in titles]


async def _turn(script: list[LLMResult], streamed: bool) -> tuple[Session, AgentLoop, _Probe]:
    probe = _Probe()
    registry = default_registry()
    registry.register(probe)
    session = Session(cwd="/tmp", model="test/model")
    loop = AgentLoop(_Scripted(script), registry, session, max_iterations=30, settings=Settings())
    if streamed:
        async for _event in loop.astream_turn("do a multi-step thing"):
            pass
    else:
        await loop.arun_turn("do a multi-step thing")
    return session, loop, probe


def _plan_results(session: Session) -> list[ToolResultBlock]:
    return [
        block
        for message in session.messages
        for block in message.blocks
        if isinstance(block, ToolResultBlock) and block.tool_use_id == "p1"
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("streamed", [False, True], ids=["buffered", "streaming"])
async def test_the_loop_refuses_the_third_plan_only_completion_and_takes_the_plan_after_work(
    streamed: bool,
) -> None:
    script = [
        _plan_call(_steps("a", "b")),
        _judge_ok(),
        _plan_call(_steps("a", "b", "c")),
        _plan_call(_steps("a", "b", "c", "d")),  # the third in a row: refused
        _probe_call(1),  # another tool ends the run
        _plan_call(_steps("a", "b", "c", "e")),  # accepted again
        _done(),
        _review_ok(),
    ]
    session, loop, probe = await _turn(script, streamed)
    results = _plan_results(session)
    assert [bool((b.data or {}).get("plan_only_bound")) for b in results] == [
        False,
        False,
        True,
        False,
    ]
    refused = results[2]
    assert refused.is_error
    assert "current step is 1 'a': do that step now" in refused.output
    assert probe.runs == 1
    # The refused plan never landed; the one sent after the probe did.
    assert [t.title for t in session.task_network.leaves()] == ["a", "b", "c", "e"]
    notes = [e for e in loop._trace.events if e.data.get("kind") == "plan_only_bound"]
    assert len(notes) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("streamed", [False, True], ids=["buffered", "streaming"])
async def test_planning_between_real_steps_is_never_refused(streamed: bool) -> None:
    script = [_plan_call(_steps("a", "b")), _judge_ok()]
    for n in range(4):
        script += [_probe_call(n), _plan_call(_steps("a", "b", *[f"x{i}" for i in range(n + 1)]))]
    script += [_done(), _review_ok()]
    session, loop, probe = await _turn(script, streamed)
    assert not any((b.data or {}).get("plan_only_bound") for b in _plan_results(session))
    assert probe.runs == 4
    assert not [e for e in loop._trace.events if e.data.get("kind") == "plan_only_bound"]
