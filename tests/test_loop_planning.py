"""Integration tests for the task-network planning wired into the agent loop.

Hermetic: a scripted :class:`Provider` (no network) drives ``update_plan`` calls and
completions, and we assert the loop (1) re-injects the live plan into later provider calls,
(2) nudges before finishing with open plan steps and then completes gracefully (degraded), and
(3) emits a ``task_update`` event on the streaming path. The same script drives both the
buffered and streaming paths (the base ``astream`` wraps ``acomplete``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from zakcode.agent.loop import _MAX_PLAN_IDLE_TURNS, AgentLoop
from zakcode.config import PermissionTier, Settings
from zakcode.events import AgentTaskUpdate
from zakcode.providers.base import Capabilities, LLMResult, Provider, ToolCall
from zakcode.session.store import Session
from zakcode.tasks import Task, TaskNetwork
from zakcode.tools.base import (
    ConcurrencyClass,
    Tool,
    ToolContext,
    ToolRegistry,
    ToolResult,
    ToolSpec,
)
from zakcode.tools.builtins.default_registry import default_registry
from zakcode.tools.builtins.update_plan import UpdatePlanTool
from zakcode.usage import Usage


class _Scripted(Provider):
    """Returns a fixed sequence of completions and records the messages it was handed."""

    def __init__(self, results: list[LLMResult]) -> None:
        self._results = results
        self.calls = 0
        self.seen: list[list] = []

    async def acomplete(self, messages, *, system=None, tools=None, **kw) -> LLMResult:
        self.seen.append(list(messages))
        i = min(self.calls, len(self._results) - 1)
        self.calls += 1
        return self._results[i]

    def count_tokens(self, messages, *, system=None) -> int:
        return 0

    def capabilities(self) -> Capabilities:
        return Capabilities()


def _plan_call(tasks: list[dict]) -> LLMResult:
    return LLMResult(
        text="",
        tool_calls=[ToolCall(id="p1", name="update_plan", arguments={"tasks": tasks})],
        usage=Usage(total_tokens=1),
    )


def _done(text: str = "all done") -> LLMResult:
    return LLMResult(text=text, tool_calls=[], usage=Usage(total_tokens=1))


def _loop(provider: Provider) -> tuple[AgentLoop, Session]:
    session = Session(cwd="/tmp", model="test/model")
    loop = AgentLoop(provider, default_registry(), session, max_iterations=20)
    return loop, session


@pytest.mark.asyncio
async def test_plan_is_persisted_and_reinjected_into_later_calls() -> None:
    provider = _Scripted([_plan_call([{"title": "A", "status": "in_progress"}, {"title": "B"}])])
    loop, session = _loop(provider)
    # Provider only ever asks for the plan, then keeps "finishing"; the gate will run.
    provider._results.append(_done())
    await loop.arun_turn("do a two-step thing")

    # The plan is stored on the session (survives the turn / would survive /resume).
    assert [t.title for t in session.task_network.tasks] == ["A", "B"]
    assert session.task_network.tasks[0].id == "1"

    # A provider call AFTER the update_plan saw the live plan re-injected (ephemeral tail),
    # but it was NOT persisted into the durable message history.
    later = provider.seen[1]
    assert any("[plan]" in m.text and "Current plan" in m.text for m in later)
    assert not any("[plan]" in m.text for m in session.messages)


@pytest.mark.asyncio
async def test_completion_gate_nudges_then_completes_degraded() -> None:
    # Lay out a plan with an open step, then keep trying to finish without resolving it.
    provider = _Scripted(
        [_plan_call([{"title": "A", "status": "done"}, {"title": "B", "status": "pending"}])]
        + [_done()] * 6
    )
    loop, _ = _loop(provider)
    result = await loop.arun_turn("two steps")

    # call1 plan; calls 2 & 3 nudged (cap=2); call 4 completes despite the open step.
    assert result.stop_reason == "completed"
    assert result.degraded is True  # finished with an unresolved plan step
    assert provider.calls == 4


@pytest.mark.asyncio
async def test_completion_gate_is_inert_when_plan_is_complete() -> None:
    provider = _Scripted(
        [_plan_call([{"title": "A", "status": "done"}, {"title": "B", "status": "done"}]), _done()]
    )
    loop, session = _loop(provider)
    result = await loop.arun_turn("two steps")

    assert result.stop_reason == "completed"
    assert result.degraded is False  # nothing left open -> no nudge, clean finish
    assert provider.calls == 2
    assert session.task_network.is_complete()


@pytest.mark.asyncio
async def test_no_plan_means_no_gate() -> None:
    provider = _Scripted([_done("just a quick answer")])
    loop, _ = _loop(provider)
    result = await loop.arun_turn("hi")
    assert result.stop_reason == "completed"
    assert result.degraded is False
    assert provider.calls == 1


@pytest.mark.asyncio
async def test_completed_plan_is_reset_at_next_turn_start() -> None:
    # Turn 1 completes a plan; turn 2 (unrelated) must start clean — the finished checklist is
    # neither carried forward nor re-injected.
    provider = _Scripted(
        [
            _plan_call([{"title": "A", "status": "done"}, {"title": "B", "status": "done"}]),
            _done("turn one finished"),
            _done("turn two answer"),
        ]
    )
    loop, session = _loop(provider)
    await loop.arun_turn("first goal")
    assert session.task_network.is_complete()

    await loop.arun_turn("a different question")
    assert session.task_network.is_empty()  # the completed plan was reset at turn start
    # The new turn's provider call saw no plan re-injected.
    assert not any("[plan]" in m.text for m in provider.seen[-1])


@pytest.mark.asyncio
async def test_incomplete_plan_persists_across_turns() -> None:
    # An UNFINISHED plan carries forward and is re-injected on the next turn (multi-turn work).
    provider = _Scripted(
        [_plan_call([{"title": "A", "status": "done"}, {"title": "B", "status": "pending"}])]
        + [_done()] * 8
    )
    loop, session = _loop(provider)
    await loop.arun_turn("start the work")
    assert not session.task_network.is_empty() and not session.task_network.is_complete()

    n_calls_after_turn1 = provider.calls
    await loop.arun_turn("keep going")
    # The first provider call of turn 2 re-injected the still-open plan.
    assert any("[plan]" in m.text for m in provider.seen[n_calls_after_turn1])


@pytest.mark.asyncio
async def test_update_plan_rejects_all_malformed_without_wiping_existing_plan() -> None:
    net = TaskNetwork(tasks=[Task(title="keep me", status="in_progress")])
    net.normalize()
    ctx = ToolContext(workspace_root=Path("/tmp"), task_network=net)
    res = await UpdatePlanTool().execute({"tasks": [1, "nope", None]}, ctx)
    assert res.is_error
    assert [t.title for t in net.tasks] == ["keep me"]  # untouched by the bad call


@pytest.mark.asyncio
async def test_streaming_emits_task_update_event() -> None:
    plan = _plan_call([{"title": "A", "status": "in_progress"}, {"title": "B"}])
    provider = _Scripted([plan] + [_done()] * 5)
    loop, _ = _loop(provider)
    events = [ev async for ev in loop.astream_turn("two steps")]
    updates = [ev for ev in events if isinstance(ev, AgentTaskUpdate)]
    assert updates, "expected a task_update event after update_plan"
    last = updates[-1]
    assert "Current plan" in last.plan
    assert last.total == 2 and [t["title"] for t in last.tasks] == ["A", "B"]


# ── R3: capability-triggered decomposition hint (decompose-on-stuck) ──────────


def test_decompose_hint_targets_a_primitive_current_step() -> None:
    loop, session = _loop(_Scripted([_done()]))
    session.task_network = TaskNetwork(tasks=[Task(title="big step")])
    session.task_network.normalize()
    assert "smaller sub-steps" in loop._decompose_hint()  # current step is primitive


def test_decompose_hint_is_empty_without_a_primitive_current_step() -> None:
    loop, session = _loop(_Scripted([_done()]))
    assert loop._decompose_hint() == ""  # no plan
    # an already-decomposed (compound) current step is not re-suggested for decomposition
    session.task_network = TaskNetwork(
        tasks=[Task(title="g", kind="compound", children=[Task(title="leaf", status="done")])]
    )
    session.task_network.normalize()
    assert loop._decompose_hint() == ""


# ── R5: opt-in plan-first gate (plan before you mutate) ───────────────────────


class _FakeWrite(Tool):
    spec = ToolSpec(
        name="write_file",
        description="fake write",
        required_permission=PermissionTier.WORKSPACE_WRITE,
        concurrency=ConcurrencyClass.NEVER_PARALLEL,
    )

    def __init__(self) -> None:
        self.runs = 0

    async def execute(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        self.runs += 1
        return ToolResult.ok("written")


def _plan_first_loop(provider: Provider, write: _FakeWrite) -> tuple[AgentLoop, Session]:
    reg = ToolRegistry()
    reg.register(write)
    reg.register(UpdatePlanTool())
    session = Session(cwd="/tmp", model="t/m")
    loop = AgentLoop(
        provider, reg, session, settings=Settings(require_plan=True), max_iterations=20
    )
    return loop, session


def _write_call() -> LLMResult:
    return LLMResult(
        text="",
        tool_calls=[ToolCall(id="w", name="write_file", arguments={"path": "a.txt"})],
        usage=Usage(total_tokens=1),
    )


@pytest.mark.asyncio
async def test_plan_first_withholds_a_mutation_until_a_plan_exists() -> None:
    write = _FakeWrite()
    # write (withheld) -> plan -> write (now allowed) -> done
    provider = _Scripted(
        [_write_call(), _plan_call([{"title": "a"}, {"title": "b"}]), _write_call(), _done()]
    )
    loop, _ = _plan_first_loop(provider, write)
    result = await loop.arun_turn("edit something")
    assert result.stop_reason == "completed"
    assert write.runs == 1  # the write ran only AFTER a plan existed


@pytest.mark.asyncio
async def test_plan_first_fails_open_after_the_cap() -> None:
    write = _FakeWrite()
    # model never plans, just keeps trying to write; after the cap the write goes through
    provider = _Scripted([_write_call()] * 6)
    loop, _ = _plan_first_loop(provider, write)
    result = await loop.arun_turn("edit something")
    assert write.runs >= 1  # not deadlocked — fails open and the write eventually executes
    assert result.stop_reason in {"completed", "max_iterations", "doom_loop"}


@pytest.mark.asyncio
async def test_plan_first_is_off_by_default() -> None:
    write = _FakeWrite()
    reg = ToolRegistry()
    reg.register(write)
    reg.register(UpdatePlanTool())
    loop = AgentLoop(
        _Scripted([_write_call(), _done()]),
        reg,
        Session(cwd="/tmp", model="t/m"),
        max_iterations=20,
    )
    await loop.arun_turn("edit something")
    assert write.runs == 1  # no gate -> the write runs immediately, no plan required


# ── issue #32: bounded auto-clear of a stale / abandoned incomplete plan ───────


def test_stale_incomplete_plan_is_dropped_after_idle_turn_threshold() -> None:
    # Unit: an unfinished plan left byte-identical across turn-starts accrues idle turns and is
    # dropped once it stalls, so it cannot re-inject + re-nudge + degrade every turn forever.
    loop, session = _loop(_Scripted([_done()]))
    session.task_network = TaskNetwork(
        tasks=[Task(title="A", status="done"), Task(title="B", status="pending")]
    )
    session.task_network.normalize()

    loop._reset_stale_or_completed_plan()  # first sighting -> record signature, counter 0
    assert not session.task_network.is_empty()
    assert session.plan_idle_turns == 0
    for k in range(1, _MAX_PLAN_IDLE_TURNS):  # unchanged turn-starts accrue idle turns
        loop._reset_stale_or_completed_plan()
        assert not session.task_network.is_empty(), f"dropped too early at idle turn {k}"
        assert session.plan_idle_turns == k
    loop._reset_stale_or_completed_plan()  # crosses the threshold -> dropped
    assert session.task_network.is_empty()
    assert session.plan_idle_turns == 0 and session.plan_signature == ""


def test_active_plan_is_never_dropped_because_each_edit_resets_the_counter() -> None:
    # Unit: any change (advance or edit) resets the idle counter, so genuine multi-turn work is
    # never auto-cleared no matter how many turns pass.
    loop, session = _loop(_Scripted([_done()]))
    net = TaskNetwork(tasks=[Task(title="A", status="pending"), Task(title="B", status="pending")])
    session.task_network = net
    net.normalize()
    for _ in range(_MAX_PLAN_IDLE_TURNS + 5):
        # advance the plan a little before each turn-start so the signature keeps changing
        net.tasks[0].status = "in_progress" if net.tasks[0].status == "pending" else "pending"
        net.normalize()
        loop._reset_stale_or_completed_plan()
        assert not session.task_network.is_empty()
        assert session.plan_idle_turns == 0  # reset by the change every turn


def test_plan_gate_nudge_offers_to_clear_the_plan() -> None:
    # The completion nudge names the escape hatch (clear the plan) for an abandoned goal.
    loop, session = _loop(_Scripted([_done()]))
    session.task_network = TaskNetwork(
        tasks=[Task(title="A", status="pending"), Task(title="B", status="pending")]
    )
    session.task_network.normalize()
    nudge = loop._plan_gate_nudge()
    assert nudge is not None
    assert "clear the whole plan with update_plan" in nudge


@pytest.mark.asyncio
async def test_abandoned_plan_stops_haunting_across_real_turns() -> None:
    # Integration (buffered): a plan left incomplete and untouched is dropped within a bounded
    # number of real turns instead of re-injecting + nudging + degrading forever.
    provider = _Scripted(
        [_plan_call([{"title": "A", "status": "done"}, {"title": "B", "status": "pending"}])]
        + [_done()] * 80
    )
    loop, session = _loop(provider)
    await loop.arun_turn("lay out a plan")
    assert not session.task_network.is_empty()

    cleared = False
    for _ in range(_MAX_PLAN_IDLE_TURNS + 3):
        await loop.arun_turn("an unrelated message")
        if session.task_network.is_empty():
            cleared = True
            break
    assert cleared, "abandoned plan kept haunting across turns"
    # Once dropped, a later turn sees no plan re-injected.
    await loop.arun_turn("later still")
    assert not any("[plan]" in m.text for m in provider.seen[-1])


@pytest.mark.asyncio
async def test_abandoned_plan_auto_clear_holds_on_streaming_path() -> None:
    # Parity: the same staleness guard fires on astream_turn (the reset runs at turn start on
    # both the buffered and streaming paths).
    provider = _Scripted(
        [_plan_call([{"title": "A", "status": "done"}, {"title": "B", "status": "pending"}])]
        + [_done()] * 80
    )
    loop, session = _loop(provider)
    async for _ in loop.astream_turn("lay out a plan"):
        pass
    assert not session.task_network.is_empty()

    cleared = False
    for _ in range(_MAX_PLAN_IDLE_TURNS + 3):
        async for _ in loop.astream_turn("unrelated"):
            pass
        if session.task_network.is_empty():
            cleared = True
            break
    assert cleared, "abandoned plan kept haunting on the streaming path"
