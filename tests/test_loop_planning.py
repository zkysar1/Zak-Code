"""Integration tests for the task-network planning wired into the agent loop.

Hermetic: a scripted :class:`Provider` (no network) drives ``update_plan`` calls and
completions, and we assert the loop (1) re-injects the live plan into later provider calls,
(2) nudges before finishing with open plan steps and then completes gracefully (degraded), and
(3) emits a ``task_update`` event on the streaming path. The same script drives both the
buffered and streaming paths (the base ``astream`` wraps ``acomplete``).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from zakcode.agent.loop import AgentLoop
from zakcode.events import AgentTaskUpdate
from zakcode.providers.base import Capabilities, LLMResult, Provider, ToolCall
from zakcode.session.store import Session
from zakcode.tasks import Task, TaskNetwork
from zakcode.tools.base import ToolContext
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
