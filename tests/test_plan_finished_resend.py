"""A finished plan sent back after the turn-start reset stays closed (ADR-0390).

Measured on ten served workers over 2.2 hours: 101 plan-only replies used 248 model-minutes; 20 of
them sent a finished plan (66 minutes) and 9 were byte-identical resends of a plan that had already
finished (33 minutes). One worker sent the same finished 12-step plan back four times in 17
minutes. Each resend put the plan back on the board the turn start had cleared (ADR-0110), so every
later call of that turn carried the "Plan complete ... answer now" line for a plan that was over.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from zakcode.tasks import TaskNetwork
from zakcode.tools.base import ToolContext
from zakcode.tools.builtins.update_plan import _COMPLETE_HINT, UpdatePlanTool

DONE = [
    {"title": "Read the failing test", "status": "done", "outcome": "the fixture is stale"},
    {
        "title": "Fix the fixture",
        "subtasks": [
            {"title": "regenerate it", "status": "done"},
            {"title": "drop the old copy", "status": "cancelled"},
        ],
    },
]


def _ctx() -> tuple[ToolContext, TaskNetwork]:
    net = TaskNetwork()
    return ToolContext(workspace_root=Path("/tmp"), task_network=net), net


def _turn_start_reset(net: TaskNetwork) -> None:
    """What the loop's turn-start reset does to a finished or abandoned plan (ADR-0110)."""
    net.tasks = []


async def test_a_finished_plan_sent_back_after_the_reset_is_not_reopened() -> None:
    ctx, net = _ctx()
    first = await UpdatePlanTool().execute({"tasks": DONE}, ctx)
    assert first.output.startswith("Plan updated: 3/3 steps done — complete.")
    _turn_start_reset(net)
    events = len(net.log)

    again = await UpdatePlanTool().execute({"tasks": DONE}, ctx)
    assert not again.is_error
    assert again.output.startswith("Not reopened: this plan already finished (3/3 steps done)")
    assert "Send update_plan again only for new work" in again.output
    assert again.hint is None  # no verdict rail: the plan's turn is over
    assert again.data == {"task_count": 0, "unchanged": True, "reopened": False}
    assert net.is_empty()  # the board stays clear, so no "answer now" line rides the turn
    assert len(net.log) == events


async def test_the_same_finished_plan_resent_in_its_own_turn_keeps_the_verdict_rail() -> None:
    ctx, net = _ctx()
    await UpdatePlanTool().execute({"tasks": DONE}, ctx)
    again = await UpdatePlanTool().execute({"tasks": DONE}, ctx)  # no reset between
    assert again.output.startswith("Plan unchanged: 3/3 steps done — complete.")
    assert again.hint == _COMPLETE_HINT
    assert net.is_complete()


async def test_a_changed_finished_plan_after_the_reset_goes_up() -> None:
    ctx, net = _ctx()
    await UpdatePlanTool().execute({"tasks": DONE}, ctx)
    _turn_start_reset(net)
    renamed = [dict(DONE[0], title="Read the failing test again"), DONE[1]]
    result = await UpdatePlanTool().execute({"tasks": renamed}, ctx)
    assert result.output.startswith("Plan updated: 3/3 steps done — complete.")
    assert not net.is_empty()


async def test_an_unfinished_plan_sent_back_after_the_reset_goes_up() -> None:
    open_plan = [{"title": "a", "status": "done"}, {"title": "b", "status": "in_progress"}]
    ctx, net = _ctx()
    await UpdatePlanTool().execute({"tasks": open_plan}, ctx)
    _turn_start_reset(net)  # the abandoned-plan drop
    result = await UpdatePlanTool().execute({"tasks": open_plan}, ctx)
    assert not result.output.startswith("Not reopened")
    assert net.current() is not None and net.current().title == "b"


async def test_a_finished_plan_after_the_model_cleared_the_board_goes_up() -> None:
    ctx, net = _ctx()
    await UpdatePlanTool().execute({"tasks": DONE}, ctx)
    cleared = await UpdatePlanTool().execute({"tasks": []}, ctx)
    assert cleared.output == "Plan cleared."
    result = await UpdatePlanTool().execute({"tasks": DONE}, ctx)
    assert result.output.startswith("Plan updated: 3/3 steps done — complete.")
    assert net.is_complete()


@pytest.mark.asyncio
async def test_the_loop_keeps_a_finished_plan_closed_across_a_turn() -> None:
    from tests.test_loop_planning import _done, _judge_ok, _loop, _plan_call, _review_ok, _Scripted

    plan = [{"title": "A", "status": "done"}, {"title": "B", "status": "done"}]
    provider = _Scripted(
        [
            _plan_call(plan),
            _judge_ok(),
            _done("turn one finished"),
            _review_ok(),  # the fresh-eyes review of turn one's finished plan (ADR-0117)
            _plan_call(plan),  # turn two opens by sending the finished plan back
            _done("turn two answer"),
        ]
    )
    loop, session = _loop(provider)
    await loop.arun_turn("first goal")
    assert session.task_network.is_complete()

    await loop.arun_turn("a different question")
    assert session.task_network.is_empty()
    outputs = [
        b.output
        for m in session.messages
        for b in m.blocks
        if type(b).__name__ == "ToolResultBlock" and "Not reopened" in b.output
    ]
    assert len(outputs) == 1
    # The call after the resend carries no plan line: nothing finished is back on the board.
    assert not any("[plan]" in m.text for m in provider.seen[-1])
