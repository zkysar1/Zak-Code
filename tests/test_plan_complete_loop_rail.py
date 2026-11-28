"""ADR-0392: once a turn-end hook has named the session's loop skill, update_plan's finished-plan
rail names that loop instead of asking for an answer to "the user's original request".

ADR-0108's rail sends a finished plan back to the person's request. A served loop session has no
such request: its turns are the loop's own passes. Served loop sessions given that rail added
steps to deliver a final verdict on the original request and re-read rules about answering it,
pass after pass. Once a hook has named the loop (``Session.loop_skill``, ADR-0187), the rail
names it and leaves the request out.

Hermetic. The tool is driven directly with a bare context for the rail itself, and the seam
reuses the ADR-0205 scenario (scripted provider, in-process hook, fake composer): the context is
the one the LOOP built, and the hook names the loop in the middle of that turn.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from tests.test_refused_stop_silences_answer_now import (
    SKILL_REASON,
    _Hook,
    _judge_ok,
    _loop,
    _plan,
    _review_ok,
    _run,
    _say,
    _Scripted,
)
from zakcode.tasks import TaskNetwork
from zakcode.tools.base import ToolContext
from zakcode.tools.builtins.update_plan import (
    _COMPLETE_HINT,
    _LOOP_COMPLETE_HINT,
    UpdatePlanTool,
)

DONE = [{"title": "a", "status": "done"}, {"title": "b", "status": "done"}]
LOOP_RAIL = _LOOP_COMPLETE_HINT.format(skill="work-loop")


def _ctx(loop_skill: Callable[[], str] | None = None) -> ToolContext:
    return ToolContext(
        workspace_root=Path("/tmp"), task_network=TaskNetwork(), loop_skill=loop_skill
    )


# ── the rail ─────────────────────────────────────────────────────────────────


async def test_a_plan_finished_in_a_loop_session_gets_the_loop_rail() -> None:
    result = await UpdatePlanTool().execute({"tasks": DONE}, _ctx(lambda: "work-loop"))
    assert result.output.startswith("Plan updated: 2/2 steps done — complete.")
    assert result.hint == LOOP_RAIL
    assert "This session runs the /work-loop loop" in result.hint
    assert "Do what /work-loop says comes next" in result.hint
    # The words served loop sessions acted on are gone, not reworded around.
    assert "request" not in result.hint and "verdict" not in result.hint


async def test_the_loop_rail_names_the_skill_without_its_arguments() -> None:
    # Session.loop_skill is "<name> <args>"; the rail is about the skill, not how it was called.
    result = await UpdatePlanTool().execute({"tasks": DONE}, _ctx(lambda: "orchestrate loop"))
    assert result.hint == _LOOP_COMPLETE_HINT.format(skill="orchestrate")


async def test_every_receipt_that_reports_a_finished_plan_carries_the_loop_rail() -> None:
    # A whole plan (above), the short form closing the last step, a finished plan sent back
    # unchanged, and a worked step the harness advances: each can be the call that finishes it.
    tool = UpdatePlanTool()

    ctx = _ctx(lambda: "work-loop")
    await tool.execute({"tasks": [{"title": "a", "status": "in_progress"}]}, ctx)
    advanced = await tool.execute({"advance": {"outcome": "did a"}}, ctx)
    assert advanced.data is not None and advanced.data["complete"] is True
    assert advanced.hint == LOOP_RAIL

    ctx = _ctx(lambda: "work-loop")
    await tool.execute({"tasks": DONE}, ctx)
    resent = await tool.execute({"tasks": DONE}, ctx)
    assert resent.output.startswith("Plan unchanged: 2/2 steps done — complete.")
    assert resent.hint == LOOP_RAIL

    ctx = _ctx(lambda: "work-loop")
    worked = [{"title": "a", "status": "pending", "outcome": "did a"}]
    await tool.execute({"tasks": worked}, ctx)
    await tool.execute({"tasks": worked}, ctx)  # asks once before the advance
    moved = await tool.execute({"tasks": worked}, ctx)
    assert moved.data is not None and moved.data["autoadvanced"] is True
    assert moved.data["complete"] is True
    assert moved.hint == LOOP_RAIL


async def test_the_rail_reads_the_loop_when_the_plan_finishes() -> None:
    # The loop builds one context per turn and a hook can name the loop skill in the middle of
    # that turn (ADR-0187): a value copied when the context was built would miss it.
    named = {"skill": ""}
    ctx = _ctx(lambda: named["skill"])
    tool = UpdatePlanTool()
    assert (await tool.execute({"tasks": DONE}, ctx)).hint == _COMPLETE_HINT
    named["skill"] = "work-loop"
    assert (await tool.execute({"tasks": [{"title": "c", "status": "done"}]}, ctx)).hint == (
        LOOP_RAIL
    )


# ── controls: nothing else moves ─────────────────────────────────────────────


@pytest.mark.parametrize(
    "loop_skill",
    [None, lambda: "", lambda: "   "],
    ids=["bare-context", "no-loop-named", "blank"],
)
async def test_a_session_with_no_loop_keeps_the_verdict_rail(
    loop_skill: Callable[[], str] | None,
) -> None:
    result = await UpdatePlanTool().execute({"tasks": DONE}, _ctx(loop_skill))
    assert result.hint == _COMPLETE_HINT


async def test_an_open_plan_in_a_loop_session_keeps_its_next_step_rail() -> None:
    plan = [{"title": "a", "status": "in_progress"}, {"title": "b"}]
    result = await UpdatePlanTool().execute({"tasks": plan}, _ctx(lambda: "work-loop"))
    assert result.hint is not None
    assert result.hint.startswith("Plan updated. Now do the step marked '<- current'.")


# ── the seam: the context the loop builds ────────────────────────────────────


@pytest.fixture
def finished_rails(monkeypatch: pytest.MonkeyPatch) -> list[str | None]:
    """The rail of every receipt that reports a finished plan, as the tool returned it to the
    loop. The tool is the real one; only its results are copied out."""
    rails: list[str | None] = []
    real = UpdatePlanTool.execute

    async def recording(self: UpdatePlanTool, args: dict[str, Any], ctx: ToolContext) -> Any:
        result = await real(self, args, ctx)
        if result.data and result.data.get("complete"):
            rails.append(result.hint)
        return result

    monkeypatch.setattr(UpdatePlanTool, "execute", recording)
    return rails


@pytest.mark.asyncio
@pytest.mark.parametrize("streamed", [False, True], ids=["buffered", "streaming"])
async def test_a_plan_finished_after_the_hook_names_the_loop_gets_the_loop_rail(
    tmp_path: Path, streamed: bool, finished_rails: list[str | None]
) -> None:
    # ADR-0208's scenario: a plan finishes, the stop taken on it is refused with a reason that
    # names Skill('cycle') with args='loop', and the model finishes another plan in the SAME
    # turn. Both turn paths build their context once, before the hook has named anything.
    provider = _Scripted(
        [
            _plan("in_progress"),  # 0
            _judge_ok(),  # 1  side
            _plan("done"),  # 2  finished before any hook named a loop: the verdict rail
            _say("done: the answer"),  # 3  refused; the harness delivers /cycle loop
            _review_ok(),  # 4  side
            _plan("in_progress"),  # 5  reopens it
            _plan("done"),  # 6  finished after the hook named the loop: the loop rail
            _say("done again: the answer"),  # 7  the hook lets the turn end
        ]
    )
    loop = _loop(provider, tmp_path, _Hook([SKILL_REASON]))
    assert loop.session.loop_skill == ""
    assert await _run(loop, "one step", streamed=streamed) == "completed"
    assert loop.session.loop_skill == "cycle loop"
    assert finished_rails == [_COMPLETE_HINT, _LOOP_COMPLETE_HINT.format(skill="cycle")]
