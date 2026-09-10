"""The update_plan result is a RECEIPT, not the plan (ADR-0124).

Measured 2026-09-10 on a live agent (gemini-2.5-flash, a 40-step plan): 37 iterations, 11.68M
tokens, $3.57 in six minutes. The full checklist reached the model TWICE per iteration — as the
result of the update_plan it had just sent, and again as the ephemeral end-of-context reminder —
and the tool-result copy is the one that PERSISTS, so every edit grew the history for the rest of
the session. The reminder is the single carrier now; the result keeps only what the model cannot
get elsewhere: the step to act on, this call's advisories, this edit's score.
"""

from __future__ import annotations

from pathlib import Path

import zakcode
from tests.test_cli_chat import FakeAgent, app, runner
from zakcode.tasks import Task, TaskNetwork
from zakcode.tools.base import ToolContext
from zakcode.tools.builtins.update_plan import UpdatePlanTool


def _ctx() -> tuple[ToolContext, TaskNetwork]:
    net = TaskNetwork()
    return ToolContext(workspace_root=Path("/tmp"), task_network=net), net


async def test_the_result_is_a_receipt_naming_the_step_in_hand() -> None:
    ctx, net = _ctx()
    result = await UpdatePlanTool().execute(
        {"tasks": [{"title": "read the roster", "status": "done"}, {"title": "pick the lineup"}]},
        ctx,
    )
    assert not result.is_error
    first = result.output.splitlines()[0]
    assert first.startswith("Plan updated: 1/2 steps done")
    assert "current:" in first and "pick the lineup" in first
    # The checklist itself is NOT in the result — the reminder carries it, once, ephemerally.
    assert "Current plan (" not in result.output
    assert "[x]" not in result.output and "[ ]" not in result.output
    assert net.render()  # the network still renders in full for the reminder and /todo


async def test_a_finished_plan_says_so_and_the_advisories_survive() -> None:
    ctx, _ = _ctx()
    await UpdatePlanTool().execute({"tasks": [{"title": "a"}, {"title": "b"}]}, ctx)
    # Resend with a step missing: the full-replace advisory ("dropped open step") is
    # call-specific information the reminder cannot carry, so it stays in the receipt.
    result = await UpdatePlanTool().execute({"tasks": [{"title": "a", "status": "done"}]}, ctx)
    assert result.output.startswith("Plan updated: 1/1 steps done — complete.")
    assert "Notes:" in result.output and "dropped" in result.output


def test_todo_shows_the_full_plan_on_demand(monkeypatch) -> None:
    class PlannedAgent(FakeAgent):
        def __init__(self, **overrides: object) -> None:
            super().__init__(**overrides)
            self.session.task_network.replace_from_author(
                [Task(title="read the roster", status="done"), Task(title="pick the lineup")]
            )

    monkeypatch.setattr(zakcode, "Agent", PlannedAgent)
    result = runner.invoke(app, ["cli"], input="hello\n/todo\n/exit\n")
    assert result.exit_code == 0, result.stdout
    assert "plan 1/2 steps done" in result.stdout
    assert "read the roster" in result.stdout and "pick the lineup" in result.stdout
