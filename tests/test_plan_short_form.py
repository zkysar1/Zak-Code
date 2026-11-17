"""ADR-0388: ``update_plan``'s short form — close the current step and start the next.

The full-replace contract (the model resends the whole plan on every call) made the commonest
edit the most expensive one: measured 2026-11-17 on ten 27B worker sessions behind one pod, the
responses that only updated the plan were 19 to 28 percent of each session's model time, a median
44 s and 668 streamed tokens each, and most of those tokens were the resent plan. ``advance``
says "the current step is finished" in a few dozen tokens. The harness builds the full plan it
stands for and installs it through the same full-replace path, so these tests pin both halves:
the short form moves the plan the way the full form would (outcome, log, dependencies, nesting,
notes kept), and everything the full form guarantees still holds (the null-close challenge,
the unchanged rail, the paged skill's next section, the plan-first gate's question).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from zakcode.agent.loop import AgentLoop
from zakcode.agent.prompt import SystemPromptBuilder
from zakcode.config import PermissionTier, load_settings
from zakcode.providers.base import Capabilities, LLMResult, Provider, ToolCall
from zakcode.session.store import Session
from zakcode.tasks import NULL_MARK, Task, TaskNetwork
from zakcode.tools.base import (
    RECEIPT_OF_CHANGE,
    ConcurrencyClass,
    Tool,
    ToolContext,
    ToolRegistry,
    ToolResult,
    ToolSpec,
)
from zakcode.tools.builtins.update_plan import (
    _ADVANCED_HINT,
    _CHALLENGED_HINT,
    _COMPLETE_HINT,
    _PLANNED_HINT,
    _UNCHANGED_HINT,
    PLAN_ADVANCE_CALL,
    PLAN_SHORT_FORM,
    UpdatePlanTool,
    authors_a_plan,
)
from zakcode.usage import Usage


def _ctx(network: TaskNetwork) -> ToolContext:
    return ToolContext(workspace_root=Path("/tmp"), task_network=network)


async def _send(network: TaskNetwork, args: dict[str, Any]) -> ToolResult:
    return await UpdatePlanTool().execute(args, _ctx(network))


async def _laid_out(steps: list[dict[str, Any]]) -> TaskNetwork:
    network = TaskNetwork()
    res = await _send(network, {"tasks": steps})
    assert not res.is_error, res.output
    return network


def _data(res: ToolResult) -> dict[str, Any]:
    assert res.data is not None, res.output
    return res.data


def _statuses(network: TaskNetwork) -> dict[str, str]:
    return {leaf.title: leaf.status for leaf in network.leaves()}


THREE = [
    {"title": "read the config", "status": "in_progress", "note": "the port is known"},
    {"title": "change the port", "note": "the file says 8081"},
    {"title": "restart and check", "note": "GET /health returns 200"},
]


# ── the short form moves the plan the way the full form would ─────────────────────────


async def test_advance_closes_the_current_step_and_starts_the_next() -> None:
    network = await _laid_out(THREE)

    res = await _send(network, {"advance": {"outcome": "port is 8080"}})

    assert not res.is_error, res.output
    assert _statuses(network) == {
        "read the config": "done",
        "change the port": "in_progress",
        "restart and check": "pending",
    }
    assert network.leaves()[0].outcome == "port is 8080"
    # The receipt names the step it closed: the model never sent the plan, so this is how it
    # learns which step the advance closed.
    assert res.output.startswith("Step 1 done ('read the config'). Plan updated: 1/3 steps done")
    assert "current: 2 change the port" in res.output
    assert _data(res)["short_form"] == "1" and _data(res)[RECEIPT_OF_CHANGE] is True
    assert res.hint == _PLANNED_HINT
    # Logged like any close and start (ADR-0110), so recent_closed and the history read it.
    details = [event.detail for event in network.log if event.kind == "step"]
    assert details[-2:] == ["in_progress -> done — port is 8080", "pending -> in_progress"]
    assert network.last_closed() is network.leaves()[0]


async def test_the_full_form_reports_no_short_form() -> None:
    network = await _laid_out(THREE)

    res = await _send(network, {"tasks": [dict(THREE[0], status="done"), *THREE[1:]]})

    assert _data(res)["short_form"] == "" and res.output.startswith("Plan updated:")


async def test_advance_walks_nested_steps_in_dependency_order() -> None:
    # "report" sits before "build" but waits on it, and "tidy" waits on nothing. The next step is
    # read from the plan as it will stand, dependencies and derived parent statuses included, so
    # "report" starts only once BOTH of build's children are done, and still before "tidy".
    network = await _laid_out(
        [
            {"title": "read", "status": "in_progress"},
            {"title": "report", "blocked_by": ["3"]},
            {
                "title": "build",
                "subtasks": [{"title": "compile"}, {"title": "link", "blocked_by": ["3.1"]}],
            },
            {"title": "tidy"},
        ]
    )

    order = []
    for _ in range(4):
        res = await _send(network, {"advance": {}})
        assert not res.is_error, res.output
        current = network.current()
        order.append(current.title if current is not None else None)

    assert order == ["compile", "link", "report", "tidy"]
    res = await _send(network, {"advance": "tidied"})
    assert network.is_complete() and res.hint == _COMPLETE_HINT
    assert res.output.startswith("Step 4 done ('tidy'). Plan updated: 5/5 steps done — complete.")


async def test_every_step_keeps_its_note_and_outcome() -> None:
    network = await _laid_out(THREE)
    await _send(network, {"advance": {"outcome": "port is 8080"}})

    await _send(network, {"advance": {"outcome": "changed in app.cfg"}})

    notes = [leaf.note for leaf in network.leaves()]
    assert notes == ["the port is known", "the file says 8081", "GET /health returns 200"]
    assert [leaf.outcome for leaf in network.leaves()][:2] == ["port is 8080", "changed in app.cfg"]


@pytest.mark.parametrize(
    ("advance", "outcome"),
    [
        pytest.param({"outcome": "  found it  "}, "found it", id="an outcome object"),
        pytest.param("found it", "found it", id="a bare outcome string"),
        pytest.param({}, "last action: grep port app.cfg ✓", id="empty: filled from evidence"),
        pytest.param(True, "last action: grep port app.cfg ✓", id="true: filled from evidence"),
    ],
)
async def test_the_outcome_is_what_was_sent_or_the_last_evidence(
    advance: Any, outcome: str
) -> None:
    network = await _laid_out(THREE)
    network.attach_evidence(network.leaves()[0], "grep port app.cfg ✓")

    await _send(network, {"advance": advance})

    assert network.leaves()[0].status == "done" and network.leaves()[0].outcome == outcome


# ── what the full form guarantees still holds ─────────────────────────────────────────


async def test_a_null_close_through_advance_is_still_challenged() -> None:
    # ADR-0116: a step with no done-condition, closed on a null result, is handed back once —
    # however the close was sent.
    network = await _laid_out(
        [{"title": "find the caller", "status": "in_progress"}, {"title": "fix it"}]
    )
    network.attach_evidence(network.leaves()[0], f"grep -rn caller src {NULL_MARK}")

    res = await _send(network, {"advance": {"outcome": "no caller"}})

    assert _statuses(network) == {"find the caller": "in_progress", "fix it": "pending"}
    assert network.leaves()[0].challenged and network.log[-1].kind == "challenged"
    assert res.output.startswith("Plan updated:")  # no "Step 1 done": it is not done
    assert "REOPENED" in res.output and res.hint == _CHALLENGED_HINT

    # The second close is accepted as deliberate.
    res = await _send(network, {"advance": {"outcome": "no caller, positive control passed"}})
    assert network.leaves()[0].status == "done" and res.output.startswith("Step 1 done")


async def test_a_resend_of_the_plan_the_advance_left_reads_unchanged() -> None:
    # The advance is recorded as what the model sent (ADR-0235), so sending back the plan it now
    # sees is a resend, and the unchanged rail (ADR-0168) answers it as before.
    network = await _laid_out(THREE)
    await _send(network, {"advance": {"outcome": "port is 8080"}})

    res = await _send(
        network,
        {
            "tasks": [
                {"title": "read the config", "status": "done"},
                {"title": "change the port", "status": "in_progress"},
                {"title": "restart and check"},
            ]
        },
    )

    assert _data(res).get("unchanged") is True and res.output.startswith("Plan unchanged")
    assert res.hint == _UNCHANGED_HINT.format(id="2")


async def test_tasks_win_over_advance_and_the_receipt_says_so() -> None:
    network = await _laid_out(THREE)

    res = await _send(
        network,
        {"tasks": [dict(THREE[0], status="done"), *THREE[1:]], "advance": {"outcome": "x"}},
    )

    assert _statuses(network)["read the config"] == "done"
    assert network.leaves()[0].outcome != "x"
    assert _data(res)["short_form"] == ""
    assert "'advance' was ignored: this call also sent 'tasks'" in res.output


# ── refusals: nothing to advance ──────────────────────────────────────────────────────


async def test_advance_refuses_when_there_is_no_plan_of_the_models() -> None:
    empty = TaskNetwork()
    res = await _send(empty, {"advance": {"outcome": "x"}})
    assert (
        res.is_error
        and "no plan of yours to advance" in res.output
        and "'tasks'" in (res.fix or "")
    )

    # The harness's request anchor (ADR-0111) is a record, not a plan: the model lays its own
    # out first, as the plan-first gate asks.
    anchored = TaskNetwork(tasks=[Task(title="the request", status="in_progress", anchor=True)])
    anchored.normalize()
    res = await _send(anchored, {"advance": {}})
    assert res.is_error and "no plan of yours to advance" in res.output
    assert anchored.leaves()[0].status == "in_progress"


async def test_advance_refuses_when_no_step_is_open() -> None:
    network = await _laid_out([{"title": "a", "status": "done"}])
    res = await _send(network, {"advance": {}})
    assert res.is_error and "every step is already finished" in res.output

    blocked = await _laid_out([{"title": "a", "status": "blocked", "note": "waiting on x"}])
    res = await _send(blocked, {"advance": {}})
    assert res.is_error and "every open step is blocked" in res.output
    assert blocked.leaves()[0].status == "blocked"


@pytest.mark.parametrize("args", [{}, {"advance": None}, {"advance": False}], ids=str)
async def test_a_call_with_neither_form_names_both(args: dict[str, Any]) -> None:
    network = await _laid_out(THREE)

    res = await _send(network, args)

    assert res.is_error and "'advance'" in res.output and '"advance"' in (res.fix or "")
    assert _statuses(network)["read the config"] == "in_progress"


# ── where the model reads about it ────────────────────────────────────────────────────


def test_the_schema_offers_advance_and_requires_neither_form() -> None:
    params = UpdatePlanTool.spec.parameters
    assert set(params["properties"]) == {"tasks", "advance"}
    assert params["properties"]["advance"]["properties"]["outcome"]["type"] == "string"
    # A backend that builds its tool-call grammar from the schema must not force 'tasks' out.
    assert "required" not in params


def test_an_advance_is_not_a_plan_of_the_models_for_the_plan_first_gate() -> None:
    # ADR-0231: the gate asks whether the call leaves a plan of the model's on the board; an
    # advance only moves one that is already there.
    assert not authors_a_plan({"advance": {"outcome": "x"}})
    assert authors_a_plan({"tasks": [{"title": "a"}], "advance": {}})


def test_the_short_form_is_said_where_the_advance_is(tmp_path: Path) -> None:
    assert PLAN_SHORT_FORM in UpdatePlanTool.spec.description
    assert PLAN_SHORT_FORM in SystemPromptBuilder().build(load_settings(workspace_root=tmp_path))
    assert PLAN_ADVANCE_CALL in PLAN_SHORT_FORM
    for hint in (_UNCHANGED_HINT, _ADVANCED_HINT):
        assert "'advance'" in hint.format(id="2")

    # ADR-0388 amendment: the receipt and the plan line lead with the short form as the call
    # itself, before the whole-plan form, and the plan line names the step it would close.
    assert _PLANNED_HINT.index(PLAN_ADVANCE_CALL) < _PLANNED_HINT.index("whole plan")
    loop = AgentLoop(_Scripted([_done()]), ToolRegistry(), Session(cwd=str(tmp_path), model="t"))
    loop.session.task_network.tasks = [Task(title="a", status="in_progress"), Task(title="b")]
    loop.session.task_network.normalize()
    current = loop.session.task_network.current()
    assert current is not None
    reminder = loop._plan_reminder()
    assert reminder is not None
    lead = f"When step {current.id} is done, call update_plan with {PLAN_ADVANCE_CALL}"
    assert lead in reminder.text
    assert reminder.text.index(lead) < reminder.text.index("whole plan")


async def test_the_call_the_model_is_shown_is_one_the_tool_takes() -> None:
    # The literal in every hint is a working call: sent as written, it closes the current step.
    network = await _laid_out(THREE)

    res = await _send(network, json.loads(PLAN_ADVANCE_CALL))

    assert not res.is_error, res.output
    assert _data(res)["short_form"] == "1"
    assert _statuses(network)["change the port"] == "in_progress"


# ── in the loop: the advance rides along like the full form ───────────────────────────


class _Scripted(Provider):
    def __init__(self, results: list[LLMResult]) -> None:
        self._results = results
        self.calls = 0

    async def acomplete(self, messages, *, system=None, tools=None, **kw) -> LLMResult:
        i = min(self.calls, len(self._results) - 1)
        self.calls += 1
        return self._results[i]

    def count_tokens(self, messages, *, system=None) -> int:
        return 0

    def capabilities(self) -> Capabilities:
        return Capabilities(context_window=8192)


class _Write(Tool):
    spec = ToolSpec(
        name="write_file",
        description="fake write",
        required_permission=PermissionTier.WORKSPACE_WRITE,
        concurrency=ConcurrencyClass.NEVER_PARALLEL,
    )

    async def execute(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        return ToolResult.ok("written")


def _batch(*calls: ToolCall) -> LLMResult:
    return LLMResult(text="", tool_calls=list(calls), usage=Usage(total_tokens=1))


def _done() -> LLMResult:
    return LLMResult(text="stopping here", usage=Usage(total_tokens=1))


def _judge_ok() -> LLMResult:
    # The decomposition judge (ADR-0050) reads one scorecard when the plan is first laid out.
    return LLMResult(
        text='{"scores": {"coverage": 0.9, "granularity": 0.9, "ordering": 0.9, "soundness": 0.9}}',
        usage=Usage(total_tokens=2),
    )


async def test_an_advance_first_in_a_batch_credits_the_work_to_the_step_it_starts(
    tmp_path: Path,
) -> None:
    # ADR-0237's pairing, short form: the plan call runs first, so the work in the same
    # response is evidence for the step the advance started.
    plan = ToolCall(
        id="p",
        name="update_plan",
        arguments={"tasks": [{"title": "a", "status": "in_progress"}, {"title": "b"}]},
    )
    advance = ToolCall(id="v", name="update_plan", arguments={"advance": {"outcome": "a is done"}})
    write = ToolCall(id="w", name="write_file", arguments={"path": "b.txt"})
    registry = ToolRegistry()
    registry.register(_Write())
    registry.register(UpdatePlanTool())
    session = Session(cwd=str(tmp_path), model="t")
    loop = AgentLoop(
        _Scripted([_batch(plan), _judge_ok(), _batch(advance, write), _done()]),
        registry,
        session,
        max_iterations=8,
    )

    await loop.arun_turn("do a then b")

    steps = {step.title: step for step in session.task_network.leaves()}
    assert steps["a"].status == "done" and steps["a"].outcome == "a is done"
    assert steps["b"].status == "in_progress"
    assert any("write_file" in line for line in steps["b"].evidence)
    assert not any("write_file" in line for line in steps["a"].evidence)
