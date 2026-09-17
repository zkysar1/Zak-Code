"""Plan hygiene in working memory (ADR-0184): a long turn stops paying for every finished
step on every iteration, and the fold is round-trip-safe under update_plan's full-replace.

User directive 2026-09-01: "completed items stay around too long — remove them from working
memory more often". Measured then: a 20-step plan at step 18 re-injected 17 done rows (title,
outcome, deps) on every iteration for the life of the turn, and a plan that completed mid-turn
kept re-injecting its checklist. The fold lives in the render; the network is never touched.
"""

from __future__ import annotations

import asyncio
import re
from pathlib import Path
from typing import Any

from zakcode.agent.loop import AgentLoop
from zakcode.messages import Message
from zakcode.providers.base import Capabilities, LLMResult, Provider
from zakcode.session.store import Session
from zakcode.tasks import COLLAPSED_ROW_RE, Task, TaskNetwork
from zakcode.tools.base import ToolContext, ToolRegistry
from zakcode.tools.builtins.update_plan import UpdatePlanTool


def _net(*tasks: Task) -> TaskNetwork:
    net = TaskNetwork()
    net.tasks = list(tasks)
    net.normalize()
    return net


def _plan(done: int = 17, total: int = 20) -> TaskNetwork:
    """A representative long plan: ``done`` closed steps with outcomes, one in progress with
    evidence, the rest pending with done-conditions."""
    steps: list[Task] = []
    for i in range(1, total + 1):
        title = f"step {i}: edit module {i}"
        note = f"done when tests for module {i} pass"
        if i <= done:
            steps.append(
                Task(
                    title=title,
                    status="done",
                    note=note,
                    outcome=f"module {i} rewritten and its tests pass",
                )
            )
        elif i == done + 1:
            steps.append(Task(title=title, status="in_progress", note=note))
        else:
            steps.append(Task(title=title, note=note))
    net = _net(*steps)
    net.tasks[2].evidence = ["fake_edit module_3.py ✓", "bash pytest -q tests/test_3.py ✓"]
    if done < total:
        net.tasks[done].evidence = ["fake_read module_18.py ✓"]
    return net


_ROW_RE = re.compile(r"^\s*\[(?P<glyph>[ x~!\-])\] (?P<body>.*?)(?:  <- current)?$")
_STATUS = {"x": "done", "~": "in_progress", " ": "pending", "!": "blocked", "-": "cancelled"}


def _echo(rendered: str, *, keep_folds: bool = True) -> list[dict[str, Any]]:
    """What a model does with the checklist: resend every row as a step — the title being the
    row's text without the id, the detail and the marker — except a folded row, which has no
    title but its ids, so it goes back verbatim (or, with ``keep_folds=False``, not at all)."""
    steps: list[dict[str, Any]] = []
    for line in rendered.splitlines()[1:]:
        m = _ROW_RE.match(line)
        assert m is not None, line
        body = m["body"]
        if COLLAPSED_ROW_RE.match(body):
            if keep_folds:
                steps.append({"title": body, "status": "done"})
            continue
        title = body.split(" — ")[0].split(" (after ")[0].split(" ", 1)[1]
        steps.append({"title": title, "status": _STATUS[m["glyph"]]})
    return steps


# ── the fold ──────────────────────────────────────────────────────────────────────────────


def test_a_run_of_closed_steps_folds_into_one_row_carrying_its_ids() -> None:
    net = _plan(17, 20)
    full = net.render()
    folded = net.render(elide_done=True)
    assert folded.startswith("Current plan (17/20 steps done):")
    assert "  [x] 1–17 (17 steps done)" in folded
    assert "step 3: edit module 3" not in folded
    assert "module 3 rewritten" not in folded
    assert (
        "  [~] 18 step 18: edit module 18 — done when tests for module 18 pass  <- current"
        in folded
    )
    assert "  [ ] 19 step 19" in folded
    assert "  [ ] 20 step 20" in folded
    assert len(folded.splitlines()) == 5
    # The default is the whole record: every consumer but the reminder keeps it.
    assert "[x] 3 step 3: edit module 3 — module 3 rewritten and its tests pass" in full
    assert len(full.splitlines()) == 21


def test_the_fold_shrinks_a_twenty_step_plan_at_step_eighteen_by_more_than_two_thirds() -> None:
    net = _plan(17, 20)
    assert len(net.render(elide_done=True)) < len(net.render()) / 3


def test_a_lone_closed_step_keeps_what_it_produced() -> None:
    net = _net(
        Task(title="probe", status="done", outcome="the path exists"),
        Task(title="fix", status="in_progress"),
        Task(title="verify", status="done", outcome="green"),
        Task(title="report"),
    )
    folded = net.render(elide_done=True)
    assert "[x] 1 probe — the path exists" in folded
    assert "[x] 3 verify — green" in folded
    assert "–" not in folded


def test_open_blocked_and_in_progress_steps_are_never_folded() -> None:
    net = _net(
        Task(title="a", status="done"),
        Task(title="b", status="done"),
        Task(title="c", status="blocked"),
        Task(title="d", status="done"),
        Task(title="e", status="cancelled"),
        Task(title="f", status="in_progress"),
        Task(title="g"),
    )
    assert net.render(elide_done=True).splitlines()[1:] == [
        "  [x] 1–2 (2 steps done)",
        "  [!] 3 c",
        "  [x] 4–5 (1 done, 1 cancelled)",
        "  [~] 6 f  <- current",
        "  [ ] 7 g",
    ]


def test_closed_compounds_fold_to_one_row_and_an_open_one_folds_only_its_closed_run() -> None:
    net = _net(
        Task(title="set up", status="done", outcome="venv ready"),
        Task(
            title="build",
            kind="compound",
            children=[
                Task(title="route", status="done"),
                Task(title="handler", status="done"),
                Task(title="wire", status="done"),
            ],
        ),
        Task(
            title="test",
            kind="compound",
            children=[
                Task(title="unit", status="done"),
                Task(title="integration", status="done"),
                Task(title="smoke", status="in_progress"),
            ],
        ),
        Task(title="ship"),
    )
    assert net.render(elide_done=True).splitlines()[1:] == [
        "  [x] 1–2 (4 steps done)",
        "  [~] 3 test",
        "    [x] 3.1–3.2 (2 steps done)",
        "    [~] 3.3 smoke  <- current",
        "  [ ] 4 ship",
    ]


def test_a_lone_closed_compound_is_one_row_with_its_subtree_counted() -> None:
    net = _net(
        Task(title="design", status="in_progress"),
        Task(
            title="build",
            kind="compound",
            children=[Task(title="a", status="done"), Task(title="b", status="cancelled")],
        ),
        Task(title="ship"),
    )
    assert net.render(elide_done=True).splitlines()[2] == "  [x] 2 build (1 done, 1 cancelled)"


# ── the reminder and the UI event ─────────────────────────────────────────────────────────


class _Provider(Provider):
    async def acomplete(
        self, messages: list[Message], *, system: str | None = None, tools: Any = None, **kw: Any
    ) -> LLMResult:
        return LLMResult(text="")

    def count_tokens(self, messages: list[Message], *, system: str | None = None) -> int:
        return 100

    def capabilities(self) -> Capabilities:
        return Capabilities(supports_tools=True, context_window=8192)


def _loop(tmp_path: Path, plan: TaskNetwork) -> AgentLoop:
    loop = AgentLoop(
        _Provider(),
        ToolRegistry(),
        Session(cwd=str(tmp_path), model="test"),
        workspace_root=tmp_path,
    )
    network = loop.session.task_network
    network.tasks = plan.tasks
    network.normalize()
    return loop


def test_the_reminder_carries_the_folded_plan_and_the_round_trip_contract(tmp_path: Path) -> None:
    loop = _loop(tmp_path, _plan(17, 20))
    reminder = loop._plan_reminder()
    assert reminder is not None
    assert "[x] 1–17 (17 steps done)" in reminder.text
    assert "module 3 rewritten" not in reminder.text
    assert "Closed steps are folded into rows like" in reminder.text
    # The UI event stays the whole record — the structured tree and the checklist alike.
    event = loop._task_update_event()
    assert event is not None
    assert "[x] 3 step 3: edit module 3" in event.plan
    assert len(event.tasks) == 20


def test_the_contract_sentence_is_absent_when_nothing_is_folded(tmp_path: Path) -> None:
    loop = _loop(tmp_path, _plan(1, 3))
    reminder = loop._plan_reminder()
    assert reminder is not None
    assert "Closed steps are folded" not in reminder.text


def test_a_plan_completed_mid_turn_is_one_line_and_the_network_stays_intact(
    tmp_path: Path,
) -> None:
    loop = _loop(tmp_path, _plan(20, 20))
    reminder = loop._plan_reminder()
    assert reminder is not None
    assert reminder.text.startswith("[plan] Plan complete (20/20 steps done)")
    assert len(reminder.text.splitlines()) == 1
    assert len(loop.session.task_network.tasks) == 20


# ── the round trip through update_plan (full-replace) ─────────────────────────────────────


def _run(net: TaskNetwork, steps: list[dict[str, Any]]) -> Any:
    ctx = ToolContext(workspace_root=Path("/tmp"), task_network=net)
    return asyncio.run(UpdatePlanTool().execute({"tasks": steps}, ctx))


def test_an_echoed_folded_row_expands_to_the_steps_it_stands_for() -> None:
    net = _plan(17, 20)
    echoed = _echo(net.render(elide_done=True))
    assert echoed[0] == {"title": "1–17 (17 steps done)", "status": "done"}
    assert len(echoed) == 4
    result = _run(net, echoed)
    assert result.output.startswith("Plan updated: 17/20 steps done")
    assert net.progress() == (17, 20)
    assert [t.id for t in net.tasks] == [str(i) for i in range(1, 21)]
    assert net.tasks[2].title == "step 3: edit module 3"
    assert net.tasks[2].evidence == ["fake_edit module_3.py ✓", "bash pytest -q tests/test_3.py ✓"]
    assert net.tasks[2].outcome == "module 3 rewritten and its tests pass"
    assert net.tasks[17].status == "in_progress"
    assert "dropped open step" not in result.output
    assert "restored" not in result.output


def test_a_folded_row_echoed_with_its_glyph_and_a_hyphen_still_expands() -> None:
    net = _plan(5, 7)
    steps = [{"title": "[x] 1-5 (5 steps done)", "status": "done"}]
    steps += _echo(net.render(elide_done=True))[1:]
    _run(net, steps)
    assert net.progress() == (5, 7)
    assert net.tasks[4].title == "step 5: edit module 5"


def test_closed_steps_left_out_of_a_resend_are_restored_from_the_record() -> None:
    net = _plan(17, 20)
    echoed = _echo(net.render(elide_done=True), keep_folds=False)
    assert len(echoed) == 3
    result = _run(net, echoed)
    assert net.progress() == (17, 20)
    assert [t.title for t in net.tasks[:3]] == [
        "step 1: edit module 1",
        "step 2: edit module 2",
        "step 3: edit module 3",
    ]
    assert net.tasks[17].status == "in_progress"
    assert net.tasks[17].title == "step 18: edit module 18"
    assert "restored 17 done step(s)" in result.output
    assert "done work is history" in result.output


def test_a_visible_done_step_left_out_is_the_models_call_but_hidden_ones_come_back() -> None:
    # ADR-0113 stands: a done step shown in full (a lone closed row) and left out of a resend
    # is a decision — no restore, no event. Only what the fold HID comes back.
    net = _net(
        Task(title="a", status="done"),
        Task(title="b", status="done"),
        Task(title="c", status="in_progress"),
        Task(title="d", status="done", outcome="shown in full"),
        Task(title="e"),
    )
    assert net.render(elide_done=True).splitlines()[1:] == [
        "  [x] 1–2 (2 steps done)",
        "  [~] 3 c  <- current",
        "  [x] 4 d — shown in full",
        "  [ ] 5 e",
    ]
    result = _run(net, [{"title": "c", "status": "in_progress"}, {"title": "e"}])
    assert [t.title for t in net.tasks] == ["a", "b", "c", "e"]
    assert "restored 2 done step(s)" in result.output
    assert not any(e.kind == "dropped" for e in net.log)  # 'd' left out is not an event


def test_a_cancelled_step_a_fold_hid_is_not_restored() -> None:
    # A cancelled section the model then drops stays dropped (the paging contract): cancelling
    # is a decision about the title, and dropping it is consistent with that decision.
    net = _net(
        Task(title="a", status="done"),
        Task(title="b", status="cancelled"),
        Task(title="c", status="in_progress"),
    )
    _run(net, [{"title": "c", "status": "in_progress"}])
    assert [t.title for t in net.tasks] == ["a", "c"]


def test_a_folded_closed_compound_echoed_as_a_leaf_expands_to_its_subtree() -> None:
    net = _net(
        Task(title="design", status="in_progress"),
        Task(
            title="build",
            kind="compound",
            children=[
                Task(title="route", status="done", outcome="GET /x"),
                Task(title="handler", status="done"),
            ],
        ),
        Task(title="ship"),
    )
    echoed = _echo(net.render(elide_done=True))
    assert echoed[1] == {"title": "build (2 steps done)", "status": "done"}
    _run(net, echoed)
    assert [t.title for t in net.tasks] == ["design", "build", "ship"]
    assert [c.title for c in net.tasks[1].children] == ["route", "handler"]
    assert net.tasks[1].children[0].outcome == "GET /x"
    assert net.progress() == (2, 4)


def test_a_dropped_request_anchor_is_not_restored() -> None:
    # ADR-0111: the model's plan supersedes the harness's request anchor; the restore must not
    # resurrect it.
    net = _net(
        Task(title="the request itself", status="done", anchor=True),
        Task(title="real step", status="in_progress"),
    )
    _run(net, [{"title": "real step", "status": "in_progress"}, {"title": "next"}])
    assert [t.title for t in net.tasks] == ["real step", "next"]


def test_a_stale_fold_whose_ids_no_longer_resolve_is_dropped_not_installed() -> None:
    net = _net(
        Task(title="a", status="done"),
        Task(title="b", status="done"),
        Task(title="c", status="in_progress"),
    )
    _run(
        net,
        [
            {"title": "4–9 (6 steps done)", "status": "done"},
            {"title": "c", "status": "in_progress"},
        ],
    )
    # The artefact is gone; the two done steps the real fold hid come back from the record.
    assert [t.title for t in net.tasks] == ["a", "b", "c"]
    assert net.progress() == (2, 3)


def test_a_same_titled_child_under_its_closed_parent_never_expands_into_it() -> None:
    # The measured parent-over-same-named-child shape (a 35B nesting "Create utils/duration.py"
    # over a same-named leaf): the compound expansion keys on the parent too, so the child leaf
    # resent under its parent stays a leaf.
    net = _net(
        Task(
            title="Create utils/duration.py",
            kind="compound",
            children=[Task(title="Create utils/duration.py", status="done")],
        ),
        Task(title="Export parse_duration", status="in_progress"),
    )
    _run(
        net,
        [
            {
                "title": "Create utils/duration.py",
                "subtasks": [{"title": "Create utils/duration.py", "status": "done"}],
            },
            {"title": "Export parse_duration", "status": "in_progress"},
        ],
    )
    assert [t.title for t in net.tasks] == ["Create utils/duration.py", "Export parse_duration"]
    child = net.tasks[0].children
    assert len(child) == 1 and not child[0].children
    assert net.progress() == (1, 2)
