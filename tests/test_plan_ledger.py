"""ADR-0110 — the plan as the goal's RECORD, and deep work that does not start without one.

Hermetic. The unit half drives :class:`TaskNetwork` directly: memory carried across the
model's full-replace, outcomes filled from evidence, closures read in chronological order,
the bounded history, harness seeding. The loop half uses a scripted provider (no network):
a deep turn that never plans gets the request anchored as its plan and closed at the
conclusion; tool calls attach as evidence to the step that was current; the re-injected
plan and the compaction note carry the request and the plan's short-term memory.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from zakcode.agent.loop import AgentLoop
from zakcode.config import PermissionTier
from zakcode.providers.base import Capabilities, LLMResult, Provider, ToolCall
from zakcode.session.store import Session
from zakcode.tasks import (
    MAX_EVIDENCE_PER_STEP,
    MAX_LOG_EVENTS,
    MAX_REQUEST_CHARS,
    Task,
    TaskNetwork,
    clip,
)
from zakcode.tools.base import (
    ConcurrencyClass,
    Tool,
    ToolContext,
    ToolRegistry,
    ToolResult,
    ToolSpec,
)
from zakcode.tools.builtins.update_plan import UpdatePlanTool
from zakcode.usage import Usage

# ── unit: the network's memory ────────────────────────────────────────────────


def _author(network: TaskNetwork, *specs: tuple[str, str] | tuple[str, str, str]) -> None:
    """Model-style full replace: ``(title, status[, outcome])`` per leaf."""
    tasks = []
    for spec in specs:
        title, status = spec[0], spec[1]
        outcome = spec[2] if len(spec) == 3 else ""
        tasks.append(Task(title=title, status=status, outcome=outcome))  # type: ignore[arg-type]
    network.replace_from_author(tasks)


def test_full_replace_carries_evidence_and_outcome_by_title_and_logs_transitions() -> None:
    net = TaskNetwork()
    _author(net, ("read the config", "in_progress"), ("fix it", "pending"))
    net.attach_evidence(net.tasks[0], "read_file app/config.py ✓")
    net.attach_evidence(net.tasks[0], "grep TIMEOUT ✓")

    # The model resends the plan (new objects, ids may shift) — the record must survive.
    _author(net, ("read the config", "done", "timeout is 30s"), ("fix it", "in_progress"))
    first, second = net.tasks
    assert first.evidence == ["read_file app/config.py ✓", "grep TIMEOUT ✓"]
    assert first.outcome == "timeout is 30s"
    assert first.status == "done" and second.status == "in_progress"

    kinds = [(e.kind, e.detail) for e in net.log]
    assert ("authored", "2 step(s): read the config; fix it") in kinds
    closes = [d for k, d in kinds if k == "step" and d.startswith("in_progress -> done")]
    assert closes == ["in_progress -> done — timeout is 30s"]
    assert any(k == "step" and d == "pending -> in_progress" for k, d in kinds)
    # Every event is stamped and sequenced.
    assert [e.seq for e in net.log] == list(range(1, len(net.log) + 1))
    assert all(e.at for e in net.log)


def test_a_step_closed_without_an_outcome_takes_its_last_evidence_line() -> None:
    net = TaskNetwork()
    _author(net, ("probe", "in_progress"))
    net.attach_evidence(net.tasks[0], "bash pytest -q ✗")
    net.attach_evidence(net.tasks[0], "bash pytest -q tests/x.py ✓")
    _author(net, ("probe", "done"))
    assert net.tasks[0].outcome == "last action: bash pytest -q tests/x.py ✓"
    # ...and the render shows what it PRODUCED, not its (empty) done-condition.
    assert "[x] 1 probe — last action: bash pytest -q tests/x.py ✓" in net.render()


def test_render_shows_outcome_on_closed_steps_and_the_note_on_open_ones() -> None:
    net = TaskNetwork()
    net.replace_from_author(
        [
            Task(title="a", status="done", note="tests pass", outcome="3 tests added"),
            Task(title="b", status="in_progress", note="GET /health returns 200"),
        ]
    )
    rendered = net.render()
    assert "[x] 1 a — 3 tests added" in rendered
    assert "[~] 2 b — GET /health returns 200" in rendered


def test_recent_closed_follows_the_log_not_document_order() -> None:
    net = TaskNetwork()
    _author(net, ("a", "pending"), ("b", "in_progress"), ("c", "pending"))
    _author(net, ("a", "pending"), ("b", "done", "b happened"), ("c", "in_progress"))
    _author(net, ("a", "done", "a happened"), ("b", "done", "b happened"), ("c", "in_progress"))
    # b closed before a, so b is OLDER even though a comes first in the document.
    assert [t.title for t in net.recent_closed(3)] == ["a", "b"]
    last = net.last_closed()
    assert last is not None and last.title == "a" and last.outcome == "a happened"


def test_evidence_and_log_are_bounded_and_the_log_counts_what_it_folded() -> None:
    net = TaskNetwork()
    _author(net, ("s", "in_progress"))
    step = net.tasks[0]
    for i in range(MAX_EVIDENCE_PER_STEP + 3):
        net.attach_evidence(step, f"call {i} ✓")
    assert len(step.evidence) == MAX_EVIDENCE_PER_STEP
    assert step.evidence[0] == "call 3 ✓"  # the oldest lines dropped

    before = len(net.log)
    for i in range(MAX_LOG_EVENTS + 5):
        net.record("step", step=step, detail=f"tick {i}")
    assert len(net.log) == MAX_LOG_EVENTS
    assert net.log_folded == before + 5
    assert net.log[-1].seq == before + MAX_LOG_EVENTS + 5  # seq keeps counting past the fold


def test_insert_before_stamps_harness_origin_and_logs_the_seed() -> None:
    net = TaskNetwork()
    net.insert_before(None, [Task(title="look around"), Task(title="decide")], reason="probe")
    assert all(t.origin == "harness" for t in net.tasks)
    assert net.log[-1].kind == "seeded" and net.log[-1].detail == "probe"
    # A later model replace keeps the provenance the model never saw.
    _author(net, ("look around", "done", "nothing odd"), ("decide", "in_progress"))
    assert all(t.origin == "harness" for t in net.tasks)


def test_model_clearing_the_plan_leaves_a_record() -> None:
    net = TaskNetwork()
    _author(net, ("a", "in_progress"), ("b", "pending"))
    net.record("cleared", detail="2 open step(s) dropped by the model: a; b")
    net.tasks = []
    net.normalize()
    assert net.is_empty()
    assert net.log[-1].kind == "cleared" and "a; b" in net.log[-1].detail


# ── loop: anchor, evidence, memory ────────────────────────────────────────────


class _Scripted(Provider):
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
        return Capabilities(context_window=8192)


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


class _FakeRead(Tool):
    spec = ToolSpec(
        name="fake_read",
        description="fake read",
        required_permission=PermissionTier.READ_ONLY,
        concurrency=ConcurrencyClass.READ_ONLY_SAFE,
    )

    async def execute(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        return ToolResult.ok("contents")


def _call(name: str, arguments: dict[str, Any], cid: str = "c1") -> LLMResult:
    return LLMResult(
        text="",
        tool_calls=[ToolCall(id=cid, name=name, arguments=arguments)],
        usage=Usage(total_tokens=1),
    )


def _plan(tasks: list[dict]) -> LLMResult:
    return _call("update_plan", {"tasks": tasks}, cid="p1")


def _judge_ok() -> LLMResult:
    return LLMResult(
        text=json.dumps(
            {"scores": {"coverage": 0.9, "granularity": 0.9, "ordering": 0.9, "soundness": 0.9}}
        ),
        usage=Usage(total_tokens=2),
    )


def _done(text: str) -> LLMResult:
    return LLMResult(text=text, tool_calls=[], usage=Usage(total_tokens=1))


def _registry(*tools: Tool) -> ToolRegistry:
    reg = ToolRegistry()
    for tool in tools:
        reg.register(tool)
    reg.register(UpdatePlanTool())
    return reg


_DEEP_REQUEST = (
    "Refactor the configuration loader so every setting is read once at startup, validated "
    "against the schema, and exposed through a typed accessor; update the three call sites in "
    "the API layer, the worker, and the CLI to use the accessor; add tests for the invalid "
    "cases (missing key, wrong type, out-of-range) and make sure the existing suite stays "
    "green; then write a short migration note for operators explaining which environment "
    "variables changed names and what the defaults are now. Keep the public module path "
    "stable so downstream imports do not break, and leave the legacy shim in place for one "
    "release with a deprecation warning that names the replacement."
)


@pytest.mark.asyncio
async def test_deep_turn_that_never_plans_gets_the_request_anchored_and_closed() -> None:
    assert len(_DEEP_REQUEST) > 600  # past the quick/deep length heuristic
    write = _FakeWrite()
    provider = _Scripted(
        [
            _call("write_file", {"path": "a.txt", "content": "x"}),
            _call("write_file", {"path": "a.txt", "content": "x"}),
            _call("write_file", {"path": "a.txt", "content": "x"}),
            _done("Result: the loader now validates every setting at startup."),
        ]
    )
    session = Session(cwd="/tmp", model="t/m")
    loop = AgentLoop(provider, _registry(write), session, max_iterations=20)
    result = await loop.arun_turn(_DEEP_REQUEST)

    # Two batches withheld for a plan; the third ran against the harness's request anchor.
    assert write.runs == 1
    assert result.stop_reason == "completed"
    net = session.task_network
    assert len(net.tasks) == 1
    anchor = net.tasks[0]
    assert anchor.anchor and anchor.origin == "harness"
    assert anchor.title.startswith("Refactor the configuration loader")
    assert anchor.evidence == ["write_file a.txt ✓"]
    # Closed at the conclusion, with the conclusion's first line as its outcome.
    assert anchor.status == "done"
    assert anchor.outcome == "Result: the loader now validates every setting at startup."
    # The request is anchored verbatim up to the bound (an ellipsis marks the cut).
    assert net.context.request == clip(_DEEP_REQUEST, MAX_REQUEST_CHARS)
    assert net.context.request.endswith("…") and len(net.context.request) == MAX_REQUEST_CHARS
    kinds = [e.kind for e in net.log]
    assert "seeded" in kinds and kinds[-1] == "step"
    assert "closed by the harness" in net.log[-1].detail
    # The plan the anchor made was visible to the model on the call after the write.
    reminder = [m.text for m in provider.seen[3] if "[plan]" in m.text]
    assert reminder and "Goal: Refactor the configuration loader" in reminder[-1]
    assert "Step 1 so far: write_file a.txt ✓" in reminder[-1]


@pytest.mark.asyncio
async def test_quick_turn_is_never_gated_and_never_anchored() -> None:
    write = _FakeWrite()
    provider = _Scripted([_call("write_file", {"path": "a.txt", "content": "x"}), _done("Done.")])
    session = Session(cwd="/tmp", model="t/m")
    loop = AgentLoop(provider, _registry(write), session, max_iterations=20)
    await loop.arun_turn("add a newline at the end of a.txt")
    assert write.runs == 1
    assert session.task_network.is_empty()
    assert provider.calls == 2  # write, then the answer — no nudge in between


@pytest.mark.asyncio
async def test_model_authored_open_steps_are_never_closed_by_the_anchor_rule() -> None:
    # The anchor rule closes ONLY a lone harness anchor. A model-authored open step still
    # goes through the plan gate (nudged, then completed degraded) — never silently closed.
    provider = _Scripted(
        [
            _plan([{"title": "A", "status": "done"}, {"title": "B", "status": "pending"}]),
            _judge_ok(),
        ]
        + [_done("all done")] * 6
    )
    session = Session(cwd="/tmp", model="t/m")
    loop = AgentLoop(provider, _registry(), session, max_iterations=20)
    result = await loop.arun_turn("two steps")
    assert result.stop_reason == "completed"
    assert session.task_network.tasks[1].status == "pending"
    assert result.degraded  # the plan was left unresolved, and the loop says so


@pytest.mark.asyncio
async def test_tool_calls_attach_to_the_current_step_and_the_reminder_carries_memory() -> None:
    provider = _Scripted(
        [
            _plan([{"title": "read it", "status": "in_progress"}, {"title": "finish"}]),
            _judge_ok(),
            _call("fake_read", {"path": "notes.md"}, cid="r1"),
            _plan(
                [
                    {"title": "read it", "status": "done"},
                    {"title": "finish", "status": "done", "outcome": "nothing to change"},
                ]
            ),
            _done("Verdict: the notes are consistent; nothing to change."),
        ]
    )
    session = Session(cwd="/tmp", model="t/m")
    loop = AgentLoop(provider, _registry(_FakeRead()), session, max_iterations=20)
    result = await loop.arun_turn("check the notes and tell me if they are consistent")
    assert result.stop_reason == "completed"

    first, second = session.task_network.tasks
    # The read ran while "read it" was current: it is that step's evidence, and — the model
    # having closed the step without an outcome — its outcome too.
    assert first.evidence == ["fake_read notes.md ✓"]
    assert first.outcome == "last action: fake_read notes.md ✓"
    assert second.outcome == "nothing to change"
    # update_plan itself is bookkeeping, never evidence.
    assert not any("update_plan" in line for step in (first, second) for line in step.evidence)

    # The plan re-injected on the call after the read carried the request and the memory.
    reminder = [m.text for m in provider.seen[3] if "[plan]" in m.text][-1]
    assert "Goal: check the notes and tell me if they are consistent" in reminder
    assert "Step 1 so far: fake_read notes.md ✓" in reminder
    # Once every step closed, the one-line completion message quotes the request (ADR-0108).
    final = [m.text for m in provider.seen[4] if "[plan]" in m.text][-1]
    assert "Plan complete" in final and "The original request was:" in final

    # The compaction position note carries the same facts across a context fold.
    note = loop._compaction_position_note()
    assert "- request: check the notes and tell me if they are consistent" in note
    assert "- closed: 2 finish — nothing to change" in note
    assert "- closed: 1 read it — last action: fake_read notes.md ✓" in note

    # And the plan's history says what happened, in order.
    kinds = [(e.kind, e.detail) for e in session.task_network.log]
    assert kinds[0][0] == "authored"
    assert ("step", "in_progress -> done — last action: fake_read notes.md ✓") in kinds
    assert ("step", "pending -> done — nothing to change") in kinds


@pytest.mark.asyncio
async def test_a_finished_plan_is_reset_at_the_next_turn_but_its_record_stays() -> None:
    provider = _Scripted(
        [
            _plan([{"title": "only step", "status": "done", "outcome": "it is done"}]),
            _judge_ok(),
            _done("Verdict: done."),
            _done("Second turn answer."),
        ]
    )
    session = Session(cwd="/tmp", model="t/m")
    loop = AgentLoop(provider, _registry(), session, max_iterations=20)
    await loop.arun_turn("first request")
    assert session.task_network.is_complete()
    await loop.arun_turn("second request")
    net = session.task_network
    assert net.is_empty()  # the finished plan left the board at the turn start
    resets = [e for e in net.log if e.kind == "reset"]
    assert resets and resets[-1].detail.startswith("completed 1/1: only step")
    # The next plan starts with the NEW request as its context.
    assert net.context.request == "second request"
