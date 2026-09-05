"""ADR-0111 — the plan's record is readable on demand, and a deep turn is anchored at its
first action.

Hermetic. The tool half drives :class:`PlanRecallTool` against a hand-built network: the
overview, one step's record, a search, a bad step id, no network. The loop half uses a
scripted provider: a deep turn whose first batch is read-only gets the request anchored
BEFORE the read runs (so the read is the anchor's evidence); the model's own plan supersedes
the anchor and the history keeps what the anchor recorded; an anchor alone does not satisfy
the mutate gate; reading the record is never evidence.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from zakcode.agent.loop import AgentLoop
from zakcode.config import PermissionTier
from zakcode.messages import ToolResultBlock
from zakcode.providers.base import Capabilities, LLMResult, Provider, ToolCall
from zakcode.session.store import Session
from zakcode.tasks import PlanContext, Task, TaskNetwork
from zakcode.tools.base import (
    ConcurrencyClass,
    Tool,
    ToolContext,
    ToolRegistry,
    ToolResult,
    ToolSpec,
)
from zakcode.tools.builtins.default_registry import default_registry
from zakcode.tools.builtins.plan_recall import PlanRecallTool
from zakcode.tools.builtins.update_plan import UpdatePlanTool
from zakcode.usage import Usage

# ── the tool ──────────────────────────────────────────────────────────────────


def _network() -> TaskNetwork:
    net = TaskNetwork(context=PlanContext(request="make the loader validate settings"))
    net.replace_from_author(
        [
            Task(title="read the loader", status="in_progress", note="know the entry points"),
            Task(title="add validation", status="pending", note="invalid config raises"),
        ]
    )
    net.attach_evidence(net.tasks[0], "read_file app/config.py ✓")
    net.attach_evidence(net.tasks[0], "grep load_settings ✓")
    net.replace_from_author(
        [
            Task(title="read the loader", status="done", outcome="two entry points, one shared"),
            Task(title="add validation", status="in_progress", note="invalid config raises"),
        ]
    )
    net.attach_evidence(net.tasks[1], "edit_file app/config.py ✓")
    net.attach_evidence(net.tasks[1], "bash pytest -q ✗")
    return net


def _ctx(tmp_path: Path, network: TaskNetwork | None) -> ToolContext:
    return ToolContext(workspace_root=tmp_path, task_network=network)


@pytest.mark.asyncio
async def test_overview_lists_request_steps_outcomes_and_recent_history(tmp_path: Path) -> None:
    result = await PlanRecallTool().execute({}, _ctx(tmp_path, _network()))
    assert not result.is_error
    out = result.output
    assert "Request: make the loader validate settings" in out
    assert "[x] 1 read the loader — two entry points, one shared · 2 tool call(s)" in out
    assert "[~] 2 add validation — invalid config raises · 2 tool call(s)" in out
    assert "Files changed (1): app/config.py" in out  # the failed pytest is not a file
    assert "History (this plan: last" in out
    assert "authored" in out and "in_progress -> done — two entry points, one shared" in out
    assert result.data == {"steps": 2, "events": len(_network().log), "folded": 0}


@pytest.mark.asyncio
async def test_overview_history_is_scoped_to_the_current_plan_unless_asked(tmp_path: Path) -> None:
    # The log outlives plans: a turn-start reset keeps it. Without `last` the overview shows
    # this plan's own events; `last` widens to the whole session's record (ADR-0112).
    net = _network()
    net.record("reset", detail="completed 2/2: read the loader; add validation")
    net.tasks = []
    net.normalize()
    net.context = PlanContext(request="now document it")
    net.replace_from_author([Task(title="write the docs", status="in_progress")])
    scoped = (await PlanRecallTool().execute({}, _ctx(tmp_path, net))).output
    assert "History (this plan: last 1 of 1; " in scoped and "from earlier plans" in scoped
    assert "two entry points" not in scoped  # the earlier plan's closure is out of scope
    widened = (await PlanRecallTool().execute({"last": 60}, _ctx(tmp_path, net))).output
    assert "History (last" in widened and "two entry points, one shared" in widened


@pytest.mark.asyncio
async def test_step_returns_full_evidence_and_that_steps_history(tmp_path: Path) -> None:
    result = await PlanRecallTool().execute({"step": "1"}, _ctx(tmp_path, _network()))
    assert not result.is_error
    out = result.output
    assert "Step 1 [done] read the loader" in out
    assert "outcome: two entry points, one shared" in out
    assert "- read_file app/config.py ✓" in out and "- grep load_settings ✓" in out
    assert "step 1: in_progress -> done — two entry points, one shared" in out
    # Another step's history is not mixed in.
    assert "pending -> in_progress" not in out


@pytest.mark.asyncio
async def test_query_searches_titles_outcomes_evidence_and_history(tmp_path: Path) -> None:
    result = await PlanRecallTool().execute({"query": "pytest"}, _ctx(tmp_path, _network()))
    assert not result.is_error
    assert "1 match(es) for 'pytest'" in result.output
    assert "step 2 evidence: bash pytest -q ✗" in result.output

    result = await PlanRecallTool().execute({"query": "ENTRY points"}, _ctx(tmp_path, _network()))
    assert "step 1 outcome: two entry points, one shared" in result.output
    assert "history #" in result.output  # the close event carried the outcome too

    result = await PlanRecallTool().execute({"query": "kubernetes"}, _ctx(tmp_path, _network()))
    assert not result.is_error and "No match for 'kubernetes'" in result.output


@pytest.mark.asyncio
async def test_unknown_step_and_missing_network_are_recoverable_errors(tmp_path: Path) -> None:
    result = await PlanRecallTool().execute({"step": "9"}, _ctx(tmp_path, _network()))
    assert result.is_error and "no step '9'" in result.output and "steps: 1, 2" in result.output
    result = await PlanRecallTool().execute({}, _ctx(tmp_path, None))
    assert result.is_error and "no task network" in result.output


def test_plan_recall_is_registered_read_only_with_its_alias() -> None:
    registry = default_registry()
    tool = registry.get("plan_recall")
    assert tool is not None and tool.spec.required_permission == PermissionTier.READ_ONLY
    assert registry.get("plan_history") is tool
    # "recall" stays free: the persistence boundary reserves it for a Mind's own memory tool.
    assert registry.get("recall") is None


# ── the loop ──────────────────────────────────────────────────────────────────


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
    reg.register(PlanRecallTool())
    return reg


_DEEP_REQUEST = (
    "Audit the notification pipeline end to end: read the producer, the queue adapter and "
    "the three consumers, list every place a message can be dropped or duplicated, check "
    "whether the retry policy matches what the runbook promises, look at how the dead-letter "
    "queue is drained and by whom, and then write up what you found as a short report with "
    "one recommendation per finding, ordered by how likely the failure is in production and "
    "how bad it would be. Do not change any code; this is an analysis. Quote file paths and "
    "line numbers so the team can go straight to each spot, and flag anything you could not "
    "verify from the code alone so it can be checked against the dashboards."
)


@pytest.mark.asyncio
async def test_deep_read_only_turn_is_anchored_before_its_first_read_runs() -> None:
    assert len(_DEEP_REQUEST) > 600
    provider = _Scripted(
        [
            _call("fake_read", {"path": "producer.py"}, cid="r1"),
            _call("fake_read", {"path": "consumer.py"}, cid="r2"),
            _done("Finding: messages can be dropped in the consumer's ack path."),
        ]
    )
    session = Session(cwd="/tmp", model="t/m")
    loop = AgentLoop(provider, _registry(_FakeRead()), session, max_iterations=20)
    result = await loop.arun_turn(_DEEP_REQUEST)
    assert result.stop_reason == "completed"
    assert provider.calls == 3  # nothing was withheld: read, read, answer

    net = session.task_network
    anchor = net.tasks[0]
    assert len(net.tasks) == 1 and anchor.anchor and anchor.origin == "harness"
    # Planted BEFORE the first batch ran, so both reads are its evidence...
    assert anchor.evidence == ["fake_read producer.py ✓", "fake_read consumer.py ✓"]
    # ...and the conclusion closed it with the finding as its outcome.
    assert anchor.status == "done"
    assert anchor.outcome == "Finding: messages can be dropped in the consumer's ack path."


@pytest.mark.asyncio
async def test_models_plan_supersedes_the_anchor_and_the_history_keeps_its_record() -> None:
    provider = _Scripted(
        [
            _call("fake_read", {"path": "producer.py"}, cid="r1"),
            _plan([{"title": "map the drops", "status": "in_progress"}, {"title": "write up"}]),
            _judge_ok(),
            _plan(
                [
                    {"title": "map the drops", "status": "done", "outcome": "two drop sites"},
                    {"title": "write up", "status": "done", "outcome": "report drafted"},
                ]
            ),
            _done("Finding: two drop sites, both in the consumer."),
        ]
    )
    session = Session(cwd="/tmp", model="t/m")
    loop = AgentLoop(provider, _registry(_FakeRead()), session, max_iterations=20)
    result = await loop.arun_turn(_DEEP_REQUEST)
    assert result.stop_reason == "completed"

    net = session.task_network
    assert [t.title for t in net.tasks] == ["map the drops", "write up"]
    assert not any(t.anchor for t in net.tasks)
    # What ran before the model's plan existed is in the history, not lost.
    replaced = [e for e in net.log if e.detail.startswith("anchor -> replaced by the model's plan")]
    assert len(replaced) == 1
    assert "(1 tool call(s) recorded) — last action: fake_read producer.py ✓" in replaced[0].detail
    # The request the plan serves is untouched by the model's full-replace.
    assert net.context.request.startswith("Audit the notification pipeline")


@pytest.mark.asyncio
async def test_an_anchor_alone_does_not_satisfy_the_mutate_gate() -> None:
    write = _FakeWrite()
    provider = _Scripted(
        [
            _call("fake_read", {"path": "producer.py"}, cid="r1"),
            _call("write_file", {"path": "report.md", "content": "x"}),
            _call("write_file", {"path": "report.md", "content": "x"}),
            _call("write_file", {"path": "report.md", "content": "x"}),
            _done("Result: the report is written."),
        ]
    )
    session = Session(cwd="/tmp", model="t/m")
    loop = AgentLoop(provider, _registry(_FakeRead(), write), session, max_iterations=20)
    result = await loop.arun_turn(_DEEP_REQUEST)
    assert result.stop_reason == "completed"
    # The read anchored the turn; the write was still withheld twice for the MODEL's plan,
    # then ran against the anchor once the nudges were spent.
    assert write.runs == 1
    assert provider.calls == 5
    anchor = session.task_network.tasks[0]
    assert anchor.anchor
    assert anchor.evidence == ["fake_read producer.py ✓", "write_file report.md ✓"]


@pytest.mark.asyncio
async def test_reading_the_record_is_never_evidence() -> None:
    provider = _Scripted(
        [
            _call("fake_read", {"path": "producer.py"}, cid="r1"),
            _call("plan_recall", {}, cid="q1"),
            _call("plan_recall", {"query": "producer"}, cid="q2"),
            _done("Finding: the producer is fine."),
        ]
    )
    session = Session(cwd="/tmp", model="t/m")
    loop = AgentLoop(provider, _registry(_FakeRead()), session, max_iterations=20)
    result = await loop.arun_turn(_DEEP_REQUEST)
    assert result.stop_reason == "completed"
    anchor = session.task_network.tasks[0]
    assert anchor.evidence == ["fake_read producer.py ✓"]  # the two recalls left no trace
    # ...and the recall answered from the live record: the request and the read it made.
    recall_outputs = [
        block.output
        for message in session.messages
        for block in message.blocks
        if isinstance(block, ToolResultBlock) and block.tool_use_id in ("q1", "q2")
    ]
    assert len(recall_outputs) == 2
    assert all("Request: Audit the notification pipeline" in out for out in recall_outputs)
    assert "· 1 tool call(s)" in recall_outputs[0]  # the overview counts the anchor's call...
    assert "step 1 evidence: fake_read producer.py ✓" in recall_outputs[1]  # ...the search names it
