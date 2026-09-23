"""ADR-0237: a plan advance rides with the next step's first call, in one response.

A response that only updates the plan costs a whole model call, and on a slow backend it is the
most expensive kind (measured 2026-09-23 on the 131k P40 pod: 17.8 and 22.3 percent of the two
worker Bodies' model time). Nothing told the model it could pair the update with the work, so it
rarely did. These tests pin the one sentence that now says so on every surface the model reads
about the plan, the batch mechanics that make the pairing safe (the plan runs first, so the work
is credited to the step it starts), and the one exception: a paged skill's section, whose next
section arrives only in the reply to the update.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from zakcode.agent.loop import AgentLoop
from zakcode.agent.prompt import SystemPromptBuilder
from zakcode.config import PermissionTier, load_settings
from zakcode.providers.base import Capabilities, LLMResult, Provider, ToolCall
from zakcode.session.store import Session
from zakcode.tasks import Task, skill_pages
from zakcode.tools.base import (
    ConcurrencyClass,
    Tool,
    ToolContext,
    ToolRegistry,
    ToolResult,
    ToolSpec,
)
from zakcode.tools.builtins.update_plan import PLAN_ADVANCE, UpdatePlanTool
from zakcode.usage import Usage


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


def _plan(a: str, b: str = "pending") -> ToolCall:
    return ToolCall(
        id="p",
        name="update_plan",
        arguments={"tasks": [{"title": "a", "status": a}, {"title": "b", "status": b}]},
    )


def _write() -> ToolCall:
    return ToolCall(id="w", name="write_file", arguments={"path": "b.txt"})


def _judge_ok() -> LLMResult:
    # The decomposition judge (ADR-0050) reads one scorecard when the plan is first laid out.
    return LLMResult(
        text='{"scores": {"coverage": 0.9, "granularity": 0.9, "ordering": 0.9, "soundness": 0.9}}',
        usage=Usage(total_tokens=2),
    )


def _done() -> LLMResult:
    return LLMResult(text="stopping here", usage=Usage(total_tokens=1))


def test_the_sentence_is_on_every_surface_that_says_how_the_plan_moves(tmp_path: Path) -> None:
    assert PLAN_ADVANCE in UpdatePlanTool.spec.description
    assert PLAN_ADVANCE in SystemPromptBuilder().build(load_settings(workspace_root=tmp_path))

    loop = AgentLoop(_Scripted([_done()]), ToolRegistry(), Session(cwd=str(tmp_path), model="t"))
    network = loop.session.task_network
    network.tasks = [Task(title="a", status="in_progress"), Task(title="b")]
    network.normalize()
    reminder = loop._plan_reminder()
    assert reminder is not None and PLAN_ADVANCE in reminder.text


async def test_the_receipt_of_a_plan_update_says_it_too() -> None:
    from zakcode.tasks import TaskNetwork

    network = TaskNetwork()
    ctx = ToolContext(workspace_root=Path("/tmp"), task_network=network)
    res = await UpdatePlanTool().execute(
        {"tasks": [{"title": "a", "status": "in_progress"}, {"title": "b"}]}, ctx
    )
    assert not res.is_error
    assert res.hint is not None and PLAN_ADVANCE in res.hint  # the rail after the receipt


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize(
    ("order", "credited"),
    [
        pytest.param("plan first", "b", id="the plan first credits the step it starts"),
        pytest.param("work first", "a", id="the work first credits the step it closes"),
    ],
)
async def test_a_paired_response_credits_the_work_to_the_step_the_batch_leaves_current(
    tmp_path: Path, streaming: bool, order: str, credited: str
) -> None:
    # Why the sentence says "update_plan first": a call's evidence belongs to the step that is
    # current when it runs (ADR-0110), and a batch runs in order.
    advance, work = _plan("done", "in_progress"), _write()
    paired = _batch(advance, work) if order == "plan first" else _batch(work, advance)
    registry = ToolRegistry()
    registry.register(_Write())
    registry.register(UpdatePlanTool())
    session = Session(cwd=str(tmp_path), model="t")
    loop = AgentLoop(
        _Scripted([_batch(_plan("in_progress")), _judge_ok(), paired, _done()]),
        registry,
        session,
        max_iterations=8,
    )
    if streaming:
        async for _event in loop.astream_turn("do a then b"):
            pass
    else:
        await loop.arun_turn("do a then b")

    steps = {step.title: step for step in session.task_network.leaves()}
    assert steps["a"].status == "done" and steps["b"].status == "in_progress"
    holders = [
        title
        for title, step in steps.items()
        if any("write_file" in line for line in step.evidence)
    ]
    assert holders == [credited]


def test_a_paged_section_says_its_update_goes_alone_and_the_last_section_does_not() -> None:
    body = "# S\n\nfront\n\n## Phase 1\n" + "a " * 4000 + "\n\n## Phase 2\n" + "b " * 4000 + "\n"
    pages = skill_pages(body, skill="demo")
    assert pages is not None and pages.count == 2
    assert "update_plan, alone in its response" in pages.render(1)
    assert "section 2 of 2 arrives in the reply to that call" in pages.render(1)
    assert "alone in its response" not in pages.render(2)
