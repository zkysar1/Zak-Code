"""ADR-0398: the plan's short form rides in the Bash call that starts the next step.

ADR-0237 asked the model to send a plan update in the same response as the next step's first call,
and ADR-0388 shrank the update to ``{"advance": ...}``; neither removes the response that does
nothing else, and the served model makes that response nearly every time (measured 2026-11-20 on
ten worker Bodies: 530 of 544 update_plan calls in 12.6 hours were alone in their response). So
Bash takes an optional ``advance`` field and the loop applies it, as a harness-made update_plan
call, before the command runs. These tests pin the mechanics: the step closes and the next starts,
the command is credited to the step it starts (as a batch of update_plan then the work is,
ADR-0110), the tool never sees the field, a refused advance still runs the command, a null field
carries nothing, a plan holding a paged skill's sections is left as it is, only Bash carries the
field, and every surface that says how the plan moves names the carried form first.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from zakcode.agent.loop import _CARRIED_ADVANCE_PAGED, AgentLoop
from zakcode.agent.prompt import SystemPromptBuilder
from zakcode.config import PermissionTier, load_settings
from zakcode.messages import ToolResultBlock
from zakcode.providers.base import Capabilities, LLMResult, Provider, ToolCall
from zakcode.session.store import Session
from zakcode.tasks import Task
from zakcode.tools.base import (
    ConcurrencyClass,
    Tool,
    ToolContext,
    ToolRegistry,
    ToolResult,
    ToolSpec,
)
from zakcode.tools.builtins.bash import BashTool
from zakcode.tools.builtins.update_plan import (
    _PLANNED_HINT,
    BASH_ADVANCE_FIELD,
    PLAN_ADVANCE_CALL,
    PLAN_ADVANCE_CARRIED,
    UpdatePlanTool,
)
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


class _Recorder(Tool):
    """A stand-in tool that records the arguments it was run with."""

    def __init__(self, name: str) -> None:
        self.spec = ToolSpec(
            name=name,
            description=f"fake {name}",
            required_permission=PermissionTier.READ_ONLY,
            concurrency=ConcurrencyClass.NEVER_PARALLEL,
        )
        self.seen: list[dict[str, Any]] = []

    async def execute(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        self.seen.append(dict(args))
        return ToolResult.ok("ran")


def _batch(*calls: ToolCall) -> LLMResult:
    return LLMResult(text="", tool_calls=list(calls), usage=Usage(total_tokens=1))


def _plan() -> ToolCall:
    return ToolCall(
        id="p",
        name="update_plan",
        arguments={"tasks": [{"title": "a", "status": "in_progress"}, {"title": "b"}]},
    )


def _judge_ok() -> LLMResult:
    # The decomposition judge (ADR-0050) reads one scorecard when the plan is first laid out.
    return LLMResult(
        text='{"scores": {"coverage": 0.9, "granularity": 0.9, "ordering": 0.9, "soundness": 0.9}}',
        usage=Usage(total_tokens=2),
    )


def _done() -> LLMResult:
    return LLMResult(text="stopping here", usage=Usage(total_tokens=1))


def _bash(**extra: Any) -> ToolCall:
    return ToolCall(id="b1", name="Bash", arguments={"command": "echo hi", **extra})


async def _run(
    tmp_path: Path, script: list[LLMResult], *, streaming: bool = False, tools: list[Tool]
) -> AgentLoop:
    registry = ToolRegistry()
    for tool in tools:
        registry.register(tool)
    registry.register(UpdatePlanTool())
    loop = AgentLoop(
        _Scripted(script), registry, Session(cwd=str(tmp_path), model="t"), max_iterations=8
    )
    if streaming:
        async for _event in loop.astream_turn("do a then b"):
            pass
    else:
        await loop.arun_turn("do a then b")
    return loop


def _result(loop: AgentLoop, call_id: str) -> ToolResultBlock:
    for message in loop.session.messages:
        for block in message.blocks:
            if isinstance(block, ToolResultBlock) and block.tool_use_id == call_id:
                return block
    raise AssertionError(f"no result for {call_id}")


def _holders(loop: AgentLoop, tool: str) -> list[str]:
    return [
        step.title
        for step in loop.session.task_network.leaves()
        if any(tool in line for line in step.evidence)
    ]


def test_bash_offers_the_field_and_does_not_require_it() -> None:
    params = BashTool.spec.parameters
    assert params["properties"]["advance"]["description"] == BASH_ADVANCE_FIELD
    assert "outcome" in params["properties"]["advance"]["properties"]
    assert params["required"] == ["command"]


@pytest.mark.parametrize("streaming", [False, True])
async def test_a_carried_advance_closes_the_step_and_credits_the_command_to_the_next(
    tmp_path: Path, streaming: bool
) -> None:
    bash = _Recorder("Bash")
    loop = await _run(
        tmp_path,
        [
            _batch(_plan()),
            _judge_ok(),
            _batch(_bash(advance={"outcome": "a read"})),
            _done(),
        ],
        streaming=streaming,
        tools=[bash],
    )

    steps = {step.title: step for step in loop.session.task_network.leaves()}
    assert steps["a"].status == "done" and steps["a"].outcome == "a read"
    assert steps["b"].status == "in_progress"
    # The command belongs to the step the advance started, as after update_plan then Bash.
    assert _holders(loop, "Bash") == ["b"]
    # The harness owns the field: the tool ran the command and never saw it.
    assert bash.seen == [{"command": "echo hi"}]
    out = _result(loop, "b1").output
    assert out.startswith("[plan, from this call's advance] Step 1 done")
    assert out.rstrip().endswith("ran")


async def test_a_refused_advance_still_runs_the_command(tmp_path: Path) -> None:
    # No plan yet: the short form has nothing to close, and says so; the command is unaffected.
    bash = _Recorder("Bash")
    loop = await _run(
        tmp_path,
        [_batch(_bash(advance={"outcome": "x"})), _done()],
        tools=[bash],
    )

    assert bash.seen == [{"command": "echo hi"}]
    block = _result(loop, "b1")
    assert not block.is_error
    assert "no plan of yours to advance" in block.output
    assert loop.session.task_network.is_anchor_only() or loop.session.task_network.is_empty()


@pytest.mark.parametrize("value", [None, False])
async def test_a_null_advance_carries_nothing(tmp_path: Path, value: Any) -> None:
    bash = _Recorder("Bash")
    loop = await _run(
        tmp_path,
        [_batch(_plan()), _judge_ok(), _batch(_bash(advance=value)), _done()],
        tools=[bash],
    )

    steps = {step.title: step.status for step in loop.session.task_network.leaves()}
    assert steps == {"a": "in_progress", "b": "pending"}
    assert bash.seen == [{"command": "echo hi"}]
    assert _result(loop, "b1").output == "ran"


async def test_a_plan_with_a_paged_skill_is_left_as_it_is(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A paged section's next part arrives only in the reply to an update sent alone (ADR-0237).
    monkeypatch.setattr(AgentLoop, "_plan_holds_paged_skill", lambda self: True)
    bash = _Recorder("Bash")
    loop = await _run(
        tmp_path,
        [_batch(_plan()), _judge_ok(), _batch(_bash(advance={"outcome": "a read"})), _done()],
        tools=[bash],
    )

    steps = {step.title: step.status for step in loop.session.task_network.leaves()}
    assert steps == {"a": "in_progress", "b": "pending"}
    assert _holders(loop, "Bash") == ["a"]
    assert bash.seen == [{"command": "echo hi"}]
    assert _result(loop, "b1").output.startswith(_CARRIED_ADVANCE_PAGED)


async def test_only_bash_carries_the_field(tmp_path: Path) -> None:
    other = _Recorder("write_file")
    call = ToolCall(id="w1", name="write_file", arguments={"path": "b.txt", "advance": {}})
    loop = await _run(
        tmp_path,
        [_batch(_plan()), _judge_ok(), _batch(call), _done()],
        tools=[other],
    )

    steps = {step.title: step.status for step in loop.session.task_network.leaves()}
    assert steps == {"a": "in_progress", "b": "pending"}
    assert other.seen == [{"path": "b.txt", "advance": {}}]


def test_every_surface_names_the_carried_form_first(tmp_path: Path) -> None:
    assert PLAN_ADVANCE_CARRIED in UpdatePlanTool.spec.description
    assert PLAN_ADVANCE_CARRIED in SystemPromptBuilder().build(
        load_settings(workspace_root=tmp_path)
    )
    assert _PLANNED_HINT.index(PLAN_ADVANCE_CARRIED) < _PLANNED_HINT.index(PLAN_ADVANCE_CALL)

    loop = AgentLoop(_Scripted([_done()]), ToolRegistry(), Session(cwd=str(tmp_path), model="t"))
    network = loop.session.task_network
    network.tasks = [Task(title="a", status="in_progress"), Task(title="b")]
    network.normalize()
    reminder = loop._plan_reminder()
    assert reminder is not None
    assert reminder.text.index(PLAN_ADVANCE_CARRIED) < reminder.text.index(PLAN_ADVANCE_CALL)
