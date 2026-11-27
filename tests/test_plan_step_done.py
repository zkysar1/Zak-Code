"""ADR-0416: a plan step closes inside the next call (``step_done``).

Measured 2026-11-27 on eleven worker sessions sharing one local model: 13.4 percent of the model
responses called update_plan and nothing else, and not one update_plan call shared a response with
another call, so ADR-0237's request to pair them did not take. Any tool but the plan's may carry
an optional ``step_done``; the harness closes the current step through update_plan's ``advance``
before the call runs, strips the argument, and runs the call, so its evidence is credited to the
step it starts. These tests pin the schema, the order, the refusals and the receipt line.
"""

from __future__ import annotations

import asyncio
import copy
from pathlib import Path
from typing import Any

import pytest

from zakcode.agent.loop import AgentLoop
from zakcode.config import PermissionTier
from zakcode.messages import ToolResultBlock, ToolUseBlock
from zakcode.providers.base import Capabilities, LLMResult, Provider, ToolCall
from zakcode.session.store import Session
from zakcode.tools.base import (
    ConcurrencyClass,
    Tool,
    ToolContext,
    ToolRegistry,
    ToolResult,
    ToolSpec,
)
from zakcode.tools.builtins.update_plan import (
    PLAN_TOOL_NAMES,
    STEP_DONE_ARG,
    STEP_DONE_NUDGE,
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


class _Write(Tool):
    """A fake write tool that records the arguments it was actually run with."""

    spec = ToolSpec(
        name="write_file",
        description="fake write",
        parameters={
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
        required_permission=PermissionTier.WORKSPACE_WRITE,
        concurrency=ConcurrencyClass.NEVER_PARALLEL,
    )

    def __init__(self) -> None:
        self.seen: list[dict[str, Any]] = []

    async def execute(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        self.seen.append(dict(args))
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


def _plan(a: str = "in_progress", b: str = "pending") -> ToolCall:
    return ToolCall(
        id="p",
        name="update_plan",
        arguments={"tasks": [{"title": "a", "status": a}, {"title": "b", "status": b}]},
    )


def _write(call_id: str = "w", **extra: Any) -> ToolCall:
    return ToolCall(id=call_id, name="write_file", arguments={"path": "b.txt", **extra})


def _loop(tmp_path: Path, results: list[LLMResult]) -> tuple[AgentLoop, _Write, Session]:
    write = _Write()
    registry = ToolRegistry()
    registry.register(write)
    registry.register(UpdatePlanTool())
    session = Session(cwd=str(tmp_path), model="t")
    return AgentLoop(_Scripted(results), registry, session, max_iterations=8), write, session


async def _run(loop: AgentLoop, prompt: str, *, streaming: bool) -> None:
    if streaming:
        async for _event in loop.astream_turn(prompt):
            pass
    else:
        await loop.arun_turn(prompt)


def _outputs(session: Session, call_id: str) -> str:
    return "\n".join(
        block.output
        for message in session.messages
        for block in message.blocks
        if isinstance(block, ToolResultBlock) and block.tool_use_id == call_id
    )


def _steps(session: Session) -> dict[str, Any]:
    return {step.title: step for step in session.task_network.leaves()}


# ── the schema ────────────────────────────────────────────────────────────────────────


def _registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(_Write())
    registry.register(UpdatePlanTool())
    return registry


def _properties(definitions: list[dict[str, Any]], name: str) -> dict[str, Any]:
    (found,) = [d for d in definitions if d["function"]["name"] == name]
    return found["function"]["parameters"].get("properties", {})


def test_every_tool_but_the_plans_declares_step_done_while_update_plan_is_offered() -> None:
    definitions = _registry().definitions()

    declared = _properties(definitions, "write_file")[STEP_DONE_ARG]
    assert declared["type"] == "string" and declared["description"]
    for name in PLAN_TOOL_NAMES:
        if any(d["function"]["name"] == name for d in definitions):
            assert STEP_DONE_ARG not in _properties(definitions, name)
    assert _properties(definitions, "write_file")["path"] == {"type": "string"}


def test_without_update_plan_no_tool_declares_it() -> None:
    definitions = _registry().definitions(allowed=["write_file"])

    assert STEP_DONE_ARG not in _properties(definitions, "write_file")


def test_the_definitions_do_not_mutate_a_spec_and_are_identical_call_after_call() -> None:
    registry = _registry()
    before = copy.deepcopy(_Write.spec.parameters)

    first = registry.definitions()
    second = registry.definitions()

    assert first == second  # the definitions render first in the prompt (ADR-0268)
    assert _Write.spec.parameters == before
    assert STEP_DONE_ARG not in _Write.spec.parameters["properties"]
    first[0]["function"]["parameters"]["properties"]["junk"] = {}  # a caller's edit stays its own
    assert "junk" not in registry.definitions()[0]["function"]["parameters"]["properties"]


# ── the close, in the loop ────────────────────────────────────────────────────────────


@pytest.mark.parametrize("streaming", [False, True], ids=["run", "stream"])
async def test_a_call_with_step_done_closes_the_step_and_credits_the_new_one(
    tmp_path: Path, streaming: bool
) -> None:
    loop, write, session = _loop(
        tmp_path,
        [
            _batch(_plan()),
            _judge_ok(),
            _batch(_write("w1")),  # work on a: the step has something to close on
            _batch(_write("w2", step_done="a is written")),
            _done(),
        ],
    )

    await _run(loop, "do a then b", streaming=streaming)

    steps = _steps(session)
    assert steps["a"].status == "done" and steps["a"].outcome == "a is written"
    assert steps["b"].status == "in_progress"
    assert write.seen == [{"path": "b.txt"}, {"path": "b.txt"}]  # the tool never saw step_done
    holders = {
        title
        for title, step in steps.items()
        if any("write_file" in line for line in step.evidence)
    }
    assert holders == {"a", "b"}  # w1 on a, w2 on the step it started
    assert len(steps["b"].evidence) == 1
    receipt = _outputs(session, "w2")
    assert receipt.startswith("[plan] Step 1 done ('a'); step 2 ('b') is current.\n")
    assert receipt.endswith("written")
    assert STEP_DONE_NUDGE not in receipt  # the nudge is the stand-alone advance's alone
    sent = [
        block
        for message in session.messages
        for block in message.blocks
        if isinstance(block, ToolUseBlock) and block.id == "w2"
    ]
    assert sent and sent[0].input[STEP_DONE_ARG] == "a is written"  # the transcript keeps it


@pytest.mark.parametrize("streaming", [False, True], ids=["run", "stream"])
async def test_a_step_with_no_work_recorded_is_not_closed_and_the_call_still_runs(
    tmp_path: Path, streaming: bool
) -> None:
    loop, write, session = _loop(
        tmp_path,
        [_batch(_plan()), _judge_ok(), _batch(_write("w", step_done="nothing yet")), _done()],
    )

    await _run(loop, "do a then b", streaming=streaming)

    steps = _steps(session)
    assert steps["a"].status == "in_progress" and steps["b"].status == "pending"
    assert write.seen == [{"path": "b.txt"}]
    assert _outputs(session, "w").startswith(
        "[plan] step_done not applied: step 1 has no work recorded on it yet.\n"
    )


@pytest.mark.parametrize("streaming", [False, True], ids=["run", "stream"])
async def test_with_no_plan_of_the_models_it_is_not_applied(
    tmp_path: Path, streaming: bool
) -> None:
    loop, write, session = _loop(tmp_path, [_batch(_write("w", step_done="all done")), _done()])

    await _run(loop, "hi", streaming=streaming)

    assert write.seen == [{"path": "b.txt"}]
    assert _outputs(session, "w").startswith("[plan] step_done not applied: there is no plan")


@pytest.mark.parametrize("empty", ["", "   "])
async def test_an_empty_value_is_stripped_and_ignored(tmp_path: Path, empty: str) -> None:
    loop, write, session = _loop(
        tmp_path,
        [
            _batch(_plan()),
            _judge_ok(),
            _batch(_write("w1")),
            _batch(_write("w2", step_done=empty)),
            _done(),
        ],
    )

    await _run(loop, "do a then b", streaming=False)

    assert _steps(session)["a"].status == "in_progress"
    assert write.seen == [{"path": "b.txt"}, {"path": "b.txt"}]
    assert "[plan]" not in _outputs(session, "w2")


async def test_a_step_of_a_paged_skill_is_not_closed_inside_a_call(tmp_path: Path) -> None:
    loop, write, session = _loop(
        tmp_path,
        [
            _batch(_plan()),
            _judge_ok(),
            _batch(_write("w1")),
            _batch(_write("w2", step_done="section done")),
            _done(),
        ],
    )
    loop._step_is_paged = lambda step: True  # type: ignore[method-assign]

    await _run(loop, "do a then b", streaming=False)

    assert _steps(session)["a"].status == "in_progress"
    assert write.seen[-1] == {"path": "b.txt"}
    assert "call update_plan alone" in _outputs(session, "w2")


async def test_closing_the_last_step_says_the_plan_is_complete(tmp_path: Path) -> None:
    loop, _write_tool, session = _loop(
        tmp_path,
        [
            _batch(_plan("done", "in_progress")),
            _judge_ok(),
            _batch(_write("w1")),
            _batch(_write("w2", step_done="b is written")),
            _done(),
        ],
    )

    await _run(loop, "do a then b", streaming=False)

    assert _steps(session)["b"].status == "done"
    assert "the plan is complete." in _outputs(session, "w2")


class _Read(Tool):
    """A fake read-only tool that yields once, so calls in one batch interleave."""

    spec = ToolSpec(
        name="read_file",
        description="fake read",
        parameters={
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
        required_permission=PermissionTier.READ_ONLY,
        concurrency=ConcurrencyClass.READ_ONLY_SAFE,
    )

    async def execute(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        await asyncio.sleep(0)
        return ToolResult.ok(f"read {args.get('path')}")


def _read(call_id: str, **extra: Any) -> ToolCall:
    return ToolCall(id=call_id, name="read_file", arguments={"path": f"{call_id}.txt", **extra})


async def test_a_batch_where_every_call_carries_step_done_closes_one_step(tmp_path: Path) -> None:
    loop, _write_tool, session = _loop(
        tmp_path,
        [
            _batch(_plan()),
            _judge_ok(),
            _batch(_read("r1")),  # work on a
            _batch(_read("r2", step_done="a is read"), _read("r3", step_done="b is read")),
            _done(),
        ],
    )
    loop.registry.register(_Read())

    await _run(loop, "do a then b", streaming=False)

    steps = _steps(session)
    assert steps["a"].status == "done" and steps["a"].outcome == "a is read"
    assert steps["b"].status == "in_progress"  # r2's work has not landed on b when r3 asks
    assert _outputs(session, "r3").startswith(
        "[plan] step_done not applied: step 2 has no work recorded on it yet.\n"
    )
    assert _outputs(session, "r3").endswith("read r3.txt")
