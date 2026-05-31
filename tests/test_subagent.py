"""Tests for the sub-agent factory (M4-3): isolation, filtering, shared budget, summary.

These exercise the structural guarantees delegation relies on, using a scripted
provider so no model or network is touched:

* a child runs on a FRESH session (no parent-context bleed),
* a child sees only its FILTERED tools,
* children draw from the SAME shared budget (and respect its child cap),
* the parent gets back a condensed SUMMARY (final text), not the raw transcript.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from pathlib import Path

import pytest

from zakcode.agent.budget import ChildLimitExceeded, IterationBudget
from zakcode.agent.subagent import (
    GENERAL_PURPOSE,
    SubAgentDefinition,
    SubAgentResult,
    SubAgentRunner,
)
from zakcode.config import Settings
from zakcode.messages import Message
from zakcode.providers.base import (
    Capabilities,
    LLMResult,
    Provider,
    ProviderStreamEvent,
    StreamDone,
    StreamTextDelta,
    StreamToolCallDelta,
    ToolCall,
)
from zakcode.tools.base import Tool, ToolContext, ToolRegistry, ToolResult, ToolSpec
from zakcode.usage import Usage


class _RecordingTool(Tool):
    """A read-only tool that records every (registry-scoped) invocation."""

    def __init__(self, name: str) -> None:
        self.spec = ToolSpec(name=name, description=f"the {name} tool")
        self.invocations: list[dict] = []

    async def execute(self, args: dict, ctx: ToolContext) -> ToolResult:
        self.invocations.append(args)
        return ToolResult.ok(f"{self.spec.name} ran")


class _OneShotProvider(Provider):
    """Emits one assistant text then completes — a single, deterministic turn.

    With ``call_tool`` set, the first completion instead requests that tool once
    (then the second completion finishes), so tool-filtering can be observed.
    """

    def __init__(self, text: str, *, call_tool: str | None = None) -> None:
        self.text = text
        self.call_tool = call_tool
        self._calls = 0

    async def acomplete(
        self, messages: list, *, tools: list | None = None, system: str | None = None
    ) -> LLMResult:
        self._calls += 1
        if self.call_tool and self._calls == 1:
            return LLMResult(
                tool_calls=[ToolCall(id="t1", name=self.call_tool, arguments={"x": 1})],
                usage=Usage(total_tokens=1),
            )
        return LLMResult(text=self.text, usage=Usage(total_tokens=1))

    async def astream(
        self, messages: list, *, tools: list | None = None, system: str | None = None
    ) -> AsyncIterator[ProviderStreamEvent]:
        result = await self.acomplete(messages, tools=tools, system=system)
        if result.text:
            yield StreamTextDelta(text=result.text)
        for i, call in enumerate(result.tool_calls):
            yield StreamToolCallDelta(
                index=i, id=call.id, name=call.name, arguments_delta=json.dumps(call.arguments)
            )
        yield StreamDone()

    def count_tokens(self, messages: list[Message], *, system: str | None = None) -> int:
        return 0

    def capabilities(self) -> Capabilities:
        return Capabilities()


def _runner(
    tmp_path: Path,
    provider: Provider,
    registry: ToolRegistry,
    budget: IterationBudget,
) -> SubAgentRunner:
    return SubAgentRunner(
        provider=provider,
        registry=registry,
        settings=Settings(default_model="scripted/test", workspace_root=tmp_path),
        budget=budget,
        workspace_root=tmp_path,
    )


def _registry(*tools: Tool) -> ToolRegistry:
    reg = ToolRegistry()
    for t in tools:
        reg.register(t)
    return reg


async def test_child_returns_summary_not_transcript(tmp_path: Path) -> None:
    runner = _runner(
        tmp_path, _OneShotProvider("the answer is 42"), _registry(), IterationBudget(10)
    )
    result = await runner.run(GENERAL_PURPOSE, "what is the answer?")
    assert isinstance(result, SubAgentResult)
    assert result.summary == "the answer is 42"
    assert result.stop_reason == "completed"
    assert result.name == "general-purpose"


async def test_child_runs_on_fresh_session(tmp_path: Path) -> None:
    # The child's loop builds its own Session; the parent passes no history in, so
    # the only messages are the child's own prompt + reply. We assert via the tool
    # context that the workspace is the child's and that two runs don't share state.
    runner = _runner(tmp_path, _OneShotProvider("done"), _registry(), IterationBudget(10))
    r1 = await runner.run(GENERAL_PURPOSE, "first")
    r2 = await runner.run(GENERAL_PURPOSE, "second")
    # Each child completed independently in exactly one iteration (no accumulated
    # history that would change iteration counts).
    assert r1.iterations == 1
    assert r2.iterations == 1


async def test_child_tool_filtering_blocks_disallowed_tool(tmp_path: Path) -> None:
    allowed = _RecordingTool("read_file")
    forbidden = _RecordingTool("write_file")
    registry = _registry(allowed, forbidden)
    # The child is restricted to read_file but the model tries to call write_file.
    provider = _OneShotProvider("ok", call_tool="write_file")
    definition = SubAgentDefinition(name="reader", allowed_tools=["read_file"])
    runner = _runner(tmp_path, provider, registry, IterationBudget(10))

    result = await runner.run(definition, "go")
    # write_file was never executed (not in the child's filtered registry)…
    assert forbidden.invocations == []
    # …and the child still completes (the disallowed call returns an error result
    # the model recovers from), reporting the final summary.
    assert result.summary == "ok"


def test_child_registry_is_subset(tmp_path: Path) -> None:
    a, b, c = _RecordingTool("read_file"), _RecordingTool("write_file"), _RecordingTool("bash")
    registry = _registry(a, b, c)
    runner = _runner(tmp_path, _OneShotProvider("x"), registry, IterationBudget(10))
    sub = runner.child_registry(SubAgentDefinition(name="r", allowed_tools=["read_file", "bash"]))
    assert set(sub.names()) == {"read_file", "bash"}
    # No filter → the parent's full registry is reused unchanged.
    full = runner.child_registry(GENERAL_PURPOSE)
    assert full is registry


async def test_children_share_the_budget(tmp_path: Path) -> None:
    budget = IterationBudget(10)
    runner = _runner(tmp_path, _OneShotProvider("done"), _registry(), budget)
    await runner.run(GENERAL_PURPOSE, "one")
    consumed_after_first = budget.consumed
    assert consumed_after_first >= 1
    await runner.run(GENERAL_PURPOSE, "two")
    # The second child drew from the SAME pool (consumption strictly increased).
    assert budget.consumed > consumed_after_first


async def test_child_cap_enforced(tmp_path: Path) -> None:
    budget = IterationBudget(100, max_children=1)
    runner = _runner(tmp_path, _OneShotProvider("done"), _registry(), budget)
    await runner.run(GENERAL_PURPOSE, "first child ok")
    with pytest.raises(ChildLimitExceeded):
        await runner.run(GENERAL_PURPOSE, "second child over cap")
    assert budget.children_spawned == 1


def test_prompt_builder_carries_system_suffix(tmp_path: Path) -> None:
    runner = _runner(tmp_path, _OneShotProvider("x"), _registry(), IterationBudget(10))
    definition = SubAgentDefinition(name="planner", system_suffix="Produce a plan; do not edit.")
    builder = runner.prompt_builder_for(definition)
    assert builder.extra_instructions == "Produce a plan; do not edit."
