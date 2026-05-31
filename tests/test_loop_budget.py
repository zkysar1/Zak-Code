"""Tests for wiring the shared IterationBudget into the AgentLoop (M4-2).

The budget is an ADDITIVE bound: with no budget the loop behaves exactly as
before (the local ``max_iterations`` cap is the only limit); with a budget, each
iteration draws one unit from the shared pool and the turn stops with
``stop_reason="max_iterations"`` when the pool empties. A single budget instance
shared by two loops bounds their *combined* iteration count — the property
sub-agent delegation will rely on.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from pathlib import Path

from zakcode.agent.budget import IterationBudget
from zakcode.agent.loop import AgentLoop
from zakcode.events import AgentDone
from zakcode.messages import Message
from zakcode.providers.base import (
    Capabilities,
    LLMResult,
    Provider,
    ProviderStreamEvent,
    StreamDone,
    StreamToolCallDelta,
    ToolCall,
)
from zakcode.session.store import Session
from zakcode.tools.base import Tool, ToolContext, ToolRegistry, ToolResult, ToolSpec
from zakcode.usage import Usage


class _AlwaysToolProvider(Provider):
    """Always requests a (harmless) tool call, so the turn only ever ends on a
    cap/budget bound — never on a natural ``completed``.

    Each call carries a *distinct* argument (the running call index) so the loop's
    doom-loop guard — which stops on three byte-identical tool-call batches in a
    row — never fires; that isolates the iteration *budget/cap* as the sole stop
    condition under test.
    """

    def __init__(self) -> None:
        self.calls = 0

    async def acomplete(
        self, messages: list, *, tools: list | None = None, system: str | None = None
    ) -> LLMResult:
        self.calls += 1
        return LLMResult(
            text="",
            tool_calls=[ToolCall(id=f"c{self.calls}", name="noop", arguments={"n": self.calls})],
            usage=Usage(total_tokens=1),
        )

    async def astream(
        self, messages: list, *, tools: list | None = None, system: str | None = None
    ) -> AsyncIterator[ProviderStreamEvent]:
        self.calls += 1
        yield StreamToolCallDelta(
            index=0,
            id=f"c{self.calls}",
            name="noop",
            arguments_delta=json.dumps({"n": self.calls}),
        )
        yield StreamDone()

    def count_tokens(self, messages: list[Message], *, system: str | None = None) -> int:
        return 0

    def capabilities(self) -> Capabilities:
        return Capabilities()


class _NoopTool(Tool):
    spec = ToolSpec(name="noop", description="does nothing")

    async def execute(self, args: dict, ctx: ToolContext) -> ToolResult:
        return ToolResult.ok("ok")


def _registry() -> ToolRegistry:
    reg = ToolRegistry()
    reg.register(_NoopTool())
    return reg


def _loop(tmp_path: Path, *, max_iterations: int, budget: IterationBudget | None) -> AgentLoop:
    return AgentLoop(
        _AlwaysToolProvider(),
        _registry(),
        Session(cwd=str(tmp_path), model="fake/test"),
        workspace_root=tmp_path,
        max_iterations=max_iterations,
        budget=budget,
    )


async def test_no_budget_uses_local_cap_unchanged(tmp_path: Path) -> None:
    loop = _loop(tmp_path, max_iterations=2, budget=None)
    result = await loop.arun_turn("go")
    assert result.stop_reason == "max_iterations"
    assert result.iterations == 2


async def test_budget_caps_iterations_below_local_cap(tmp_path: Path) -> None:
    budget = IterationBudget(3)
    loop = _loop(tmp_path, max_iterations=100, budget=budget)
    result = await loop.arun_turn("go")
    assert result.stop_reason == "max_iterations"
    assert result.iterations == 3  # the budget, not the local cap, stopped it
    assert budget.remaining == 0
    assert budget.consumed == 3


async def test_shared_budget_bounds_two_loops_combined(tmp_path: Path) -> None:
    # One pool of 5 shared by two loops: the first drains it, the second gets none.
    budget = IterationBudget(5)
    first = _loop(tmp_path, max_iterations=100, budget=budget)
    r1 = await first.arun_turn("go")
    assert r1.iterations == 5
    assert budget.remaining == 0

    second = _loop(tmp_path, max_iterations=100, budget=budget)
    r2 = await second.arun_turn("go")
    assert r2.iterations == 0  # pool already empty → stops before any iteration
    assert r2.stop_reason == "max_iterations"


async def test_partially_consumed_pool_bounds_a_later_loop(tmp_path: Path) -> None:
    budget = IterationBudget(5)
    budget.consume(3)  # a sibling already drew 3 of the shared pool
    loop = _loop(tmp_path, max_iterations=100, budget=budget)
    result = await loop.arun_turn("go")
    assert result.iterations == 2  # exactly the remaining headroom


async def test_budget_bounds_the_streaming_path_too(tmp_path: Path) -> None:
    budget = IterationBudget(2)
    loop = _loop(tmp_path, max_iterations=100, budget=budget)
    events = [e async for e in loop.astream_turn("go")]
    done = events[-1]
    assert isinstance(done, AgentDone)
    assert done.stop_reason == "max_iterations"
    assert done.iterations == 2
    assert budget.remaining == 0
