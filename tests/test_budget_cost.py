"""Cost/token budget stop condition (parity #4).

The shared IterationBudget gains optional cost/token ceilings bounding cumulative spend
across a turn and its whole sub-agent tree; once crossed the loop stops with
stop_reason="budget_exhausted" (a non-vetoable hard bound, like max_iterations). Hermetic.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from zakcode.agent.budget import IterationBudget
from zakcode.agent.loop import AgentLoop
from zakcode.config import load_settings
from zakcode.providers.base import Capabilities, LLMResult, Provider
from zakcode.session.store import Session
from zakcode.tools.base import ToolRegistry
from zakcode.usage import Usage

# ── budget unit ───────────────────────────────────────────────────────────────


def test_cost_ceiling_exhausts() -> None:
    b = IterationBudget(100, max_cost_usd=0.10)
    b.add_usage(cost_usd=0.04, total_tokens=10)
    assert not b.over_budget()
    b.add_usage(cost_usd=0.07, total_tokens=10)  # cumulative 0.11 >= 0.10
    assert b.cost_exhausted() and b.over_budget()
    assert b.cost_spent == pytest.approx(0.11)


def test_token_ceiling_exhausts() -> None:
    b = IterationBudget(100, max_tokens=1000)
    b.add_usage(total_tokens=600)
    assert not b.over_budget()
    b.add_usage(total_tokens=500)  # cumulative 1100 >= 1000
    assert b.tokens_exhausted() and b.over_budget()


def test_no_ceiling_never_exhausts() -> None:
    b = IterationBudget(100)  # no cost/token ceilings
    b.add_usage(cost_usd=999.0, total_tokens=10**9)
    assert not b.over_budget()


def test_negative_usage_clamped() -> None:
    b = IterationBudget(100, max_cost_usd=1.0)
    b.add_usage(cost_usd=-5.0, total_tokens=-100)  # junk record cannot reduce the total
    assert b.cost_spent == 0.0 and b.tokens_spent == 0


def test_shared_across_tree() -> None:
    """Parent and child draw from one cost cap (mirrors the shared-iteration property)."""
    shared = IterationBudget(100, max_cost_usd=0.10)
    shared.add_usage(cost_usd=0.06)  # "parent" spends
    shared.add_usage(cost_usd=0.05)  # "child" spends on the SAME budget -> 0.11
    assert shared.over_budget()


# ── loop integration ─────────────────────────────────────────────────────────


class CostProvider(Provider):
    """Returns a fixed cost per call so cumulative spend is predictable."""

    def __init__(self, cost_per_call: float, tokens_per_call: int = 100) -> None:
        self._cost = cost_per_call
        self._tokens = tokens_per_call
        self.calls = 0

    async def acomplete(self, messages, *, system=None, tools=None, **kw) -> LLMResult:
        self.calls += 1
        # Always asks for a tool so the turn would continue absent a budget stop.
        from zakcode.providers.base import ToolCall

        return LLMResult(
            text="working",
            tool_calls=[ToolCall(id=f"c{self.calls}", name="noop", arguments={})],
            usage=Usage(total_tokens=self._tokens, cost_usd=self._cost),
        )

    async def astream(self, messages, *, system=None, tools=None, **kw):
        # Streaming twin: a tool-call delta + a usage event (so the per-event budget fold
        # at the StreamUsage handler runs) + done. Keeps the turn going absent a budget stop.
        import json

        from zakcode.providers.base import StreamDone, StreamToolCallDelta, StreamUsage

        self.calls += 1
        yield StreamToolCallDelta(
            index=0, id=f"c{self.calls}", name="noop", arguments_delta=json.dumps({})
        )
        yield StreamUsage(usage=Usage(total_tokens=self._tokens, cost_usd=self._cost))
        yield StreamDone(finish_reason="tool_calls")

    def count_tokens(self, messages, *, system=None) -> int:
        return 0

    def capabilities(self) -> Capabilities:
        return Capabilities(supports_tools=True, context_window=8192)


def _loop(provider: Provider, budget: IterationBudget) -> AgentLoop:
    settings = load_settings(workspace_root=Path.cwd())
    session = Session(cwd="/tmp/work", model="test/model")
    return AgentLoop(
        provider, ToolRegistry(), session, settings=settings, budget=budget, max_iterations=50
    )


def test_loop_stops_on_cost_budget() -> None:
    # No "noop" tool registered → each tool call errors, but the turn keeps iterating;
    # the cost ceiling is what must stop it. 0.04/call, ceiling 0.10 → stop after call 3.
    budget = IterationBudget(50, max_cost_usd=0.10)
    provider = CostProvider(cost_per_call=0.04)
    result = asyncio.run(_loop(provider, budget).arun_turn("go"))
    assert result.stop_reason == "budget_exhausted"
    assert provider.calls == 3  # 0.04 + 0.04 + 0.04 = 0.12 >= 0.10
    assert not result.degraded  # a clean hard bound, not a struggle (like max_iterations)


def test_loop_stops_on_token_budget() -> None:
    budget = IterationBudget(50, max_tokens=250)
    provider = CostProvider(cost_per_call=0.0, tokens_per_call=100)
    result = asyncio.run(_loop(provider, budget).arun_turn("go"))
    assert result.stop_reason == "budget_exhausted"
    assert provider.calls == 3  # 100*3 = 300 >= 250


def test_loop_no_ceiling_never_budget_stops() -> None:
    """With no cost/token ceiling, a turn is never stopped by budget_exhausted (it ends on
    its own terms — here the doom guard, since the scripted batch repeats)."""
    budget = IterationBudget(50)  # no cost ceiling
    provider = CostProvider(cost_per_call=1.0)
    result = asyncio.run(_loop(provider, budget).arun_turn("go"))
    assert result.stop_reason != "budget_exhausted"


def test_streaming_stops_on_cost_budget() -> None:
    budget = IterationBudget(50, max_cost_usd=0.10)
    provider = CostProvider(cost_per_call=0.04)
    loop = _loop(provider, budget)
    done = asyncio.run(_drain(loop, "go"))
    assert done.stop_reason == "budget_exhausted"
    assert provider.calls == 3  # 0.04 x 3 = 0.12 >= 0.10


def test_streaming_stops_on_token_budget() -> None:
    budget = IterationBudget(50, max_tokens=250)
    provider = CostProvider(cost_per_call=0.0, tokens_per_call=100)
    done = asyncio.run(_drain(_loop(provider, budget), "go"))
    assert done.stop_reason == "budget_exhausted"
    assert provider.calls == 3  # 100 x 3 = 300 >= 250


async def _drain(loop: AgentLoop, text: str) -> Any:
    last = None
    async for ev in loop.astream_turn(text):
        last = ev
    return last  # the terminal AgentDone


# ── pairing coverage for the budget_exhausted break ─────────────────────────


def _tool_use_ids(session: Session) -> set[str]:
    """Collect all tool_use block IDs from the persisted session."""
    ids: set[str] = set()
    for m in session.messages:
        for b in m.blocks:
            if hasattr(b, "type") and b.type == "tool_use":
                ids.add(b.id)
    return ids


def _tool_result_ids(session: Session) -> set[str]:
    """Collect all tool_result block IDs from the persisted session."""
    ids: set[str] = set()
    for m in session.messages:
        for b in m.blocks:
            if hasattr(b, "type") and b.type == "tool_result":
                ids.add(b.tool_use_id)
    return ids


def _synthetic_budget_results(session: Session) -> list[Any]:
    """Return all tool_result blocks carrying data={'budget_exhausted': True}."""
    return [
        b
        for m in session.messages
        if m.role == "tool"
        for b in m.blocks
        if getattr(b, "data", None) and b.data.get("budget_exhausted")
    ]


def test_buffered_budget_pairs_dangling_tool_use() -> None:
    """When the budget-crossing iteration returns tool calls, the synthetic error
    tool_results pair them so the session has no dangling tool_use blocks."""
    budget = IterationBudget(50, max_cost_usd=0.10)
    provider = CostProvider(cost_per_call=0.04)
    loop = _loop(provider, budget)
    result = asyncio.run(loop.arun_turn("go"))
    assert result.stop_reason == "budget_exhausted"
    # Every tool_use in the session must have a matching tool_result.
    assert _tool_use_ids(loop.session) == _tool_result_ids(loop.session)
    # The final iteration's tool calls were never executed; they got synthetic
    # error results with is_error=True and data={'budget_exhausted': True}.
    synthetics = _synthetic_budget_results(loop.session)
    assert len(synthetics) >= 1  # at least the over-budget batch
    assert all(b.is_error for b in synthetics)
    assert all(b.data == {"budget_exhausted": True} for b in synthetics)


def test_streaming_budget_pairs_dangling_tool_use() -> None:
    """Streaming twin: same pairing guarantee as the buffered path."""
    budget = IterationBudget(50, max_cost_usd=0.10)
    provider = CostProvider(cost_per_call=0.04)
    loop = _loop(provider, budget)
    done = asyncio.run(_drain(loop, "go"))
    assert done.stop_reason == "budget_exhausted"
    assert _tool_use_ids(loop.session) == _tool_result_ids(loop.session)
    synthetics = _synthetic_budget_results(loop.session)
    assert len(synthetics) >= 1
    assert all(b.is_error for b in synthetics)
    assert all(b.data == {"budget_exhausted": True} for b in synthetics)


class TextOnlyOverBudgetProvider(Provider):
    """Returns tool calls on early iterations, then a text-only response on the
    iteration that crosses the budget -- so the guard fires without tool_calls."""

    def __init__(self) -> None:
        self.calls = 0

    async def acomplete(self, messages, *, system=None, tools=None, **kw) -> LLMResult:
        self.calls += 1
        if self.calls == 1:
            # First call: tool call, pushes budget to 0.05 (under 0.10 ceiling).
            from zakcode.providers.base import ToolCall

            return LLMResult(
                text="working",
                tool_calls=[ToolCall(id=f"c{self.calls}", name="noop", arguments={})],
                usage=Usage(total_tokens=50, cost_usd=0.05),
            )
        # Second call: text-only (no tool calls), pushes budget to 0.11 (over 0.10).
        return LLMResult(
            text="done",
            tool_calls=[],
            usage=Usage(total_tokens=50, cost_usd=0.06),
        )

    async def astream(self, messages, *, system=None, tools=None, **kw):
        import json

        from zakcode.providers.base import StreamDone, StreamTextDelta, StreamToolCallDelta, StreamUsage

        self.calls += 1
        if self.calls == 1:
            yield StreamToolCallDelta(
                index=0, id=f"c{self.calls}", name="noop", arguments_delta=json.dumps({})
            )
            yield StreamUsage(usage=Usage(total_tokens=50, cost_usd=0.05))
            yield StreamDone(finish_reason="tool_calls")
        else:
            yield StreamTextDelta(text="done")
            yield StreamUsage(usage=Usage(total_tokens=50, cost_usd=0.06))
            yield StreamDone(finish_reason="stop")

    def count_tokens(self, messages, *, system=None) -> int:
        return 0

    def capabilities(self) -> Capabilities:
        return Capabilities(supports_tools=True, context_window=8192)


def test_buffered_no_tool_calls_budget_stop_no_synthetic() -> None:
    """When the budget-crossing iteration returns NO tool calls, the guard is
    conditional: no synthetic tool_result message is injected."""
    budget = IterationBudget(50, max_cost_usd=0.10)
    provider = TextOnlyOverBudgetProvider()
    loop = _loop(provider, budget)
    result = asyncio.run(loop.arun_turn("go"))
    assert result.stop_reason == "budget_exhausted"
    # No synthetic budget_exhausted results should exist -- the final iteration
    # had no tool calls, so there was nothing to pair.
    synthetics = _synthetic_budget_results(loop.session)
    assert len(synthetics) == 0
    # The session should still be clean: all tool_use ids paired.
    assert _tool_use_ids(loop.session) == _tool_result_ids(loop.session)


def test_streaming_no_tool_calls_budget_stop_no_synthetic() -> None:
    """Streaming twin of the no-tool-calls guard test."""
    budget = IterationBudget(50, max_cost_usd=0.10)
    provider = TextOnlyOverBudgetProvider()
    loop = _loop(provider, budget)
    done = asyncio.run(_drain(loop, "go"))
    assert done.stop_reason == "budget_exhausted"
    synthetics = _synthetic_budget_results(loop.session)
    assert len(synthetics) == 0
    assert _tool_use_ids(loop.session) == _tool_result_ids(loop.session)


def test_facade_builds_budget_from_settings_without_delegation() -> None:
    """A plain Agent (no sub-agents) still gets a cost cap when configured."""
    import zakcode

    agent = zakcode.Agent(
        provider=CostProvider(cost_per_call=0.04),
        max_cost_usd=0.10,
        workspace_root=Path.cwd(),
    )
    result = agent.run_turn("go")
    assert result.stop_reason == "budget_exhausted"
