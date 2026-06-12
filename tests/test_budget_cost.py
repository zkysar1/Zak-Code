"""Cost/token budget stop condition (parity #4).

The shared IterationBudget gains optional cost/token ceilings bounding cumulative spend
across a turn and its whole sub-agent tree; once crossed the loop stops with
stop_reason="budget_exhausted" (a non-vetoable hard bound, like max_iterations). Hermetic.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

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
