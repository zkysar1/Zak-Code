"""Tests for the loop's read-only batch parallelism and shared-budget refunds."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from zakcode.agent.budget import IterationBudget
from zakcode.config import PermissionTier
from zakcode.evals.harness import ScriptedProvider, call_tool, make_agent, reply
from zakcode.providers.base import LLMResult, ToolCall
from zakcode.tools.base import ConcurrencyClass, Tool, ToolContext, ToolResult, ToolSpec


class _ConcurrencyProbe(Tool):
    """A READ_ONLY_SAFE tool that records the peak number of overlapping executions."""

    spec = ToolSpec(
        name="probe",
        description="concurrency probe",
        required_permission=PermissionTier.READ_ONLY,
        concurrency=ConcurrencyClass.READ_ONLY_SAFE,
    )

    def __init__(self) -> None:
        self.active = 0
        self.peak = 0

    async def execute(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        self.active += 1
        self.peak = max(self.peak, self.active)
        await asyncio.sleep(0.02)  # hold the slot so overlaps are observable
        self.active -= 1
        return ToolResult.ok("ok")


def _batch(*calls: ToolCall) -> LLMResult:
    return LLMResult(tool_calls=list(calls), finish_reason="tool_calls")


async def test_all_read_only_batch_runs_concurrently(tmp_path: Path) -> None:
    probe = _ConcurrencyProbe()
    calls = [ToolCall(id=str(i), name="probe", arguments={}) for i in range(3)]
    provider = ScriptedProvider(script=[_batch(*calls), reply("done")])
    agent = make_agent(
        provider, workspace_root=str(tmp_path), permission_mode="allow", extra_tools=[probe]
    )

    await agent.arun_turn("read three things")

    assert probe.peak == 3  # all three overlapped → executed in parallel


async def test_mixed_batch_runs_sequentially(tmp_path: Path) -> None:
    probe = _ConcurrencyProbe()
    # A write_file in the batch makes it not-all-read-only → sequential execution.
    calls = [
        ToolCall(id="1", name="probe", arguments={}),
        ToolCall(id="2", name="probe", arguments={}),
        ToolCall(id="3", name="write_file", arguments={"path": "x.txt", "content": "y"}),
    ]
    provider = ScriptedProvider(script=[_batch(*calls), reply("done")])
    agent = make_agent(
        provider, workspace_root=str(tmp_path), permission_mode="allow", extra_tools=[probe]
    )

    await agent.arun_turn("read then write")

    assert probe.peak == 1  # no overlap → ran one at a time
    assert (tmp_path / "x.txt").read_text() == "y"


# ── shared-budget refunds via the loop ────────────────────────────────────────


def _agent_with_budget(provider: ScriptedProvider, tmp_path: Path, budget: IterationBudget):
    from zakcode import Agent
    from zakcode.permissions import PermissionPolicy

    return Agent(
        provider=provider,
        permission_policy=PermissionPolicy("allow"),
        budget=budget,
        default_model="scripted/test",
        context_window=8192,
        workspace_root=str(tmp_path),
    )


async def test_empty_completion_refunds_budget(tmp_path: Path) -> None:
    budget = IterationBudget(5)
    agent = _agent_with_budget(ScriptedProvider(script=[reply("")]), tmp_path, budget)
    await agent.arun_turn("hi")
    # The single iteration was a truly empty completion → its unit was refunded.
    assert budget.consumed == 0


async def test_normal_completion_does_not_refund(tmp_path: Path) -> None:
    budget = IterationBudget(5)
    agent = _agent_with_budget(
        ScriptedProvider(script=[reply("here is the answer")]), tmp_path, budget
    )
    await agent.arun_turn("hi")
    assert budget.consumed == 1  # produced text → counts


async def test_denied_batch_refunds_budget(tmp_path: Path) -> None:
    from zakcode import Agent
    from zakcode.permissions import PermissionPolicy

    budget = IterationBudget(5)
    # deny mode → the write is blocked (no work done) on iter 1; iter 2 completes.
    provider = ScriptedProvider(
        script=[call_tool("write_file", {"path": "x.txt", "content": "y"}), reply("ok")]
    )
    agent = Agent(
        provider=provider,
        permission_policy=PermissionPolicy("deny"),
        budget=budget,
        default_model="scripted/test",
        context_window=8192,
        workspace_root=str(tmp_path),
    )
    await agent.arun_turn("write a file")
    # iter1 (denied) refunded, iter2 (reply) counts → net 1, not 2.
    assert budget.consumed == 1


# ── concurrency safety: READ_ONLY_SAFE must also be READ_ONLY tier ─────────────


def test_toolspec_rejects_read_only_safe_with_write_tier() -> None:
    import pytest
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        ToolSpec(
            name="bad",
            description="mis-declared",
            required_permission=PermissionTier.WORKSPACE_WRITE,
            concurrency=ConcurrencyClass.READ_ONLY_SAFE,
        )


def test_loop_guard_excludes_non_read_only_tier_from_parallel(tmp_path: Path) -> None:
    # Even if a spec bypassed validation (model_construct), the loop's own guard keeps
    # a non-READ_ONLY tool off the parallel path.
    from zakcode import Agent
    from zakcode.permissions import PermissionPolicy

    sneaky = ToolSpec.model_construct(
        name="sneaky",
        description="bypassed validation",
        parameters={"type": "object", "properties": {}},
        required_permission=PermissionTier.WORKSPACE_WRITE,
        concurrency=ConcurrencyClass.READ_ONLY_SAFE,
    )

    class _Sneaky(Tool):
        spec = sneaky

        async def execute(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
            return ToolResult.ok("ok")

    agent = Agent(
        provider=ScriptedProvider(script=[reply("ok")]),
        permission_policy=PermissionPolicy("allow"),
        default_model="scripted/test",
        context_window=8192,
        workspace_root=str(tmp_path),
    )
    call = ToolCall(id="1", name="sneaky", arguments={})
    agent.registry.register(_Sneaky())
    assert agent.loop._is_read_only_safe(call) is False  # tier check keeps it sequential
