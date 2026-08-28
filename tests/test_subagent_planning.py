"""Tests for the planner-role decomposition (R4) and structured sub-agent handoff (R6)."""

from __future__ import annotations

from typing import Any

from zakcode.agent.budget import IterationBudget
from zakcode.agent.subagent import GENERAL_PURPOSE, PLAN, SubAgentRunner
from zakcode.config import Settings
from zakcode.messages import Message
from zakcode.providers.base import Capabilities, LLMResult, Provider
from zakcode.tools.builtins.default_registry import default_registry


class _Stub(Provider):
    async def acomplete(self, messages, *, system=None, tools=None, **kw: Any) -> LLMResult:
        return LLMResult(text="")

    def count_tokens(self, messages: list[Message], *, system: str | None = None) -> int:
        return 0

    def capabilities(self) -> Capabilities:
        return Capabilities(context_window=8192)


def _runner() -> SubAgentRunner:
    return SubAgentRunner(
        provider=_Stub(),
        registry=default_registry(),
        settings=Settings(),
        budget=IterationBudget(50),
    )


# ── R4: the read-only planner can produce a STRUCTURED plan, on the planner model ──


def test_plan_subagent_can_use_update_plan_and_stays_read_only() -> None:
    assert "update_plan" in PLAN.allowed_tools  # can structure the plan, not just prose
    # update_plan is READ_ONLY, so Plan Mode is still schema-enforced read-only on the workspace.
    assert "write_file" not in PLAN.allowed_tools and "bash" not in PLAN.allowed_tools
    assert PLAN.system_suffix is not None and "update_plan" in PLAN.system_suffix


def test_planner_role_model_routing_is_expressible() -> None:
    # Mirrors the facade: model_roles['planner'] flows onto the plan sub-agent definition.
    routed = PLAN.model_copy(update={"model": "cheap/planner"})
    assert routed.model == "cheap/planner"
    assert "update_plan" in routed.allowed_tools  # still structured-planning capable


# ── R6: every sub-agent gets a self-contained structured-handoff instruction ──


def test_general_subagent_gets_the_handoff_instruction() -> None:
    builder = _runner().prompt_builder_for(GENERAL_PURPOSE)
    assert builder.extra_instructions is not None
    assert "Handoff:" in builder.extra_instructions


def test_plan_subagent_keeps_its_suffix_and_adds_the_handoff() -> None:
    builder = _runner().prompt_builder_for(PLAN)
    assert builder.extra_instructions is not None
    assert "PLAN MODE" in builder.extra_instructions  # its own specialization survives
    assert "Handoff:" in builder.extra_instructions  # and the shared handoff is appended
