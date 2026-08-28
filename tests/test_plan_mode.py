"""Plan Mode tests (M4-5): the planner's write tools are absent from its schema.

The M4 exit criterion is structural, not behavioral: a planner sub-agent must be
*unable to call* write tools because they are not in the tool schema it is handed,
not merely refused at runtime. These tests pin that down at the registry/schema
level and through the facade.
"""

from __future__ import annotations

from pathlib import Path
from typing import cast

from zakcode import Agent
from zakcode.agent.budget import IterationBudget
from zakcode.agent.subagent import GENERAL_PURPOSE, PLAN, READ_ONLY_TOOLS, SubAgentRunner
from zakcode.config import Settings
from zakcode.providers.base import Provider
from zakcode.tools.builtins.default_registry import default_registry

_WRITE_TOOLS = {"write_file", "edit_file", "bash"}


def _runner(tmp_path: Path) -> SubAgentRunner:
    # These tests inspect registries/schemas only; no turn is run, so the stub
    # provider is never invoked.
    return SubAgentRunner(
        provider=cast(Provider, object()),
        registry=default_registry(),
        settings=Settings(
            default_model="scripted/test", context_window=8192, workspace_root=tmp_path
        ),
        budget=IterationBudget(10),
        workspace_root=tmp_path,
    )


def test_plan_definition_is_read_only() -> None:
    # The planner gets the read-only tools plus update_plan (itself READ_ONLY — it only touches the
    # in-memory plan), so it can structure a plan but still cannot edit the workspace.
    assert PLAN.allowed_tools == [*READ_ONLY_TOOLS, "update_plan"]
    assert _WRITE_TOOLS.isdisjoint(set(PLAN.allowed_tools))
    assert PLAN.system_suffix and "PLAN MODE" in PLAN.system_suffix


def test_planner_registry_omits_write_tools(tmp_path: Path) -> None:
    planner_registry = _runner(tmp_path).child_registry(PLAN)
    names = set(planner_registry.names())
    assert set(READ_ONLY_TOOLS) <= names  # read-only tools present…
    assert _WRITE_TOOLS.isdisjoint(names)  # …and every write tool absent entirely


def test_planner_schema_offers_no_write_tools(tmp_path: Path) -> None:
    # The definitions sent to the model (the actual schema) contain no write tool.
    defs = _runner(tmp_path).child_registry(PLAN).definitions()
    schema_names = {d["function"]["name"] for d in defs}
    assert _WRITE_TOOLS.isdisjoint(schema_names)
    assert "read_file" in schema_names


def test_general_purpose_still_has_write_tools(tmp_path: Path) -> None:
    names = set(_runner(tmp_path).child_registry(GENERAL_PURPOSE).names())
    assert names >= _WRITE_TOOLS


def test_facade_exposes_plan_subagent_type(tmp_path: Path) -> None:
    agent = Agent(
        settings=Settings(
            default_model="scripted/test", context_window=8192, workspace_root=tmp_path
        ),
        enable_subagents=True,
    )
    spawner = agent.loop.spawner
    assert spawner is not None
    assert "plan" in spawner.available_types()
    assert "general-purpose" in spawner.available_types()
    assert spawner.default_type() == "general-purpose"
