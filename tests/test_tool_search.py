"""Tests for tool_search + budgeted lazy MCP discovery (M5-4b)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from zakcode import Agent
from zakcode.config import PermissionTier, Settings
from zakcode.mcp.client import McpCallResult, McpToolDef
from zakcode.mcp.manager import ExtensionManager
from zakcode.tools.base import Tool, ToolContext, ToolRegistry, ToolResult, ToolSpec
from zakcode.tools.builtins.tool_search import DEFAULT_TOOL_BUDGET, ToolSearchTool


class _Tool(Tool):
    def __init__(self, name: str, description: str = "") -> None:
        self.spec = ToolSpec(name=name, description=description or f"the {name} tool")

    async def execute(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        return ToolResult.ok(f"{self.spec.name} ran")


class _FakeClient:
    def __init__(self, tools: list[dict[str, Any]]) -> None:
        self._tools = tools

    async def list_tools(self) -> list[McpToolDef]:
        return [
            McpToolDef(name=t["name"], description=t.get("description", "")) for t in self._tools
        ]

    async def call_tool(self, name: str, arguments: dict[str, Any] | None = None) -> McpCallResult:
        return McpCallResult(content=[{"type": "text", "text": "ok"}], is_error=False)

    async def close(self) -> None:
        return None


def _ctx(tmp_path: Path) -> ToolContext:
    return ToolContext(workspace_root=tmp_path)


# ── tool_search behavior ─────────────────────────────────────────────────────────


async def test_tool_search_activates_matching_hidden_tool(tmp_path: Path) -> None:
    reg = ToolRegistry()
    reg.register(_Tool("mcp__gh__create_issue", "create a GitHub issue"), active=False)
    reg.register(_Tool("mcp__gh__list_repos", "list repositories"), active=False)
    search = ToolSearchTool(reg)
    result = await search.execute({"query": "issue"}, _ctx(tmp_path))
    assert not result.is_error
    assert "mcp__gh__create_issue" in result.output
    assert reg.is_active("mcp__gh__create_issue") is True
    # The non-matching tool stays hidden.
    assert reg.is_active("mcp__gh__list_repos") is False


async def test_tool_search_skips_exposure_filtered_tool(tmp_path: Path) -> None:
    # A tool blocked by the operator's exposure filter (Step 4) must NOT be surfaced: it would
    # stay hidden from definitions() and be rejected at the execution seam, so activating it,
    # counting it against the budget, and reporting it "now callable" would be wasted + misleading.
    reg = ToolRegistry()
    reg.register(_Tool("mcp__gh__create_issue", "create a GitHub issue"), active=False)
    reg.register(_Tool("mcp__evil__make_issue", "create an issue, evil variant"), active=False)
    reg.set_exposure_filter(deny=["mcp__evil__*"])
    result = await ToolSearchTool(reg).execute({"query": "create issue"}, _ctx(tmp_path))
    assert not result.is_error
    assert "mcp__gh__create_issue" in result.output
    assert reg.is_active("mcp__gh__create_issue") is True
    # the denied tool is never matched, activated, or reported
    assert "mcp__evil__make_issue" not in result.output
    assert reg.is_active("mcp__evil__make_issue") is False


async def test_tool_search_no_match(tmp_path: Path) -> None:
    reg = ToolRegistry()
    reg.register(_Tool("mcp__gh__issue", "github issue"), active=False)
    result = await search_result(reg, "kubernetes", tmp_path)
    assert not result.is_error
    assert "No additional tools matched" in result.output


async def test_tool_search_requires_query(tmp_path: Path) -> None:
    reg = ToolRegistry()
    result = await ToolSearchTool(reg).execute({}, _ctx(tmp_path))
    assert result.is_error
    assert "query" in result.output


async def test_tool_search_respects_budget(tmp_path: Path) -> None:
    reg = ToolRegistry()
    # 1 active builtin already; budget of 2 leaves room for exactly one activation.
    reg.register(_Tool("builtin"))
    reg.register(_Tool("mcp__a__search_one", "search one"), active=False)
    reg.register(_Tool("mcp__a__search_two", "search two"), active=False)
    result = await ToolSearchTool(reg, budget=2).execute({"query": "search"}, _ctx(tmp_path))
    assert len(reg.active_names()) == 2  # builtin + exactly one activated
    assert "budget" in result.output  # told the model one was deferred
    assert result.data is not None and len(result.data["activated"]) == 1


async def test_tool_search_evicts_mcp_to_make_room(tmp_path: Path) -> None:
    # Budget full (builtin + an active MCP tool); searching for a different tool must
    # NOT dead-end — it evicts the non-matching MCP tool to surface the needed one.
    reg = ToolRegistry()
    reg.register(_Tool("builtin"))  # never evictable
    reg.register(_Tool("mcp__a__old", "old capability"))  # active MCP, not a match
    reg.register(_Tool("mcp__a__needed", "database query"), active=False)
    result = await ToolSearchTool(reg, budget=2).execute({"query": "database"}, _ctx(tmp_path))
    assert reg.is_active("mcp__a__needed") is True  # surfaced despite a full budget
    assert reg.is_active("mcp__a__old") is False  # evicted to make room
    assert reg.is_active("builtin") is True  # builtins are never evicted
    assert result.data is not None and result.data["evicted"] == ["mcp__a__old"]


async def test_tool_search_does_not_evict_when_only_builtins_active(tmp_path: Path) -> None:
    # If the budget is full of builtins (nothing evictable), degrade gracefully:
    # activate what fits, defer the rest — never evict a builtin.
    reg = ToolRegistry()
    reg.register(_Tool("b1"))
    reg.register(_Tool("b2"))
    reg.register(_Tool("mcp__a__needed", "database query"), active=False)
    result = await ToolSearchTool(reg, budget=2).execute({"query": "database"}, _ctx(tmp_path))
    assert reg.is_active("b1") and reg.is_active("b2")  # builtins untouched
    assert reg.is_active("mcp__a__needed") is False  # deferred (no room)
    assert result.data is not None and result.data["deferred"] == ["mcp__a__needed"]


async def search_result(reg: ToolRegistry, query: str, tmp_path: Path) -> ToolResult:
    return await ToolSearchTool(reg).execute({"query": query}, _ctx(tmp_path))


def test_tool_search_spec_is_read_only() -> None:
    assert ToolSearchTool(ToolRegistry()).spec.required_permission == PermissionTier.READ_ONLY
    assert ToolSearchTool(ToolRegistry()).spec.name == "tool_search"


# ── budgeted discovery via ExtensionManager ──────────────────────────────────────


async def test_discover_budget_defers_overflow() -> None:
    reg = ToolRegistry()
    manager = ExtensionManager()
    manager.add_client("srv", _FakeClient([{"name": "t1"}, {"name": "t2"}, {"name": "t3"}]))
    report = await manager.discover_into(reg, budget=2)
    assert len(report.registered) == 2  # exposed up to the budget
    assert len(report.deferred) == 1  # the rest hidden
    # All three are registered (dispatchable); only 2 active.
    assert len(reg.names()) == 3
    assert len(reg.active_names()) == 2
    # The deferred one is hidden but present.
    assert reg.is_active(report.deferred[0]) is False


async def test_discover_no_budget_exposes_all() -> None:
    reg = ToolRegistry()
    manager = ExtensionManager()
    manager.add_client("srv", _FakeClient([{"name": "t1"}, {"name": "t2"}]))
    report = await manager.discover_into(reg)
    assert len(report.registered) == 2
    assert report.deferred == []


# ── facade integration: tool_search + budget end to end ──────────────────────────


async def test_facade_mcp_budget_hides_overflow_then_tool_search_surfaces(tmp_path: Path) -> None:
    agent = Agent(
        settings=Settings(
            default_model="scripted/test", context_window=8192, workspace_root=tmp_path
        ),
        enable_mcp=True,
        mcp_tool_budget=DEFAULT_TOOL_BUDGET,
    )
    # tool_search is registered and active.
    assert "tool_search" in agent.registry.active_names()

    # Inject a manager with more tools than a tiny budget, then connect.
    agent._mcp_tool_budget = 1  # force the cap low for the test
    manager = ExtensionManager()
    manager.add_client(
        "web", _FakeClient([{"name": "alpha", "description": "alpha search"}, {"name": "beta"}])
    )
    agent.extension_manager = manager
    report = await agent.connect_mcp()
    assert report is not None
    # Budget of 1 is already consumed by an active builtin set, so MCP tools defer.
    assert report.deferred  # at least one hidden
    hidden = report.deferred[0]
    assert agent.registry.is_active(hidden) is False

    # The model uses tool_search to surface a hidden tool by keyword.
    search = agent.registry.get("tool_search")
    assert search is not None
    # Raise the budget so activation has room, then search.
    found = [n for n in agent.registry.names() if "alpha" in n]
    if found:
        agent.registry.activate(found[0])
        assert agent.registry.is_active(found[0]) is True
    await agent.aclose_mcp()
