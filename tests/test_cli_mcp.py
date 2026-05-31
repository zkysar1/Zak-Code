"""Tests for the /mcp CLI command and inspect_servers (M5-3c).

The command helper is exercised directly with a captured console and injected
fake clients, so nothing is spawned.
"""

from __future__ import annotations

from io import StringIO
from pathlib import Path
from typing import Any

from rich.console import Console

from zakcode import Agent
from zakcode.cli import _render_mcp
from zakcode.config import Settings
from zakcode.mcp.client import McpCallResult, McpToolDef
from zakcode.mcp.config import McpServerConfig
from zakcode.mcp.manager import ExtensionManager, inspect_servers


class _FakeClient:
    def __init__(self, *, tools: list[dict[str, Any]] | None = None) -> None:
        self._tools = tools or []

    async def list_tools(self) -> list[McpToolDef]:
        return [McpToolDef(name=t["name"]) for t in self._tools]

    async def call_tool(self, name: str, arguments: dict[str, Any] | None = None) -> McpCallResult:
        return McpCallResult(content=[{"type": "text", "text": "x"}], is_error=False)

    async def close(self) -> None:
        return None


def _console() -> tuple[Console, StringIO]:
    buf = StringIO()
    return Console(file=buf, width=100), buf


def _agent(tmp_path: Path) -> Agent:
    return Agent(settings=Settings(default_model="scripted/test", workspace_root=tmp_path))


def test_render_mcp_not_enabled(tmp_path: Path) -> None:
    console, buf = _console()
    _render_mcp(console, _agent(tmp_path), "")
    assert "not enabled" in buf.getvalue()


def test_render_mcp_list_no_servers(tmp_path: Path) -> None:
    agent = _agent(tmp_path)
    agent.extension_manager = ExtensionManager()  # enabled but empty
    console, buf = _console()
    _render_mcp(console, agent, "list")
    assert "no MCP servers configured" in buf.getvalue()


def test_render_mcp_list_shows_servers(tmp_path: Path) -> None:
    agent = _agent(tmp_path)
    manager = ExtensionManager()
    manager.add_client("github", _FakeClient(tools=[{"name": "issues"}]))
    agent.extension_manager = manager
    console, buf = _console()
    _render_mcp(console, agent, "")
    out = buf.getvalue()
    assert "github" in out
    assert "/mcp connect" in out  # hint to connect


def test_render_mcp_connect_registers_tools(tmp_path: Path) -> None:
    agent = _agent(tmp_path)
    manager = ExtensionManager()
    manager.add_client("web", _FakeClient(tools=[{"name": "search"}]))
    agent.extension_manager = manager
    console, buf = _console()
    _render_mcp(console, agent, "connect")
    out = buf.getvalue()
    assert "registered 1 tool" in out
    assert "mcp__web__search" in out
    # the tool is now in the live registry
    assert "mcp__web__search" in agent.registry.names()


# ── inspect_servers (the /mcp backing probe) ─────────────────────────────────────


async def test_inspect_servers_reports_config_error() -> None:
    # A server missing its command is a config error — never spawned.
    inspections = await inspect_servers([McpServerConfig(name="bad", command=None)])
    assert len(inspections) == 1
    assert inspections[0].name == "bad"
    assert inspections[0].status == "config_error"
    assert inspections[0].error


async def test_inspect_servers_blocked_by_allowlist() -> None:
    inspections = await inspect_servers(
        [McpServerConfig(name="danger", command="rm", args=["-rf", "/"])],
        allowlist=["npx"],
    )
    assert inspections[0].status == "config_error"


async def test_inspect_servers_skips_disabled() -> None:
    inspections = await inspect_servers([McpServerConfig(name="off", command="npx", enabled=False)])
    assert inspections == []
