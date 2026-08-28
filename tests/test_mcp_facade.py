"""Tests for MCP wiring into the Agent facade (M5-3b).

Constructing an Agent is side-effect-free (no process/network), so these build real
Agents with a scripted model string. Discovery is exercised with an injected
:class:`ExtensionManager` holding fake clients, so nothing is actually spawned.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from zakcode import Agent
from zakcode.config import Settings
from zakcode.mcp.client import McpCallResult, McpToolDef
from zakcode.mcp.config import McpServerConfig
from zakcode.mcp.manager import ExtensionManager, build_extension_manager


class _FakeClient:
    def __init__(self, *, tools: list[dict[str, Any]] | None = None) -> None:
        self._tools = tools or []
        self.closed = False

    async def list_tools(self) -> list[McpToolDef]:
        return [
            McpToolDef(name=t["name"], description=t.get("description", "")) for t in self._tools
        ]

    async def call_tool(self, name: str, arguments: dict[str, Any] | None = None) -> McpCallResult:
        return McpCallResult(content=[{"type": "text", "text": f"ran {name}"}], is_error=False)

    async def close(self) -> None:
        self.closed = True


def _settings(tmp_path: Path) -> Settings:
    return Settings(default_model="scripted/test", context_window=8192, workspace_root=tmp_path)


# ── build_extension_manager (pure: constructs transports, never spawns) ───────────


def test_build_manager_wires_enabled_servers(monkeypatch: Any) -> None:
    monkeypatch.setenv("TK", "v")
    servers = [
        McpServerConfig(name="a", command="npx", args=["x"], env={"T": "${TK}"}),
        McpServerConfig(name="b", command="uvx"),
    ]
    manager, errors = build_extension_manager(servers)
    assert set(manager.server_names) == {"a", "b"}
    assert errors == {}


def test_build_manager_skips_disabled() -> None:
    servers = [
        McpServerConfig(name="on", command="npx"),
        McpServerConfig(name="off", command="npx", enabled=False),
    ]
    manager, errors = build_extension_manager(servers)
    assert manager.server_names == ["on"]


def test_build_manager_collects_config_errors_without_raising() -> None:
    servers = [
        McpServerConfig(name="good", command="npx"),
        McpServerConfig(name="nocmd", command=None),  # missing command
    ]
    manager, errors = build_extension_manager(servers)
    assert manager.server_names == ["good"]
    assert "nocmd" in errors


def test_build_manager_enforces_allowlist() -> None:
    servers = [McpServerConfig(name="danger", command="rm", args=["-rf", "/"])]
    manager, errors = build_extension_manager(servers, allowlist=["npx", "uvx"])
    assert manager.server_names == []
    assert "danger" in errors


# ── Agent facade ─────────────────────────────────────────────────────────────────


def test_agent_without_mcp_has_no_manager(tmp_path: Path) -> None:
    agent = Agent(settings=_settings(tmp_path))
    assert agent.extension_manager is None
    assert agent.mcp_config_errors == {}


async def test_agent_connect_mcp_returns_none_when_disabled(tmp_path: Path) -> None:
    agent = Agent(settings=_settings(tmp_path))
    assert await agent.connect_mcp() is None


def test_agent_enable_mcp_builds_manager_without_spawning(tmp_path: Path) -> None:
    # A valid server config builds a (not-yet-started) client; nothing is spawned.
    agent = Agent(
        settings=_settings(tmp_path),
        enable_mcp=True,
        mcp_servers=[McpServerConfig(name="srv", command="npx")],
    )
    assert agent.extension_manager is not None
    assert agent.extension_manager.server_names == ["srv"]
    assert agent.mcp_config_errors == {}


def test_agent_enable_mcp_records_config_errors(tmp_path: Path) -> None:
    agent = Agent(
        settings=_settings(tmp_path),
        enable_mcp=True,
        mcp_servers=[McpServerConfig(name="bad", command=None)],
    )
    assert agent.extension_manager is not None
    assert "bad" in agent.mcp_config_errors


async def test_agent_connect_mcp_registers_tools_into_registry(tmp_path: Path) -> None:
    # Inject a manager with a fake client so connect spawns nothing but still
    # exercises the real discover_into path that mutates the agent's registry.
    agent = Agent(settings=_settings(tmp_path))
    manager = ExtensionManager()
    manager.add_client("web", _FakeClient(tools=[{"name": "search"}, {"name": "fetch"}]))
    agent.extension_manager = manager

    report = await agent.connect_mcp()
    assert report is not None
    assert set(report.registered) == {"mcp__web__search", "mcp__web__fetch"}
    # The tools are now in the SAME registry the loop uses.
    assert "mcp__web__search" in agent.registry.names()
    assert agent.mcp_report is report
    await agent.aclose_mcp()


async def test_agent_connect_mcp_is_graceful_on_failed_server(tmp_path: Path) -> None:
    class _BadClient(_FakeClient):
        async def list_tools(self) -> list[McpToolDef]:
            raise RuntimeError("unreachable")

    agent = Agent(settings=_settings(tmp_path))
    manager = ExtensionManager()
    manager.add_client("ok", _FakeClient(tools=[{"name": "t"}]))
    manager.add_client("bad", _BadClient())
    agent.extension_manager = manager

    report = await agent.connect_mcp()
    assert report is not None
    assert report.registered == ["mcp__ok__t"]
    assert "bad" in report.failed
