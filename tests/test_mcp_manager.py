"""Tests for MCP tool adaptation, qualified-name routing, and discovery (M5-2)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from zakcode.config import PermissionTier
from zakcode.mcp.client import McpCallResult, MCPClient, McpToolDef
from zakcode.mcp.manager import (
    MAX_TOOL_NAME_LEN,
    ExtensionManager,
    McpTool,
    parse_qualified_name,
    qualified_tool_name,
)
from zakcode.tools.base import ConcurrencyClass, ToolContext, ToolRegistry


class _FakeClient:
    """A structural McpClientProtocol with no process/network."""

    def __init__(
        self,
        *,
        tools: list[dict[str, Any]] | None = None,
        results: dict[str, McpCallResult] | None = None,
        fail: bool = False,
    ) -> None:
        self._tools = tools or []
        self._results = results or {}
        self._fail = fail
        self.closed = False
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def list_tools(self) -> list[McpToolDef]:
        if self._fail:
            raise RuntimeError("server down")
        return [
            McpToolDef(
                name=t["name"],
                description=t.get("description", ""),
                input_schema=t.get("inputSchema", {"type": "object", "properties": {}}),
            )
            for t in self._tools
        ]

    async def call_tool(self, name: str, arguments: dict[str, Any] | None = None) -> McpCallResult:
        self.calls.append((name, arguments or {}))
        if name in self._results:
            return self._results[name]
        return McpCallResult(content=[{"type": "text", "text": f"ran {name}"}], is_error=False)

    async def close(self) -> None:
        self.closed = True


class _InlineTransport:
    """A minimal in-memory MCP server transport (no cross-test-module import).

    Self-contained so the real :class:`MCPClient` can be driven through discovery
    without importing another test module (which pytest would dual-import, causing
    order-dependent failures).
    """

    def __init__(self, tools: list[dict[str, Any]]) -> None:
        self._tools = tools
        self._outbox: list[dict[str, Any]] = []

    async def start(self) -> None:
        return None

    async def send(self, message: dict[str, Any]) -> None:
        if "id" not in message:
            return  # notification
        rid = message["id"]
        method = message.get("method")
        if method == "initialize":
            self._outbox.append(
                {
                    "jsonrpc": "2.0",
                    "id": rid,
                    "result": {
                        "protocolVersion": "2025-06-18",
                        "capabilities": {"tools": {}},
                        "serverInfo": {"name": "inline", "version": "1"},
                    },
                }
            )
        elif method == "tools/list":
            self._outbox.append({"jsonrpc": "2.0", "id": rid, "result": {"tools": self._tools}})
        else:
            self._outbox.append(
                {"jsonrpc": "2.0", "id": rid, "error": {"code": -32601, "message": "no"}}
            )

    async def receive(self) -> dict[str, Any] | None:
        return self._outbox.pop(0) if self._outbox else None

    async def close(self) -> None:
        return None


def _ctx(tmp_path: Path) -> ToolContext:
    return ToolContext(workspace_root=tmp_path)


# ── qualified naming ─────────────────────────────────────────────────────────────


def test_qualified_tool_name_basic() -> None:
    assert qualified_tool_name("github", "create_issue") == "mcp__github__create_issue"


def test_qualified_tool_name_sanitizes_invalid_chars() -> None:
    name = qualified_tool_name("my.server", "do:it")
    assert name == "mcp__my_server__do_it"
    assert all(c.isalnum() or c in "_-" for c in name)


def test_qualified_tool_name_respects_64_char_limit() -> None:
    name = qualified_tool_name("server", "x" * 100)
    assert len(name) <= MAX_TOOL_NAME_LEN
    # Distinct long names stay distinct via the hash suffix.
    other = qualified_tool_name("server", "y" * 100)
    assert name != other


def test_parse_qualified_name_roundtrip() -> None:
    assert parse_qualified_name("mcp__github__create_issue") == ("github", "create_issue")


def test_parse_qualified_name_rejects_non_qualified() -> None:
    assert parse_qualified_name("read_file") is None
    assert parse_qualified_name("mcp__onlyserver") is None


# ── McpTool adapter ──────────────────────────────────────────────────────────────


async def test_mcp_tool_execute_roundtrip(tmp_path: Path) -> None:
    client = _FakeClient()
    tool = McpTool(
        server="srv",
        definition=McpToolDef(name="echo", description="echo it"),
        client=client,
    )
    assert tool.spec.name == "mcp__srv__echo"
    assert tool.spec.required_permission == PermissionTier.DANGER_FULL_ACCESS
    assert tool.spec.concurrency == ConcurrencyClass.NEVER_PARALLEL
    result = await tool.execute({"text": "hi"}, _ctx(tmp_path))
    assert result.is_error is False
    assert result.output == "ran echo"
    assert client.calls == [("echo", {"text": "hi"})]  # the REMOTE name was used


async def test_mcp_tool_execute_surfaces_error_result(tmp_path: Path) -> None:
    client = _FakeClient(
        results={"boom": McpCallResult(content=[{"type": "text", "text": "bad"}], is_error=True)}
    )
    tool = McpTool(server="srv", definition=McpToolDef(name="boom"), client=client)
    result = await tool.execute({}, _ctx(tmp_path))
    assert result.is_error is True
    assert result.output == "bad"


async def test_mcp_tool_execute_never_raises_on_client_failure(tmp_path: Path) -> None:
    class _Boom(_FakeClient):
        async def call_tool(
            self, name: str, arguments: dict[str, Any] | None = None
        ) -> McpCallResult:
            raise RuntimeError("transport exploded")

    tool = McpTool(server="srv", definition=McpToolDef(name="x"), client=_Boom())
    result = await tool.execute({}, _ctx(tmp_path))
    assert result.is_error is True
    assert "transport exploded" in result.output


# ── ExtensionManager discovery + registry routing ────────────────────────────────


async def test_discover_registers_qualified_tools() -> None:
    manager = ExtensionManager()
    manager.add_client("srv", _FakeClient(tools=[{"name": "alpha"}, {"name": "beta"}]))
    registry = ToolRegistry()
    report = await manager.discover_into(registry)
    assert set(report.registered) == {"mcp__srv__alpha", "mcp__srv__beta"}
    assert report.failed == {}
    assert "mcp__srv__alpha" in registry.names()
    # The MCP tool definitions are exposed to the model alongside builtins.
    schema_names = {d["function"]["name"] for d in registry.definitions()}
    assert {"mcp__srv__alpha", "mcp__srv__beta"} <= schema_names


async def test_discover_is_graceful_on_failed_server() -> None:
    manager = ExtensionManager()
    manager.add_client("good", _FakeClient(tools=[{"name": "ok"}]))
    manager.add_client("bad", _FakeClient(fail=True))
    registry = ToolRegistry()
    report = await manager.discover_into(registry)
    assert report.registered == ["mcp__good__ok"]  # the good server still discovered
    assert "bad" in report.failed
    assert "server down" in report.failed["bad"]


async def test_mcp_tool_dispatches_through_registry(tmp_path: Path) -> None:
    # The whole point: once registered, an MCP tool runs via the SAME registry.execute
    # path as a builtin — the loop cannot tell the difference.
    manager = ExtensionManager()
    client = _FakeClient(tools=[{"name": "search"}])
    manager.add_client("web", client)
    registry = ToolRegistry()
    await manager.discover_into(registry)
    result = await registry.execute("mcp__web__search", {"q": "zak"}, _ctx(tmp_path))
    assert result.is_error is False
    assert result.output == "ran search"
    assert client.calls == [("search", {"q": "zak"})]


async def test_aclose_closes_all_clients() -> None:
    manager = ExtensionManager()
    a, b = _FakeClient(), _FakeClient()
    manager.add_client("a", a)
    manager.add_client("b", b)
    await manager.aclose()
    assert a.closed and b.closed


async def test_discover_against_real_client() -> None:
    # Prove the *real* MCPClient (not just a fake) flows through discovery + routing,
    # driven over a self-contained inline transport (no cross-test-module import).
    transport = _InlineTransport(tools=[{"name": "ping", "description": "ping"}])
    manager = ExtensionManager()
    manager.add_client("real", MCPClient(transport))
    registry = ToolRegistry()
    report = await manager.discover_into(registry)
    assert report.registered == ["mcp__real__ping"]
