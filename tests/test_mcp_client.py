"""Tests for the hand-rolled MCP client + stdio transport (M5-1).

Most tests drive the client against an in-memory ``FakeServer`` transport (no
process, no network) so they are fast and deterministic. One integration test
spawns a real Python subprocess speaking newline-delimited JSON-RPC over stdio, to
prove :class:`StdioTransport`'s real framing end to end.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

from zakcode.mcp.client import McpCallResult, MCPClient, McpToolDef
from zakcode.mcp.jsonrpc import JSONRPCError, MCPProtocolError
from zakcode.mcp.transport import StdioTransport


class FakeServer:
    """An in-memory MCP server implementing the Transport protocol for tests."""

    def __init__(
        self,
        *,
        tools: list[dict[str, Any]] | None = None,
        call_results: dict[str, dict[str, Any]] | None = None,
        error_on: str | None = None,
    ) -> None:
        self.tools = tools if tools is not None else []
        self.call_results = call_results or {}
        self.error_on = error_on  # a method name to answer with a JSON-RPC error
        self.sent: list[dict[str, Any]] = []
        self.initialize_count = 0
        self.started = False
        self.closed = False
        self._outbox: list[dict[str, Any]] = []
        #: extra messages (e.g. notifications) to emit before the next response
        self.pending_notifications: list[dict[str, Any]] = []

    async def start(self) -> None:
        self.started = True

    async def send(self, message: dict[str, Any]) -> None:
        self.sent.append(message)
        if "id" not in message:
            return  # a notification from the client — nothing to answer
        rid = message["id"]
        method = message.get("method")
        # Drain any queued server-initiated notifications ahead of the response.
        self._outbox.extend(self.pending_notifications)
        self.pending_notifications = []
        if self.error_on is not None and method == self.error_on:
            self._outbox.append(
                {"jsonrpc": "2.0", "id": rid, "error": {"code": -32000, "message": "boom"}}
            )
            return
        if method == "initialize":
            self.initialize_count += 1
            self._outbox.append(
                {
                    "jsonrpc": "2.0",
                    "id": rid,
                    "result": {
                        "protocolVersion": "2025-06-18",
                        "capabilities": {"tools": {}},
                        "serverInfo": {"name": "fake", "version": "1.0"},
                    },
                }
            )
        elif method == "tools/list":
            self._outbox.append({"jsonrpc": "2.0", "id": rid, "result": {"tools": self.tools}})
        elif method == "tools/call":
            params = message.get("params", {})
            name = params.get("name")
            res = self.call_results.get(
                name, {"content": [{"type": "text", "text": f"ran {name}"}], "isError": False}
            )
            self._outbox.append({"jsonrpc": "2.0", "id": rid, "result": res})
        else:
            self._outbox.append(
                {"jsonrpc": "2.0", "id": rid, "error": {"code": -32601, "message": "no method"}}
            )

    async def receive(self) -> dict[str, Any] | None:
        if not self._outbox:
            return None
        return self._outbox.pop(0)

    async def close(self) -> None:
        self.closed = True


async def test_initialize_is_idempotent() -> None:
    server = FakeServer()
    client = MCPClient(server)
    await client.initialize()
    await client.initialize()
    assert client.initialized is True
    assert server.initialize_count == 1  # handshake runs exactly once
    # The protocol's initialized acknowledgement was sent.
    assert any(m.get("method") == "notifications/initialized" for m in server.sent)
    assert client.server_info == {"name": "fake", "version": "1.0"}


async def test_list_tools_parses_definitions() -> None:
    server = FakeServer(
        tools=[
            {
                "name": "search",
                "description": "search the web",
                "inputSchema": {"type": "object", "properties": {"q": {"type": "string"}}},
            }
        ]
    )
    client = MCPClient(server)
    defs = await client.list_tools()
    assert len(defs) == 1
    assert isinstance(defs[0], McpToolDef)
    assert defs[0].name == "search"
    assert defs[0].description == "search the web"
    assert defs[0].input_schema["properties"]["q"]["type"] == "string"


async def test_list_tools_skips_malformed_entries() -> None:
    server = FakeServer(tools=[{"no_name": True}, {"name": "ok"}])
    client = MCPClient(server)
    defs = await client.list_tools()
    assert [d.name for d in defs] == ["ok"]


async def test_call_tool_roundtrip() -> None:
    server = FakeServer()
    client = MCPClient(server)
    result = await client.call_tool("echo", {"text": "hi"})
    assert isinstance(result, McpCallResult)
    assert result.is_error is False
    assert result.text == "ran echo"
    # The arguments were forwarded.
    call = next(m for m in server.sent if m.get("method") == "tools/call")
    assert call["params"]["arguments"] == {"text": "hi"}


async def test_call_tool_error_result_is_flagged() -> None:
    server = FakeServer(
        call_results={"boom": {"content": [{"type": "text", "text": "nope"}], "isError": True}}
    )
    client = MCPClient(server)
    result = await client.call_tool("boom")
    assert result.is_error is True
    assert result.text == "nope"


async def test_jsonrpc_error_response_raises() -> None:
    server = FakeServer(error_on="tools/list")
    client = MCPClient(server)
    with pytest.raises(JSONRPCError) as exc:
        await client.list_tools()
    assert exc.value.code == -32000


async def test_interleaved_notification_is_skipped() -> None:
    server = FakeServer(tools=[{"name": "t"}])
    # The server emits a log notification before answering tools/list.
    server.pending_notifications = [
        {"jsonrpc": "2.0", "method": "notifications/message", "params": {"level": "info"}}
    ]
    client = MCPClient(server)
    defs = await client.list_tools()
    assert [d.name for d in defs] == ["t"]  # the notification did not derail the response


class _SilentTransport:
    """A transport that never returns a message — simulates a dead/closed server."""

    async def start(self) -> None:
        return None

    async def send(self, message: dict[str, Any]) -> None:
        return None

    async def receive(self) -> dict[str, Any] | None:
        return None  # immediate EOF

    async def close(self) -> None:
        return None


async def test_eof_during_request_raises_protocol_error() -> None:
    client = MCPClient(_SilentTransport())
    with pytest.raises(MCPProtocolError):
        await client.initialize()


# ── integration: a real subprocess over real stdio framing ───────────────────────

_FAKE_SERVER_SCRIPT = """\
import sys, json

def main() -> None:
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        msg = json.loads(line)
        mid = msg.get("id")
        method = msg.get("method")
        if method == "notifications/initialized":
            continue  # notification: no response
        if method == "initialize":
            result = {
                "protocolVersion": "2025-06-18",
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "itest", "version": "1"},
            }
        elif method == "tools/list":
            result = {"tools": [{"name": "echo", "description": "echo",
                                 "inputSchema": {"type": "object",
                                                 "properties": {"text": {"type": "string"}}}}]}
        elif method == "tools/call":
            args = (msg.get("params") or {}).get("arguments") or {}
            result = {"content": [{"type": "text", "text": args.get("text", "")}], "isError": False}
        else:
            sys.stdout.write(json.dumps(
                {"jsonrpc": "2.0", "id": mid, "error": {"code": -32601, "message": "no"}}) + "\\n")
            sys.stdout.flush()
            continue
        sys.stdout.write(json.dumps({"jsonrpc": "2.0", "id": mid, "result": result}) + "\\n")
        sys.stdout.flush()

main()
"""


async def test_stdio_transport_against_real_subprocess(tmp_path: Path) -> None:
    script = tmp_path / "fake_mcp_server.py"
    script.write_text(_FAKE_SERVER_SCRIPT, encoding="utf-8")
    transport = StdioTransport(sys.executable, [str(script)])
    client = MCPClient(transport, request_timeout=15.0)
    try:
        await client.initialize()
        assert client.server_info["name"] == "itest"
        defs = await client.list_tools()
        assert [d.name for d in defs] == ["echo"]
        result = await client.call_tool("echo", {"text": "round-trip"})
        assert result.text == "round-trip"
    finally:
        await client.close()


# A server whose FIRST stdout line is raw non-UTF-8 bytes, then a valid JSON line.
_BAD_BYTES_SERVER_SCRIPT = """\
import sys, json
# Emit invalid UTF-8 bytes followed by a newline, on the raw stdout buffer.
sys.stdout.buffer.write(b"\\xff\\xfe garbage\\n")
sys.stdout.buffer.flush()
# Then a well-formed JSON-RPC message.
sys.stdout.write(json.dumps({"jsonrpc": "2.0", "id": 1, "result": {"ok": True}}) + "\\n")
sys.stdout.flush()
"""


async def test_stdio_transport_survives_non_utf8_bytes(tmp_path: Path) -> None:
    # Regression: a misbehaving server emitting non-UTF-8 bytes must NOT crash the
    # transport with UnicodeDecodeError. The bad line degrades to a recoverable
    # MCPProtocolError (invalid JSON), and the NEXT, valid line still parses.
    script = tmp_path / "bad_bytes_server.py"
    script.write_text(_BAD_BYTES_SERVER_SCRIPT, encoding="utf-8")
    transport = StdioTransport(sys.executable, [str(script)])
    try:
        await transport.start()
        with pytest.raises(MCPProtocolError):
            await transport.receive()  # the garbage line: invalid JSON, not a crash
        msg = await transport.receive()  # the following valid line parses fine
        assert msg == {"jsonrpc": "2.0", "id": 1, "result": {"ok": True}}
    finally:
        await transport.close()
