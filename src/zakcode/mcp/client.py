"""A minimal MCP client: initialize once, list tools, call tools.

Clean-room implementation over a :class:`~zakcode.mcp.transport.Transport`. One
client drives one MCP server connection. :meth:`initialize` performs the MCP
handshake exactly once and is idempotent; :meth:`list_tools` and :meth:`call_tool`
ensure initialization first, so callers can use them directly.

Concurrency: this client issues one request at a time (callers await each call), so
:meth:`_request` simply reads messages until it sees the response with the matching
id, ignoring any interleaved server notifications (logging, progress, …).
"""

from __future__ import annotations

import asyncio
import itertools
from typing import Any

from pydantic import BaseModel, Field

from zakcode.mcp.jsonrpc import (
    MCPProtocolError,
    is_response_for,
    make_notification,
    make_request,
    result_or_raise,
)
from zakcode.mcp.transport import Transport
from zakcode.version import __version__

#: MCP protocol revision this client advertises. Servers echo the version they
#: support; we record theirs but do not hard-fail on a mismatch.
PROTOCOL_VERSION = "2025-06-18"


class McpToolDef(BaseModel):
    """A tool definition as advertised by an MCP server (``tools/list``)."""

    name: str
    description: str = ""
    input_schema: dict[str, Any] = Field(
        default_factory=lambda: {"type": "object", "properties": {}}
    )


class McpCallResult(BaseModel):
    """The result of an MCP ``tools/call`` — a list of content blocks plus an error flag."""

    content: list[dict[str, Any]] = Field(default_factory=list)
    is_error: bool = False

    @property
    def text(self) -> str:
        """Concatenated text of all ``type: "text"`` content blocks."""
        parts = [
            str(block.get("text", ""))
            for block in self.content
            if isinstance(block, dict) and block.get("type") == "text"
        ]
        return "\n".join(parts)


class MCPClient:
    """Drive one MCP server over a :class:`Transport` (initialize-once, list, call)."""

    def __init__(
        self,
        transport: Transport,
        *,
        client_name: str = "zakcode",
        client_version: str = __version__,
        protocol_version: str = PROTOCOL_VERSION,
        request_timeout: float | None = None,
    ) -> None:
        self._transport = transport
        self._client_name = client_name
        self._client_version = client_version
        self._protocol_version = protocol_version
        self._request_timeout = request_timeout
        self._counter = itertools.count(1)
        self._initialized = False
        self.server_info: dict[str, Any] = {}
        self.server_capabilities: dict[str, Any] = {}
        self.server_protocol_version: str = ""

    @property
    def initialized(self) -> bool:
        return self._initialized

    async def initialize(self) -> None:
        """Run the MCP handshake exactly once (idempotent).

        Sends ``initialize``, records the server's info/capabilities, then sends the
        ``notifications/initialized`` acknowledgement the protocol requires.
        """
        if self._initialized:
            return
        await self._transport.start()
        result = await self._request(
            "initialize",
            {
                "protocolVersion": self._protocol_version,
                "capabilities": {},
                "clientInfo": {"name": self._client_name, "version": self._client_version},
            },
        )
        self.server_info = dict(result.get("serverInfo", {}))
        self.server_capabilities = dict(result.get("capabilities", {}))
        self.server_protocol_version = str(result.get("protocolVersion", ""))
        await self._transport.send(make_notification("notifications/initialized"))
        self._initialized = True

    async def list_tools(self) -> list[McpToolDef]:
        """Return the tools the server exposes (initializing first if needed)."""
        await self.initialize()
        result = await self._request("tools/list")
        tools = result.get("tools", [])
        defs: list[McpToolDef] = []
        for raw in tools:
            if not isinstance(raw, dict) or "name" not in raw:
                continue
            defs.append(
                McpToolDef(
                    name=str(raw["name"]),
                    description=str(raw.get("description", "")),
                    input_schema=raw.get("inputSchema") or {"type": "object", "properties": {}},
                )
            )
        return defs

    async def call_tool(self, name: str, arguments: dict[str, Any] | None = None) -> McpCallResult:
        """Invoke an MCP tool by name and normalize its result."""
        await self.initialize()
        result = await self._request("tools/call", {"name": name, "arguments": arguments or {}})
        content = result.get("content", [])
        if not isinstance(content, list):
            content = []
        return McpCallResult(content=content, is_error=bool(result.get("isError", False)))

    async def close(self) -> None:
        """Close the underlying transport."""
        await self._transport.close()
        self._initialized = False

    async def _request(self, method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        request_id = next(self._counter)
        await self._transport.send(make_request(request_id, method, params))

        async def _read_until_response() -> dict[str, Any]:
            while True:
                msg = await self._transport.receive()
                if msg is None:
                    raise MCPProtocolError(
                        f"MCP server closed the connection while awaiting {method!r}"
                    )
                if is_response_for(msg, request_id):
                    return result_or_raise(msg)
                # Otherwise an interleaved notification or unrelated message — skip it.

        if self._request_timeout is None:
            return await _read_until_response()
        return await asyncio.wait_for(_read_until_response(), timeout=self._request_timeout)


__all__ = ["MCPClient", "McpToolDef", "McpCallResult", "PROTOCOL_VERSION"]
