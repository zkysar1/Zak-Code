"""MCP (Model Context Protocol) client subsystem (M5).

Zak Code is an MCP *host*: it connects to external MCP servers and slots the tools
they expose into the same :class:`~zakcode.tools.base.ToolRegistry` the agent loop
already uses, under qualified names ``mcp__<server>__<tool>`` — so an MCP tool is
indistinguishable from a builtin to the loop.

This is a clean-room implementation: a minimal JSON-RPC 2.0 client over a pluggable
:class:`~zakcode.mcp.transport.Transport` (stdio today; streamable-HTTP can slot in
later), with no dependency on the official ``mcp`` SDK. That keeps the core
dependency-free and lets every test run against an in-memory fake transport.

Public surface is re-exported lazily so importing this package never spawns a
subprocess or touches the network.
"""

from __future__ import annotations

from zakcode.mcp.client import McpCallResult, MCPClient, McpToolDef
from zakcode.mcp.jsonrpc import JSONRPCError, MCPError, MCPProtocolError
from zakcode.mcp.manager import (
    DiscoveryReport,
    ExtensionManager,
    McpClientProtocol,
    McpTool,
    parse_qualified_name,
    qualified_tool_name,
)
from zakcode.mcp.transport import StdioTransport, Transport

__all__ = [
    "MCPClient",
    "McpToolDef",
    "McpCallResult",
    "Transport",
    "StdioTransport",
    "MCPError",
    "JSONRPCError",
    "MCPProtocolError",
    "ExtensionManager",
    "McpTool",
    "McpClientProtocol",
    "DiscoveryReport",
    "qualified_tool_name",
    "parse_qualified_name",
]
