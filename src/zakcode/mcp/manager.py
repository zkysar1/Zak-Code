"""Slot MCP servers' tools into the agent's :class:`ToolRegistry` (M5-2).

An MCP tool becomes a plain :class:`~zakcode.tools.base.Tool` registered under a
**qualified name** ``mcp__<server>__<tool>``, so the agent loop dispatches it exactly
like a builtin — there is no separate routing path. :class:`ExtensionManager` owns the
client connections and discovers tools into a registry; a server that fails to list
its tools is recorded (graceful degradation) and never crashes the host.

The manager and the tool depend only on :class:`McpClientProtocol` (structurally
satisfied by :class:`~zakcode.mcp.client.MCPClient`), so both are testable against an
in-memory fake without a process or network.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import re
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, Field

from zakcode.config import PermissionTier
from zakcode.mcp.client import McpCallResult, McpToolDef
from zakcode.tools.base import ConcurrencyClass, Tool, ToolContext, ToolResult, ToolSpec

#: MCP tool names must match ``^[a-zA-Z0-9_-]{1,64}$``; we enforce the charset + length.
MAX_TOOL_NAME_LEN = 64
_QUALIFIED_PREFIX = "mcp"
_INVALID_NAME_CHARS = re.compile(r"[^a-zA-Z0-9_-]")
_EMPTY_SCHEMA: dict[str, Any] = {"type": "object", "properties": {}}


@runtime_checkable
class McpClientProtocol(Protocol):
    """The slice of an MCP client the manager and tools depend on."""

    async def list_tools(self) -> list[McpToolDef]: ...
    async def call_tool(
        self, name: str, arguments: dict[str, Any] | None = None
    ) -> McpCallResult: ...
    async def close(self) -> None: ...


def _sanitize(part: str) -> str:
    return _INVALID_NAME_CHARS.sub("_", part)


def qualified_tool_name(server: str, tool: str) -> str:
    """Build the registry name ``mcp__<server>__<tool>`` (MCP-charset-safe, <=64 chars).

    Invalid characters are replaced with ``_``. If the result would exceed the MCP
    64-char limit, it is truncated and given a short stable hash suffix so distinct
    tools never collide.
    """
    base = f"{_QUALIFIED_PREFIX}__{_sanitize(server)}__{_sanitize(tool)}"
    if len(base) <= MAX_TOOL_NAME_LEN:
        return base
    digest = hashlib.sha256(base.encode("utf-8")).hexdigest()[:6]
    keep = MAX_TOOL_NAME_LEN - len(digest) - 1
    return f"{base[:keep]}_{digest}"


def parse_qualified_name(name: str) -> tuple[str, str] | None:
    """Split ``mcp__<server>__<tool>`` into ``(server, tool)``; ``None`` if not qualified."""
    prefix = _QUALIFIED_PREFIX + "__"
    if not name.startswith(prefix):
        return None
    server, sep, tool = name[len(prefix) :].partition("__")
    if not sep or not server or not tool:
        return None
    return server, tool


class McpTool(Tool):
    """A :class:`Tool` adapter that forwards a call to an MCP server tool.

    The spec's ``name`` is the qualified registry name; the *remote* (server-side)
    tool name is kept separately for the actual ``tools/call``. ``execute`` never
    raises — any client/transport failure becomes an error :class:`ToolResult` so the
    loop keeps going and the model can react.
    """

    def __init__(
        self,
        *,
        server: str,
        definition: McpToolDef,
        client: McpClientProtocol,
        required_permission: PermissionTier = PermissionTier.DANGER_FULL_ACCESS,
    ) -> None:
        self._client = client
        self._server = server
        self._remote_name = definition.name
        schema = definition.input_schema if isinstance(definition.input_schema, dict) else None
        self.spec = ToolSpec(
            name=qualified_tool_name(server, definition.name),
            description=(
                definition.description
                or f"MCP tool {definition.name!r} provided by server {server!r}."
            ),
            parameters=schema or dict(_EMPTY_SCHEMA),
            # External, side-effecting, and untrusted by default: require the strongest
            # tier so it is gated/approved unless the operator widens the mode.
            required_permission=required_permission,
            # One MCP connection serves one request at a time; keep MCP calls serial.
            concurrency=ConcurrencyClass.NEVER_PARALLEL,
        )

    async def execute(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        try:
            result = await self._client.call_tool(self._remote_name, args)
        except Exception as exc:  # noqa: BLE001 — handlers never raise; surface as a result
            return ToolResult.error(
                f"MCP tool {self.spec.name!r} failed: {type(exc).__name__}: {exc}"
            )
        if result.text:
            output = result.text
        elif result.content:
            output = json.dumps(result.content)
        else:
            output = ""
        return ToolResult(
            output=output, is_error=result.is_error, data={"mcp_content": result.content}
        )


class DiscoveryReport(BaseModel):
    """What an :meth:`ExtensionManager.discover_into` pass registered and what failed."""

    registered: list[str] = Field(default_factory=list)  # qualified tool names
    failed: dict[str, str] = Field(default_factory=dict)  # server name -> error string


class ExtensionManager:
    """Owns MCP client connections and discovers their tools into a registry."""

    def __init__(
        self, *, mcp_permission_tier: PermissionTier = PermissionTier.DANGER_FULL_ACCESS
    ) -> None:
        self._clients: dict[str, McpClientProtocol] = {}
        self._tier = mcp_permission_tier

    def add_client(self, server: str, client: McpClientProtocol) -> None:
        """Register a (lazy) client for ``server``. It is connected on discovery."""
        self._clients[server] = client

    @property
    def server_names(self) -> list[str]:
        return list(self._clients)

    async def discover_into(self, registry: Any) -> DiscoveryReport:
        """List each server's tools and register them as :class:`McpTool` s.

        A server whose ``list_tools`` fails is recorded in the report's ``failed`` map
        and skipped — discovery of the other servers continues (graceful degradation).
        ``registry`` is a :class:`~zakcode.tools.base.ToolRegistry`.
        """
        report = DiscoveryReport()
        for server, client in self._clients.items():
            try:
                defs = await client.list_tools()
            except Exception as exc:  # noqa: BLE001 — one bad server must not abort the rest
                report.failed[server] = f"{type(exc).__name__}: {exc}"
                continue
            for definition in defs:
                tool = McpTool(
                    server=server,
                    definition=definition,
                    client=client,
                    required_permission=self._tier,
                )
                if registry.get(tool.spec.name) is not None:
                    continue  # already present (idempotent re-discovery / collision)
                registry.register(tool)
                report.registered.append(tool.spec.name)
        return report

    async def aclose(self) -> None:
        """Close every client connection (best-effort)."""
        for client in self._clients.values():
            with contextlib.suppress(Exception):
                await client.close()


__all__ = [
    "McpClientProtocol",
    "McpTool",
    "ExtensionManager",
    "DiscoveryReport",
    "qualified_tool_name",
    "parse_qualified_name",
    "MAX_TOOL_NAME_LEN",
]
