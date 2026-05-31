"""Declarative configuration for MCP servers (M5-3).

MCP servers are declared in a small JSON file rather than the env-var ``Settings``
model, because a *list* of structured server entries does not fit 12-factor env
config. The shape is compatible with the common ``mcpServers`` convention::

    {
      "mcpServers": {
        "github": {
          "command": "npx",
          "args": ["-y", "@modelcontextprotocol/server-github"],
          "env": {"GITHUB_TOKEN": "${GITHUB_TOKEN}"}
        }
      }
    }

**Secrets never live in this file.** An ``env`` value of the form ``${VAR}`` is
expanded from the host environment at spawn time (:func:`resolve_env`); the
resolved value is passed to the child process and never written back to disk. Plain
literal values are allowed (operator's choice) but the ``${VAR}`` form is the
documented way to pass tokens.

Loading and resolution are pure and dependency-free so they unit-test without
spawning anything; :func:`build_transport` constructs (but does not start) a
:class:`~zakcode.mcp.transport.StdioTransport`.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from zakcode.mcp.transport import StdioTransport, Transport

#: ``${VAR}`` reference in an env value, expanded from the host environment.
_ENV_REF = re.compile(r"^\$\{([A-Za-z_][A-Za-z0-9_]*)\}$")


class McpConfigError(Exception):
    """Raised when MCP configuration is malformed or references a missing secret."""


class McpServerConfig(BaseModel):
    """One declared MCP server."""

    name: str
    transport: Literal["stdio"] = "stdio"
    command: str | None = None
    args: list[str] = Field(default_factory=list)
    #: child env: each value is a literal or a ``${VAR}`` reference to the host env.
    env: dict[str, str] = Field(default_factory=dict)
    cwd: str | None = None
    enabled: bool = True


def resolve_env(config: McpServerConfig) -> dict[str, str]:
    """Resolve a server's ``env`` to concrete values (expanding ``${VAR}`` refs).

    A ``${VAR}`` value is read from the host environment; a missing referenced
    variable raises :class:`McpConfigError` (fail loud rather than silently spawn a
    server without its credential). Literal values pass through unchanged.
    """
    resolved: dict[str, str] = {}
    for key, value in config.env.items():
        match = _ENV_REF.match(value)
        if match is None:
            resolved[key] = value
            continue
        var = match.group(1)
        if var not in os.environ:
            raise McpConfigError(
                f"MCP server {config.name!r} needs env var {var!r} (referenced by "
                f"{key}=${{{var}}}) but it is not set in the environment"
            )
        resolved[key] = os.environ[var]
    return resolved


def is_command_allowed(command: str, allowlist: list[str] | None) -> bool:
    """Whether ``command`` may be auto-spawned.

    ``allowlist=None`` means "no restriction" (the operator declared the servers
    themselves, so they are trusted). A non-``None`` allowlist permits only commands
    whose base name (or full string) is listed — the guard for auto-installed
    servers where the command might come from a less-trusted source.
    """
    if allowlist is None:
        return True
    base = os.path.basename(command)
    return command in allowlist or base in allowlist


def build_transport(config: McpServerConfig, *, allowlist: list[str] | None = None) -> Transport:
    """Construct (but do not start) a transport for ``config``.

    Raises :class:`McpConfigError` for an unsupported transport, a missing command,
    or a command blocked by ``allowlist``. Env ``${VAR}`` refs are resolved here.
    """
    if config.transport != "stdio":
        raise McpConfigError(
            f"MCP server {config.name!r}: transport {config.transport!r} is not supported yet "
            "(only 'stdio')"
        )
    if not config.command:
        raise McpConfigError(f"MCP server {config.name!r}: stdio transport requires a 'command'")
    if not is_command_allowed(config.command, allowlist):
        raise McpConfigError(
            f"MCP server {config.name!r}: command {config.command!r} is not in the allowlist"
        )
    return StdioTransport(
        config.command,
        config.args,
        env=resolve_env(config),
        cwd=config.cwd,
    )


def parse_mcp_config(data: dict[str, Any]) -> list[McpServerConfig]:
    """Parse a config ``dict`` into server configs.

    Accepts either ``{"mcpServers": {...}}`` (the common convention) or
    ``{"servers": {...}}``; each entry's key is its server name. A malformed entry
    raises :class:`McpConfigError`.
    """
    table = data.get("mcpServers")
    if table is None:
        table = data.get("servers", {})
    if not isinstance(table, dict):
        raise McpConfigError("MCP config: 'mcpServers'/'servers' must be an object")
    servers: list[McpServerConfig] = []
    for name, entry in table.items():
        if not isinstance(entry, dict):
            raise McpConfigError(f"MCP server {name!r}: entry must be an object")
        try:
            servers.append(McpServerConfig(name=name, **entry))
        except Exception as exc:  # noqa: BLE001 — normalize pydantic errors to our type
            raise McpConfigError(f"MCP server {name!r}: {exc}") from exc
    return servers


def load_mcp_config(path: str | Path) -> list[McpServerConfig]:
    """Load MCP server configs from a JSON file.

    A missing file yields ``[]`` (MCP is opt-in; absence is not an error). A present
    but malformed file raises :class:`McpConfigError`.
    """
    p = Path(path)
    if not p.is_file():
        return []
    try:
        raw = p.read_text(encoding="utf-8")
    except OSError as exc:
        raise McpConfigError(f"could not read MCP config {p}: {exc}") from exc
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise McpConfigError(f"invalid JSON in MCP config {p}: {exc}") from exc
    if not isinstance(data, dict):
        raise McpConfigError(f"MCP config {p}: top level must be a JSON object")
    return parse_mcp_config(data)


def default_config_paths(workspace_root: str | Path) -> list[Path]:
    """Candidate MCP config locations, project-first then user-home.

    Project ``<workspace>/.zakcode/mcp.json`` overrides (is listed before) the user
    ``~/.config/zakcode/mcp.json``.
    """
    return [
        Path(workspace_root) / ".zakcode" / "mcp.json",
        Path.home() / ".config" / "zakcode" / "mcp.json",
    ]


def discover_config(workspace_root: str | Path) -> list[McpServerConfig]:
    """Load servers from the first existing default config path (project, then user)."""
    for candidate in default_config_paths(workspace_root):
        servers = load_mcp_config(candidate)
        if servers:
            return servers
    return []


__all__ = [
    "McpServerConfig",
    "McpConfigError",
    "resolve_env",
    "is_command_allowed",
    "build_transport",
    "parse_mcp_config",
    "load_mcp_config",
    "default_config_paths",
    "discover_config",
]
