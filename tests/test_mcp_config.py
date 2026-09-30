"""Tests for MCP server configuration parsing, env-key resolution, and the
command allowlist (M5-3). All pure — nothing is spawned."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import ANY

import pytest

from zakcode.mcp.config import (
    McpConfigError,
    McpServerConfig,
    build_transport,
    discover_config,
    is_command_allowed,
    load_mcp_config,
    parse_mcp_config,
    resolve_env,
)
from zakcode.mcp.transport import StdioTransport


def test_parse_mcpservers_shape() -> None:
    servers = parse_mcp_config(
        {"mcpServers": {"github": {"command": "npx", "args": ["-y", "srv"]}}}
    )
    assert len(servers) == 1
    assert servers[0].name == "github"
    assert servers[0].command == "npx"
    assert servers[0].args == ["-y", "srv"]
    assert servers[0].enabled is True


def test_parse_servers_alias_shape() -> None:
    servers = parse_mcp_config({"servers": {"a": {"command": "x"}}})
    assert [s.name for s in servers] == ["a"]


def test_parse_rejects_non_object_entry() -> None:
    with pytest.raises(McpConfigError):
        parse_mcp_config({"mcpServers": {"bad": "not-an-object"}})


def test_parse_empty_is_empty() -> None:
    assert parse_mcp_config({}) == []


# ── env-key resolution (secrets via ${VAR}) ──────────────────────────────────────


def test_resolve_env_expands_var(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MY_TOKEN", "s3cret")
    cfg = McpServerConfig(name="s", command="x", env={"TOKEN": "${MY_TOKEN}"})
    assert resolve_env(cfg) == {"TOKEN": "s3cret"}


def test_resolve_env_passes_literals_through() -> None:
    cfg = McpServerConfig(name="s", command="x", env={"MODE": "fast"})
    assert resolve_env(cfg) == {"MODE": "fast"}


def test_resolve_env_missing_var_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("NOPE_TOKEN", raising=False)
    cfg = McpServerConfig(name="s", command="x", env={"TOKEN": "${NOPE_TOKEN}"})
    with pytest.raises(McpConfigError):
        resolve_env(cfg)


def test_config_does_not_store_secret_values() -> None:
    # The config object itself only holds the ${VAR} reference, never the secret.
    cfg = McpServerConfig(name="s", command="x", env={"TOKEN": "${MY_TOKEN}"})
    assert cfg.env["TOKEN"] == "${MY_TOKEN}"


# ── command allowlist ────────────────────────────────────────────────────────────


def test_allowlist_none_allows_anything() -> None:
    assert is_command_allowed("/usr/bin/anything", None) is True


def test_allowlist_matches_basename_or_full() -> None:
    assert is_command_allowed("/usr/bin/npx", ["npx"]) is True
    assert is_command_allowed("npx", ["npx"]) is True
    assert is_command_allowed("/usr/bin/rm", ["npx", "uvx"]) is False


# ── build_transport ──────────────────────────────────────────────────────────────


def test_build_transport_returns_stdio(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TK", "v")
    cfg = McpServerConfig(name="s", command="npx", args=["a"], env={"X": "${TK}"})
    transport = build_transport(cfg)
    assert isinstance(transport, StdioTransport)


def test_build_transport_rejects_missing_command() -> None:
    cfg = McpServerConfig(name="s", command=None)
    with pytest.raises(McpConfigError):
        build_transport(cfg)


def test_build_transport_enforces_allowlist() -> None:
    cfg = McpServerConfig(name="s", command="rm", args=["-rf", "/"])
    with pytest.raises(McpConfigError):
        build_transport(cfg, allowlist=["npx", "uvx"])


def test_build_transport_rejects_unsupported_transport() -> None:
    cfg = McpServerConfig(name="s", transport="stdio", command="x")
    # Force an unsupported value past the model to exercise the guard.
    object.__setattr__(cfg, "transport", "http")
    with pytest.raises(McpConfigError):
        build_transport(cfg)


# ── file loading ─────────────────────────────────────────────────────────────────


def test_load_missing_file_is_empty(tmp_path: Path) -> None:
    assert load_mcp_config(tmp_path / "nope.json") == []


def test_load_valid_file(tmp_path: Path) -> None:
    p = tmp_path / "mcp.json"
    p.write_text(json.dumps({"mcpServers": {"s": {"command": "x"}}}), encoding="utf-8")
    servers = load_mcp_config(p)
    assert [s.name for s in servers] == ["s"]


#: Two ways a present file is unreadable: broken JSON, and bytes that are not UTF-8.
@pytest.mark.parametrize(
    "broken", [b"{not json", b'{"mcpServers": {}}\xff'], ids=["not-json", "not-utf8"]
)
def test_load_invalid_json_raises(tmp_path: Path, broken: bytes) -> None:
    p = tmp_path / "mcp.json"
    p.write_bytes(broken)
    with pytest.raises(McpConfigError) as info:
        load_mcp_config(p)
    assert str(p) in str(info.value)  # the error names the file


def test_load_reports_a_file_it_cannot_even_check(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # In a directory the user cannot enter, Path.is_file() raises PermissionError (Python
    # re-raises EACCES from stat). It must surface as the same named McpConfigError as a bad
    # read, so discovery skips and reports it instead of the session failing to start.
    p = tmp_path / "mcp.json"

    def denied(self: Path) -> bool:
        raise PermissionError(13, "Permission denied", str(self))

    monkeypatch.setattr(Path, "is_file", denied)
    with pytest.raises(McpConfigError) as info:
        load_mcp_config(p)
    assert str(p) in str(info.value)


def test_load_non_object_top_level_raises(tmp_path: Path) -> None:
    p = tmp_path / "mcp.json"
    p.write_text("[1, 2, 3]", encoding="utf-8")
    with pytest.raises(McpConfigError):
        load_mcp_config(p)


def test_discover_prefers_project_over_user(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A project config under <workspace>/.zakcode/mcp.json is discovered.
    proj = tmp_path / ".zakcode"
    proj.mkdir()
    (proj / "mcp.json").write_text(
        json.dumps({"mcpServers": {"proj": {"command": "x"}}}), encoding="utf-8"
    )
    servers, errors = discover_config(tmp_path)
    assert [s.name for s in servers] == ["proj"]
    assert errors == {}


def test_discover_empty_when_no_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # No project config and a home dir without one → empty (MCP is opt-in).
    monkeypatch.setattr(Path, "home", lambda: tmp_path / "home")
    assert discover_config(tmp_path) == ([], {})


# A present config file that cannot be read is SKIPPED and REPORTED by discovery, never
# raised, so the session still starts and the next location is still consulted. Claude Code
# does the same with a broken project .mcp.json: it starts, still loads the user's other
# servers, and its MCP view reports the parse error.
@pytest.mark.parametrize(
    "broken", [b"{not json", b'{"mcpServers": {}}\xff'], ids=["not-json", "not-utf8"]
)
def test_discover_skips_an_unreadable_project_config_and_reports_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, broken: bytes
) -> None:
    workspace, home = tmp_path / "ws", tmp_path / "home"
    project = workspace / ".zakcode" / "mcp.json"
    project.parent.mkdir(parents=True)
    project.write_bytes(broken)
    user = home / ".config" / "zakcode" / "mcp.json"
    user.parent.mkdir(parents=True)
    user.write_text(json.dumps({"mcpServers": {"user": {"command": "x"}}}), encoding="utf-8")
    monkeypatch.setattr(Path, "home", lambda: home)
    servers, errors = discover_config(workspace)
    assert [s.name for s in servers] == ["user"]  # the next location still loads
    assert list(errors) == [str(project)]  # keyed by the file that failed
    assert str(project) in errors[str(project)]  # and the message names it


def test_discover_reports_every_unreadable_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace, home = tmp_path / "ws", tmp_path / "home"
    project = workspace / ".zakcode" / "mcp.json"
    user = home / ".config" / "zakcode" / "mcp.json"
    for p in (project, user):
        p.parent.mkdir(parents=True)
        p.write_text("{not json", encoding="utf-8")
    monkeypatch.setattr(Path, "home", lambda: home)
    assert discover_config(workspace) == ([], {str(project): ANY, str(user): ANY})
