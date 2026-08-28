"""Tests for the /plugins CLI command renderer (M6-5)."""

from __future__ import annotations

import json
from io import StringIO
from pathlib import Path

from rich.console import Console

from zakcode import Agent
from zakcode.cli import _render_plugins
from zakcode.config import Settings

_PLUGIN_BODY = """\
from zakcode.commands import CommandResult
from zakcode.tools.base import Tool, ToolResult, ToolSpec


class HelloTool(Tool):
    spec = ToolSpec(name="plugin_hello", description="a plugin tool")

    async def execute(self, args, ctx):
        return ToolResult.ok("hi")


def register(ctx):
    ctx.register_tool(HelloTool())
"""


def _console() -> tuple[Console, StringIO]:
    buf = StringIO()
    return Console(file=buf, width=100), buf


def _write_plugin(workspace: Path, name: str, *, body: str = _PLUGIN_BODY) -> None:
    pdir = workspace / ".zakcode" / "plugins" / name
    pdir.mkdir(parents=True)
    (pdir / "plugin.json").write_text(json.dumps({"name": name}), encoding="utf-8")
    (pdir / "plugin.py").write_text(body, encoding="utf-8")


def _agent(tmp_path: Path, **kw: object) -> Agent:
    return Agent(
        settings=Settings(
            default_model="scripted/test", context_window=8192, workspace_root=tmp_path
        ),
        **kw,
    )


def test_render_plugins_not_enabled(tmp_path: Path) -> None:
    console, buf = _console()
    _render_plugins(console, _agent(tmp_path))
    assert "not enabled" in buf.getvalue()


def test_render_plugins_loaded(tmp_path: Path) -> None:
    _write_plugin(tmp_path, "demo")
    agent = _agent(tmp_path, enable_plugins=True, trusted_plugins=["demo"])
    console, buf = _console()
    _render_plugins(console, agent)
    out = buf.getvalue()
    assert "demo" in out
    assert "tools" in out  # contribution summary shown


def test_render_plugins_skipped_untrusted(tmp_path: Path) -> None:
    _write_plugin(tmp_path, "demo")
    agent = _agent(tmp_path, enable_plugins=True)  # untrusted → skipped
    console, buf = _console()
    _render_plugins(console, agent)
    out = buf.getvalue()
    assert "demo" in out
    assert "untrusted" in out


def test_render_plugins_none_discovered(tmp_path: Path) -> None:
    agent = _agent(tmp_path, enable_plugins=True)  # enabled but nothing on disk
    console, buf = _console()
    _render_plugins(console, agent)
    assert "no plugins discovered" in buf.getvalue()
