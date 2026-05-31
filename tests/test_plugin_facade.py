"""Tests for plugin wiring into the Agent facade (M6-4).

Builds a real project plugin under a tmp workspace and constructs an Agent with
``enable_plugins=True``; asserts the plugin's contributions land in the agent's live
registry / command registry, and that trust gating + error isolation hold. No model
or network is involved (Agent construction is side-effect-free beyond plugin load).
"""

from __future__ import annotations

import json
from pathlib import Path

from zakcode import Agent
from zakcode.config import Settings

_PLUGIN_BODY = """\
from zakcode.commands import CommandResult
from zakcode.tools.base import Tool, ToolResult, ToolSpec


class HelloTool(Tool):
    spec = ToolSpec(name="plugin_hello", description="a tool from a plugin")

    async def execute(self, args, ctx):
        return ToolResult.ok("hello from plugin")


def register(ctx):
    ctx.register_tool(HelloTool())
    ctx.register_command("greet", lambda arg: CommandResult.ok(f"hi {arg}"), description="greet")
"""


def _write_plugin(workspace: Path, name: str, *, body: str = _PLUGIN_BODY) -> None:
    pdir = workspace / ".zakcode" / "plugins" / name
    pdir.mkdir(parents=True)
    (pdir / "plugin.json").write_text(json.dumps({"name": name}), encoding="utf-8")
    (pdir / "plugin.py").write_text(body, encoding="utf-8")


def _agent(tmp_path: Path, **kw: object) -> Agent:
    return Agent(settings=Settings(default_model="scripted/test", workspace_root=tmp_path), **kw)


def test_no_plugins_by_default(tmp_path: Path) -> None:
    _write_plugin(tmp_path, "p")
    agent = _agent(tmp_path)  # enable_plugins defaults False
    assert agent.plugin_report is None
    assert "plugin_hello" not in agent.registry.names()


def test_trusted_plugin_contributes_tool_and_command(tmp_path: Path) -> None:
    _write_plugin(tmp_path, "p")
    agent = _agent(tmp_path, enable_plugins=True, trusted_plugins=["p"])
    assert agent.plugin_report is not None
    assert agent.plugin_report.loaded == ["p"]
    # Contributions are in the SAME registries the loop/CLI use.
    assert "plugin_hello" in agent.registry.names()
    assert agent.command_registry.has("greet")
    assert agent.command_registry.get("greet").source == "p"


def test_untrusted_plugin_is_not_loaded(tmp_path: Path) -> None:
    _write_plugin(tmp_path, "p")
    agent = _agent(tmp_path, enable_plugins=True)  # no trusted_plugins → untrusted
    assert agent.plugin_report is not None
    assert agent.plugin_report.loaded == []
    assert "p" in agent.plugin_report.skipped
    assert "plugin_hello" not in agent.registry.names()


def test_broken_plugin_is_isolated(tmp_path: Path) -> None:
    _write_plugin(tmp_path, "good")
    _write_plugin(tmp_path, "bad", body="def register(ctx):\n    raise RuntimeError('boom')\n")
    agent = _agent(tmp_path, enable_plugins=True, trusted_plugins=["good", "bad"])
    assert agent.plugin_report is not None
    assert agent.plugin_report.loaded == ["good"]
    assert "bad" in agent.plugin_report.failed
    assert "plugin_hello" in agent.registry.names()  # the good plugin still loaded


def test_discovery_error_recorded(tmp_path: Path) -> None:
    # A plugin dir with bad JSON is a discovery error, surfaced on the agent.
    pdir = tmp_path / ".zakcode" / "plugins" / "broken"
    pdir.mkdir(parents=True)
    (pdir / "plugin.json").write_text("{not json", encoding="utf-8")
    agent = _agent(tmp_path, enable_plugins=True)
    assert "broken" in agent.plugin_discovery_errors


def test_command_registry_exists_without_plugins(tmp_path: Path) -> None:
    # The command registry is always present (plugins or not) so the CLI can use it.
    agent = _agent(tmp_path)
    assert agent.command_registry is not None
    assert agent.command_registry.names() == []
