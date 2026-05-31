"""Tests for the plugin contract + PluginManager load engine (M6-2).

Plugins are injected as in-memory :class:`Plugin` objects (a manifest + a register
callable), so these exercise trust/enable gating, error isolation, and the
contribution API with no filesystem or import.
"""

from __future__ import annotations

from typing import Any

from zakcode.commands import CommandRegistry, CommandResult
from zakcode.config import Settings
from zakcode.hooks import HookEvent, HookManager
from zakcode.plugins import Plugin, PluginContext, PluginManager, PluginManifest
from zakcode.tools.base import Tool, ToolContext, ToolResult, ToolSpec


class _SampleTool(Tool):
    spec = ToolSpec(name="sample_tool", description="a sample plugin tool")

    async def execute(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        return ToolResult.ok("sample ran")


def _engine() -> tuple[ToolRegistry, HookManager, CommandRegistry, Settings]:
    from zakcode.tools.builtins.default_registry import default_registry

    return default_registry(), HookManager(), CommandRegistry(), Settings(default_model="x/y")


def _manifest(
    name: str = "sample", *, enabled: bool = True, trusted: bool = True
) -> PluginManifest:
    return PluginManifest(name=name, enabled=enabled, trusted=trusted)


# ── PluginContext API ────────────────────────────────────────────────────────────


def test_context_registers_tool_command_hook() -> None:
    registry, hooks, commands, settings = _engine()
    ctx = PluginContext(
        plugin_name="p",
        registry=registry,
        hook_manager=hooks,
        command_registry=commands,
        settings=settings,
    )

    ctx.register_tool(_SampleTool())
    ctx.register_command("hello", lambda arg: CommandResult.ok("hi"), description="greet")

    def _cb(**kwargs: Any) -> None:
        return None

    ctx.register_hook(HookEvent.POST_TOOL_USE, _cb)

    assert "sample_tool" in registry.names()
    assert commands.has("hello")
    assert commands.get("hello").source == "p"  # attributed to the plugin
    assert HookEvent.POST_TOOL_USE in hooks.in_process
    assert ctx.registered_tools == ["sample_tool"]
    assert ctx.registered_commands == ["hello"]
    assert ctx.registered_hooks == [HookEvent.POST_TOOL_USE.value]


def test_register_hook_accepts_string_event() -> None:
    registry, hooks, commands, settings = _engine()
    ctx = PluginContext(
        plugin_name="p",
        registry=registry,
        hook_manager=hooks,
        command_registry=commands,
        settings=settings,
    )
    ctx.register_hook("PreToolUse", lambda **k: None)
    assert HookEvent.PRE_TOOL_USE in hooks.in_process


# ── PluginManager load/trust/isolation ───────────────────────────────────────────


def _load(manager: PluginManager) -> Any:
    registry, hooks, commands, settings = _engine()
    report = manager.load_into(
        registry=registry,
        hook_manager=hooks,
        command_registry=commands,
        settings=settings,
    )
    return report, registry, commands


def test_trusted_enabled_plugin_loads() -> None:
    def register(ctx: PluginContext) -> None:
        ctx.register_tool(_SampleTool())
        ctx.register_command("p_cmd", lambda arg: CommandResult.ok("ok"))

    manager = PluginManager()
    manager.add(Plugin(manifest=_manifest("good"), register=register))
    report, registry, commands = _load(manager)

    assert report.loaded == ["good"]
    assert "sample_tool" in registry.names()
    assert commands.has("p_cmd")
    assert report.contributions["good"]["tools"] == ["sample_tool"]
    assert report.contributions["good"]["commands"] == ["p_cmd"]


def test_untrusted_plugin_is_skipped() -> None:
    ran = []
    manager = PluginManager()
    manager.add(
        Plugin(
            manifest=_manifest("untrusted", trusted=False),
            register=lambda ctx: ran.append(1),
        )
    )
    report, _, _ = _load(manager)
    assert report.loaded == []
    assert "untrusted" in report.skipped
    assert "untrusted" in report.skipped["untrusted"]
    assert ran == []  # its code never ran


def test_disabled_plugin_is_skipped() -> None:
    ran = []
    manager = PluginManager()
    manager.add(
        Plugin(
            manifest=_manifest("off", enabled=False),
            register=lambda ctx: ran.append(1),
        )
    )
    report, _, _ = _load(manager)
    assert report.skipped.get("off") == "disabled"
    assert ran == []


def test_broken_plugin_is_isolated() -> None:
    def good(ctx: PluginContext) -> None:
        ctx.register_command("works", lambda arg: CommandResult.ok("ok"))

    def boom(ctx: PluginContext) -> None:
        raise RuntimeError("plugin exploded")

    manager = PluginManager()
    manager.add(Plugin(manifest=_manifest("boom"), register=boom))
    manager.add(Plugin(manifest=_manifest("good"), register=good))
    report, _, commands = _load(manager)

    # The broken plugin is recorded as failed; the good one still loaded.
    assert "boom" in report.failed
    assert "plugin exploded" in report.failed["boom"]
    assert report.loaded == ["good"]
    assert commands.has("works")


def test_plugin_names() -> None:
    manager = PluginManager()
    manager.add(Plugin(manifest=_manifest("a"), register=lambda ctx: None))
    manager.add(Plugin(manifest=_manifest("b"), register=lambda ctx: None))
    assert manager.plugin_names == ["a", "b"]


# late import so the helper above can reference it without an unused top-level import
from zakcode.tools.base import ToolRegistry  # noqa: E402
