"""Plugin subsystem (M6) — first-party extensions via a single ``register(ctx)``.

A plugin is a manifest plus a ``register(ctx)`` entrypoint. When loaded, it is
handed a :class:`PluginContext` whose narrow API lets it contribute **tools**,
**hooks**, and **commands** — without importing core internals or touching core
files. Loading is **trust-gated** (a plugin runs only if its manifest is both
``enabled`` and ``trusted``) and **error-isolated** (an import or ``register``
failure is recorded, never crashing startup).

This module is the contract + the load engine. Filesystem / entry-point discovery
(turning directories and installed packages into :class:`Plugin` objects) lives in
:mod:`zakcode.plugins.discovery` so the core load/isolation logic here is testable
with injected plugins and no I/O.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from zakcode.commands import CommandHandler, CommandRegistry
from zakcode.hooks import HookEvent, HookManager
from zakcode.tools.base import Tool, ToolRegistry

if TYPE_CHECKING:
    from zakcode.config import Settings

logger = logging.getLogger("zakcode.plugins")


class PluginManifest(BaseModel):
    """Declarative description of a plugin (its ``plugin.json``)."""

    name: str
    version: str = "0.0.0"
    description: str = ""
    #: Module to import for the ``register`` entrypoint — a ``<name>.py`` file inside
    #: the plugin directory, or a dotted import path for an installed package.
    module: str = "plugin"
    #: The plugin participates only if BOTH are true. ``trusted`` is the explicit
    #: opt-in that must be set by the operator before third-party code runs.
    enabled: bool = True
    trusted: bool = False


#: A plugin's entrypoint: ``register(ctx)`` — contributes via the context's API.
RegisterFn = Callable[["PluginContext"], None]


@dataclass
class Plugin:
    """A discovered plugin: its manifest plus the resolved ``register`` callable."""

    manifest: PluginManifest
    register: RegisterFn
    #: Where it came from (directory path or entry-point name) — for provenance.
    origin: str = ""


class PluginContext:
    """The narrow API handed to a plugin's ``register(ctx)``.

    A plugin contributes capability **only** through these methods; it never imports
    or mutates core internals. Each registration is recorded so the load report can
    attribute contributions (and `/plugins` can show provenance). Registration
    methods are namespaced/attributed to the plugin via ``source``.
    """

    def __init__(
        self,
        *,
        plugin_name: str,
        registry: ToolRegistry,
        hook_manager: HookManager,
        command_registry: CommandRegistry,
        settings: Settings,
    ) -> None:
        self.plugin_name = plugin_name
        self.settings = settings
        self._registry = registry
        self._hook_manager = hook_manager
        self._command_registry = command_registry
        self.registered_tools: list[str] = []
        self.registered_commands: list[str] = []
        self.registered_hooks: list[str] = []

    def register_tool(self, tool: Tool) -> None:
        """Contribute a :class:`~zakcode.tools.base.Tool` to the agent's registry."""
        self._registry.register(tool)
        self.registered_tools.append(tool.name)

    def register_command(
        self, name: str, handler: CommandHandler, *, description: str = ""
    ) -> None:
        """Contribute a slash command (attributed to this plugin as its ``source``)."""
        self._command_registry.register(
            name, handler, description=description, source=self.plugin_name
        )
        self.registered_commands.append(name.lstrip("/").lower())

    def register_hook(self, event: HookEvent | str, callback: Any) -> None:
        """Register an in-process lifecycle hook callback for ``event``."""
        ev = event if isinstance(event, HookEvent) else HookEvent(event)
        self._hook_manager.register(ev, callback)
        self.registered_hooks.append(ev.value)

    def log(self, message: str) -> None:
        """Emit a log line attributed to this plugin (never raises)."""
        logger.info("[plugin %s] %s", self.plugin_name, message)


class PluginLoadReport(BaseModel):
    """What a :meth:`PluginManager.load_into` pass loaded, skipped, and failed."""

    loaded: list[str] = Field(default_factory=list)
    skipped: dict[str, str] = Field(default_factory=dict)  # name -> reason
    failed: dict[str, str] = Field(default_factory=dict)  # name -> error
    #: Per-loaded-plugin contribution summary: name -> {tools, commands, hooks}.
    contributions: dict[str, dict[str, list[str]]] = Field(default_factory=dict)


class PluginManager:
    """Holds discovered plugins and loads the trusted+enabled ones into the agent."""

    def __init__(self) -> None:
        self._plugins: list[Plugin] = []

    def add(self, plugin: Plugin) -> None:
        self._plugins.append(plugin)

    @property
    def plugin_names(self) -> list[str]:
        return [p.manifest.name for p in self._plugins]

    def load_into(
        self,
        *,
        registry: ToolRegistry,
        hook_manager: HookManager,
        command_registry: CommandRegistry,
        settings: Settings,
    ) -> PluginLoadReport:
        """Run each trusted+enabled plugin's ``register(ctx)`` against the live engine.

        Skipped: a plugin that is disabled or not trusted (recorded with a reason,
        never loaded — and, for a directory plugin, never even imported). Failed: a
        plugin whose import or ``register`` raises (recorded; the others still load).
        On success the plugin's contributions are recorded for provenance.
        """
        report = PluginLoadReport()
        for plugin in self._plugins:
            name = plugin.manifest.name
            if not plugin.manifest.enabled:
                report.skipped[name] = "disabled"
                continue
            if not plugin.manifest.trusted:
                report.skipped[name] = "untrusted (set trusted=true to enable)"
                continue
            ctx = PluginContext(
                plugin_name=name,
                registry=registry,
                hook_manager=hook_manager,
                command_registry=command_registry,
                settings=settings,
            )
            try:
                # resolve() imports the module for a directory plugin — done HERE,
                # after the trust gate above, so untrusted modules are never imported.
                register_fn = plugin.resolve()
                register_fn(ctx)
            except Exception as exc:  # noqa: BLE001 — one bad plugin must not crash startup
                report.failed[name] = f"{type(exc).__name__}: {exc}"
                logger.warning("plugin %r failed to register: %s", name, exc)
                continue
            report.loaded.append(name)
            report.contributions[name] = {
                "tools": ctx.registered_tools,
                "commands": ctx.registered_commands,
                "hooks": ctx.registered_hooks,
            }
        return report


__all__ = [
    "PluginManifest",
    "RegisterFn",
    "Plugin",
    "PluginContext",
    "PluginLoadReport",
    "PluginManager",
]
