"""Tests for the ToolRegistry active-set (M5-4a): lazy registration + activation.

The active-set is the mechanism behind MCP lazy discovery / the tool budget: a tool
can be registered dispatchable-but-hidden, then surfaced on demand. Default
registration stays active, so all prior behavior is unchanged.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from zakcode.tools.base import Tool, ToolContext, ToolRegistry, ToolResult, ToolSpec


class _Tool(Tool):
    def __init__(self, name: str) -> None:
        self.spec = ToolSpec(name=name, description=f"the {name} tool")

    async def execute(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        return ToolResult.ok(f"{self.spec.name} ran")


def _schema_names(registry: ToolRegistry) -> set[str]:
    return {d["function"]["name"] for d in registry.definitions()}


def test_default_registration_is_active() -> None:
    reg = ToolRegistry()
    reg.register(_Tool("a"))
    reg.register(_Tool("b"))
    assert reg.active_names() == ["a", "b"]
    assert _schema_names(reg) == {"a", "b"}
    assert reg.is_active("a") is True


def test_inactive_registration_is_dispatchable_but_hidden() -> None:
    reg = ToolRegistry()
    reg.register(_Tool("shown"))
    reg.register(_Tool("hidden"), active=False)
    # Hidden tool is registered (names) but not exposed (definitions/active).
    assert set(reg.names()) == {"shown", "hidden"}
    assert reg.active_names() == ["shown"]
    assert _schema_names(reg) == {"shown"}
    assert reg.is_active("hidden") is False


async def test_inactive_tool_still_executes(tmp_path: Path) -> None:
    # Hidden ≠ disabled: if the model somehow calls it, dispatch still works.
    reg = ToolRegistry()
    reg.register(_Tool("hidden"), active=False)
    result = await reg.execute("hidden", {}, ToolContext(workspace_root=tmp_path))
    assert result.output == "hidden ran"


def test_activate_exposes_a_hidden_tool() -> None:
    reg = ToolRegistry()
    reg.register(_Tool("h"), active=False)
    assert _schema_names(reg) == set()
    assert reg.activate("h") is True
    assert _schema_names(reg) == {"h"}
    assert reg.activate("does-not-exist") is False


def test_deactivate_hides_an_active_tool() -> None:
    reg = ToolRegistry()
    reg.register(_Tool("a"))
    assert reg.deactivate("a") is True
    assert reg.active_names() == []
    assert _schema_names(reg) == set()
    assert reg.deactivate("missing") is False


def test_activation_respects_aliases() -> None:
    reg = ToolRegistry()
    reg.register(_Tool("canonical"), aliases=["alias"], active=False)
    assert reg.is_active("alias") is False
    assert reg.activate("alias") is True  # activate via alias
    assert reg.is_active("canonical") is True
    assert _schema_names(reg) == {"canonical"}


def test_active_names_preserve_registration_order() -> None:
    reg = ToolRegistry()
    for n in ["z", "a", "m"]:
        reg.register(_Tool(n))
    assert reg.active_names() == ["z", "a", "m"]
