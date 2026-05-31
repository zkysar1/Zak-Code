"""Factory for the default tool registry (all built-in tools, pre-registered)."""

from __future__ import annotations

from zakcode.tools.base import ToolRegistry
from zakcode.tools.builtins.bash import BashTool
from zakcode.tools.builtins.edit import EditFileTool
from zakcode.tools.builtins.glob import GlobTool
from zakcode.tools.builtins.grep import GrepTool
from zakcode.tools.builtins.list_dir import ListDirTool
from zakcode.tools.builtins.read_file import ReadFileTool
from zakcode.tools.builtins.write_file import WriteFileTool


def default_registry() -> ToolRegistry:
    """Build a :class:`ToolRegistry` populated with every built-in tool.

    Friendly aliases (``read`` / ``write`` / ``ls``) are registered alongside the
    canonical tool names so the model can use either form.
    """
    registry = ToolRegistry()
    registry.register(ReadFileTool(), aliases=["read"])
    registry.register(WriteFileTool(), aliases=["write"])
    registry.register(EditFileTool(), aliases=["edit"])
    registry.register(ListDirTool(), aliases=["ls"])
    registry.register(GlobTool())
    registry.register(GrepTool())
    registry.register(BashTool())
    return registry
