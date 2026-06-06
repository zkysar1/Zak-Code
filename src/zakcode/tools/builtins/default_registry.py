"""Factory for the default tool registry (all built-in tools, pre-registered)."""

from __future__ import annotations

from zakcode.tools.base import ToolRegistry
from zakcode.tools.builtins.bash import BashTool
from zakcode.tools.builtins.edit import EditFileTool
from zakcode.tools.builtins.glob import GlobTool
from zakcode.tools.builtins.grep import GrepTool
from zakcode.tools.builtins.list_dir import ListDirTool
from zakcode.tools.builtins.powershell import PowerShellTool
from zakcode.tools.builtins.read_file import ReadFileTool
from zakcode.tools.builtins.write_file import WriteFileTool


def default_registry() -> ToolRegistry:
    """Build a :class:`ToolRegistry` populated with every built-in tool.

    Each tool is registered under a small set of guessable aliases (POSIX muscle-memory like
    ``cat`` / ``ls`` / ``grep`` aliases) so a model that guesses a sibling name still resolves
    to the right tool ("learn one, guess the rest"). Aliases are a SILENT fallback — they are
    not advertised in the prompt (that would cost tokens and dilute the one-canonical-name
    signal); the registry just routes them. See ToolRegistry.register's collision guard.
    """
    registry = ToolRegistry()
    registry.register(ReadFileTool(), aliases=["read", "cat"])
    registry.register(WriteFileTool(), aliases=["write"])
    registry.register(EditFileTool(), aliases=["edit"])
    registry.register(ListDirTool(), aliases=["ls", "dir"])
    registry.register(GlobTool(), aliases=["find"])
    registry.register(GrepTool(), aliases=["search", "rg"])
    registry.register(BashTool(), aliases=["sh", "shell"])
    registry.register(PowerShellTool(), aliases=["pwsh"])
    return registry
