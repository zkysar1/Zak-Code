"""Tool registry and built-in tools.

A small, sharp, composable tool set the agent uses to act on the workspace (read/write
files, run commands, search code). :mod:`zakcode.tools.base` defines the Tool contract,
JSON-schema exposure, and the registry; concrete tools live in ``builtins/``.

Status: contracts implemented (M0 Phase A); the built-in tools land next. See
``docs/ARCHITECTURE.md`` and ``docs/PARITY.md``.
"""

from zakcode.tools.base import (
    ConcurrencyClass,
    Tool,
    ToolContext,
    ToolRegistry,
    ToolResult,
    ToolSpec,
)

__all__ = [
    "ConcurrencyClass",
    "Tool",
    "ToolContext",
    "ToolRegistry",
    "ToolResult",
    "ToolSpec",
]
