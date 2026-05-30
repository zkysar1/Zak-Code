"""Tooling subsystem: the tool contract, the registry, and built-in tools.

The contract lives in :mod:`zakcode.tools.base` and the assembled default tool set in
:mod:`zakcode.tools.builtins.default_registry`; both are re-exported here for convenience.
See ``docs/ARCHITECTURE.md``.
"""

from zakcode.tools.base import (
    ConcurrencyClass,
    Tool,
    ToolContext,
    ToolRegistry,
    ToolResult,
    ToolSpec,
)
from zakcode.tools.builtins.default_registry import default_registry

__all__ = [
    "ConcurrencyClass",
    "Tool",
    "ToolContext",
    "ToolRegistry",
    "ToolResult",
    "ToolSpec",
    "default_registry",
]
