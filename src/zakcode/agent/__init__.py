"""The agent runtime: the ReAct loop, system-prompt assembly, and orchestration glue.

The concrete types live in :mod:`zakcode.agent.loop` and :mod:`zakcode.agent.prompt`;
this package re-exports them for convenience. See ``docs/ARCHITECTURE.md``.
"""

from zakcode.agent.loop import DEFAULT_MAX_ITERATIONS, AgentLoop, TurnResult
from zakcode.agent.prompt import (
    DYNAMIC_BOUNDARY,
    MAX_MEMORY_FILE_CHARS,
    MAX_MEMORY_TOTAL_CHARS,
    MEMORY_FILENAME,
    SystemPromptBuilder,
    discover_memory,
)

__all__ = [
    "DEFAULT_MAX_ITERATIONS",
    "DYNAMIC_BOUNDARY",
    "MAX_MEMORY_FILE_CHARS",
    "MAX_MEMORY_TOTAL_CHARS",
    "MEMORY_FILENAME",
    "AgentLoop",
    "SystemPromptBuilder",
    "TurnResult",
    "discover_memory",
]
