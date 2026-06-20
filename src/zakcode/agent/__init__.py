"""The agent runtime: the ReAct loop, system-prompt assembly, and orchestration glue.

The concrete types live in :mod:`zakcode.agent.loop` and :mod:`zakcode.agent.prompt`;
this package re-exports them for convenience. See ``docs/ARCHITECTURE.md``.
"""

from zakcode.agent.loop import DEFAULT_MAX_ITERATIONS, AgentLoop, TurnResult
from zakcode.agent.prompt import (
    AGENT_GUIDE_FILENAMES,
    DYNAMIC_BOUNDARY,
    MAX_CONTEXT_FILE_CHARS,
    MAX_CONTEXT_TOTAL_CHARS,
    README_FILENAME,
    SystemPromptBuilder,
    discover_context,
)

__all__ = [
    "AGENT_GUIDE_FILENAMES",
    "DEFAULT_MAX_ITERATIONS",
    "DYNAMIC_BOUNDARY",
    "MAX_CONTEXT_FILE_CHARS",
    "MAX_CONTEXT_TOTAL_CHARS",
    "README_FILENAME",
    "AgentLoop",
    "SystemPromptBuilder",
    "TurnResult",
    "discover_context",
]
