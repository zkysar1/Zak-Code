"""The framework agents a new chat can start, read from the workspace (ADR-0351).

A seed workspace keeps its agents under ``agents/<name>/``. The workspace's rules and skills reach
every chat, but an agent's identity and what it has learned (its knowledge, its lessons, its
guardrails) load only when ``/start <name>`` starts it, and the start ties it to that one chat, so
every new chat begins without one. A question typed before that is answered without the agent. The
web page and the terminal both offer to start an agent on a new chat, and both ask this module which
agents there are and what starts each, so the two never disagree.

An agent is a directory under ``agents/`` that holds a ``self.md``. The offer is made only where the
workspace holds the framework's start skill, since that is what the offer runs.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

#: The framework's start skill, relative to the seed root: what every offer runs.
START_SKILL = Path(".claude") / "skills" / "start" / "SKILL.md"

#: A name that can stand in a ``/start`` line as it is: one word, with no markup in it.
_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")


def startable_agents(workspace_root: str | os.PathLike[str]) -> list[str]:
    """The agents a new chat in ``workspace_root`` can start, sorted; ``[]`` for none.

    Read from the disk on every call, so an agent made in one chat is offered in the next. A
    workspace without the start skill has none to offer, and so does one whose ``agents/`` cannot
    be read: the offer is a help, never a reason for a chat to fail.
    """
    root = Path(workspace_root)
    try:
        if not (root / START_SKILL).is_file():
            return []
        return sorted(
            child.name
            for child in (root / "agents").iterdir()
            if _NAME.fullmatch(child.name) and (child / "self.md").is_file()
        )
    except OSError:
        return []


def start_line(agent: str) -> str:
    """The message that starts ``agent`` in assistant mode, as a chat's first line."""
    return f"/start {agent} --mode assistant"
