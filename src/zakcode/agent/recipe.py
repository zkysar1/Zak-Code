"""The Recipe Cursor (Slice 2): a verify-before-finish gate for create-and-run tasks.

A weak local model writes a file, claims success, and ends the turn over code it never
actually ran (or that is broken) — failures #2 (incoherent plan) and #5 (broken final
state). The cursor makes the HARNESS, not the model, own "done": once the model has
written a runnable (``.py``) file this turn, the turn may not end ``completed`` until the
model has RUN that file successfully (a ``bash``/``powershell`` result with no error whose
command references the written file). If it cannot get there within ``attempt_cap``
nudges, the turn ends gracefully as ``recipe_stalled`` rather than deadlocking or claiming
a false success.

Pure per-turn state: the loop creates one cursor per turn, feeds it each iteration's
``(calls, results)`` via :meth:`observe`, and consults :meth:`needs_verification` at the
turn's natural completion point. No provider/transport/vendor knowledge — the *model*
performs the run (so the correct interpreter, e.g. ``py``, is used via the workspace
rules), and the harness only enforces that a real run happened.
"""

from __future__ import annotations

import os

from zakcode.messages import ToolResultBlock
from zakcode.providers.base import ToolCall

_WRITE_TOOLS = {"write_file", "edit_file"}
_RUN_TOOLS = {"bash", "powershell"}


def _python_path(call: ToolCall, result: ToolResultBlock) -> str | None:
    """The ``.py`` path a successful write/edit touched, else ``None``."""
    path: str | None = None
    if isinstance(result.data, dict):
        candidate = result.data.get("path")
        if isinstance(candidate, str):
            path = candidate
    if path is None:
        candidate = call.arguments.get("path")
        path = candidate if isinstance(candidate, str) else None
    return path if (path is not None and path.endswith(".py")) else None


class RecipeCursor:
    """Per-turn 'verify what you wrote before finishing' gate (see the module docstring)."""

    def __init__(self, *, enabled: bool, attempt_cap: int = 3) -> None:
        self.enabled = enabled
        self.attempt_cap = max(0, attempt_cap)
        self.wrote_runnable = False  # a .py was created/edited this turn
        self.verified = False  # ...and then run successfully (command referenced it)
        self.nudges = 0  # how many times we have pushed the model to verify
        self._targets: set[str] = set()  # basenames of .py files written this turn

    def observe(self, calls: list[ToolCall], results: list[ToolResultBlock]) -> None:
        """Update state from one iteration's *successful* tool calls."""
        if not self.enabled:
            return
        by_id = {r.tool_use_id: r for r in results}
        for call in calls:
            result = by_id.get(call.id)
            if result is None or result.is_error:
                continue
            if call.name in _WRITE_TOOLS:
                path = _python_path(call, result)
                if path is not None:
                    self.wrote_runnable = True
                    self.verified = False  # a fresh write must be re-verified
                    self._targets.add(os.path.basename(path))
            elif call.name in _RUN_TOOLS and self.wrote_runnable:
                command = call.arguments.get("command")
                if isinstance(command, str) and any(t and t in command for t in self._targets):
                    self.verified = True

    def needs_verification(self) -> bool:
        """True when the turn should not end yet: a runnable file written, not yet run."""
        return self.enabled and self.wrote_runnable and not self.verified

    def can_nudge(self) -> bool:
        """Whether another corrective nudge is allowed before giving up (recipe_stalled)."""
        return self.nudges < self.attempt_cap

    def nudge(self) -> str:
        """Consume one attempt and return the one-step corrective instruction to inject."""
        self.nudges += 1
        return (
            "You created or edited a Python file but have not run it successfully yet. "
            "Run it now (use `py <file>`) to verify it works; if it errors, fix the file "
            "and run it again. Do not finish until the program runs without error."
        )
