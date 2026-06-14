"""The project-verifier gate (R1): gate completion on the workspace's own health check.

The :class:`~zakcode.agent.recipe.RecipeCursor` verifies that a freshly *written script* runs.
This gate is its complement at the project scale: when an operator/mind/skill configures a
**verify command** (e.g. ``uv run poe check``, ``pytest -q``, ``npm test``) and a turn actually
**changed code**, the turn may not end ``completed`` until that command has run successfully. It
operationalizes the research finding that *external, sound verification beats model self-critique*
— tests/linters/type-checkers are exactly the sound verifiers a coding agent affords, and a model
left to self-attest "done" abandons or skips that check a large fraction of the time.

Pure, per-turn state mirroring :class:`RecipeCursor` / :class:`~zakcode.agent.stuck.StuckTracker`:
the loop builds one gate per turn, feeds it each iteration's ``(calls, results)`` via
:meth:`observe`, and consults :meth:`needs_verification` at the turn's completion point. It holds
**no provider/transport knowledge** and never runs anything itself — the loop performs the run (so
permission gating and the right shell are reused) and reports the outcome back here.

Deliberately **domain-agnostic**: the engine never guesses the command. With no ``command`` the
gate is wholly inert, so an ordinary turn (and the entire existing recipe-gate behavior) is
unchanged.
"""

from __future__ import annotations

import shlex

from zakcode.messages import ToolResultBlock
from zakcode.providers.base import ToolCall

#: Tools whose successful result means code changed this turn (arms the gate).
_WRITE_TOOLS = {"write_file", "edit_file"}
#: Tools that can execute the verify command.
_RUN_TOOLS = {"bash", "powershell"}


def _commands_match(ran: str, verify: str) -> bool:
    """Whether the shell command ``ran`` is (or wraps) the configured ``verify`` command.

    Token-prefix match after a tolerant split, so ``uv run poe check`` matches when the model
    runs exactly that, and a wrapped form (``cd repo && uv run poe check``) still counts because
    the verify tokens appear as a contiguous run. Conservative: never matches a command that does
    not actually contain the verify token sequence.
    """
    try:
        ran_tokens = shlex.split(ran)
        verify_tokens = shlex.split(verify)
    except ValueError:
        ran_tokens, verify_tokens = ran.split(), verify.split()
    if not verify_tokens or len(verify_tokens) > len(ran_tokens):
        return False
    n = len(verify_tokens)
    return any(ran_tokens[i : i + n] == verify_tokens for i in range(len(ran_tokens) - n + 1))


class VerificationGate:
    """Per-turn 'the project's checks must pass before finishing' gate (see module docstring)."""

    def __init__(self, *, command: str | None, attempt_cap: int = 2) -> None:
        self.command = command.strip() if command and command.strip() else None
        self.attempt_cap = max(0, attempt_cap)
        self.code_changed = False  # a successful write/edit happened this turn
        self.passed = False  # the verify command has run successfully this turn
        self.attempts = 0  # verification attempts spent (harness runs + nudges) toward the cap
        self.harness_runs = 0  # how many harness-issued verification runs were issued
        self.last_output = ""  # output of the most recent FAILED verify run (for the nudge)

    @property
    def enabled(self) -> bool:
        """True only when a verify command is configured; otherwise the gate is wholly inert."""
        return self.command is not None

    def observe(self, calls: list[ToolCall], results: list[ToolResultBlock]) -> None:
        """Update state from one iteration's tool calls (model-issued or harness-issued)."""
        if not self.enabled:
            return
        by_id = {r.tool_use_id: r for r in results}
        for call in calls:
            result = by_id.get(call.id)
            if result is None:
                continue
            if call.name in _WRITE_TOOLS and not result.is_error:
                self.code_changed = True
            elif call.name in _RUN_TOOLS:
                command = call.arguments.get("command")
                if not isinstance(command, str) or not _commands_match(command, self.command or ""):
                    continue
                if result.is_error:
                    self.last_output = result.output or ""
                else:
                    self.passed = True
                    self.last_output = ""

    def needs_verification(self) -> bool:
        """True when the turn should not finish yet: code changed but checks haven't passed."""
        return self.enabled and self.code_changed and not self.passed

    def can_attempt(self) -> bool:
        """Whether another verification attempt is allowed before giving up."""
        return self.attempts < self.attempt_cap

    def consume_attempt(self) -> None:
        """Count one verification attempt (a harness-issued run or a nudge) toward the cap."""
        self.attempts += 1

    def nudge(self) -> str:
        """The one-step corrective instruction to inject when the model must verify itself."""
        message = (
            f"You changed code but have not verified the project still passes its checks. "
            f"Run `{self.command}` and make sure it succeeds before finishing; if it fails, fix "
            "the cause and run it again."
        )
        if self.last_output:
            tail = self.last_output.strip()[-500:]
            message += f"\nThe last run failed with:\n{tail}"
        return message


__all__ = ["VerificationGate"]
