"""The permission model — deny-first authorization for tool use.

This is the security core of Zak Code. Every privileged tool call is authorized
**here**, on a code path the model cannot reach: a prompt-injected model can ask
to run ``rm -rf /`` all it likes, but the decision is made by :class:`PermissionPolicy`
against the *operator's* configured posture, not by anything the model emits
(see ``docs/GUARDRAILS.md`` §2–§5).

Two-layer design, so it is trivially testable:

* :meth:`PermissionPolicy.decide` is **pure** — given a tool spec, its arguments,
  and the mode, it returns a static :class:`PermissionDecision` (allow/ask/deny)
  with a human reason. No I/O, no prompting, no state. The full authorization
  matrix is unit-tested through this.
* :meth:`PermissionPolicy.authorize` wraps ``decide`` with the stateful parts:
  per-session "allow/deny for the rest of the conversation" memory, and — when a
  decision is *ask* — prompting an injected :class:`PermissionPrompter`. With no
  prompter available it **fails closed** (ask → deny).

Authorization combines two independent checks, and the stricter wins:

1. **Tier vs. mode.** Each tool declares the narrowest
   :class:`~zakcode.config.PermissionTier` it needs. The operator's
   :class:`PermissionMode` sets a ceiling of what auto-allows; anything above the
   ceiling either prompts (``ask``/``acceptEdits``/``allow``) or is denied outright
   (``deny``). An unknown tool (no spec) is treated as the **strongest** tier —
   fail-closed.
2. **Dangerous-pattern blocklist.** Shell commands matching :data:`DANGEROUS_PATTERNS`
   (``rm -rf /``, ``sudo``, ``mkfs``, fork bombs, raw device writes, ``DROP TABLE`` …)
   can only ever *tighten* a decision: an otherwise-allowed catastrophic command is
   escalated to a confirmation prompt (or denied in ``deny`` mode). It never loosens.
"""

from __future__ import annotations

import json
import re
from enum import StrEnum
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from pydantic import BaseModel, Field

from zakcode.config import PermissionTier

if TYPE_CHECKING:
    from zakcode.tools.base import ToolSpec


class PermissionMode(StrEnum):
    """The operator's chosen authorization posture (from ``ZAKCODE_PERMISSION_MODE``).

    Ordered loosest-last by how much auto-allows:

    * ``deny`` — only read-only tools run; anything that writes or executes is
      blocked outright (no prompt).
    * ``ask`` — the safe default: read-only auto-allows; writing/executing prompts.
    * ``acceptEdits`` — read + workspace-write auto-allow; executing (shell) prompts.
    * ``allow`` — everything auto-allows (still subject to the dangerous-pattern
      blocklist, which escalates catastrophic commands to a prompt).
    """

    DENY = "deny"
    ASK = "ask"
    ACCEPT_EDITS = "acceptEdits"
    ALLOW = "allow"

    @classmethod
    def parse(cls, value: str | PermissionMode | None) -> PermissionMode:
        """Parse a mode string, falling back to the safe default (``ask``).

        Accepts the canonical values plus a couple of friendly spellings
        (``accept-edits``/``accept_edits``). An unknown/empty value resolves to
        ``ask`` rather than something more permissive — fail toward safe.
        """
        if isinstance(value, PermissionMode):
            return value
        if not value:
            return cls.ASK
        normalized = value.strip().lower().replace("-", "").replace("_", "")
        for mode in cls:
            if mode.value.lower().replace("-", "").replace("_", "") == normalized:
                return mode
        return cls.ASK


class PermissionDecision(StrEnum):
    """The static (pre-prompt) verdict for a tool call."""

    ALLOW = "allow"
    ASK = "ask"
    DENY = "deny"


#: Highest tier each mode auto-allows without prompting. A required tier at or
#: below the ceiling allows; above it, the mode decides (prompt vs. deny).
_MODE_CEILING: dict[PermissionMode, PermissionTier] = {
    PermissionMode.DENY: PermissionTier.READ_ONLY,
    PermissionMode.ASK: PermissionTier.READ_ONLY,
    PermissionMode.ACCEPT_EDITS: PermissionTier.WORKSPACE_WRITE,
    PermissionMode.ALLOW: PermissionTier.DANGER_FULL_ACCESS,
}

#: What a tier *above* the ceiling resolves to, per mode. ``deny`` blocks outright;
#: every other mode prompts (and ``allow`` never reaches here — its ceiling is max).
_ABOVE_CEILING: dict[PermissionMode, PermissionDecision] = {
    PermissionMode.DENY: PermissionDecision.DENY,
    PermissionMode.ASK: PermissionDecision.ASK,
    PermissionMode.ACCEPT_EDITS: PermissionDecision.ASK,
    PermissionMode.ALLOW: PermissionDecision.ASK,
}


#: Catastrophic shell-command patterns. A match forces at least a confirmation
#: prompt (or a hard deny in ``deny`` mode), regardless of the tier/mode verdict.
#: Patterns are intentionally conservative (aimed at clear footguns) so they rarely
#: false-positive on benign commands; the tier/mode gate is the primary control.
DANGEROUS_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (
        re.compile(r"\brm\s+(-[a-z]*r[a-z]*\s+|-[a-z]*f[a-z]*\s+).*(/|~|\$HOME)"),
        "recursive/force remove of a root or home path",
    ),
    (re.compile(r"(^|\s)sudo(\s|$)"), "privilege escalation (sudo)"),
    (re.compile(r"\bmkfs\b|\bformat\s+[a-z]:"), "filesystem format"),
    (re.compile(r":\s*\(\s*\)\s*\{.*\}\s*;\s*:"), "shell fork bomb"),
    (re.compile(r"\bdd\b.*\bof=/dev/"), "raw write to a block device"),
    (re.compile(r">\s*/dev/sd[a-z]"), "overwrite of a block device"),
    (re.compile(r"\bchmod\s+-R\s+0?777\b"), "recursive world-writable permissions"),
    (re.compile(r"\bDROP\s+(TABLE|DATABASE|SCHEMA)\b", re.IGNORECASE), "destructive SQL (DROP)"),
    (
        re.compile(r"\bgit\s+(push\s+.*--force|reset\s+--hard)\b"),
        "destructive git (force push / hard reset)",
    ),
    (re.compile(r"\bgit\s+clean\s+-[a-z]*f"), "git clean -f (deletes untracked files)"),
    (
        re.compile(r"\bcurl\b.*\|\s*(sudo\s+)?(ba)?sh\b|\bwget\b.*\|\s*(ba)?sh\b"),
        "pipe-to-shell of remote content",
    ),
]

#: Argument keys whose string values are scanned against DANGEROUS_PATTERNS.
_SHELL_ARG_KEYS = ("command", "cmd", "script")


class PermissionRequest(BaseModel):
    """What the operator is being asked to confirm (passed to a prompter)."""

    tool_name: str
    tier: PermissionTier
    arguments: dict = Field(default_factory=dict)
    reason: str = ""


class PermissionOutcome(StrEnum):
    """An operator's answer to a confirmation prompt.

    The ``*_session`` variants are remembered so the same call is not re-prompted
    for the rest of the conversation (fighting approval fatigue).
    """

    ALLOW_ONCE = "allow_once"
    ALLOW_SESSION = "allow_session"
    DENY_ONCE = "deny_once"
    DENY_SESSION = "deny_session"


@runtime_checkable
class PermissionPrompter(Protocol):
    """Asks the operator to confirm an escalated tool call.

    Implemented by clients (the CLI shows a console prompt; tests use a scripted
    one). The core never imports a UI; it only depends on this protocol.
    """

    async def confirm(self, request: PermissionRequest) -> PermissionOutcome: ...


class PermissionPolicy:
    """Authorizes tool calls against the operator's mode, in the core.

    Construct with a mode and an optional :class:`PermissionPrompter`. Hold one per
    session so "allow for the rest of the conversation" decisions persist.
    """

    def __init__(
        self,
        mode: PermissionMode | str = PermissionMode.ASK,
        *,
        prompter: PermissionPrompter | None = None,
        dangerous_patterns: list[tuple[re.Pattern[str], str]] | None = None,
    ) -> None:
        self.mode = PermissionMode.parse(mode)
        self.prompter = prompter
        self.dangerous_patterns = (
            DANGEROUS_PATTERNS if dangerous_patterns is None else dangerous_patterns
        )
        self._session_allow: set[str] = set()
        self._session_deny: set[str] = set()

    # ── pure decision ─────────────────────────────────────────────────────────

    @staticmethod
    def _required_tier(spec: ToolSpec | None) -> PermissionTier:
        # Fail-closed: an unknown tool (no spec) is treated as the most dangerous.
        if spec is None:
            return PermissionTier.DANGER_FULL_ACCESS
        return spec.required_permission

    def _dangerous_reason(self, arguments: dict) -> str | None:
        for key in _SHELL_ARG_KEYS:
            value = arguments.get(key)
            if isinstance(value, str):
                for pattern, description in self.dangerous_patterns:
                    if pattern.search(value):
                        return description
        return None

    def decide(self, spec: ToolSpec | None, arguments: dict) -> tuple[PermissionDecision, str]:
        """Return the static (pre-prompt, stateless) verdict and a human reason.

        The dangerous-pattern check only ever tightens the tier/mode verdict.
        """
        tier = self._required_tier(spec)
        ceiling = _MODE_CEILING[self.mode]
        base = PermissionDecision.ALLOW if tier <= ceiling else _ABOVE_CEILING[self.mode]

        danger = self._dangerous_reason(arguments)
        if danger is not None:
            if self.mode is PermissionMode.DENY or base is PermissionDecision.DENY:
                return (PermissionDecision.DENY, f"blocked dangerous command: {danger}")
            # Force a confirmation even if tier/mode would have auto-allowed.
            return (PermissionDecision.ASK, f"dangerous command: {danger}")

        if base is PermissionDecision.ALLOW:
            return (base, "")
        if base is PermissionDecision.DENY:
            return (base, f"'{tier.name}' is blocked in '{self.mode.value}' mode")
        return (base, f"'{tier.name}' requires confirmation in '{self.mode.value}' mode")

    # ── stateful authorization ────────────────────────────────────────────────

    @staticmethod
    def _key(tool_name: str, arguments: dict) -> str:
        try:
            args = json.dumps(arguments, sort_keys=True, default=str)
        except (TypeError, ValueError):
            args = repr(sorted(arguments.items()))
        return f"{tool_name}::{args}"

    async def authorize(self, spec: ToolSpec | None, arguments: dict) -> tuple[bool, str]:
        """Authorize one tool call. Returns ``(allowed, reason)``.

        Applies, in order: session deny memory → session allow memory → the static
        :meth:`decide` verdict → (for *ask*) a prompt. With no prompter, *ask*
        fails closed (denied). A denial is never an exception — the caller turns it
        into an error ``ToolResult`` so the loop continues and the model can adapt.
        """
        tool_name = spec.name if spec is not None else "<unknown>"
        key = self._key(tool_name, arguments)

        if key in self._session_deny:
            return (False, "denied for this session")
        if key in self._session_allow:
            return (True, "allowed for this session")

        decision, reason = self.decide(spec, arguments)
        if decision is PermissionDecision.ALLOW:
            return (True, "")
        if decision is PermissionDecision.DENY:
            return (False, reason)

        # decision is ASK → need the operator. No prompter ⇒ fail closed.
        if self.prompter is None:
            return (False, f"requires confirmation but none is available: {reason}")

        request = PermissionRequest(
            tool_name=tool_name,
            tier=self._required_tier(spec),
            arguments=arguments,
            reason=reason,
        )
        outcome = await self.prompter.confirm(request)
        if outcome is PermissionOutcome.ALLOW_SESSION:
            self._session_allow.add(key)
            return (True, "")
        if outcome is PermissionOutcome.ALLOW_ONCE:
            return (True, "")
        if outcome is PermissionOutcome.DENY_SESSION:
            self._session_deny.add(key)
            return (False, "denied for this session by the operator")
        return (False, "denied by the operator")


__all__ = [
    "PermissionMode",
    "PermissionDecision",
    "PermissionRequest",
    "PermissionOutcome",
    "PermissionPrompter",
    "PermissionPolicy",
    "DANGEROUS_PATTERNS",
]
