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
   escalated to a confirmation prompt — except in ``autonomous`` mode, where it is a
   deterministic hard DENY (no prompt exists to escalate to) — and is denied outright
   in ``deny`` mode. It never loosens.
"""

from __future__ import annotations

import json
import logging
import os
import posixpath
import re
from collections.abc import Callable
from datetime import UTC, datetime
from enum import StrEnum
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from pydantic import BaseModel, Field

from zakcode.config import PermissionTier
from zakcode.deps_gate import display_name, installed_specs

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
    * ``autonomous`` — the unattended posture (audit P0-2a / D12): everything
      auto-allows and NOTHING ever prompts. A dangerous-pattern match is a
      deterministic hard DENY — returned as a recoverable tool error the model can
      adapt to, identical with or without a prompter, which is what makes the mode
      testable and safe headless. Contrast ``allow``, where a present operator may
      interactively approve a catastrophic command; ``autonomous`` never can.
    * ``bypassPermissions`` — the dangerously-skip posture (the Claude Code
      ``--dangerously-skip-permissions`` analog, field-driven 2026-08-28: an
      unattended Mind runner stalled forever on an interactive y/a/n prompt).
      NOTHING ever prompts, and everything the other modes would ESCALATE is
      ALLOWED instead: undeclared package installs, protected-path writes, and
      confirm-on-use tools all pass. Only two refusals survive, both as
      recoverable errors: the catastrophic-command blocklist (uniform in every
      mode, never waived anywhere) and explicit whole-tool config denies. Where
      ``autonomous`` is never-prompt-fail-CLOSED, this is never-prompt-fail-OPEN —
      for deployments whose own stack (hooks, gates, guardrails) is the guardrail
      layer. An explicit per-tool tighten (``tool_trust_overrides``) is still
      honored: the operator who writes both has asked for that tool to prompt.
    """

    DENY = "deny"
    ASK = "ask"
    ACCEPT_EDITS = "acceptEdits"
    ALLOW = "allow"
    AUTONOMOUS = "autonomous"
    BYPASS = "bypassPermissions"

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
    PermissionMode.AUTONOMOUS: PermissionTier.DANGER_FULL_ACCESS,
    PermissionMode.BYPASS: PermissionTier.DANGER_FULL_ACCESS,
}

#: What a tier *above* the ceiling resolves to, per mode. ``deny`` blocks outright;
#: every other mode prompts (``allow``/``autonomous`` never reach here — their ceiling
#: is max; ``autonomous`` maps to DENY anyway: it must never produce a prompt).
_ABOVE_CEILING: dict[PermissionMode, PermissionDecision] = {
    PermissionMode.DENY: PermissionDecision.DENY,
    PermissionMode.ASK: PermissionDecision.ASK,
    PermissionMode.ACCEPT_EDITS: PermissionDecision.ASK,
    PermissionMode.ALLOW: PermissionDecision.ASK,
    PermissionMode.AUTONOMOUS: PermissionDecision.DENY,
    PermissionMode.BYPASS: PermissionDecision.ALLOW,  # unreachable (ceiling is max); totality
}

#: Looseness rank for grant-restore filtering (D12): a persisted grant recorded under
#: mode X is honored only while the ACTIVE mode is at least as loose as X — a session
#: resumed under a TIGHTER posture re-asks (or denies) instead of inheriting grants.
_MODE_LOOSENESS: dict[PermissionMode, int] = {
    PermissionMode.DENY: 0,
    PermissionMode.ASK: 1,
    PermissionMode.ACCEPT_EDITS: 2,
    PermissionMode.ALLOW: 3,
    PermissionMode.AUTONOMOUS: 4,
    PermissionMode.BYPASS: 5,
}


# Recursive root/home ``rm`` detection (catastrophic-command blocklist). A match forces at least a
# confirmation prompt (a hard deny in ``deny``/``autonomous`` mode), regardless of the tier verdict.
# In CODE (not DANGEROUS_PATTERNS) to stay ReDoS-proof -- see _is_dangerous_recursive_rm.
def _names_root_or_home(arg: str) -> bool:
    """True if an ``rm`` target names the filesystem ROOT, a TOP-LEVEL directory or a HOME.

    ``/``, ``/*``, ``/etc``, ``/etc/*``, ``/opt/..``, ``~``, ``~user``, ``~/*``, ``$HOME``,
    ``${HOME}``, ``/root`` and ``/home/<user>`` (each with an optional trailing ``/`` or ``/*``)
    are the footgun. A DEEPER absolute or home-relative path — ``/opt/app/build``, ``/tmp/x``,
    ``~/.cache/pip``, ``$HOME/tmp`` — is an ordinary target (ADR-0085): the workspace, temp and
    cache directories an agent legitimately cleans are absolute paths, and flagging every
    ``/…`` hard-denied all of them in ``autonomous`` mode. Pure string work, no filesystem.
    """
    a = arg.strip("'\"")
    if a.startswith(("~", "$HOME", "${HOME}")):
        head, _, rest = a.partition("/")
        if head.startswith("~") or head in ("$HOME", "${HOME}"):
            return rest.strip("/") in ("", "*")
        return False  # $HOMEDIR/x and the like: not the home variable
    if not a.startswith("/"):
        return False
    parts = [p for p in posixpath.normpath(a).split("/") if p]
    if parts and parts[-1] == "*":
        parts.pop()
    if len(parts) <= 1:
        return True  # "/", "/*", "/etc", "/etc/*", "/root"
    return parts[0] == "home" and len(parts) == 2  # "/home/<user>", "/home/<user>/*"


def _is_dangerous_recursive_rm(command: str) -> bool:
    """True iff ``command`` runs ``rm`` RECURSIVELY against a root/home path.

    Detected by TOKENISATION (not a regex) so it can't ReDoS on a long ``rm -r -r …`` run and is
    not fooled by a glued ``-rf/etc`` form or by flags after the path. A recursive delete of a
    RELATIVE subdir (``rm -rf ./build``) is NOT flagged. Each shell segment (split on ``; & | ( )``)
    is examined on its own, so ``rm -rf build && echo /etc`` is not mis-flagged. A recursive flag is
    required: a short flag bundle containing r/R (-r/-rf/-fr/-Rf, in any position) or --recursive.
    """
    for segment in re.split(r"[;&|()]+", command):
        tokens = segment.split()
        rm_at = next((k for k, t in enumerate(tokens) if t.rsplit("/", 1)[-1] == "rm"), None)
        if rm_at is None:
            continue
        recursive = dangerous = False
        for arg in tokens[rm_at + 1 :]:
            if arg.startswith("--"):
                if arg.split("=", 1)[0] in ("--recursive", "--recurse"):
                    recursive = True
            elif arg.startswith("-"):
                run = re.match(r"-[a-zA-Z]*", arg)  # leading short-flag bundle (linear, no ReDoS)
                end = run.end() if run else 1
                flags, glued = arg[1:end], arg[end:]
                if any(c in "rR" for c in flags):
                    recursive = True
                if glued and _names_root_or_home(glued):  # glued path, e.g. -rf/etc, -rf~
                    dangerous = True
            elif _names_root_or_home(arg):
                dangerous = True
        if recursive and dangerous:
            return True
    return False


#: Patterns are intentionally conservative (aimed at clear footguns) so they rarely
#: false-positive on benign commands; the tier/mode gate is the primary control.
DANGEROUS_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    # Recursive root/home ``rm`` is NOT a regex here -- see :func:`_is_dangerous_recursive_rm`, a
    # tokeniser checked alongside this list in ``_dangerous_reason``. (A single regex for "rm,
    # recursively, any flag form, path at any position" needs two overlapping token-eaters -> O(n^2)
    # ReDoS on a long ``rm -r -r … `` run, and still misses glued ``-rf/etc``. Code is ReDoS-proof.)
    (re.compile(r"(^|\s)sudo(\s|$)"), "privilege escalation (sudo)"),
    (re.compile(r"\bmkfs\b|\bformat\s+[A-Za-z]:", re.IGNORECASE), "filesystem format"),
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
    # VCS-internals destruction (ADR-0038). Field incident 2026-08-27: an unattended small
    # model, chasing a phantom "stale ref", ran ``rm -f .git/objects/pack/*``, ``rm -rf
    # .git/refs/mind`` and finally re-cloned the repository — none of it matched the
    # recursive-root ``rm`` check (relative paths are exempt) or the destructive-git entry.
    # There is no agent task that needs to delete or move ``.git`` itself; ``git`` plumbing
    # (``update-ref -d``, ``fetch``, ``gc`` without ``--prune=now``) stays available.
    (
        re.compile(
            r"\bgit\s+(?:gc\b[^;&|\n]*--prune=now|prune\b(?!-packed)"
            r"|reflog\s+expire\b[^;&|\n]*--expire(?:-unreachable)?=now)"
        ),
        "destructive git maintenance (permanently discards unreachable history)",
    ),
    (
        re.compile(
            r"\b(?:rm|rmdir|mv|shred|unlink|truncate)\b[^;&|\n]*"
            r"(?:^|[\s\"'=])(?:[^\s\"';&|]*[\\/])?\.git(?=[\\/\s\"']|$)",
            re.IGNORECASE,
        ),
        "destruction of VCS internals (.git)",
    ),
    (
        re.compile(r">{1,2}\s*(?:[^\s;&|]*[\\/])?\.git[\\/]"),
        "shell write into VCS internals (.git/)",
    ),
    (
        re.compile(r"\bcurl\b.*\|\s*(sudo\s+)?(ba)?sh\b|\bwget\b.*\|\s*(ba)?sh\b"),
        "pipe-to-shell of remote content",
    ),
    # ── PowerShell equivalents (the powershell tool shares this blocklist) ──────
    (
        # Recursive Remove-Item of any absolute drive path or the profile/home (e.g.
        # Remove-Item -Recurse -Force C:\Windows\System32, or -r C:\Users\X). Accepts the
        # -Recurse abbreviations (-r / -rec / ... — all unambiguous prefixes PowerShell
        # honors). A single-file Remove-Item, and recursive deletes of relative paths,
        # stay benign.
        re.compile(
            r"\b(Remove-Item|ri|rd|rmdir|del|erase)\b"
            r"[^\n]*\s-r(?:e(?:c(?:u(?:r(?:s(?:e)?)?)?)?)?)?\b"
            r"[^\n]*([A-Za-z]:\\|\$HOME|\$env:USERPROFILE|~)",
            re.IGNORECASE,
        ),
        "recursive Remove-Item of a drive or home path",
    ),
    (
        # cmd.exe recursive delete (bash runs through cmd.exe on Windows): rd/rmdir/del/
        # erase with the /s switch targeting an absolute drive or profile path, in any
        # argument order. Plain `del file.txt` (no /s and no drive path) stays benign.
        re.compile(
            r"\b(rd|rmdir|del|erase)\b"
            r"(?=[^\n]*\s/[sS]\b)"
            r"(?=[^\n]*([A-Za-z]:\\|%USERPROFILE%|%HOMEPATH%))",
            re.IGNORECASE,
        ),
        "recursive delete (cmd.exe /s) of a drive or profile path",
    ),
    (
        re.compile(r"\b(Format-Volume|Clear-Disk|Remove-Partition)\b", re.IGNORECASE),
        "destructive disk operation (format / clear / remove-partition)",
    ),
    (
        # Download-and-execute: iwr/curl/Invoke-WebRequest piped into iex/Invoke-Expression.
        re.compile(
            r"\b(iwr|curl|Invoke-WebRequest|Invoke-RestMethod|irm)\b[^\n]*\|\s*"
            r"(iex|Invoke-Expression)\b",
            re.IGNORECASE,
        ),
        "download-and-execute (web request piped to Invoke-Expression)",
    ),
]


def scan_command_danger(command: str) -> str | None:
    """The catastrophic-blocklist reason ``command`` matches, or None.

    The shared scan over the built-in :data:`DANGEROUS_PATTERNS` regexes PLUS the recursive
    root/home ``rm`` tokeniser (:func:`_is_dangerous_recursive_rm`). Used by the settings-hook
    loader; :class:`PermissionPolicy` runs the same scan over its operator-extended instance list.
    """
    for pattern, reason in DANGEROUS_PATTERNS:
        if pattern.search(command):
            return reason
    if _is_dangerous_recursive_rm(command):
        return "recursive remove of a root or home path"
    return None


#: Argument keys whose string values are scanned against DANGEROUS_PATTERNS.
_SHELL_ARG_KEYS = ("command", "cmd", "script")

#: Protected-path floor (SELF-REMEDIATION Step 2). A write whose target matches one of these
#: sensitive locations is NEVER auto-allowed: it escalates to a confirmation prompt — and a hard
#: DENY in ``autonomous`` — even under ``allow``/``acceptEdits`` mode or a per-tool/session grant
#: (the safety check runs *before* the allow rules). These are the locations an unattended agent
#: must not silently modify: VCS internals (repo/hook corruption), the secrets file, the
#: installed dependencies (supply-chain tampering), and the agent's OWN config (a
#: self-permission-escalation vector). Matched against a write/edit tool's file-path arg;
#: ``.env.example`` / ``.gitignore`` / ``.github/`` deliberately do NOT match. Tighten-only, like
#: :data:`DANGEROUS_PATTERNS`. (Shell-driven writes are NOT scanned — a path in an arbitrary
#: command is usually a read/execute; that residual is the Step 3 sandbox's job.)
PROTECTED_PATH_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (
        re.compile(r"(?:^|[\\/\s\"'=>])\.git[\\/]", re.IGNORECASE),
        "VCS internals (.git/)",
    ),
    (
        # ``.env`` and layered/backup secrets (``.env.local``, ``.env.production.local``,
        # ``.env.local.bak``, ``.env~``) — any chain of dotted segments — but NOT a template
        # whose FIRST suffix is example/sample/template/dist (``.env.example``). The negative
        # lookahead guards only the first segment; ``~`` (editor backup) is an accepted terminator.
        re.compile(
            r"(?:^|[\\/\s\"'=>])"
            r"\.env(?:\.(?!example|sample|template|dist)\w+(?:\.\w+)*)?"
            r"(?=$|[\\/\s\"'~])",
            re.IGNORECASE,
        ),
        "secrets file (.env)",
    ),
    (
        re.compile(r"(?:^|[\\/\s\"'=>])(?:\.venv|venv|site-packages)[\\/]", re.IGNORECASE),
        "virtualenv / installed packages",
    ),
    # ``.claude/`` is deliberately NOT a built-in protected class (ADR-0029). The agent's
    # config — skills, rules, CLAUDE.md, the settings files themselves — is agent-editable by
    # default; a framework that wants any of it protected declares ``deny Edit|Write(glob)``
    # rules in ``.claude/settings.json`` / ``settings.local.json``, which ingest as
    # ``extra_protected_paths`` (always-on) — an Edit/Write deny binds writes, a Read deny
    # binds reads and writes (ADR-0030). The operator's settings
    # are the single authority over the agent's config; the engine hardcodes no opinion.
]

#: Argument keys whose string values are filesystem paths checked against
#: :data:`PROTECTED_PATH_PATTERNS` (write/edit tools use ``path``; the shell-arg keys are also
#: scanned so a redirect into a protected path is caught best-effort).
_FILE_ARG_KEYS = ("path", "file_path", "filename", "dest", "destination", "output")

#: Built-in protected classes that are WRITE-sensitive only: reading them is normal operation
#: (git tooling reads ``.git/``, the venv is read on every import), so READ_ONLY-tier tools
#: skip them. Field bug (2026-08-26, ADR-0028): a read was hard-denied as a "write to a
#: protected path", derailing the turn. ``.env`` is deliberately NOT here — reading a secrets
#: file is itself the leak (the floor's charter: "read/rewrite secrets"). Operator/settings-
#: ingested extras are also never here: a CC ``deny Read(glob)`` rule ingests as a protected
#: path, so extras must bind reads too.
_WRITE_ONLY_PROTECTED = frozenset(
    {
        "VCS internals (.git/)",
        "virtualenv / installed packages",
    }
)

#: Description suffix marking a COMPILED extra pattern as write-sensitive only (ADR-0030).
#: Settings-ingested ``deny Edit|Write|MultiEdit(glob)`` gestures compile with
#: ``compile_protected_paths(..., write_only=True)`` so READ_ONLY-tier tools skip them —
#: CC semantics: an Edit deny does not block reading the path. ``deny Read(glob)`` gestures
#: and operator ``ZAKCODE_PROTECTED_PATHS`` regexes (no verb available — strict) never
#: carry it and bind reads and writes alike (strict — no verb available for env regexes).
_WRITE_ONLY_MARK = " (write-only)"

logger = logging.getLogger("zakcode.permissions")


def compile_deny_patterns(specs: list[str]) -> list[tuple[re.Pattern[str], str]]:
    """Compile operator-supplied deny regex strings into blocklist patterns.

    This is the small "grammar" for adding project- or operator-specific denies
    (e.g. from ``ZAKCODE_DENIED_COMMANDS``): each string is a case-insensitive
    regex matched against shell-command arguments. Added patterns can only ever
    *tighten* the verdict (they ride the same escalate-to-prompt / deny path as the
    built-in blocklist and are never able to loosen it). An invalid regex is skipped
    with a warning rather than raising.
    """
    compiled: list[tuple[re.Pattern[str], str]] = []
    for spec in specs:
        if not isinstance(spec, str) or not spec.strip():
            continue
        try:
            compiled.append((re.compile(spec, re.IGNORECASE), f"operator deny rule: {spec}"))
        except re.error as exc:
            logger.warning("ignoring invalid deny pattern %r: %s", spec, exc)
    return compiled


def compile_protected_paths(
    specs: list[str], *, write_only: bool = False
) -> list[tuple[re.Pattern[str], str]]:
    """Compile operator-supplied protected-path regex strings (e.g. ``ZAKCODE_PROTECTED_PATHS``).

    Each string is a case-insensitive regex appended to the built-in
    :data:`PROTECTED_PATH_PATTERNS`; it can only ever tighten (a write matching it escalates /
    hard-denies in autonomous, never loosens). An invalid regex is skipped with a warning.
    With ``write_only=True`` (settings-ingested Edit/Write/MultiEdit denies) the compiled
    descriptions carry :data:`_WRITE_ONLY_MARK`, so READ_ONLY-tier tools skip them.
    """
    mark = _WRITE_ONLY_MARK if write_only else ""
    compiled: list[tuple[re.Pattern[str], str]] = []
    for spec in specs:
        if not isinstance(spec, str) or not spec.strip():
            continue
        try:
            compiled.append(
                (re.compile(spec, re.IGNORECASE), f"operator protected path: {spec}{mark}")
            )
        except re.error as exc:
            logger.warning("ignoring invalid protected-path pattern %r: %s", spec, exc)
    return compiled


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


def parse_permission_answer(answer: str) -> PermissionOutcome | None:
    """Map an operator's permission answer to an outcome, or ``None`` if unrecognized.

    The ONE y/a/n answer grammar for every surface that reads answers off the say
    contract — the CLI's console/mux prompt and the server's inbox prompter parse
    identically. Accepts the option numbers (``1`` / ``2`` / ``3``), the single-key
    hints (``y`` / ``a`` / ``n``), and the spelled-out phrases shown in the prompt
    (``allow once`` / ``allow for session`` / ``deny``) plus common synonyms, so an
    operator who types the words they see is understood rather than silently denied.
    """
    a = answer.strip().lower()
    if a in (
        "2",
        "a",
        "always",
        "session",
        "allow session",
        "allow for session",
        "allow for the session",
        "allow always",
    ):
        return PermissionOutcome.ALLOW_SESSION
    if a in ("1", "y", "yes", "o", "once", "allow", "allow once"):
        return PermissionOutcome.ALLOW_ONCE
    if a in ("3", "n", "no", "d", "deny"):
        return PermissionOutcome.DENY_ONCE
    return None


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
        extra_dangerous_patterns: list[tuple[re.Pattern[str], str]] | None = None,
        confirm_tools: set[str] | None = None,
        tool_mode_overrides: dict[str, PermissionMode | str] | None = None,
        declared_packages: Callable[[], set[str]] | None = None,
        protected_path_patterns: list[tuple[re.Pattern[str], str]] | None = None,
        extra_protected_paths: list[tuple[re.Pattern[str], str]] | None = None,
        extra_denied_tools: set[str] | frozenset[str] | None = None,
        workspace_root: str | os.PathLike[str] | None = None,
    ) -> None:
        self.mode = PermissionMode.parse(mode)
        self.prompter = prompter
        # The root a RELATIVE file-path argument resolves against for the protected-path scan
        # (ADR-0031). CC matches path rules against absolute paths (its file tools take
        # absolute paths), and a framework's ``*/`` -prefixed deny globs need a parent
        # segment — so the scan tests the workspace-resolved absolute form alongside the raw
        # argument, or a relative spelling of a denied path walks straight past the rule.
        # String math only (``normpath``), never a filesystem call: decide() stays pure.
        self._workspace_root: str | None = (
            os.path.normpath(os.fspath(workspace_root)) if workspace_root is not None else None
        )
        # Per-tool trust overrides (audit P0-2b / D12): the named tool is judged under
        # ITS mode instead of the session mode — both directions (loosen or tighten).
        # The dangerous-command floor in an autonomous SESSION is never loosenable.
        self.tool_mode_overrides: dict[str, PermissionMode] = {
            name: PermissionMode.parse(value) for name, value in (tool_mode_overrides or {}).items()
        }
        # ``dangerous_patterns`` REPLACES the baseline (advanced); the common path
        # leaves it None and uses the built-in list. ``extra_dangerous_patterns`` is
        # APPENDED, so operator/project deny rules (see ``compile_deny_patterns``)
        # can only ever tighten the blocklist, never remove a built-in footgun.
        base = DANGEROUS_PATTERNS if dangerous_patterns is None else dangerous_patterns
        self.dangerous_patterns = [*base, *(extra_dangerous_patterns or [])]
        # Tools the operator wants to confirm on EVERY call regardless of their declared tier
        # (e.g. ``web_fetch`` egress when ``web_fetch_confirm`` is on). Like the dangerous-pattern
        # check this can only TIGHTEN a verdict — but unlike it, the prompt is session-grantable.
        self._confirm_tools = set(confirm_tools or ())
        # Declared-vs-undeclared dependency gate (self-remediation Step 1, see
        # ``docs/SELF-REMEDIATION.md``). A callable that returns the project's declared
        # package set (read lazily from manifests). ``None`` = gate OFF (the default, so the
        # pure decision matrix and every existing caller are unchanged). Like the dangerous
        # and confirm checks this only ever TIGHTENS: an install of a package no manifest
        # vouches for escalates to ASK (a hard DENY in ``autonomous`` — there is no execution
        # sandbox yet). The provider is invoked only when a command actually names an install.
        self._declared_packages = declared_packages
        # Protected-path floor (self-remediation Step 2): the built-in PROTECTED_PATH_PATTERNS
        # plus any operator/project extras (``extra_protected_paths``), appended so they only
        # ever TIGHTEN. A write to one of these never auto-allows — it re-decides to ASK (hard
        # DENY in autonomous) even under ``allow`` mode or a grant, closing the "a blanket grant
        # silently waives a sensitive write" hole. ``protected_path_patterns`` REPLACES the
        # baseline (used by child_view to pass the already-combined list); ``extra_*`` appends.
        protected_base = (
            PROTECTED_PATH_PATTERNS if protected_path_patterns is None else protected_path_patterns
        )
        self.protected_path_patterns = [*protected_base, *(extra_protected_paths or [])]
        # Whole-tool denials (e.g. an ingested CC ``deny: ["WebFetch"]``): the named tool is denied
        # UNCONDITIONALLY in decide(), regardless of its tier — unlike a ``deny`` mode override,
        # which only lowers the tier ceiling and so cannot bind a read-only tool. Tighten-only.
        self._denied_tools: frozenset[str] = frozenset(extra_denied_tools or ())
        # 'allow' grants are keyed by TOOL NAME — a grant covers the whole tool for the
        # session (the operator is not re-prompted for every new path/command). 'deny'
        # grants stay keyed by the exact (tool, args) so a block is narrow.
        self._session_allow: set[str] = set()
        self._session_deny: set[str] = set()
        # Persistable record of every operator grant (audit P0-2d / D12 / Q5): the
        # sets above drive runtime decisions; this log is what survives a restart
        # (exported into the session document, rehydrated via restore_grants).
        self._grant_log: list[dict[str, str]] = []

    def child_view(self) -> PermissionPolicy:
        """A policy for a delegated child: same immutable posture, isolated session grants.

        Shares this policy's mode, prompter, and the full (baseline + operator) dangerous-
        pattern blocklist, but starts with its OWN empty per-session allow/deny sets — so an
        ALLOW_SESSION (or DENY_SESSION) decision answered inside an "isolated" child does not
        silently widen or narrow the parent or its sibling children. The deny-first gate and
        the never-waivable catastrophic blocklist are unchanged. (audit2 #10)
        """
        return PermissionPolicy(
            self.mode,
            prompter=self.prompter,
            dangerous_patterns=self.dangerous_patterns,
            confirm_tools=self._confirm_tools,
            tool_mode_overrides=dict(self.tool_mode_overrides),
            declared_packages=self._declared_packages,
            protected_path_patterns=self.protected_path_patterns,
            extra_denied_tools=self._denied_tools,
            workspace_root=self._workspace_root,
        )

    # ── public read accessors (so clients never touch the private sets) ────────

    def session_allow(self) -> list[str]:
        """The per-session 'allow' grants (tool names), sorted (a read-only copy)."""
        return sorted(self._session_allow)

    def session_deny(self) -> list[str]:
        """The per-session 'deny' blocks (exact-call keys), sorted (a read-only copy)."""
        return sorted(self._session_deny)

    def auto_allows(self, spec: ToolSpec | None, arguments: dict) -> bool:
        """True iff a call would be allowed WITHOUT prompting (allow-mode or already granted).

        Used by the recipe harness-run to fire a synthetic verification command ONLY where
        it would not raise an uninitiated prompt. Never relaxes the gate: a dangerous
        command is not auto-allowed even under a session grant (it re-decides to ASK/DENY).
        """
        tool_name = spec.name if spec is not None else "<unknown>"
        if self._key(tool_name, arguments) in self._session_deny:
            return False
        if (
            tool_name in self._session_allow
            and self._dangerous_reason(arguments) is None
            and not self._undeclared_install(arguments)
            and self._protected_path_reason(arguments) is None
        ):
            # Mirror authorize()'s grant guard (stack review minor #1): a grant may
            # only ever resolve ASK→ALLOW, so this read-only hint must also re-decide —
            # else the recipe harness-run could auto-fire a call that authorize would
            # statically DENY (e.g. a grant restored into a tighter posture). A grant
            # likewise cannot waive the dependency gate (an undeclared install re-decides
            # below, exactly like the dangerous floor).
            decision, _ = self.decide(spec, arguments)
            return decision is not PermissionDecision.DENY
        decision, _ = self.decide(spec, arguments)
        return decision is PermissionDecision.ALLOW

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
                # Recursive root/home rm is a tokeniser check (ReDoS-proof), always on -- a built-in
                # footgun is never removable via a custom dangerous_patterns list.
                if _is_dangerous_recursive_rm(value):
                    return "recursive remove of a root or home path"
        return None

    def dangerous_reason(self, arguments: dict) -> str | None:
        """Public: the catastrophic-blocklist reason ``arguments`` match, or None.

        Lets a caller re-check the NEVER-WAIVABLE blocklist against arguments that changed
        after authorization — e.g. a PreToolUse hook that rewrote a command. (audit3 #5)
        """
        return self._dangerous_reason(arguments)

    def _protected_path_reason(self, arguments: dict, *, read_only: bool = False) -> str | None:
        """The protected-path floor ``arguments`` matches, or None (Step 2).

        Scans the FILE-PATH args (``path``/``file_path``/…) against
        :data:`PROTECTED_PATH_PATTERNS`. Pure, no I/O. With ``read_only=True`` (the calling
        tool's tier is READ_ONLY) the write-sensitive patterns are skipped — the
        :data:`_WRITE_ONLY_PROTECTED` built-ins (reading ``.git/`` or the venv is normal
        operation) and every :data:`_WRITE_ONLY_MARK`-compiled extra (an ingested Edit/Write
        deny does not block reading, per CC semantics) — while secrets (``.env``),
        ``deny Read(...)``-ingested extras, and operator env regexes still bind.
        Shell args are deliberately NOT scanned:
        a path inside an arbitrary command is far more often a READ/EXECUTE (``.venv/bin/python
        x``, ``cat .git/config``) than a write, so matching it over-blocks legitimate commands —
        the same "don't parse shell intent" lesson as the dependency gate. A shell-driven write
        to a protected path is part of the best-effort residual the Step 3 sandbox contains.
        """
        for key in _FILE_ARG_KEYS:
            value = arguments.get(key)
            if not isinstance(value, str):
                continue
            # Tighten-only union: the raw argument AND (for a relative path, when a workspace
            # root is known) its resolved absolute form. Either matching binds.
            candidates = [value]
            if self._workspace_root is not None and value and not os.path.isabs(value):
                candidates.append(os.path.normpath(os.path.join(self._workspace_root, value)))
            for pattern, description in self.protected_path_patterns:
                if read_only and (
                    description in _WRITE_ONLY_PROTECTED or description.endswith(_WRITE_ONLY_MARK)
                ):
                    continue
                if any(pattern.search(candidate) for candidate in candidates):
                    return description
        return None

    def protected_path_reason(self, arguments: dict, *, read_only: bool = False) -> str | None:
        """Public: the protected-path reason ``arguments`` matches, or None.

        Mirrors :meth:`dangerous_reason` so the loop can re-apply the protected-path floor to a
        PreToolUse-rewritten call (so a hook can't rewrite a benign edit into a write to
        ``.env`` / ``.git/`` / the venv). File-path args only; pass
        ``read_only=True`` for a READ_ONLY-tier tool so write-sensitive built-ins don't bind.
        ``None`` unconditionally under session-wide ``bypassPermissions`` — that mode waives
        the protected-path floor everywhere, the loop's rewrite re-check included.
        """
        if self.mode is PermissionMode.BYPASS:
            return None
        return self._protected_path_reason(arguments, read_only=read_only)

    def _undeclared_install(self, arguments: dict) -> list[str]:
        """Package-install targets in the command that NO project manifest declares.

        Always empty when the gate is off (no ``declared_packages`` provider injected), so
        the pure decision matrix is unchanged by default. When on, it parses each shell
        argument for an explicit install (:func:`zakcode.deps_gate.installed_specs`) and —
        only if some package is actually named — reads the declared set and returns the ones
        not in it. The manifest read is wrapped: a read failure yields an empty declared set
        (so every named package counts as undeclared → escalate), never a crash.
        """
        if self._declared_packages is None:
            return []
        if self.mode is PermissionMode.BYPASS:
            # Bypass runs with the dependency gate OFF — one switch read by decide(), the
            # grant fast-path, and the loop's post-rewrite re-check alike.
            return []
        specs: list[str] = []
        seen: set[str] = set()
        for key in _SHELL_ARG_KEYS:
            value = arguments.get(key)
            if isinstance(value, str) and value.strip():
                for spec in installed_specs(value):
                    if spec not in seen:
                        seen.add(spec)
                        specs.append(spec)
        if not specs:  # no explicit install named → no manifest read, gate is a no-op
            return []
        try:
            declared = self._declared_packages()
        except Exception:  # pragma: no cover - never let a manifest read crash authorization
            logger.warning(
                "dependency gate: declared-package read failed; treating none as declared"
            )
            declared = set()
        # ``specs`` and ``declared`` are ecosystem-tagged; compare tagged, return the human name.
        return [display_name(spec) for spec in specs if spec not in declared]

    def undeclared_install_reason(self, arguments: dict) -> str | None:
        """Public: a reason if ``arguments`` install a package no manifest declares, else None.

        Mirrors :meth:`dangerous_reason` so a caller can re-apply the dependency gate to
        arguments that changed AFTER authorization — e.g. the loop's re-check of a command a
        PreToolUse hook rewrote (audit3 #5), so a hook can't turn an authorized ``echo hi`` into
        an undeclared ``pip install evil``. Returns None when the gate is off or all declared.
        """
        undeclared = self._undeclared_install(arguments)
        if not undeclared:
            return None
        return "undeclared package install: " + ", ".join(undeclared)

    def undeclared_install_targets(self, arguments: dict) -> list[str]:
        """Public: the undeclared install targets in ``arguments`` (display names), or ``[]``.

        The list form of :meth:`undeclared_install_reason`, for callers that must COMPARE two
        commands — the loop's post-rewrite re-check blocks only the targets a hook rewrite
        INTRODUCED relative to the authorized original, so a hook that rewrites every command
        (a Mind deployment's env prepend) cannot nullify an operator-approved install.
        """
        return self._undeclared_install(arguments)

    def decide(self, spec: ToolSpec | None, arguments: dict) -> tuple[PermissionDecision, str]:
        """Return the static (pre-prompt, stateless) verdict and a human reason.

        The dangerous-pattern check only ever tightens the tier/mode verdict. A
        per-tool trust override (``tool_mode_overrides``) substitutes the EFFECTIVE
        mode for the named tool — but when the SESSION mode is ``autonomous``, a
        dangerous-pattern match is a hard DENY regardless of any per-tool loosening
        (D12 invariant: overrides cannot loosen the dangerous floor in autonomous).

        The dependency gate (when enabled) is the same shape of tighten-only check,
        applied after the dangerous-pattern one: an install of a package the project's
        manifests/lockfile do not declare escalates to ASK (hard DENY in autonomous).
        """
        tool_name = spec.name if spec is not None else "<unknown>"
        # A configuration-denied tool is denied UNCONDITIONALLY, regardless of tier — a whole-tool
        # deny (e.g. an ingested ``deny: ["WebFetch"]``) must bind even a read-only tool, which a
        # mere ``deny`` mode override cannot (that only lowers the tier ceiling). Tighten-only.
        if tool_name in self._denied_tools:
            return (PermissionDecision.DENY, f"'{tool_name}' is denied by configuration")
        mode = self.tool_mode_overrides.get(tool_name, self.mode)
        tier = self._required_tier(spec)
        ceiling = _MODE_CEILING[mode]
        base = PermissionDecision.ALLOW if tier <= ceiling else _ABOVE_CEILING[mode]

        # Operator-requested confirm-on-every-call tools (e.g. web_fetch egress). Tighten-only,
        # like the dangerous-pattern check: blocked outright in ``deny`` mode, otherwise escalated
        # to a (session-grantable) prompt. Never loosens an existing ASK/DENY. ``autonomous``
        # never prompts, so a required confirmation there fails closed deterministically.
        confirm_escalated = False
        if (
            tool_name in self._confirm_tools
            and base is PermissionDecision.ALLOW
            # ``bypassPermissions`` waives per-call confirmations outright (its whole point
            # is that nothing prompts and nothing fails closed for lack of a prompter).
            and PermissionMode.BYPASS not in (mode, self.mode)
        ):
            if mode is PermissionMode.DENY:
                return (PermissionDecision.DENY, f"'{tool_name}' is blocked in 'deny' mode")
            if PermissionMode.AUTONOMOUS in (mode, self.mode):
                return (
                    PermissionDecision.DENY,
                    f"'{tool_name}' requires per-call confirmation, which 'autonomous' "
                    "mode never prompts for",
                )
            base = PermissionDecision.ASK
            confirm_escalated = True

        danger = self._dangerous_reason(arguments)
        if danger is not None:
            # D12: in autonomous OR bypass (session-wide or per-tool), a catastrophic match
            # is a deterministic hard DENY — never a prompt, attended or headless. Bypass
            # waives prompts and gates, never the catastrophic floor: that floor is uniform
            # in every mode, which is what keeps it one rule with no exceptions.
            if {PermissionMode.AUTONOMOUS, PermissionMode.BYPASS} & {mode, self.mode}:
                return (PermissionDecision.DENY, f"blocked dangerous command: {danger}")
            if mode is PermissionMode.DENY or base is PermissionDecision.DENY:
                return (PermissionDecision.DENY, f"blocked dangerous command: {danger}")
            # Force a confirmation even if tier/mode would have auto-allowed.
            return (PermissionDecision.ASK, f"dangerous command: {danger}")

        # Declared-vs-undeclared dependency gate (self-remediation Step 1). Tighten-only,
        # exactly like the dangerous-pattern check above: an install of a package no manifest
        # vouches for is the typosquatting / malicious-dependency / injection-escalation
        # vector. In ``autonomous`` (session-wide or per-tool) it is a deterministic hard DENY
        # — there is no execution sandbox yet — surfaced as a recoverable tool error. In
        # ``deny`` (or when tier/mode already denies) it stays DENY; otherwise it escalates to
        # a (session-grantable) prompt. Declared installs and lockfile/local installs pass
        # through untouched. The gate is OFF unless a provider was injected (default).
        undeclared = self._undeclared_install(arguments)
        # ``bypassPermissions`` runs with the dependency gate off (session-wide bypass
        # already short-circuits inside _undeclared_install; this also covers a per-tool
        # bypass override under a stricter session mode).
        if undeclared and PermissionMode.BYPASS not in (mode, self.mode):
            targets = ", ".join(undeclared)
            if PermissionMode.AUTONOMOUS in (mode, self.mode):
                return (
                    PermissionDecision.DENY,
                    f"blocked undeclared package install: {targets} "
                    "(not in the project's manifests/lockfile)",
                )
            if mode is PermissionMode.DENY or base is PermissionDecision.DENY:
                return (
                    PermissionDecision.DENY,
                    f"blocked undeclared package install: {targets}",
                )
            return (
                PermissionDecision.ASK,
                f"undeclared package install: {targets} (not in the project's manifests/lockfile)",
            )

        # Protected-path floor (self-remediation Step 2). Same tighten-only shape: a write to a
        # sensitive location (.git/, .env, the venv) never auto-allows — it
        # escalates to a (session-grantable) prompt, and is a hard DENY in ``autonomous`` (no
        # prompt; an unattended agent must not silently corrupt the repo, read/rewrite secrets,
        # or tamper with installed deps). READ_ONLY-tier tools skip
        # the write-sensitive built-ins (reading a skill file is what a skill file is for) but
        # still bind on secrets and operator extras. The grant fast-paths in
        # authorize()/auto_allows() re-decide so a blanket grant cannot waive this.
        read_only = tier is PermissionTier.READ_ONLY
        # ``bypassPermissions`` waives the protected-path floor too — dangerously, by name:
        # the operator has declared the surrounding stack the guardrail layer.
        protected = (
            None
            if PermissionMode.BYPASS in (mode, self.mode)
            else self._protected_path_reason(arguments, read_only=read_only)
        )
        if protected is not None:
            verb = "read of" if read_only else "write to"
            if PermissionMode.AUTONOMOUS in (mode, self.mode):
                return (PermissionDecision.DENY, f"blocked {verb} a protected path: {protected}")
            if mode is PermissionMode.DENY or base is PermissionDecision.DENY:
                return (PermissionDecision.DENY, f"blocked {verb} a protected path: {protected}")
            return (PermissionDecision.ASK, f"{verb} a protected path: {protected}")

        if base is PermissionDecision.ALLOW:
            return (base, "")
        if base is PermissionDecision.DENY:
            return (base, f"'{tier.name}' is blocked in '{mode.value}' mode")
        if confirm_escalated:
            return (base, f"'{tool_name}' requires confirmation before reaching the network")
        return (base, f"'{tier.name}' requires confirmation in '{mode.value}' mode")

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
        # A session 'allow' grant covers the whole TOOL for the rest of the session, so
        # the operator is not re-prompted for every new path/command. FOUR things a grant
        # can NEVER waive (D12: grants resolve ASK→ALLOW only): the dangerous-command
        # blocklist (a dangerous call re-decides below), an UNDECLARED package install (the
        # dependency gate — a blanket "allow bash" must not silently authorize fetching and
        # running an unvetted package), a write to a PROTECTED PATH (.git/.env/venv/config —
        # Step 2: a blanket grant must not silently authorize a sensitive write), and a static
        # DENY — re-decide so a grant restored into a tighter posture cannot override a block.
        if (
            tool_name in self._session_allow
            and self._dangerous_reason(arguments) is None
            and not self._undeclared_install(arguments)
            and self._protected_path_reason(arguments) is None
        ):
            decision, reason = self.decide(spec, arguments)
            if decision is PermissionDecision.DENY:
                return (False, reason)
            return (True, "allowed for this session")

        decision, reason = self.decide(spec, arguments)
        if decision is PermissionDecision.ALLOW:
            return (True, "")
        if decision is PermissionDecision.DENY:
            # Operator-facing audit trail (audit P1-5; D12 asks for the autonomous
            # hard-deny to be logged): the model only sees the recoverable error.
            logger.warning("denied %r in '%s' mode: %s", tool_name, self.mode.value, reason)
            return (False, reason)

        # decision is ASK → need the operator. No prompter ⇒ fail closed.
        if self.prompter is None:
            logger.warning("denied %r (no prompter to escalate to): %s", tool_name, reason)
            return (False, f"requires confirmation but none is available: {reason}")

        request = PermissionRequest(
            tool_name=tool_name,
            tier=self._required_tier(spec),
            arguments=arguments,
            reason=reason,
        )
        outcome = await self.prompter.confirm(request)
        if outcome is PermissionOutcome.ALLOW_SESSION:
            self._session_allow.add(tool_name)  # grant the whole tool, not the exact args
            self._record_grant("allow", tool_name, "*")
            return (True, "")
        if outcome is PermissionOutcome.ALLOW_ONCE:
            return (True, "")
        if outcome is PermissionOutcome.DENY_SESSION:
            self._session_deny.add(key)
            self._record_grant("deny", tool_name, key)
            return (False, "denied for this session by the operator")
        return (False, "denied by the operator")

    # ── grant persistence (audit P0-2d / D12 / Q5) ─────────────────────────────

    def _record_grant(self, kind: str, tool: str, args_scope: str) -> None:
        self._grant_log.append(
            {
                "kind": kind,
                "tool": tool,
                "args_scope": args_scope,
                "mode_at_grant": self.mode.value,
                "timestamp": datetime.now(UTC).isoformat(),
            }
        )

    def export_grants(self) -> list[dict[str, str]]:
        """The operator's session grants as persistable records.

        Record shape (D12): ``{kind: allow|deny, tool, args_scope ('*' for a
        whole-tool allow; the exact ``tool::args`` key for a deny), mode_at_grant,
        timestamp}``. The loop snapshots this into the session document on every
        persist, so grants survive a restart.
        """
        return [dict(record) for record in self._grant_log]

    def restore_grants(self, records: list[dict[str, str]] | None) -> int:
        """Rehydrate persisted grants under the D12 constraints; returns how many held.

        An 'allow' grant is honored only when the ACTIVE mode is at least as loose as
        the mode it was granted under — a session resumed under a tighter posture
        re-asks (or denies) instead of inheriting looser-mode grants. 'deny' grants
        are always kept (keeping a block can only tighten). Grants can still never
        waive the dangerous blocklist or a static DENY (see :meth:`authorize`).
        Malformed records are skipped, never fatal — dropping a grant only re-asks.
        """
        kept = 0
        for record in records or []:
            if not isinstance(record, dict):
                continue
            if record in self._grant_log:
                continue  # dedup: a doc with repeated records must not grow the log
            kind = record.get("kind")
            tool = record.get("tool")
            scope = record.get("args_scope")
            if kind == "deny" and isinstance(scope, str) and scope:
                self._session_deny.add(scope)
                self._grant_log.append(dict(record))
                kept += 1
                continue
            if kind != "allow" or not isinstance(tool, str) or not tool:
                continue
            granted_under = PermissionMode.parse(record.get("mode_at_grant"))
            if _MODE_LOOSENESS[self.mode] >= _MODE_LOOSENESS[granted_under]:
                self._session_allow.add(tool)
                self._grant_log.append(dict(record))
                kept += 1
        return kept


__all__ = [
    "PermissionMode",
    "PermissionDecision",
    "PermissionRequest",
    "PermissionOutcome",
    "PermissionPrompter",
    "PermissionPolicy",
    "DANGEROUS_PATTERNS",
    "scan_command_danger",
    "compile_deny_patterns",
]
