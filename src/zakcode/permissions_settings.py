"""Ingest Claude Code ``permissions`` rules into Zak Code's deny-first policy (Phase 3).

Claude Code stores its authorization posture as a ``permissions`` block in
``.claude/settings.json`` (and the per-machine ``.claude/settings.local.json``):
three arrays — ``allow`` / ``deny`` / ``ask`` — of *gestures* in the grammar
``ToolName(pattern)`` (or a bare ``ToolName`` for the whole tool). A framework built on
Claude Code (e.g. a constitutional rule set that denies ``git reset --hard`` and protects
its own state files) ships its safety policy this way.

This module reads those gestures and TRANSLATES them into the existing tightening seams of
:class:`~zakcode.permissions.PermissionPolicy` — never a new enforcement path. Three seams,
all of which the policy treats as *tighten-only* (they ride the same escalate-to-prompt /
hard-deny machinery as the built-in floor and can never loosen it):

================================  ==============================================================
CC gesture                        Zak Code seam
================================  ==============================================================
``deny Bash(cmd-glob)``           a command deny **regex string** → ``compile_deny_patterns``
                                  → ``extra_dangerous_patterns``
``deny Read|Edit|Write|          a protected-path **regex string** → ``compile_protected_paths``
MultiEdit(path-glob)``            → ``extra_protected_paths``
``deny ToolName`` / ``           ``extra_denied_tools`` — an unconditional, tier-independent deny
ToolName(*)``
``allow ToolName(...)``           a ``tool_mode_overrides`` entry pinning that tool to ``allow``
``ask  ...``                      the default posture (prompt) — recorded as skipped, no-op
================================  ==============================================================

THE SECURITY INVARIANT (the whole point). The never-waivable safety FLOOR is preserved regardless of
what is ingested. A whole-tool ``deny`` denies that tool UNCONDITIONALLY (``extra_denied_tools``,
tier-independent — even a read-only tool). A whole-tool ``allow`` is a per-tool ``allow`` *mode*:
it can loosen a benign call for that tool, but it still runs — before any allow
decision — the never-waivable catastrophic-command floor and the protected-path floor: a
catastrophic command (``rm -rf /``) escalates to a confirmation prompt (a hard DENY in an
``autonomous`` session, and a fail-closed DENY when no prompter is present), and a write to a
protected path (``.env`` / ``.git/`` / the venv / the agent's config) does likewise. So an
ingested ``allow: ["Bash(*)"]`` can NEVER auto-allow ``rm -rf /`` or a write to ``.env``. The
floor is enforced by :meth:`PermissionPolicy.decide`, which runs the dangerous-pattern and
protected-path checks *ahead of* the allow rules; this module only feeds that machinery and
adds nothing that could bypass it. See ``tests/test_cc_permissions.py`` for the proof.

Discovery mirrors :mod:`zakcode.hooks.settings_loader`: both ``.claude/settings.json`` and
``.claude/settings.local.json`` are read (local layered after shared); a missing file is a
silent no-op; a parse error is recorded in the returned ``errors`` dict, never raised. Any
gesture with no clean, tightening mapping (an exotic per-arg glob, a CC-only tool such as
``AskUserQuestion``) is recorded in ``errors`` with a reason — never silently dropped or
mis-mapped. This module is OFF by default; the Agent wires it in only behind
``settings_permissions`` (env ``ZAKCODE_SETTINGS_PERMISSIONS``).
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

from zakcode.hooks import _CLAUDE_CODE_TOOL_NAMES

logger = logging.getLogger("zakcode.permissions_settings")

#: Inverse of :data:`zakcode.hooks._CLAUDE_CODE_TOOL_NAMES` — a Claude Code tool name (``"Bash"``,
#: ``"Read"``, ``"MultiEdit"``) → the canonical Zak Code tool it gates (``"bash"``, ``"read_file"``,
#: ``"edit_file"``). Built from the single source of truth in ``hooks`` so the two never drift; one
#: CC name maps to exactly one Zak tool (``Edit`` and ``MultiEdit`` both fold onto ``edit_file``).
_CC_TO_ZAK_TOOL: dict[str, str] = {
    cc_name: zak_name
    for zak_name, cc_names in _CLAUDE_CODE_TOOL_NAMES.items()
    for cc_name in cc_names
}

#: The CC tools whose ``(pattern)`` is a SHELL-COMMAND glob (matched against the command string),
#: not a filesystem path. Only ``Bash`` today (``powershell`` has no distinct CC counterpart).
_COMMAND_TOOLS: frozenset[str] = frozenset({"Bash"})

#: The CC tools whose ``(pattern)`` is a FILESYSTEM-PATH glob (matched against a write/edit/read
#: tool's path argument) → a protected-path rule. These are the file tools the protected-path floor
#: already understands (it scans ``path``/``file_path``/… args).
_PATH_TOOLS: frozenset[str] = frozenset({"Read", "Edit", "Write", "MultiEdit"})


@dataclass
class IngestedPermissions:
    """The translated, tighten-only seams ready to hand to :class:`PermissionPolicy`.

    Each field maps 1:1 to a ``PermissionPolicy`` constructor seam. ``denied_command_regexes``
    and ``protected_path_regexes`` are raw regex STRINGS (compiled by ``compile_deny_patterns`` /
    ``compile_protected_paths`` at wiring time, exactly like the operator ``denied_commands`` /
    ``protected_paths`` settings); ``tool_mode_overrides`` is the per-tool mode map. All are
    additive and tighten-only — see the module docstring's security invariant.
    """

    #: Regex strings for ``extra_dangerous_patterns`` (via ``compile_deny_patterns``).
    denied_command_regexes: list[str] = field(default_factory=list)
    #: Regex strings for ``extra_protected_paths`` (via ``compile_protected_paths``).
    protected_path_regexes: list[str] = field(default_factory=list)
    #: ``tool_mode_overrides`` entries — ALLOW gestures only (a whole-tool DENY goes to
    #: ``denied_tools`` so it binds unconditionally, regardless of the tool's tier).
    tool_mode_overrides: dict[str, str] = field(default_factory=dict)
    #: Whole-tool DENY gestures → ``PermissionPolicy(extra_denied_tools=...)``: a tier-independent
    #: unconditional deny (a mode override could not bind a read-only tool like ``web_fetch``).
    denied_tools: set[str] = field(default_factory=set)

    def is_empty(self) -> bool:
        """True when nothing mappable was found (so the Agent can skip wiring entirely)."""
        return not (
            self.denied_command_regexes
            or self.protected_path_regexes
            or self.tool_mode_overrides
            or self.denied_tools
        )


def _parse_gesture(gesture: str) -> tuple[str, str | None]:
    """Split a CC gesture into ``(tool_name, pattern)``; ``pattern`` is None for a bare tool.

    ``"Bash(git reset --hard*)"`` → ``("Bash", "git reset --hard*")``;
    ``"Read"`` → ``("Read", None)``; ``"Bash()"`` → ``("Bash", "")``. The pattern may itself
    contain parentheses (e.g. a shell substitution), so we split on the FIRST ``(`` and require a
    trailing ``)``; anything malformed returns ``(gesture, None)`` and is handled upstream.
    """
    text = gesture.strip()
    open_paren = text.find("(")
    if open_paren == -1:
        return (text, None)  # bare ``ToolName``
    if not text.endswith(")"):
        # Malformed — no closing paren. Treat the whole thing as a (probably-unknown) tool name so
        # the caller records it rather than silently parsing a half-gesture.
        return (text, None)
    tool = text[:open_paren].strip()
    pattern = text[open_paren + 1 : -1]
    return (tool, pattern)


def _is_whole_tool_pattern(pattern: str | None) -> bool:
    """True when a pattern denotes the WHOLE tool (a bare tool, ``()``, ``*``, or ``**``).

    These map to a per-tool mode override rather than a command/path regex — a ``*``-only path
    pattern would otherwise compile to ``.*`` and protect EVERY path (breaking all writes), and a
    ``*``-only command pattern would match every command. Routing them to a tool-mode override
    expresses the same intent ("the whole tool") through the right seam.
    """
    if pattern is None:
        return True
    stripped = pattern.strip()
    return stripped in ("", "*", "**")


def _glob_to_command_regex(glob: str) -> str:
    """Translate a CC ``Bash(...)`` command glob into a case-insensitive deny-regex fragment.

    The result is fed to :func:`compile_deny_patterns` and matched with ``re.search`` against the
    command string, so it is intentionally a SUBSTRING matcher anchored at the start of a command
    token: ``"git reset --hard*"`` → ``\\bgit\\ reset\\ --hard.*`` (matches ``git reset --hard X``).
    ``*`` → ``.*`` (commands are not path-segmented), ``?`` → ``.``; every other character is regex-
    escaped. A leading word boundary keeps ``git`` from matching ``mygit`` while still allowing a
    match mid-command (``&& git reset --hard``) — over-matching here only ever TIGHTENS.
    """
    out: list[str] = []
    for ch in glob:
        if ch == "*":
            out.append(".*")
        elif ch == "?":
            out.append(".")
        else:
            out.append(_re_escape_char(ch))
    body = "".join(out)
    # Anchor at a word boundary when the glob starts with a word char, so a command name is matched
    # as a token (not as a substring of a longer word). Globs starting with a non-word char (a flag,
    # a quote) are left unanchored.
    if glob[:1].isalnum() or glob[:1] == "_":
        return r"\b" + body
    return body


def _glob_to_path_regex(glob: str) -> str:
    """Translate a CC path glob (``Read``/``Edit``/``Write``/``MultiEdit``) into a protected-path
    regex fragment, fed to :func:`compile_protected_paths` and matched with ``re.search``.

    Path semantics, chosen to be tighten-only and cross-platform:

    * ``**`` → ``.*`` (crosses directory separators);
    * ``*``  → ``[^/\\\\]*`` (within a single path segment);
    * ``?``  → ``.``;
    * ``/``  → ``[/\\\\]`` so a POSIX-written rule (``*/world/*.jsonl``) also matches a Windows
      path (``...\\world\\x.jsonl``);
    * every other character is regex-escaped.

    The fragment is anchored on the left to a path BOUNDARY (start, separator, or a quote/space/
    ``=``) — mirroring the built-in :data:`PROTECTED_PATH_PATTERNS` prefix — so ``world/x.jsonl``
    matches ``a/world/x.jsonl`` and ``"world/x.jsonl"`` but not ``otherworld/x.jsonl``.
    """
    out: list[str] = []
    i = 0
    n = len(glob)
    while i < n:
        ch = glob[i]
        if ch == "*":
            if i + 1 < n and glob[i + 1] == "*":  # ``**`` — crosses separators
                out.append(".*")
                i += 2
                # Swallow a separator immediately after ``**`` so ``**/x`` also matches a bare
                # ``x`` at the root (CC's ``**/`` means "zero or more leading dirs").
                if i < n and glob[i] in "/\\":
                    out.append(r"[/\\]?")
                    i += 1
                continue
            out.append(r"[^/\\]*")  # ``*`` — within one segment
        elif ch == "?":
            out.append(".")
        elif ch in "/\\":
            out.append(r"[/\\]")
        else:
            out.append(_re_escape_char(ch))
        i += 1
    body = "".join(out)
    # Left boundary like the built-in protected-path patterns: start, a path separator, or a
    # surrounding quote / whitespace / ``=`` / ``>`` (a redirect or assignment).
    return r"(?:^|[\\/\s\"'=>])" + body


def _re_escape_char(ch: str) -> str:
    """Regex-escape a single character (re.escape over one char), so glob literals stay literal."""
    return re.escape(ch)


def _translate_gestures(
    gestures: object,
    decision: str,
    source: str,
    out: IngestedPermissions,
    errors: dict[str, str],
) -> None:
    """Translate one ``permissions.<decision>`` array into ``out``; record skips in ``errors``.

    ``decision`` is ``"deny"`` | ``"allow"`` | ``"ask"``. ``source`` tags the errors-dict key
    (the originating filename) so a skipped gesture is traceable. Anything unmappable is recorded,
    never dropped or mis-mapped (the security contract: a rule we cannot honor TIGHTENINGLY must be
    visible, not silently ignored).
    """
    if gestures is None:
        return
    if not isinstance(gestures, list):
        errors[f"{source}:{decision}"] = (
            f"expected a list of gestures, got {type(gestures).__name__}"
        )
        return

    for raw in gestures:
        if not isinstance(raw, str) or not raw.strip():
            continue
        key = f"{source}:{decision}:{raw}"
        tool, pattern = _parse_gesture(raw)
        zak_tool = _CC_TO_ZAK_TOOL.get(tool)

        # ``ask`` is the default posture (prompt). With no precise per-pattern "always prompt this
        # tool" seam, an ``ask`` gesture is effectively a no-op beyond documentation, so record it
        # as skipped (a benign skip — it does not loosen anything).
        if decision == "ask":
            errors[key] = "ask gesture: the default (prompt) posture already applies; no-op"
            continue

        if zak_tool is None:
            # A CC-only tool with no Zak counterpart (``AskUserQuestion``, ``Monitor``, ``Task``
            # with no mapping, …). Cannot be honored tighteningly — record, never mis-map.
            errors[key] = f"no Zak Code tool maps to CC tool {tool!r}; gesture skipped"
            continue

        # Whole-tool gesture (bare ``ToolName``, ``ToolName()``, ``*``, ``**``). A DENY denies the
        # tool UNCONDITIONALLY (``denied_tools`` → the policy's tier-independent deny; a mode
        # override can't bind a read-only tool). An ALLOW pins the tool to 'allow' mode (the
        # catastrophic / protected-path floor still runs first). A deny beats an allow.
        if _is_whole_tool_pattern(pattern):
            if decision == "deny":
                out.denied_tools.add(zak_tool)
                out.tool_mode_overrides.pop(zak_tool, None)  # a deny supersedes any prior allow
            elif zak_tool in out.denied_tools:
                errors[key] = (
                    f"allow for tool {zak_tool!r} ignored: a deny for the same tool already "
                    "applies (deny wins — tighten-only)"
                )
            else:
                out.tool_mode_overrides[zak_tool] = "allow"
            continue

        assert pattern is not None  # _is_whole_tool_pattern handled None/empty above

        if decision == "allow":
            # A per-PATTERN allow (e.g. ``Write(**/.claude/skills/**)``) has no tightening
            # pattern-scoped seam — Zak's allow seam is per-tool. Pinning the whole tool to 'allow'
            # on the strength of one narrow pattern would over-allow the tool's OTHER calls, which
            # is a LOOSENING we must not perform silently. Record it as skipped instead.
            errors[key] = (
                f"per-pattern allow {raw!r} has no tighten-only mapping "
                "(allow is per-tool, not per-pattern); skipped"
            )
            continue

        # decision == "deny" with a concrete pattern.
        if tool in _COMMAND_TOOLS:
            out.denied_command_regexes.append(_glob_to_command_regex(pattern))
        elif tool in _PATH_TOOLS:
            out.protected_path_regexes.append(_glob_to_path_regex(pattern))
        else:
            # A mapped tool that is neither a command nor a path tool (e.g. ``Grep(...)``,
            # ``WebFetch(...)``) — the deny is pattern-scoped but we have no pattern-scoped deny
            # seam for it. A whole-tool deny WOULD be safe (tightening), but it would deny far more
            # than the gesture asked; record the narrower intent as skipped rather than over-deny.
            errors[key] = (
                f"deny {raw!r}: tool {tool!r} has no per-pattern deny seam "
                "(only Bash commands and file paths are pattern-mappable); skipped"
            )


def load_settings_permissions(
    workspace_root: Path,
) -> tuple[IngestedPermissions, dict[str, str]]:
    """Read CC ``permissions`` from the workspace settings files and translate them.

    Discovery mirrors :func:`zakcode.hooks.settings_loader.load_settings_hooks`:
    ``<workspace>/.claude/settings.json`` then ``<workspace>/.claude/settings.local.json`` (the
    per-machine local overrides, layered AFTER the shared file). A missing file is a silent no-op;
    a JSON/OS read error is recorded in ``errors`` (keyed by path), never raised.

    Returns ``(ingested, errors)`` where ``ingested`` is the tighten-only
    :class:`IngestedPermissions` to feed the policy and ``errors`` maps a per-gesture (or per-file)
    key to a human reason a gesture was skipped. The caller wires ``ingested`` into the
    ``PermissionPolicy`` construction behind
    the ``settings_permissions`` flag; it can never loosen the policy (see the module docstring).
    """
    out = IngestedPermissions()
    errors: dict[str, str] = {}

    candidates = [
        workspace_root / ".claude" / "settings.json",
        # Per-machine local overrides (Claude Code's settings.local.json), read AFTER the shared
        # settings.json so its rules add to the project's. Both contribute (tighten-only union).
        workspace_root / ".claude" / "settings.local.json",
    ]
    for settings_path in candidates:
        if not settings_path.is_file():
            continue
        try:
            data = json.loads(settings_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            errors[str(settings_path)] = f"parse error: {exc}"
            continue
        if not isinstance(data, dict):
            continue
        permissions = data.get("permissions")
        if not isinstance(permissions, dict):
            continue

        source = settings_path.name
        # Order matters only for the deny-beats-allow guard in _set_tool_mode: process ``deny``
        # first so a tool denied here cannot be re-opened by an ``allow`` in the same file.
        _translate_gestures(permissions.get("deny"), "deny", source, out, errors)
        _translate_gestures(permissions.get("allow"), "allow", source, out, errors)
        _translate_gestures(permissions.get("ask"), "ask", source, out, errors)

    return out, errors


__all__ = [
    "IngestedPermissions",
    "load_settings_permissions",
]
