"""The tool-name spellings that resolve to the same tool (ADR-0190).

Since ADR-0190 the canonical names of the counterpart tools are Claude Code's (``Read``,
``Write``, ``Edit``, ``LS``, ``Glob``, ``Grep``, ``Bash``, ``WebFetch``, ``WebSearch``,
``Skill``, ``ScheduleWakeup``). The names they carried before are aliases in the registry —
and this table is the registry-free half of that promise: a hook matcher, a permission rule,
a resumed transcript or a bare registry built by hand may still say ``write_file``, and every
reader that keys on a tool name resolves it through :func:`canonical_tool_name` first.
"""

from __future__ import annotations

#: The pre-ADR-0190 canonical names → today's. Tools whose shape stayed Zak Code's own
#: (``task``, ``update_plan``, ``plan_recall``) are not here: their names did not change.
PRE_0190_TOOL_NAMES: dict[str, str] = {
    "use_skill": "Skill",
    "read_file": "Read",
    "write_file": "Write",
    "edit_file": "Edit",
    "bash": "Bash",
    "glob": "Glob",
    "grep": "Grep",
    "list_dir": "LS",
    "web_search": "WebSearch",
    "web_fetch": "WebFetch",
    "schedule_wakeup": "ScheduleWakeup",
}


def canonical_tool_name(name: str) -> str:
    """``name`` as the tool is named today: a pre-0190 spelling resolves, anything else is
    returned unchanged (so an unknown name stays visibly unknown)."""
    return PRE_0190_TOOL_NAMES.get(name, name)
