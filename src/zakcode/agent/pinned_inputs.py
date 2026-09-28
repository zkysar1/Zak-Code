"""The workspace inputs a session's system prompt is built from, kept for the session (ADR-0260).

A session's system prompt carries text read from the workspace: the identity, the rules, the
output style, the skills catalog, the project guides and the workspace survey. The session pins
that text the first time its prompt is built (:meth:`SystemPromptBuilder.build`'s ``pins``), so a
restart into a new build, a resume, or a served turn sends the same prompt instead of reading the
workspace again. An engine that caches by exact prefix would otherwise re-process the whole
conversation whenever any of those files changed in between.

The model still has to learn what changed. This module compares what each input was read from,
item by item (one rule, one guide file, one skill), with what the session last knew, and writes
the one message that tells the model: each change quoted as a diff, or the new text, within a
bound. The survey is pinned without a notice: it lists files, and a changed listing carries no
instruction the session needs.
"""

from __future__ import annotations

import difflib

from zakcode.rules import MAX_RULE_FILE_CHARS
from zakcode.session.store import Session

#: How much of one item's change the notice quotes, and of all the changes together. One item
#: may be as long as a rule or guide file the prompt itself would carry, so a new rule arrives
#: whole: a small model does not fetch what it is only pointed at. Past either bound an item is
#: named with where to read it instead, so a large rewrite cannot fill the window with one message.
MAX_ITEM_CHARS = MAX_RULE_FILE_CHARS
MAX_NOTICE_CHARS = 2 * MAX_ITEM_CHARS

_HEADER = (
    "[harness] Files that feed your system prompt changed since this session began. The system "
    "prompt keeps the text the session started with (changing it would make the server re-read "
    "the whole conversation). The changes below apply from now on: where they differ from the "
    "system prompt, follow them."
)
_WHERE = (
    "A change that is named but not quoted can be read in full: read_rule returns a rule's "
    "current text, the Skill tool loads a skill, and a guide or identity file can be read with "
    "your file-read tool."
)


def _diff(old: str, new: str) -> str:
    """The changed lines of ``old`` → ``new`` with one line of context, headers dropped."""
    lines = list(difflib.unified_diff(old.splitlines(), new.splitlines(), lineterm="", n=1))
    return "\n".join(lines[2:])  # the first two lines are the ---/+++ file headers


def _quote(text: str, room: int) -> str | None:
    """``text`` indented under its item line, or ``None`` when it does not fit in ``room``."""
    if not text.strip() or len(text) > min(room, MAX_ITEM_CHARS):
        return None
    return "\n".join(f"    {line}" for line in text.splitlines())


def change_notice(session: Session, items: dict[str, dict[str, str]]) -> str | None:
    """Record ``items`` as what the session knows, and return the notice of what changed.

    ``items`` maps an input's name (a key of ``session.prompt_inputs``) to its items: a label
    such as ``rule tabs`` or ``guide CLAUDE.md`` → the text it contributes. An input the
    session has not pinned yet is recorded silently, because the next prompt build reads it
    from the workspace anyway; so is one the session has no earlier record of. ``None`` when
    nothing the session pinned has changed.
    """
    known = session.prompt_input_items
    lines: list[str] = []
    unquoted = 0
    room = MAX_NOTICE_CHARS
    for name, now in items.items():
        before = known.get(name)
        known[name] = dict(now)
        if name not in session.prompt_inputs or before is None or before == now:
            continue
        for label in sorted(set(before) | set(now)):
            old, new = before.get(label), now.get(label)
            if old == new:
                continue
            if new is None:
                lines.append(f"- {label} was removed.")
                continue
            what, body = ("is new", new) if old is None else ("changed", _diff(old, new))
            quoted = _quote(body, room)
            if quoted is None:
                lines.append(f"- {label} {what} (not quoted here: too long).")
                unquoted += 1
            else:
                lines.append(f"- {label} {what}:\n{quoted}")
                room -= len(quoted)
    if not lines:
        return None
    tail = f"\n\n{_WHERE}" if unquoted else ""
    return f"{_HEADER}\n\n" + "\n".join(lines) + tail


__all__ = ["MAX_ITEM_CHARS", "MAX_NOTICE_CHARS", "change_notice"]
