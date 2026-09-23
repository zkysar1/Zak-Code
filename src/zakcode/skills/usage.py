"""How often each skill has been chosen: the ranking the skill listing budget keeps
descriptions by (ADR-0222).

Claude Code bounds the skill listing it shows the model and, over budget, drops descriptions
"starting with the skills you invoke least". This module is zakcode's record of the same
thing: one JSON object at ``<zakcode home>/skill-usage.json`` mapping a skill's name to how
many times a model's ``Skill`` call or an operator's ``/<name>`` delivered its body. A
harness re-entry (a turn-end hook's named skill, a fired wake-up) is the loop running itself,
not anyone choosing a skill, so the caller does not count it.

The file is READ once, when an Agent is built: the catalogue sits in the cached prompt prefix
and must not move inside a session. It is WRITTEN after each counted delivery, by a
read-modify-write and an atomic replace, under a lock that serializes this process's writers
(a served process runs many sessions). Two PROCESSES counting at the same instant can lose
one increment. This is a ranking, and a lost count only delays a change of rank, so there is
no cross-process lock.

Nothing here raises. A missing, unreadable or malformed file reads as "no usage yet", and a
write that fails is logged and dropped: a usage count is never worth failing a skill load.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import threading
from pathlib import Path

from zakcode.config import zakcode_home

logger = logging.getLogger("zakcode.skills.usage")

#: The usage file's name inside the zakcode home (``ZAKCODE_HOME`` or ``~/.zakcode``).
USAGE_FILENAME = "skill-usage.json"

#: Serializes this process's read-modify-write cycles (see the module docstring).
_WRITE_LOCK = threading.Lock()


def usage_path() -> Path:
    """Where the counts live: ``<zakcode home>/skill-usage.json``."""
    return zakcode_home() / USAGE_FILENAME


def load_skill_usage(path: Path | None = None) -> dict[str, int]:
    """Every recorded skill's count, or ``{}`` when nothing usable is recorded.

    Only entries shaped ``name -> non-negative int`` are kept, so a hand-edited or
    half-written file costs the entries it broke, never the load.
    """
    target = path if path is not None else usage_path()
    try:
        raw = json.loads(target.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except (OSError, UnicodeError, ValueError) as exc:
        logger.warning("skill usage at %s is unreadable; ranking without it: %s", target, exc)
        return {}
    if not isinstance(raw, dict):
        return {}
    return {
        name: count
        for name, count in raw.items()
        # bool is an int subclass; a true/false count is a malformed entry, not 1 or 0.
        if isinstance(name, str)
        and isinstance(count, int)
        and not isinstance(count, bool)
        and count >= 0
    }


def record_skill_use(name: str, path: Path | None = None) -> None:
    """Add one to ``name``'s count. Never raises (see the module docstring)."""
    target = path if path is not None else usage_path()
    with _WRITE_LOCK:
        counts = load_skill_usage(target)
        counts[name] = counts.get(name, 0) + 1
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp_name = tempfile.mkstemp(dir=str(target.parent), prefix=".skill-usage-")
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as handle:
                    json.dump(counts, handle, sort_keys=True, indent=0)
                os.replace(tmp_name, target)
            except BaseException:
                # The replace never happened: take the temp file with us, keep the old counts.
                Path(tmp_name).unlink(missing_ok=True)
                raise
        except OSError as exc:
            logger.warning("could not record a use of skill %r at %s: %s", name, target, exc)


__all__ = ["USAGE_FILENAME", "load_skill_usage", "record_skill_use", "usage_path"]
