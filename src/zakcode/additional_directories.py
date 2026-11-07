"""Directories a workspace's settings files add to the file tools' reach (ADR-0375).

Claude Code's ``permissions.additionalDirectories`` is an array of paths in a settings file. Each
names a directory outside the one a session starts in that the file tools may also read and
write. A path is taken as written when it is absolute, has a leading ``~`` expanded, and otherwise
resolves from the directory the session runs in (here: the workspace root).

The same grant is made on the command line by ``--extra-root`` and in code by
``Agent(extra_workspace_roots=...)``. All three land in one list, and every file tool judges a
path against that list with the ``..`` and symlink protection it applies to the workspace root.

Reach, not authority. The permission policy never sees these roots: a path inside one is decided
like any workspace path, so the protected-path floor (``.env``, ``.git/``), an ingested ``deny``
rule and the tier of the tool bind there exactly as they do in the workspace. Declared config is
live, with no trust prompt, as hooks (ADR-0025) and permission gestures (ADR-0029) already are.

Discovery is the hook loader's: ``<workspace>/.claude/settings.json``, then
``.claude/settings.local.json``, then ``.zakcode/settings.json``. The lists from the three are
joined in that order, and a directory named twice counts once. A file that does not parse is
skipped without a word here, because the hook and permission loaders already report it. A
directory that does not exist grants nothing and is left out (logged at DEBUG).
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

from zakcode.hooks.settings_loader import _settings_candidates

logger = logging.getLogger("zakcode.additional_directories")

_KEY = "permissions.additionalDirectories"


def load_additional_directories(workspace_root: Path) -> tuple[list[Path], dict[str, str]]:
    """The existing directories the workspace's settings files add to the sandbox.

    Returns ``(roots, errors)``: ``roots`` are resolved absolute directories in read order with
    repeats removed; ``errors`` maps ``<file>:<key>`` (or ``<file>:<key>[i]`` for one entry) to
    the reason a value was skipped. Never raises.
    """
    roots: list[Path] = []
    errors: dict[str, str] = {}
    seen: set[Path] = set()
    for settings_path in _settings_candidates(workspace_root):
        if not settings_path.is_file():
            continue
        try:
            data = json.loads(settings_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError, OSError):
            continue
        permissions = data.get("permissions") if isinstance(data, dict) else None
        entries = (
            permissions.get("additionalDirectories") if isinstance(permissions, dict) else None
        )
        if entries is None:
            continue
        where = f"{settings_path.name}:{_KEY}"
        if not isinstance(entries, list):
            errors[where] = f"expected a list of paths, got {type(entries).__name__}; skipped"
            continue
        for index, entry in enumerate(entries):
            if not isinstance(entry, str) or not entry.strip():
                errors[f"{where}[{index}]"] = "not a path string; skipped"
                continue
            path = Path(os.path.expanduser(entry.strip()))
            if not path.is_absolute():
                path = workspace_root / path
            try:
                resolved = path.resolve()
            except (OSError, RuntimeError):
                logger.debug("%s: %r cannot be resolved; skipped", where, entry)
                continue
            if not resolved.is_dir():
                logger.debug("%s: %r is not an existing directory; skipped", where, entry)
                continue
            if resolved not in seen:
                seen.add(resolved)
                roots.append(resolved)
    return roots, errors


__all__ = ["load_additional_directories"]
