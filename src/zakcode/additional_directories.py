"""Directories a workspace's settings files add to the file tools' reach (ADR-0375).

Claude Code's ``permissions.additionalDirectories`` is an array of paths in a settings file. Each
names a directory outside the one a session starts in that the file tools may also read and
write. Here a path is taken as written when it is absolute, has a leading ``~`` expanded, and
otherwise resolves from the workspace root.

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
skipped without a word here, because the hook and permission loaders already report it. A key that
is ``null`` declares nothing, as an absent one does. A directory that does not exist, or that the
process cannot resolve or enter (an over-long name, an embedded NUL, a symlink loop, a parent it
may not traverse), grants nothing and is left out (logged at DEBUG). So does a path spelled as a
Windows network (UNC) path, which is not looked up at all: looking one up contacts the host it
names and sends that host the user's credentials. That test reads the entry as written, so a ``~``
that expands to a share and a link or junction into one are not caught.

This loader never raises: a file it cannot take is skipped. Four kinds of file are among them: an
integer of thousands of digits, nesting thousands deep, and a ``.claude`` or a ``.zakcode``
directory the process cannot enter. Other loaders raise on those four (permission settings, hooks,
the status line and identity; which one depends on the file and on whether a permission policy was
injected), so an Agent over one does not start, with or without this loader.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

from zakcode.hooks.settings_loader import _settings_candidates

logger = logging.getLogger("zakcode.additional_directories")

_KEY = "permissions.additionalDirectories"


def load_additional_directories(workspace_root: Path) -> tuple[list[Path], dict[str, str]]:
    """The existing directories the workspace's settings files add to the sandbox.

    Returns ``(roots, errors)``: ``roots`` are resolved absolute directories in read order with
    repeats removed; ``errors`` maps ``<dir>/<file>:<key>`` (or ``<dir>/<file>:<key>[i]`` for one
    entry, ``<dir>`` being ``.claude`` or ``.zakcode``) to the reason a value was skipped. Never
    raises.
    """
    roots: list[Path] = []
    errors: dict[str, str] = {}
    seen: set[Path] = set()
    for settings_path in _settings_candidates(workspace_root):
        # is_file() is inside the guard, as in the hook loader: in a directory the process cannot
        # enter it raises PermissionError, and raised here it would stop the agent starting.
        try:
            if not settings_path.is_file():
                continue
            data = json.loads(settings_path.read_text(encoding="utf-8"))
        except (ValueError, OSError, RecursionError):
            # ValueError is a JSONDecodeError, a UnicodeDecodeError, and the plain ValueError json
            # raises for an integer of more than 4300 digits; RecursionError is nesting too deep.
            continue
        permissions = data.get("permissions") if isinstance(data, dict) else None
        entries = (
            permissions.get("additionalDirectories") if isinstance(permissions, dict) else None
        )
        if entries is None:
            continue
        # The directory is part of the label: .claude/settings.json and .zakcode/settings.json share
        # a file name, and two bad values at one position must not collapse into one entry.
        where = f"{settings_path.parent.name}/{settings_path.name}:{_KEY}"
        if not isinstance(entries, list):
            errors[where] = f"expected a list of paths, got {type(entries).__name__}; skipped"
            continue
        for index, entry in enumerate(entries):
            if not isinstance(entry, str) or not entry.strip():
                errors[f"{where}[{index}]"] = "not a path string; skipped"
                continue
            text = entry.strip()
            if sys.platform == "win32" and text.replace("/", "\\").startswith("\\\\"):
                # A network (UNC) path names another machine. Looking it up contacts that host and,
                # on Windows, hands it the user's credentials, so it is left out unseen; Claude Code
                # refuses one as a working directory for the same reason. This reads the entry as
                # written, so a ``~`` that expands to a share, or a link into one, is not caught.
                logger.debug("%s: %r is a network path; skipped", where, entry)
                continue
            try:
                path = Path(os.path.expanduser(text))
                if not path.is_absolute():
                    path = workspace_root / path
                resolved = path.resolve()
                is_dir = resolved.is_dir()
            except (OSError, RuntimeError, ValueError):
                # A name the platform rejects (over-long, an embedded NUL), a symlink loop, or a
                # parent the process may not enter: there is nothing to grant, as for a missing
                # directory. is_dir() raises on those (Python 3.11, 3.12) rather than say False.
                logger.debug("%s: %r cannot be resolved; skipped", where, entry)
                continue
            if not is_dir:
                logger.debug("%s: %r is not an existing directory; skipped", where, entry)
                continue
            if resolved not in seen:
                seen.add(resolved)
                roots.append(resolved)
    return roots, errors


__all__ = ["load_additional_directories"]
