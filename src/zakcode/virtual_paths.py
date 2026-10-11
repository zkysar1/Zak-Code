"""The host framework's virtual path prefixes, and the directories they stand for (ADR-0282).

A framework Mind names its files with two virtual prefixes: ``world/…`` is the directory its
``agents/<name>/local-paths.conf`` calls ``WORLD_PATH``, ``meta/…`` is ``META_PATH``, and any other
path belongs to the workspace. The framework's own hook rewrites the prefix before its file
tools run; this is the same rewrite for zakcode's, so a path the framework hands the model
(``retrieve.sh`` answers with ``world/knowledge/tree/…``) opens on the first try.

Pure string math, no filesystem call: the permission floor uses it too (ADR-0031), and a
policy decision stays pure.
"""

from __future__ import annotations

import os
from collections.abc import Mapping

#: Each virtual prefix and the ``local-paths.conf`` key whose directory it stands for.
VIRTUAL_PREFIX_KEYS: dict[str, str] = {"world": "WORLD_PATH", "meta": "META_PATH"}


def expand_virtual_prefix(path: str, virtual_roots: Mapping[str, str | os.PathLike[str]]) -> str:
    """``path`` with a leading ``world/`` or ``meta/`` replaced by the directory declared for it.

    Every other path comes back unchanged, and so does a prefix the workspace declared no
    directory for. The remainder is joined as a relative path, so ``world//x`` cannot turn
    into an absolute ``/x`` that discards the root; what the result may reach is still
    decided by the caller's sandbox check, which sees an ordinary path.
    """
    for prefix, root in virtual_roots.items():
        head = prefix + "/"
        if path.startswith(head):
            return os.path.join(os.fspath(root), path[len(head) :].lstrip("/\\"))
    return path
