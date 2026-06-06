"""Operator-authored agent identity (``self.md``).

A "mind" running on Zak Code is *skills + rules + memory*; its IDENTITY — who the agent is,
in a sentence or three — is the operator-authored ``self.md``. This module discovers that
file and hands its text to the system-prompt builder's identity slot, where it **replaces**
the default identity line. It mirrors rule discovery: pure I/O, defensive (never raises),
and bounded so a small-model context is never crowded.

Resolution order (first hit wins; project beats user):

1. ``<workspace>/.zakcode/self.md``
2. ``<workspace>/self.md``
3. ``~/.config/zakcode/self.md``

``.claude/`` is intentionally NOT scanned — ``self.md`` is a zakcode-native concept, distinct
from the ``ZAK.md`` memory the dynamic prompt tier already discovers.
"""

from __future__ import annotations

import os
from pathlib import Path

from zakcode.rules import _split_frontmatter

#: Hard cap on identity length (~3 sentences + slack). Small-model discipline: the identity
#: must not crowd a 3B context. A longer file is truncated to this many characters.
MAX_IDENTITY_CHARS = 2_048

#: Filename of the operator-authored agent identity.
IDENTITY_FILENAME = "self.md"


def identity_paths(workspace_root: str | os.PathLike[str]) -> list[Path]:
    """Candidate ``self.md`` locations, highest precedence first."""
    ws = Path(workspace_root)
    return [
        ws / ".zakcode" / IDENTITY_FILENAME,
        ws / IDENTITY_FILENAME,
        Path.home() / ".config" / "zakcode" / IDENTITY_FILENAME,
    ]


def load_identity(workspace_root: str | os.PathLike[str]) -> tuple[str | None, str | None]:
    """Discover and load the operator agent identity.

    Returns ``(identity_text, error)``: the first existing ``self.md`` (frontmatter stripped,
    truncated to :data:`MAX_IDENTITY_CHARS`) with ``error=None``; ``(None, None)`` when no file
    exists at all; or ``(None, error)`` when a file exists but is unreadable or empty after
    stripping frontmatter. Never raises, and never returns an empty-string identity (which
    would blank the default identity line) — an empty file is reported as an error instead.
    """
    for path in identity_paths(workspace_root):
        if not path.is_file():
            continue
        try:
            raw = path.read_text(encoding="utf-8")
        except OSError as exc:
            return None, f"could not read {path}: {exc}"
        _meta, body = _split_frontmatter(raw)
        body = body.strip()
        if not body:
            return None, f"{path} has no identity text"
        return body[:MAX_IDENTITY_CHARS], None
    return None, None


__all__ = ["IDENTITY_FILENAME", "MAX_IDENTITY_CHARS", "identity_paths", "load_identity"]
