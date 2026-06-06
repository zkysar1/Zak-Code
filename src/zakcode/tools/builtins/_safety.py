"""Path-safety helpers shared by the file-touching built-in tools.

Every filesystem tool must confine itself to the workspace. :func:`resolve_in_workspace`
turns a (possibly relative) user-supplied path into an absolute, symlink-resolved
path and rejects anything that escapes the workspace root.

Multi-root support (M-3): :func:`resolve_in_workspace_roots` accepts a list of
allowed roots. A path is accepted if it resolves under ANY of them; the first
matching root wins for relative-path interpretation. This enables cross-repo skill
execution (e.g. a claude-mind skill reading files from the mind repo, an external
world directory, and an external meta directory — all separate filesystem roots).
"""

from __future__ import annotations

import ntpath
import os
import re
from collections.abc import Sequence
from pathlib import Path


class PathEscapeError(Exception):
    """Raised when a requested path resolves outside the workspace root."""


def _reject_alternate_data_stream(path: str) -> None:
    """On Windows, reject an NTFS alternate-data-stream marker (a ``:`` outside the optional
    drive prefix), e.g. ``public.txt:secret``. Such a path stays inside the workspace (the
    colon binds to the leaf) but reads a hidden stream that ``list_dir``/``glob`` never show.
    POSIX has no ADS and allows ``:`` in filenames, so this is a no-op there. (audit3 #9)
    """
    if os.name != "nt":
        return
    _drive, rest = ntpath.splitdrive(path)
    if ":" in rest:
        raise PathEscapeError(
            f"Path {path!r} contains an alternate-data-stream marker (':') and is refused"
        )


def resolve_in_workspace(path: str, workspace_root: Path) -> Path:
    """Resolve ``path`` against ``workspace_root`` and confine it to the workspace.

    Relative paths are interpreted relative to ``workspace_root``. The result is
    fully resolved (symlinks + ``..`` collapsed via realpath) and verified to live
    inside the resolved workspace root. This defends against three escape vectors:

    * ``..`` traversal (collapsed by :meth:`Path.resolve`),
    * absolute paths pointing outside the root, and
    * symlinks inside the root whose target lives outside it (resolved away by
      :meth:`Path.resolve`, then re-checked against the root).

    Raises:
        PathEscapeError: if ``path`` is empty/non-string or the resolved path is
            outside the workspace.
    """
    if not isinstance(path, str):
        raise PathEscapeError(f"Path must be a string, got {type(path).__name__}")
    if not path:
        raise PathEscapeError("Path must not be empty")
    _reject_alternate_data_stream(path)

    root = Path(workspace_root).resolve()
    candidate = Path(path)
    combined = candidate if candidate.is_absolute() else root / candidate
    # strict=False: resolve symlinks/.. even when the leaf does not exist yet.
    resolved = combined.resolve()

    if resolved != root and root not in resolved.parents:
        raise PathEscapeError(f"Path {path!r} resolves outside the workspace root {str(root)!r}")
    return resolved


def _is_inside(resolved: Path, root: Path) -> bool:
    """Return True if ``resolved`` is the root itself or lives under it."""
    return resolved == root or root in resolved.parents


def resolve_in_workspace_roots(path: str, roots: Sequence[Path]) -> Path:
    """Resolve ``path`` against one or more workspace roots.

    Absolute paths are checked against every root — accepted if the resolved path
    lives inside any of them. Relative paths are tried against each root in order;
    the first root that contains the resolved result wins.

    This preserves the security boundary of :func:`resolve_in_workspace` (traversal
    escape, symlink escape, absolute-outside rejection) while allowing a tool to
    access files spread across multiple trusted roots.

    Raises:
        PathEscapeError: if ``path`` is empty/non-string, ``roots`` is empty, or
            the resolved path is outside all roots.
    """
    if not isinstance(path, str):
        raise PathEscapeError(f"Path must be a string, got {type(path).__name__}")
    if not path:
        raise PathEscapeError("Path must not be empty")
    if not roots:
        raise PathEscapeError("At least one workspace root is required")
    _reject_alternate_data_stream(path)

    resolved_roots = [Path(r).resolve() for r in roots]
    candidate = Path(path)

    if candidate.is_absolute():
        resolved = candidate.resolve()
        for root in resolved_roots:
            if _is_inside(resolved, root):
                return resolved
        roots_str = ", ".join(str(r) for r in resolved_roots)
        raise PathEscapeError(f"Path {path!r} resolves outside all workspace roots ({roots_str})")

    # Relative path: try each root as a base; first match wins.
    for root in resolved_roots:
        resolved = (root / candidate).resolve()
        if _is_inside(resolved, root):
            return resolved

    roots_str = ", ".join(str(r) for r in resolved_roots)
    raise PathEscapeError(f"Path {path!r} resolves outside all workspace roots ({roots_str})")


def resolve_path(path: str, workspace_root: Path, extra_roots: Sequence[Path] = ()) -> Path:
    """Resolve ``path`` using the single-root or multi-root resolver as appropriate.

    When ``extra_roots`` is empty, delegates to :func:`resolve_in_workspace` (the
    original single-root behavior). When non-empty, delegates to
    :func:`resolve_in_workspace_roots` with ``[workspace_root] + list(extra_roots)``.

    This is the convenience entry point tools should prefer — it transparently picks
    the right code path without each tool having to branch on the extra-roots list.
    """
    if extra_roots:
        return resolve_in_workspace_roots(path, [workspace_root, *extra_roots])
    return resolve_in_workspace(path, workspace_root)


# ── content write firewall (deterministic, refuse-only) ──────────────────────
# A small local model sometimes writes a SHELL COMMAND into a file instead of the
# file's text — e.g. content="$(cat other.py)", expecting substitution to run.
# write_file stores content literally, so the file is corrupted. These conservative
# checks refuse such writes BEFORE any bytes land. They fire only when the WHOLE
# content is the mistake, so a file that merely *contains* "$(" is never affected.

_WHOLE_CMD_SUB_RE = re.compile(r"^\$\(.*\)$", re.DOTALL)
_WHOLE_BACKTICK_RE = re.compile(r"^`[^`]+`$", re.DOTALL)


def check_literal_content(content: str) -> str | None:
    """Refuse content that is wholly a shell command-substitution / backtick command.

    Returns an error message, or ``None`` if the content is acceptable. Conservative:
    only fires when ``content.strip()`` is *entirely* ``$(...)`` or a single backtick
    command, never when the content merely contains those characters.
    """
    stripped = content.strip()
    if stripped and (_WHOLE_CMD_SUB_RE.match(stripped) or _WHOLE_BACKTICK_RE.match(stripped)):
        return (
            "Refusing to write a shell command as file content: write_file/edit_file store "
            "text LITERALLY and do not evaluate $(...) or backticks. Pass the actual file "
            "contents, not a command that would produce them."
        )
    return None


def check_python_syntax(path: str, content: str) -> str | None:
    """For a ``.py`` ``path``, refuse ``content`` that does not parse (parse-only).

    Uses ``compile(..., "exec")`` — no execution, no new permission tier. Returns an
    error message, or ``None`` if the content is empty, non-Python, or valid.
    """
    if not path.endswith(".py") or not content.strip():
        return None
    try:
        compile(content, path, "exec")
    except SyntaxError as exc:
        where = f" (line {exc.lineno})" if exc.lineno else ""
        return f"Refusing to write invalid Python to {path}: {exc.msg}{where}."
    return None
