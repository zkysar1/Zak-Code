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

from collections.abc import Sequence
from pathlib import Path


class PathEscapeError(Exception):
    """Raised when a requested path resolves outside the workspace root."""


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
