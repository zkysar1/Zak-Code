"""Path-safety helpers shared by the file-touching built-in tools.

Every filesystem tool must confine itself to the workspace. :func:`resolve_in_workspace`
turns a (possibly relative) user-supplied path into an absolute, symlink-resolved
path and rejects anything that escapes the workspace root.
"""

from __future__ import annotations

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
