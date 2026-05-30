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
    inside the resolved workspace root.

    Raises:
        PathEscapeError: if the resolved path is outside the workspace.
    """
    root = Path(workspace_root).resolve()
    candidate = Path(path)
    combined = candidate if candidate.is_absolute() else root / candidate
    resolved = combined.resolve()

    if resolved != root and root not in resolved.parents:
        raise PathEscapeError(f"Path {path!r} resolves outside the workspace root {str(root)!r}")
    return resolved
