"""User-facing file artifact metadata and workspace-safe resolution.

Artifacts are the machine-readable receipts for files a tool created and a client can
download or preview. The model still sees ordinary tool text; clients get this structured
side channel so binary/user-facing files do not have to travel through prompts.
"""

from __future__ import annotations

import hashlib
import mimetypes
import os
from pathlib import Path

from pydantic import BaseModel, Field

_MIME_OVERRIDES = {
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    ".xlsm": "application/vnd.ms-excel.sheet.macroEnabled.12",
    ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
}

_TEXT_SUFFIXES = {
    ".cfg",
    ".csv",
    ".css",
    ".env",
    ".html",
    ".ini",
    ".js",
    ".json",
    ".log",
    ".md",
    ".py",
    ".toml",
    ".ts",
    ".txt",
    ".xml",
    ".yaml",
    ".yml",
}


class ArtifactError(Exception):
    """Base class for artifact metadata/path failures."""


class ArtifactPathError(ArtifactError):
    """Raised when an artifact path is malformed or escapes the workspace."""


class ArtifactChangedError(ArtifactError):
    """Raised when the file no longer matches the recorded artifact metadata."""


class ArtifactRef(BaseModel):
    """A stable, downloadable reference to a file under the workspace root.

    ``path`` is always workspace-relative with POSIX separators. The absolute filesystem
    path stays server-side and is re-derived through the workspace sandbox before download.
    """

    id: str = Field(description="Stable id derived from path, size, and content hash.")
    path: str = Field(description="Workspace-relative path using POSIX separators.")
    filename: str
    mime_type: str = "application/octet-stream"
    size: int = Field(ge=0)
    sha256: str = Field(description="SHA-256 digest of the file bytes.")
    kind: str = "binary"
    created_by_tool: str | None = None


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mime_for(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in _MIME_OVERRIDES:
        return _MIME_OVERRIDES[suffix]
    guessed, _encoding = mimetypes.guess_type(path.name)
    return guessed or "application/octet-stream"


def _kind_for(path: Path, mime_type: str) -> str:
    suffix = path.suffix.lower()
    if mime_type.startswith("image/"):
        return "image"
    if mime_type.startswith("text/") or suffix in _TEXT_SUFFIXES:
        return "text"
    if suffix in {".doc", ".docx", ".odt", ".rtf"}:
        return "document"
    if suffix in {".xls", ".xlsx", ".xlsm", ".ods"}:
        return "spreadsheet"
    if suffix in {".ppt", ".pptx", ".odp"}:
        return "presentation"
    if suffix == ".pdf":
        return "pdf"
    if suffix in {".zip", ".tar", ".gz", ".7z"}:
        return "archive"
    return "binary"


def _workspace_relative(path: Path, workspace_root: Path) -> str:
    try:
        rel = path.relative_to(workspace_root)
    except ValueError as exc:
        raise ArtifactPathError("artifact path is outside the workspace") from exc
    return rel.as_posix()


def artifact_from_path(
    path: str | os.PathLike[str],
    *,
    workspace_root: str | os.PathLike[str],
    created_by_tool: str | None = None,
) -> ArtifactRef:
    """Build an :class:`ArtifactRef` for an existing file under ``workspace_root``."""

    root = Path(workspace_root).resolve()
    resolved = Path(path).resolve(strict=True)
    if not resolved.is_file():
        raise ArtifactPathError("artifact path is not a file")

    rel = _workspace_relative(resolved, root)
    size = resolved.stat().st_size
    sha = _sha256_file(resolved)
    identity = hashlib.sha256(f"{rel}\0{size}\0{sha}".encode()).hexdigest()[:24]
    mime_type = _mime_for(resolved)
    return ArtifactRef(
        id=identity,
        path=rel,
        filename=resolved.name,
        mime_type=mime_type,
        size=size,
        sha256=sha,
        kind=_kind_for(resolved, mime_type),
        created_by_tool=created_by_tool,
    )


def resolve_artifact_path(
    artifact: ArtifactRef,
    *,
    workspace_root: str | os.PathLike[str],
    verify_digest: bool = True,
) -> Path:
    """Resolve and optionally verify an artifact against the current workspace bytes."""

    rel = artifact.path
    if "\x00" in rel:
        raise ArtifactPathError("artifact path contains a NUL byte")
    if os.name == "nt" and ":" in rel:
        raise ArtifactPathError("artifact path contains a Windows stream separator")
    rel_path = Path(rel)
    if rel_path.is_absolute():
        raise ArtifactPathError("artifact path must be workspace-relative")

    root = Path(workspace_root).resolve()
    resolved = (root / rel_path).resolve()
    _workspace_relative(resolved, root)
    if not resolved.exists() or not resolved.is_file():
        raise ArtifactPathError("artifact file is no longer available")

    if verify_digest:
        size = resolved.stat().st_size
        sha = _sha256_file(resolved)
        if size != artifact.size or sha != artifact.sha256:
            raise ArtifactChangedError("artifact file changed since it was recorded")
    return resolved


__all__ = [
    "ArtifactChangedError",
    "ArtifactError",
    "ArtifactPathError",
    "ArtifactRef",
    "artifact_from_path",
    "resolve_artifact_path",
]
