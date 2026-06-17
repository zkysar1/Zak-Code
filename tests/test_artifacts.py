"""Tests for workspace-scoped artifact metadata and download resolution."""

from __future__ import annotations

from pathlib import Path

import pytest

from zakcode.artifacts import (
    ArtifactChangedError,
    ArtifactPathError,
    artifact_from_path,
    resolve_artifact_path,
)


def test_artifact_from_path_records_download_metadata(tmp_path: Path) -> None:
    path = tmp_path / "reports" / "summary.docx"
    path.parent.mkdir()
    path.write_bytes(b"fake office bytes")

    artifact = artifact_from_path(path, workspace_root=tmp_path, created_by_tool="create_docx")

    assert artifact.path == "reports/summary.docx"
    assert artifact.filename == "summary.docx"
    assert artifact.mime_type.endswith("wordprocessingml.document")
    assert artifact.kind == "document"
    assert artifact.size == len(b"fake office bytes")
    assert artifact.created_by_tool == "create_docx"
    assert len(artifact.sha256) == 64
    assert resolve_artifact_path(artifact, workspace_root=tmp_path) == path.resolve()


def test_artifact_from_path_classifies_macro_enabled_excel(tmp_path: Path) -> None:
    path = tmp_path / "budget.xlsm"
    path.write_bytes(b"fake office bytes")

    artifact = artifact_from_path(path, workspace_root=tmp_path)

    assert artifact.kind == "spreadsheet"
    assert artifact.mime_type == "application/vnd.ms-excel.sheet.macroEnabled.12"


def test_artifact_from_path_rejects_files_outside_workspace(tmp_path: Path) -> None:
    outside = tmp_path.parent / "outside-artifact.txt"
    outside.write_text("nope", encoding="utf-8")

    with pytest.raises(ArtifactPathError):
        artifact_from_path(outside, workspace_root=tmp_path)


def test_resolve_artifact_path_detects_changed_bytes(tmp_path: Path) -> None:
    path = tmp_path / "out.txt"
    path.write_text("first", encoding="utf-8")
    artifact = artifact_from_path(path, workspace_root=tmp_path)

    path.write_text("second", encoding="utf-8")

    with pytest.raises(ArtifactChangedError):
        resolve_artifact_path(artifact, workspace_root=tmp_path)
