"""Tests for ``write_text_atomic`` — publish a text file whole, or not at all (ADR-0318).

The concurrency half lives in ``test_server_staged_write_race.py``, where it is driven through
the two routes that depend on it. What is pinned here is the failure contract the routes
cannot reach: a write that dies midway must not leave its temp file behind or touch the file
it was meant to replace.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from zakcode.session.atomic_write import write_text_atomic


def test_write_text_atomic_publishes_and_replaces(tmp_path: Path) -> None:
    target = tmp_path / ".nudge"
    write_text_atomic(target, "first\n")
    assert target.read_text(encoding="utf-8") == "first\n"

    write_text_atomic(target, "second\n")
    assert target.read_text(encoding="utf-8") == "second\n"
    # Only the published file is left: no temp survives a successful write.
    assert sorted(entry.name for entry in tmp_path.iterdir()) == [".nudge"]


def test_write_text_atomic_failed_rename_cleans_up_and_keeps_the_old_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / ".observation"
    target.write_text("old\n", encoding="utf-8")

    def refuse(src: object, dst: object) -> None:  # noqa: ARG001
        raise PermissionError("rename refused")

    monkeypatch.setattr(os, "replace", refuse)
    with pytest.raises(PermissionError, match="rename refused"):
        write_text_atomic(target, "new\n")

    assert target.read_text(encoding="utf-8") == "old\n"
    assert sorted(entry.name for entry in tmp_path.iterdir()) == [".observation"]
