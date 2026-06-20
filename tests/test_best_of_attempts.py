"""Tests for seam B's isolation + diff-apply helpers — the safe core that touches the workspace.

The critical safety property is that adoption applies ONLY the diff: a file the attempt didn't touch
is never overwritten. Plus the size cap, the source-only copy, and the off-by-default no-op guard.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from zakcode.agent.best_of_attempts import (
    _unchanged_since_snapshot,
    apply_diff,
    changed_paths,
    copy_source,
    run_best_of_attempts,
)


def test_copy_source_skips_vcs_deps_caches(tmp_path: Path) -> None:
    src = tmp_path / "src"
    src.mkdir()
    (src / "app.py").write_text("code", encoding="utf-8")
    (src / ".git").mkdir()
    (src / ".git" / "HEAD").write_text("ref", encoding="utf-8")
    (src / "node_modules").mkdir()
    (src / "node_modules" / "x.js").write_text("dep", encoding="utf-8")
    dest = tmp_path / "dest"
    assert copy_source(src, dest) is True
    assert (dest / "app.py").read_text(encoding="utf-8") == "code"
    assert not (dest / ".git").exists() and not (dest / "node_modules").exists()


def test_copy_source_size_cap_skips(tmp_path: Path) -> None:
    src = tmp_path / "src"
    src.mkdir()
    (src / "big.txt").write_text("x" * 1000, encoding="utf-8")
    dest = tmp_path / "dest"
    assert copy_source(src, dest, max_bytes=100) is False  # over the cap → skip
    assert not dest.exists()  # nothing copied


def test_changed_paths_detects_add_modify_delete(tmp_path: Path) -> None:
    snap, att = tmp_path / "snap", tmp_path / "att"
    snap.mkdir()
    att.mkdir()
    (snap / "keep.py").write_text("same", encoding="utf-8")
    (snap / "edit.py").write_text("old", encoding="utf-8")
    (snap / "gone.py").write_text("bye", encoding="utf-8")
    (att / "keep.py").write_text("same", encoding="utf-8")  # unchanged
    (att / "edit.py").write_text("new", encoding="utf-8")  # modified
    (att / "added.py").write_text("hi", encoding="utf-8")  # added
    changed, deleted = changed_paths(att, snap)
    assert set(changed) == {Path("edit.py"), Path("added.py")}  # keep.py NOT flagged
    assert deleted == [Path("gone.py")]


def test_apply_diff_only_touches_the_diff(tmp_path: Path) -> None:
    ws, att = tmp_path / "ws", tmp_path / "att"
    ws.mkdir()
    att.mkdir()
    (ws / "keep.py").write_text("original", encoding="utf-8")  # untouched → MUST stay
    (ws / "edit.py").write_text("old", encoding="utf-8")
    (ws / "gone.py").write_text("bye", encoding="utf-8")
    (att / "edit.py").write_text("new", encoding="utf-8")
    (att / "added.py").write_text("hi", encoding="utf-8")
    apply_diff([Path("edit.py"), Path("added.py")], [Path("gone.py")], att, ws)
    assert (ws / "keep.py").read_text(encoding="utf-8") == "original"  # never overwritten
    assert (ws / "edit.py").read_text(encoding="utf-8") == "new"  # applied
    assert (ws / "added.py").read_text(encoding="utf-8") == "hi"  # added
    assert not (ws / "gone.py").exists()  # deleted


def test_unchanged_since_snapshot_detects_concurrent_edit(tmp_path: Path) -> None:
    snap, ws = tmp_path / "snap", tmp_path / "ws"
    snap.mkdir()
    ws.mkdir()
    (snap / "a.py").write_text("orig", encoding="utf-8")
    (ws / "a.py").write_text("orig", encoding="utf-8")
    assert _unchanged_since_snapshot([Path("a.py")], ws, snap) is True  # workspace == snapshot
    (ws / "a.py").write_text("user edited", encoding="utf-8")  # concurrent edit during the retry
    assert _unchanged_since_snapshot([Path("a.py")], ws, snap) is False  # diverged → guard trips


class _Settings:
    def __init__(self, *, best_of_attempts: int, verify_command: str | None) -> None:
        self.best_of_attempts = best_of_attempts
        self.verify_command = verify_command


class _Agent:
    def __init__(self, **kw: Any) -> None:
        self.settings = _Settings(**kw)


async def test_retry_is_noop_when_not_configured() -> None:
    # n <= 1 → off; or no verifier → off. Either returns the original untouched (no isolation runs).
    off = _Agent(best_of_attempts=1, verify_command="pytest")
    assert await run_best_of_attempts(off, "task", "ORIGINAL") == "ORIGINAL"
    no_verifier = _Agent(best_of_attempts=3, verify_command=None)
    assert await run_best_of_attempts(no_verifier, "task", "ORIGINAL") == "ORIGINAL"
