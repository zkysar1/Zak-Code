"""Tests for operator-authored agent identity (self.md) discovery."""

from __future__ import annotations

from pathlib import Path

import pytest

from zakcode.identity import MAX_IDENTITY_CHARS, identity_paths, load_identity


def _fake_home(monkeypatch: pytest.MonkeyPatch, home: Path) -> None:
    # Point Path.home() at a controlled dir so the user-level ~/.config/zakcode/self.md
    # candidate is deterministic across machines.
    monkeypatch.setattr(Path, "home", lambda: home)


def test_no_self_md_returns_none(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_home(monkeypatch, tmp_path / "home")
    assert load_identity(tmp_path / "ws") == (None, None)


def test_precedence_project_zakcode_beats_workspace_beats_user(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / "home"
    (home / ".config" / "zakcode").mkdir(parents=True)
    (home / ".config" / "zakcode" / "self.md").write_text("USER", encoding="utf-8")
    _fake_home(monkeypatch, home)

    ws = tmp_path / "ws"
    (ws / ".zakcode").mkdir(parents=True)
    (ws / "self.md").write_text("WS", encoding="utf-8")
    (ws / ".zakcode" / "self.md").write_text("ZAKCODE", encoding="utf-8")

    assert load_identity(ws) == ("ZAKCODE", None)
    (ws / ".zakcode" / "self.md").unlink()
    assert load_identity(ws) == ("WS", None)
    (ws / "self.md").unlink()
    assert load_identity(ws) == ("USER", None)


def test_frontmatter_stripped(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_home(monkeypatch, tmp_path / "home")
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "self.md").write_text("---\nname: Vinheim\n---\nYou are Vinheim.", encoding="utf-8")
    assert load_identity(ws) == ("You are Vinheim.", None)


def test_oversize_truncated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_home(monkeypatch, tmp_path / "home")
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "self.md").write_text("x" * (MAX_IDENTITY_CHARS + 500), encoding="utf-8")
    text, err = load_identity(ws)
    assert err is None and text is not None and len(text) == MAX_IDENTITY_CHARS


def test_empty_file_is_error_not_blank(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # An existing-but-empty self.md must NOT blank the default identity:
    # it returns (None, error), never ('', None).
    _fake_home(monkeypatch, tmp_path / "home")
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "self.md").write_text("---\nname: x\n---\n   \n", encoding="utf-8")  # frontmatter only
    text, err = load_identity(ws)
    assert text is None and err is not None


def test_identity_paths_order(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_home(monkeypatch, tmp_path / "home")
    paths = identity_paths(tmp_path / "ws")
    assert paths[0] == tmp_path / "ws" / ".zakcode" / "self.md"
    assert paths[1] == tmp_path / "ws" / "self.md"
    assert paths[2] == tmp_path / "home" / ".config" / "zakcode" / "self.md"
