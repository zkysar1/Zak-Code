"""Tests for agent_identity_dir extension to identity discovery (PR-T6).

These tests cover the new parameter only.  The existing test_identity.py
tests verify unchanged 3-path behavior and are not modified.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from zakcode.identity import MAX_IDENTITY_CHARS, identity_paths, load_identity


def _fake_home(monkeypatch: pytest.MonkeyPatch, home: Path) -> None:
    monkeypatch.setattr(Path, "home", lambda: home)


def test_identity_paths_default_unchanged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """agent_identity_dir=None produces the same 3-path list as before."""
    _fake_home(monkeypatch, tmp_path / "home")
    paths = identity_paths(tmp_path / "ws", agent_identity_dir=None)
    assert len(paths) == 3
    assert paths[0] == tmp_path / "ws" / ".zakcode" / "self.md"


def test_identity_paths_with_agent_dir_relative(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Relative agent_identity_dir is resolved against workspace_root."""
    _fake_home(monkeypatch, tmp_path / "home")
    paths = identity_paths(tmp_path / "ws", agent_identity_dir="agents/omni")
    assert len(paths) == 4
    assert paths[0] == tmp_path / "ws" / "agents" / "omni" / "self.md"
    # Original three still present.
    assert paths[1] == tmp_path / "ws" / ".zakcode" / "self.md"


def test_identity_paths_with_agent_dir_absolute(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Absolute agent_identity_dir is used as-is."""
    _fake_home(monkeypatch, tmp_path / "home")
    abs_dir = tmp_path / "external" / "agents" / "omni"
    paths = identity_paths(tmp_path / "ws", agent_identity_dir=abs_dir)
    assert len(paths) == 4
    assert paths[0] == abs_dir / "self.md"


def test_load_identity_from_agent_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """self.md found at agent_identity_dir wins over workspace candidates."""
    _fake_home(monkeypatch, tmp_path / "home")
    ws = tmp_path / "ws"
    ws.mkdir()
    # Create both a workspace self.md and an agent-dir self.md.
    (ws / "self.md").write_text("workspace identity", encoding="utf-8")
    agent_dir = ws / "agents" / "omni"
    agent_dir.mkdir(parents=True)
    (agent_dir / "self.md").write_text(
        "---\nname: omni\n---\nI am the omni agent.", encoding="utf-8"
    )
    text, err = load_identity(ws, agent_identity_dir="agents/omni")
    assert err is None
    assert text == "I am the omni agent."


def test_load_identity_agent_dir_fallthrough(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Missing self.md at agent_identity_dir falls through to workspace."""
    _fake_home(monkeypatch, tmp_path / "home")
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "self.md").write_text("workspace identity", encoding="utf-8")
    # Agent dir exists but has no self.md.
    agent_dir = ws / "agents" / "omni"
    agent_dir.mkdir(parents=True)
    text, err = load_identity(ws, agent_identity_dir="agents/omni")
    assert err is None
    assert text == "workspace identity"


def test_load_identity_agent_dir_truncation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """MAX_IDENTITY_CHARS truncation applies to agent-dir self.md too."""
    _fake_home(monkeypatch, tmp_path / "home")
    ws = tmp_path / "ws"
    agent_dir = ws / "agents" / "omni"
    agent_dir.mkdir(parents=True)
    (agent_dir / "self.md").write_text(
        "x" * (MAX_IDENTITY_CHARS + 100), encoding="utf-8"
    )
    text, err = load_identity(ws, agent_identity_dir="agents/omni")
    assert err is None
    assert text is not None and len(text) == MAX_IDENTITY_CHARS
