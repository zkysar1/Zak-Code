"""The served mind's session store lives under the workspace (ADR-0032).

Hermetic: ``HOME``/``USERPROFILE`` point at a throwaway "home" so the assertion that the
user store stays untouched is a real one, and the app runs against a scripted agent
factory that never turns.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from zakcode.config import Settings
from zakcode.server.app import create_app
from zakcode.session.store import Session, SessionStore


@pytest.fixture
def fake_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    monkeypatch.delenv("ZAKCODE_HOME", raising=False)
    return home


def _no_turns(session: Session, model: str | None, prompter: object = None) -> object:  # noqa: ARG001
    raise AssertionError("this test never runs a turn")


def _served(workspace: Path) -> TestClient:
    settings = Settings(default_model="scripted/test", workspace_root=workspace)
    return TestClient(create_app(settings=settings, agent_factory=_no_turns))


def test_for_workspace_re_roots_the_user_layout_at_the_workspace(tmp_path: Path) -> None:
    ws = tmp_path / "ws"
    ws.mkdir()
    store = SessionStore.for_workspace(ws)
    assert store.base_dir == ws.resolve() / ".zakcode" / "sessions"
    assert store.base_dir.is_dir()
    assert store.list() == []
    # Self-ignoring: a workspace that is also a git checkout never commits transcripts.
    assert (store.base_dir / ".gitignore").read_text(encoding="utf-8") == "*\n"


def test_for_workspace_never_clobbers_an_existing_ignore(tmp_path: Path) -> None:
    ws = tmp_path / "ws"
    custom = ws / ".zakcode" / "sessions" / ".gitignore"
    custom.parent.mkdir(parents=True)
    custom.write_text("!keep-me.json\n", encoding="utf-8")
    SessionStore.for_workspace(ws)
    assert custom.read_text(encoding="utf-8") == "!keep-me.json\n"


def test_served_app_stores_sessions_under_the_workspace(fake_home: Path, tmp_path: Path) -> None:
    ws = tmp_path / "ws"
    ws.mkdir()
    client = _served(ws)

    sid = client.post("/sessions").json()["id"]

    assert (ws / ".zakcode" / "sessions" / f"{sid}.json").is_file()
    assert [s["id"] for s in client.get("/sessions").json()] == [sid]
    # A fresh process against the same workspace sees the same conversation.
    assert SessionStore.for_workspace(ws).load(sid).id == sid
    # ...and the serving host's own user store was never touched.
    assert not (fake_home / ".zakcode" / "sessions").exists()


def test_two_served_workspaces_are_isolated(fake_home: Path, tmp_path: Path) -> None:
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir()
    b.mkdir()
    saved = Session(cwd=".", model="scripted/test")
    SessionStore.for_workspace(a).save(saved)

    assert [s["id"] for s in _served(a).get("/sessions").json()] == [saved.id]
    assert _served(b).get("/sessions").json() == []
    assert not (fake_home / ".zakcode").exists()


def test_terminal_client_default_is_still_the_user_home(fake_home: Path) -> None:
    assert SessionStore().base_dir == fake_home / ".zakcode" / "sessions"
