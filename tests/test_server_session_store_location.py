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


# ===========================================================================
# The hook-transcript projection follows the store (ADR-0061)
# ===========================================================================
#
# `_cc_transcript_path()` renders the FULL conversation for hooks that read
# `transcript_path`. It rooted at `Path.home()`, so every mind served by one host user
# pooled its conversation text into one shared directory — the cross-workspace leak
# ADR-0032 closed for the store itself while its projection went on bypassing it. These
# assert the projection now shares the store's lifetime and isolation, and that the
# terminal client's path is byte-for-byte what it always was.


def _agent_on(workspace: Path, store: SessionStore | None):
    from zakcode import Agent

    return Agent(
        settings=Settings(default_model="scripted/test", workspace_root=workspace),
        session_store=store,
    )


def _spoken(agent) -> None:
    from zakcode.messages import Message

    agent.session.add_message(Message.user("remember: the pearl holds"))
    agent.session.add_message(Message.assistant_text("held"))


def test_served_transcript_projection_lands_under_the_workspace(
    fake_home: Path, tmp_path: Path
) -> None:
    ws = tmp_path / "ws"
    ws.mkdir()
    agent = _agent_on(ws, SessionStore.for_workspace(ws))
    _spoken(agent)

    path = Path(agent.loop._cc_transcript_path())

    assert path.parent == ws.resolve() / ".zakcode" / "transcripts"
    assert path.read_text(encoding="utf-8").strip()
    # The host user's home holds no copy of this mind's conversation.
    assert not (fake_home / ".zakcode" / "transcripts").exists()


def test_two_served_workspaces_do_not_share_a_transcript_directory(
    fake_home: Path, tmp_path: Path
) -> None:
    # The isolation ADR-0032 gives the store, for the projection of the same conversation.
    first, second = tmp_path / "a", tmp_path / "b"
    for ws in (first, second):
        ws.mkdir()
    one = Path(_agent_on(first, SessionStore.for_workspace(first)).loop._cc_transcript_path())
    two = Path(_agent_on(second, SessionStore.for_workspace(second)).loop._cc_transcript_path())

    assert one.parent != two.parent
    assert one.parent == first.resolve() / ".zakcode" / "transcripts"
    assert two.parent == second.resolve() / ".zakcode" / "transcripts"


def test_served_transcript_directory_ignores_itself(fake_home: Path, tmp_path: Path) -> None:
    # A served workspace is often a git checkout; the conversation must never be committable.
    ws = tmp_path / "ws"
    ws.mkdir()
    path = Path(_agent_on(ws, SessionStore.for_workspace(ws)).loop._cc_transcript_path())

    assert (path.parent / ".gitignore").read_text(encoding="utf-8") == "*\n"


def test_transcript_projection_never_clobbers_an_existing_ignore(
    fake_home: Path, tmp_path: Path
) -> None:
    ws = tmp_path / "ws"
    (ws / ".zakcode" / "transcripts").mkdir(parents=True)
    (ws / ".zakcode" / "transcripts" / ".gitignore").write_text("mine\n", encoding="utf-8")

    _agent_on(ws, SessionStore.for_workspace(ws)).loop._cc_transcript_path()

    ignore = ws / ".zakcode" / "transcripts" / ".gitignore"
    assert ignore.read_text(encoding="utf-8") == "mine\n"


def test_terminal_client_transcript_path_is_unchanged(fake_home: Path, tmp_path: Path) -> None:
    # The terminal client's store is ~/.zakcode/sessions, so its sibling is the historical
    # ~/.zakcode/transcripts. This is the regression that would break every terminal hook.
    ws = tmp_path / "project"
    ws.mkdir()
    path = Path(_agent_on(ws, SessionStore()).loop._cc_transcript_path())

    assert path.parent == fake_home / ".zakcode" / "transcripts"


def test_storeless_loop_keeps_the_user_home(fake_home: Path, tmp_path: Path) -> None:
    # A bare Agent injects no store; nothing about that path changes.
    ws = tmp_path / "project"
    ws.mkdir()
    path = Path(_agent_on(ws, None).loop._cc_transcript_path())

    assert path.parent == fake_home / ".zakcode" / "transcripts"
