"""Tests for session state and on-disk persistence."""

from __future__ import annotations

from zakcode.messages import Message, TextBlock, ToolResultBlock, ToolUseBlock
from zakcode.session import Session, SessionStore
from zakcode.usage import Usage


def _build_session() -> Session:
    """Build a session with a representative message + usage history."""
    session = Session(cwd="/work/project", model="ollama/llama3")
    session.add_message(Message.user("read foo.py and summarize it"))
    session.add_message(
        Message(
            role="assistant",
            blocks=[
                TextBlock(text="let me read it"),
                ToolUseBlock(id="call-1", name="read", input={"path": "foo.py"}),
            ],
        )
    )
    session.add_message(
        Message.tool_results([ToolResultBlock(tool_use_id="call-1", output="print('hi')")])
    )
    session.add_message(Message.assistant_text("It prints 'hi'."))
    session.add_usage(Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15, cost_usd=0.001))
    session.add_usage(Usage(prompt_tokens=20, completion_tokens=8, total_tokens=28, cost_usd=0.002))
    return session


def test_cumulative_usage_sums() -> None:
    session = _build_session()
    total = session.cumulative_usage()
    assert total.prompt_tokens == 30
    assert total.completion_tokens == 13
    assert total.total_tokens == 43
    assert total.cost_usd == 0.003


def test_cumulative_usage_empty() -> None:
    session = Session(cwd="/tmp", model="m")
    assert session.cumulative_usage() == Usage()


def test_save_load_roundtrip(tmp_path) -> None:
    store = SessionStore(base_dir=tmp_path)
    session = _build_session()

    path = store.save(session)
    assert path.exists()
    assert path == tmp_path / f"{session.id}.json"

    loaded = store.load(session.id)
    assert loaded == session
    assert loaded.messages == session.messages
    assert loaded.cumulative_usage() == session.cumulative_usage()


def test_no_temp_file_left_behind(tmp_path) -> None:
    store = SessionStore(base_dir=tmp_path)
    store.save(_build_session())
    assert not list(tmp_path.glob("*.tmp"))
    assert len(list(tmp_path.glob("*.json"))) == 1


def test_list_includes_id(tmp_path) -> None:
    store = SessionStore(base_dir=tmp_path)
    session = _build_session()
    store.save(session)
    assert session.id in store.list()


def test_resume_matches_load(tmp_path) -> None:
    store = SessionStore(base_dir=tmp_path)
    session = _build_session()
    store.save(session)
    assert store.resume(session.id) == store.load(session.id)


def test_default_base_dir_under_home(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    store = SessionStore()
    assert store.base_dir.exists()
    assert store.base_dir.name == "sessions"
