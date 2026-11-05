"""A deleted conversation goes to the trash, not away (ADR-0358).

``SessionStore.trash`` moves a conversation's document, its transcript and what else it owns into
``<store parent>/trash/<id>/``, and says in ``deleted.json`` where each came from, so a deletion is
undone by moving them back. The document goes first and alone decides the outcome: one that cannot
be moved leaves everything where it was. ``save(keep_time=True)`` saves a change that is not a new
message without moving the chat up the list.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import pytest

from zakcode.messages import Message
from zakcode.session import store as store_module
from zakcode.session.store import Session, SessionStore


def _chat(store: SessionStore) -> Session:
    session = Session(cwd=".", model="m")
    session.add_message(Message.user("an ask"))
    store.save(session)
    return session


def test_a_missing_chat_is_not_trashed(tmp_path: Path) -> None:
    store = SessionStore(base_dir=tmp_path / "sessions")

    assert store.trash("f" * 32) is None
    assert not (tmp_path / "trash").exists()


def test_a_chat_deleted_again_after_it_was_put_back_gets_a_folder_of_its_own(
    tmp_path: Path,
) -> None:
    store = SessionStore(base_dir=tmp_path / "sessions")
    chat = _chat(store)
    first = store.trash(chat.id)
    assert first == tmp_path / "trash" / chat.id
    # Undone as deleted.json says: the document moved back to where it came from.
    record = json.loads((first / "deleted.json").read_text(encoding="utf-8"))
    os.replace(first / f"{chat.id}.json", record["moved"][f"{chat.id}.json"])
    assert store.load(chat.id).messages[0].text == "an ask"

    second = store.trash(chat.id)

    assert second == tmp_path / "trash" / f"{chat.id}-2"
    assert (second / f"{chat.id}.json").is_file()
    assert (tmp_path / "trash" / ".gitignore").read_text(encoding="utf-8") == "*\n"


def test_a_document_that_cannot_be_moved_leaves_everything_where_it_was(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = SessionStore(base_dir=tmp_path / "sessions")
    chat = _chat(store)
    transcript = store.transcript_path(chat.id)
    transcript.parent.mkdir(parents=True)
    transcript.write_text("{}\n", encoding="utf-8")

    def refused(tmp: Path, path: Path) -> None:
        raise PermissionError("held open")

    monkeypatch.setattr(store_module, "replace_retrying", refused)

    with pytest.raises(PermissionError):
        store.trash(chat.id)

    assert store.exists(chat.id)
    assert transcript.exists()
    assert list((tmp_path / "trash").iterdir()) == [tmp_path / "trash" / ".gitignore"]


def test_a_transcript_that_cannot_be_moved_stays_and_is_said_so(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = SessionStore(base_dir=tmp_path / "sessions")
    chat = _chat(store)
    transcript = store.transcript_path(chat.id)
    transcript.parent.mkdir(parents=True)
    transcript.write_text("{}\n", encoding="utf-8")
    uploads = tmp_path / "uploads" / chat.id
    uploads.mkdir(parents=True)
    moving = shutil.move

    def move(source: Path, target: Path) -> object:
        if Path(source) == transcript:
            raise PermissionError("held open")
        return moving(source, target)

    monkeypatch.setattr(store_module.shutil, "move", move)

    folder = store.trash(chat.id, also={"uploads": uploads})

    assert folder is not None
    assert not store.exists(chat.id)  # the chat is deleted all the same
    assert transcript.exists()
    assert (folder / "uploads").is_dir()
    record = json.loads((folder / "deleted.json").read_text(encoding="utf-8"))
    assert record["left"] == {transcript.name: str(transcript)}
    assert set(record["moved"]) == {f"{chat.id}.json", "uploads"}


def test_a_save_that_keeps_its_time_leaves_the_file_dated_as_it_was(tmp_path: Path) -> None:
    store = SessionStore(base_dir=tmp_path / "sessions")
    chat = _chat(store)
    path = store.base_dir / f"{chat.id}.json"
    os.utime(path, (1_790_000_000, 1_790_000_000))

    chat.title = "Trip"
    store.save(chat, keep_time=True)

    assert path.stat().st_mtime == 1_790_000_000
    assert store.load(chat.id).title == "Trip"
    store.save(chat)
    assert path.stat().st_mtime > 1_790_000_000
