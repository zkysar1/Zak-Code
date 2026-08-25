"""Unit tests for the shared say-inbox contract (``zakcode.session.say_inbox``)."""

from __future__ import annotations

from pathlib import Path

from zakcode.session.say_inbox import read_say, requeue_say, say_path, say_pending, write_say


def test_write_then_read_is_exactly_once(tmp_path: Path) -> None:
    inbox = say_path(tmp_path)
    assert write_say(inbox, "hello")
    assert say_pending(inbox)
    assert read_say(inbox) == "hello"
    assert not say_pending(inbox)
    assert read_say(inbox) is None


def test_single_slot_refuses_second_write(tmp_path: Path) -> None:
    inbox = say_path(tmp_path)
    assert write_say(inbox, "first")
    assert not write_say(inbox, "second")
    assert read_say(inbox) == "first"
    assert write_say(inbox, "second")


def test_requeue_skipped_when_newer_message_pending(tmp_path: Path) -> None:
    inbox = say_path(tmp_path)
    assert write_say(inbox, "newer")
    requeue_say(inbox, "older-recovered")
    assert read_say(inbox) == "newer"


def test_requeue_restores_message_into_empty_slot(tmp_path: Path) -> None:
    inbox = say_path(tmp_path)
    requeue_say(inbox, "recovered")
    assert read_say(inbox) == "recovered"


def test_write_creates_parent_and_read_strips(tmp_path: Path) -> None:
    inbox = say_path(tmp_path / "deep" / "workspace")
    assert write_say(inbox, "  padded  ")
    assert read_say(inbox) == "padded"


def test_empty_file_reads_as_no_say(tmp_path: Path) -> None:
    inbox = say_path(tmp_path)
    inbox.write_text("\n", encoding="utf-8")
    assert read_say(inbox) is None
    assert not inbox.exists()
