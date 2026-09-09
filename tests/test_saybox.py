"""The cockpit say box's editor (ADR-0119).

Operator report 2026-09-09: "if I paste a lot into it, I can't select and delete a large
chunk, there is no paste or delete-all shortcut, and when I scroll up to get previous
prompts that looks nasty." The box was a bare prompt session rebuilt on every message —
no history, no continuation prompt, a five-row pane a paste overflowed, and Ctrl+C
closing the pane outright. These drive the new editor through a pipe: every key it
binds, the paste tokens, the history, the pane geometry, and the close gesture.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput

from zakcode.cli.saybox import (
    MAX_ROWS,
    MIN_ROWS,
    PASTE_TOKEN_RE,
    PasteStore,
    SayBoxEditor,
    fold_lines,
    needed_rows,
)
from zakcode.session.say_inbox import write_say

BIG = "\n".join(f"line {i}" for i in range(1, 11))
PASTE = f"\x1b[200~{BIG}\x1b[201~"  # a bracketed paste, as the terminal sends it


class _Box:
    """One editor driven by a pipe: ``run(keys)`` returns ``(kind, text)``."""

    def __init__(self, tmp_path: Path, history: Path | None = None) -> None:
        self.inbox = tmp_path / ".say"
        self.interrupt = tmp_path / ".interrupt"
        self.history = history
        self.resizes: list[int] = []
        self.now = 0.0
        self.editor: SayBoxEditor | None = None

    def run(self, keys: str, default: str = "") -> tuple[str, str]:
        with create_pipe_input() as pipe:
            self.editor = SayBoxEditor(
                self.inbox,
                self.interrupt,
                history_path=self.history,
                resize=self.resizes.append,
                columns=lambda: 40,
                clock=lambda: self.now,
                input=pipe,
                output=DummyOutput(),
            )
            pipe.send_text(keys)
            return self.editor.prompt(default)


@pytest.fixture
def box(tmp_path: Path) -> _Box:
    return _Box(tmp_path)


# ── paste tokens ──────────────────────────────────────────────────────────────


def test_large_paste_collapses_to_a_token_and_expands_on_send(box: _Box) -> None:
    kind, text = box.run(f"hello {PASTE} world\r")
    assert kind == "line"
    assert text == f"hello {BIG} world"
    # the pane never grew: the buffer held one line with a token, not ten lines
    assert box.resizes == [MIN_ROWS]


def test_small_paste_stays_literal(box: _Box) -> None:
    _, text = box.run("\x1b[200~a\r\nb\x1b[201~\r")
    assert text == "a\nb"  # CRLF normalized, no token for a two-line paste


def test_backspace_and_delete_remove_a_token_as_one_unit(box: _Box) -> None:
    _, text = box.run(f"a {PASTE}\x7f b\r")
    assert text == "a  b"
    # Ctrl+A to the line start, two rights past "x ", Delete on the token
    _, text = box.run(f"x {PASTE}\x01\x1b[C\x1b[C\x1b[3~ b\r")
    assert text == "x  b"


def test_paste_store_contract() -> None:
    store = PasteStore()
    assert store.collapse("short") == "short"
    token = store.collapse(BIG)
    assert PASTE_TOKEN_RE.fullmatch(token) and "10 lines" in token
    long_line = "x" * 500
    assert "500 chars" in store.collapse(long_line)
    assert store.expand(f"say: {token}") == f"say: {BIG}"
    assert store.expand("⟪pasted #9 · 3 lines⟫") == "⟪pasted #9 · 3 lines⟫"  # unknown stays
    assert PasteStore.token_spanning(f"ab{token}cd", 2) == (2, 2 + len(token))
    assert PasteStore.token_spanning(f"ab{token}cd", 1) is None


# ── editing keys ──────────────────────────────────────────────────────────────


def test_ctrl_u_clears_everything_and_ctrl_z_restores(box: _Box) -> None:
    _, text = box.run("some text\x15after\r")
    assert text == "after"
    assert box.editor is not None and "Ctrl+Z" in box.editor.status
    _, text = box.run("some text\x15\x1a more\r")
    assert text == "some text more"


def test_ctrl_j_inserts_a_newline_and_enter_sends(box: _Box) -> None:
    _, text = box.run("abc\ndef\r")
    assert text == "abc\ndef"


# ── Esc: recall or stop ───────────────────────────────────────────────────────


def test_esc_with_nothing_pending_sends_a_stop_and_keeps_the_typed_text(box: _Box) -> None:
    kind, text = box.run("half typed\x1b")
    assert (kind, text) == ("interrupt-sent", "half typed")
    assert box.interrupt.exists()


def test_esc_recalls_a_pending_message_into_the_buffer(box: _Box) -> None:
    write_say(box.inbox, "pending msg")
    kind, text = box.run("\x1b more\r")
    assert (kind, text) == ("line", "pending msg more")
    assert not box.inbox.exists()
    assert box.editor is not None and "recalled" in box.editor.status


# ── Ctrl+C: clear first, close on a quick second press ────────────────────────


def test_ctrl_c_clears_the_input_instead_of_closing(box: _Box) -> None:
    _, text = box.run("typed\x03again\r")
    assert text == "again"
    assert box.editor is not None and "closes the box" in box.editor.status


def test_ctrl_c_on_an_empty_input_only_arms_the_close(box: _Box) -> None:
    _, text = box.run("\x03x\r")
    assert text == "x"
    assert box.editor is not None and "again to close" in box.editor.status


def test_double_ctrl_c_closes_the_box(box: _Box) -> None:
    with pytest.raises(KeyboardInterrupt):
        box.run("\x03\x03")


def test_slow_second_ctrl_c_does_not_close(tmp_path: Path) -> None:
    b = _Box(tmp_path)
    with create_pipe_input() as pipe:
        ed = SayBoxEditor(
            b.inbox, b.interrupt, clock=lambda: b.now, input=pipe, output=DummyOutput()
        )
        ticks = iter([0.0, 5.0, 5.0, 5.0])
        ed._clock = lambda: next(ticks, 5.0)
        pipe.send_text("\x03\x03x\r")
        assert ed.prompt() == ("line", "x")


# ── history ───────────────────────────────────────────────────────────────────


def test_history_persists_across_editors_and_up_recalls_it(tmp_path: Path) -> None:
    hist = tmp_path / "say-history"
    _Box(tmp_path, hist).run("first message\r")
    assert "first message" in hist.read_text(encoding="utf-8")
    kind, text = _Box(tmp_path, hist).run("\x1b[A\r")
    assert (kind, text) == ("line", "first message")


def test_history_records_the_real_text_behind_a_paste(tmp_path: Path) -> None:
    hist = tmp_path / "say-history"
    _Box(tmp_path, hist).run(f"{PASTE}\r")
    b = _Box(tmp_path, hist)
    _, text = b.run("\x1b[A\r")
    assert text == BIG
    assert b.resizes == [12, MIN_ROWS]  # grew for the recalled text, shrank after the send


# ── geometry and folding ──────────────────────────────────────────────────────


def test_needed_rows_clamps_between_min_and_max() -> None:
    assert needed_rows("a", 40) == MIN_ROWS
    assert needed_rows(BIG, 40) == 12  # 10 lines + toolbar + breath
    assert needed_rows("x" * 100, 40) == MIN_ROWS  # 3 wrapped rows + 2
    assert needed_rows("\n".join(["y"] * 40), 40) == MAX_ROWS


def test_fold_lines_keeps_short_text_and_folds_long() -> None:
    assert fold_lines("a\nb") == "a\nb"
    assert fold_lines("\n".join(map(str, range(9)))) == "0\n1\n2\n3\n4\n5\n… (+3 more lines)"
    assert fold_lines("\n".join(map(str, range(7)))).endswith("(+1 more line)")


def test_default_text_that_is_large_comes_back_as_a_token(box: _Box) -> None:
    # a refused (busy) message is carried back into the next prompt collapsed
    _, text = box.run(" tail\r", default=BIG)
    assert text == f"{BIG} tail"
    assert box.resizes == [MIN_ROWS]


def test_resize_seam_is_optional() -> None:
    fn: Callable[[int], None] | None = None
    with create_pipe_input() as pipe:
        ed = SayBoxEditor(
            Path("/nonexistent/.say"),
            Path("/nonexistent/.i"),
            resize=fn,
            input=pipe,
            output=DummyOutput(),
        )
        pipe.send_text("ok\r")
        assert ed.prompt() == ("line", "ok")
