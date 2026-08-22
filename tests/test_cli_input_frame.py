"""The framed input prompt (read_prompt) — visible border, label, interrupt-safe close."""

from __future__ import annotations

import pytest
from rich.console import Console

from zakcode.cli._layout import _frame_width, read_prompt


def _console(width: int = 60) -> Console:
    return Console(record=True, width=width, force_terminal=True, legacy_windows=False)


def test_read_prompt_draws_frame_and_returns_input(monkeypatch):
    console = _console()
    monkeypatch.setattr(console, "input", lambda *a, **k: "hello coach")
    assert read_prompt(console) == "hello coach"
    out = console.export_text()
    assert "your message" in out
    # Top and bottom borders both present (unicode corners on a capable console).
    assert "╭" in out and "╮" in out
    assert "╰" in out and "╯" in out


def test_read_prompt_closes_frame_on_interrupt(monkeypatch):
    console = _console()

    def _boom(*a, **k):
        raise KeyboardInterrupt

    monkeypatch.setattr(console, "input", _boom)
    with pytest.raises(KeyboardInterrupt):
        read_prompt(console)
    out = console.export_text()
    # The bottom border still printed — the frame never dangles open.
    assert "╰" in out and "╯" in out


def test_frame_width_bounds():
    assert _frame_width(_console(width=200)) == 100  # readability cap
    assert _frame_width(_console(width=20)) == 24  # floor beats tiny consoles
    assert _frame_width(_console(width=80)) == 76  # margin-adjusted
