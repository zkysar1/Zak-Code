"""The framed input prompt (read_prompt) — visible border, label, interrupt-safe close."""

from __future__ import annotations

import pytest
from rich.console import Console

from zakcode.cli._layout import _frame_width, read_prompt


def _console(width: int = 60) -> Console:
    return Console(record=True, width=width, force_terminal=True, legacy_windows=False)


@pytest.fixture(autouse=True)
def _no_ambient_frame_env(monkeypatch):
    # The frame is env-toggleable; keep every test hermetic against the runner's shell.
    monkeypatch.delenv("ZAKCODE_INPUT_FRAME", raising=False)


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


def test_input_frame_off_suppresses_all_chrome(monkeypatch):
    # An embedding cockpit with its own input box sets ZAKCODE_INPUT_FRAME=off:
    # the read still returns the line, but no border, label, or chevron renders.
    monkeypatch.setenv("ZAKCODE_INPUT_FRAME", "off")
    console = _console()
    monkeypatch.setattr(console, "input", lambda *a, **k: "hello coach")
    assert read_prompt(console) == "hello coach"
    out = console.export_text()
    assert "your message" not in out
    for glyph in ("╭", "╮", "╰", "╯", "›"):
        assert glyph not in out


def test_input_frame_on_values_keep_frame(monkeypatch):
    # Unset is covered by the frame tests above; any non-off value keeps the frame.
    monkeypatch.setenv("ZAKCODE_INPUT_FRAME", "on")
    console = _console()
    monkeypatch.setattr(console, "input", lambda *a, **k: "still framed")
    assert read_prompt(console) == "still framed"
    out = console.export_text()
    assert "your message" in out and "╭" in out
