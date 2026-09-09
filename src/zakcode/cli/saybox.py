"""The cockpit say box's editor — one persistent, paste-aware prompt (ADR-0119).

The say box is the ONE place an operator types inside the cockpit, so it has to behave
like an editor, not like a bare ``input()``: a large paste collapses into a single token
(``⟪pasted #1 · 120 lines⟫``) that deletes as one unit and expands back to the real text
on send; the box remembers what was said (a history file, ``↑``/``↓`` recall at the top
and bottom line, ghost-text suggestions from history); ``Ctrl+U`` clears the whole
input and ``Ctrl+Z`` brings it back; ``Ctrl+J`` inserts a newline; ``Ctrl+C`` clears
first and closes only on a second press; a status toolbar names the keys; and the pane
grows with the text instead of scrolling a five-row window.

Everything terminal-facing goes through injectable seams (``resize``, ``columns``,
``clock``, prompt_toolkit's ``input``/``output``) so the editor is testable with a pipe.
"""

from __future__ import annotations

import math
import re
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

#: A paste with more lines than this, or more characters, collapses into a token.
PASTE_COLLAPSE_LINES = 3
PASTE_COLLAPSE_CHARS = 400
#: The token a collapsed paste leaves in the buffer. Angle-quote brackets are not on a
#: keyboard, so a typed message can never accidentally spell one.
PASTE_TOKEN_RE = re.compile(r"⟪pasted #(\d+) · \d+ (?:lines|chars)⟫")
#: Pane geometry: the box starts at the minimum and grows with the text up to the cap.
MIN_ROWS = 5
MAX_ROWS = 16
#: Second Ctrl+C within this window closes the box; the first only clears the input.
CTRL_C_CLOSE_WINDOW_S = 2.0
#: Prefix smuggled through the prompt result when Esc sent a stop signal; the remainder
#: is the operator's half-typed text, restored into the next prompt.
INTERRUPT_SENTINEL = "\x00zakcode-interrupt\x00"

KEY_HELP = "Enter send · Ctrl+J newline · Ctrl+U clear · Ctrl+Z undo · ↑↓ history · Esc stop"


class PasteStore:
    """Where the text behind each paste token lives for the life of the box."""

    def __init__(self) -> None:
        self._items: dict[int, str] = {}

    def collapse(self, text: str) -> str:
        """Return ``text`` unchanged when small, else store it and return its token."""
        lines = text.count("\n") + 1
        if lines <= PASTE_COLLAPSE_LINES and len(text) <= PASTE_COLLAPSE_CHARS:
            return text
        number = len(self._items) + 1
        self._items[number] = text
        size = f"{lines} lines" if lines > 1 else f"{len(text)} chars"
        return f"⟪pasted #{number} · {size}⟫"

    def expand(self, text: str) -> str:
        """Substitute every known token with its text (unknown tokens stay literal)."""
        return PASTE_TOKEN_RE.sub(lambda m: self._items.get(int(m.group(1)), m.group(0)), text)

    @staticmethod
    def token_spanning(text: str, index: int) -> tuple[int, int] | None:
        """The ``(start, end)`` of the token containing character ``index``, if any."""
        for m in PASTE_TOKEN_RE.finditer(text):
            if m.start() <= index < m.end():
                return m.span()
        return None


def needed_rows(
    text: str, columns: int, *, minimum: int = MIN_ROWS, maximum: int = MAX_ROWS
) -> int:
    """Pane rows for ``text``: its wrapped lines plus the toolbar and one breath, clamped."""
    width = max(1, columns - 2)  # the "▸ " / "· " gutter
    rows = sum(max(1, math.ceil(len(line) / width)) for line in text.split("\n"))
    return max(minimum, min(maximum, rows + 2))


def fold_lines(text: str, *, head: int = 6) -> str:
    """The first ``head`` lines of ``text`` plus a ``… (+N more lines)`` marker."""
    lines = text.split("\n")
    if len(lines) <= head:
        return text
    hidden = len(lines) - head
    return "\n".join(lines[:head]) + f"\n… (+{hidden} more line{'s' if hidden != 1 else ''})"


class SayBoxEditor:
    """One prompt_toolkit session that lives as long as the say box does.

    ``prompt()`` returns ``(kind, text)`` — ``"line"`` with the fully expanded message,
    or ``"interrupt-sent"`` with the preserved half-typed input after Esc sent a stop
    signal. Raises ``KeyboardInterrupt`` on the closing double Ctrl+C.
    """

    def __init__(
        self,
        inbox: Path,
        interrupt_fp: Path,
        *,
        history_path: Path | None = None,
        resize: Callable[[int], None] | None = None,
        columns: Callable[[], int] | None = None,
        clock: Callable[[], float] = time.monotonic,
        input: Any = None,  # noqa: A002 — prompt_toolkit's own parameter name
        output: Any = None,
    ) -> None:
        from prompt_toolkit import PromptSession
        from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
        from prompt_toolkit.filters import has_selection
        from prompt_toolkit.history import FileHistory, InMemoryHistory
        from prompt_toolkit.key_binding import KeyBindings
        from prompt_toolkit.keys import Keys
        from prompt_toolkit.styles import Style

        from zakcode.session.say_inbox import read_say, request_interrupt

        self.pastes = PasteStore()
        self.status = ""
        self._resize = resize or (lambda rows: None)
        self._columns = columns or (lambda: 80)
        self._clock = clock
        self._rows = 0
        self._last_ctrl_c = -1e9
        self._submitting = False
        bindings = KeyBindings()

        @bindings.add("escape", eager=True)
        def _esc(event: Any) -> None:
            # Esc: RECALL a message still sitting unconsumed in the inbox (it comes back
            # into the buffer for editing), otherwise STOP the running agent.
            recalled = read_say(inbox)
            buf = event.current_buffer
            if recalled is not None:
                buf.insert_text(self.pastes.collapse(recalled))
                self.status = "recalled the pending message — edit and send again"
            else:
                request_interrupt(interrupt_fp)
                event.app.exit(result=INTERRUPT_SENTINEL + buf.text)

        @bindings.add("enter")
        def _enter(event: Any) -> None:
            # Enter always sends. The buffer carries paste TOKENS; the message (and the
            # history entry) carries the real text.
            buf = event.current_buffer
            self._submitting = True
            buf.text = self.pastes.expand(buf.text)
            buf.validate_and_handle()

        @bindings.add("c-j")
        def _newline(event: Any) -> None:
            event.current_buffer.insert_text("\n")

        @bindings.add("c-u")
        def _clear(event: Any) -> None:
            buf = event.current_buffer
            if buf.text:
                buf.save_to_undo_stack()
                buf.text = ""
                self.status = "input cleared · Ctrl+Z brings it back"

        @bindings.add("c-z")
        def _undo(event: Any) -> None:
            event.current_buffer.undo()

        @bindings.add("c-c")
        def _ctrl_c(event: Any) -> None:
            # Like a shell: the first press clears the line, a quick second press
            # closes the box. Never an instant exit — Ctrl+C-to-copy is a habit.
            now = self._clock()
            buf = event.current_buffer
            if buf.text:
                buf.save_to_undo_stack()
                buf.text = ""
                self._last_ctrl_c = now
                self.status = "input cleared · ctrl-c again within 2s closes the box"
                return
            if now - self._last_ctrl_c <= CTRL_C_CLOSE_WINDOW_S:
                event.app.exit(exception=KeyboardInterrupt())
                return
            self._last_ctrl_c = now
            self.status = "ctrl-c again to close the message box"

        @bindings.add("backspace", filter=~has_selection)
        def _backspace(event: Any) -> None:
            buf = event.current_buffer
            span = PasteStore.token_spanning(buf.text, buf.cursor_position - 1)
            if span is None:
                buf.delete_before_cursor(event.arg)
                return
            self._delete_span(buf, span)

        @bindings.add("delete", filter=~has_selection)
        def _delete(event: Any) -> None:
            buf = event.current_buffer
            span = PasteStore.token_spanning(buf.text, buf.cursor_position)
            if span is None:
                buf.delete(event.arg)
                return
            self._delete_span(buf, span)

        @bindings.add(Keys.BracketedPaste)
        def _paste(event: Any) -> None:
            data = event.data.replace("\r\n", "\n").replace("\r", "\n")
            event.current_buffer.insert_text(self.pastes.collapse(data))

        history = FileHistory(str(history_path)) if history_path is not None else InMemoryHistory()
        self._session: Any = PromptSession(
            key_bindings=bindings,
            multiline=True,
            history=history,
            auto_suggest=AutoSuggestFromHistory(),
            prompt_continuation=lambda width, line_number, wrap_count: (
                "· " if wrap_count == 0 else "  "
            ),
            bottom_toolbar=self._toolbar,
            style=Style.from_dict({"bottom-toolbar": "noreverse fg:#8a8a8a"}),
            input=input,
            output=output,
        )
        self._session.default_buffer.on_text_changed += self._on_text_changed

    # ── seams ──────────────────────────────────────────────────────────────────

    @staticmethod
    def _delete_span(buf: Any, span: tuple[int, int]) -> None:
        start, end = span
        buf.save_to_undo_stack()
        buf.text = buf.text[:start] + buf.text[end:]
        buf.cursor_position = start

    def _toolbar(self) -> str:
        return f" {self.status}   {KEY_HELP}" if self.status else f" {KEY_HELP}"

    def _on_text_changed(self, buf: Any) -> None:
        if self._submitting:
            return
        rows = needed_rows(buf.text, self._columns())
        if rows != self._rows:
            self._rows = rows
            self._resize(rows)

    def prompt(self, default: str = "") -> tuple[str, str]:
        self._submitting = False
        try:
            text = self._session.prompt("▸ ", default=self.pastes.collapse(default))
        finally:
            self._submitting = False
            if self._rows != MIN_ROWS:
                self._rows = MIN_ROWS
                self._resize(MIN_ROWS)
        if text.startswith(INTERRUPT_SENTINEL):
            return ("interrupt-sent", text[len(INTERRUPT_SENTINEL) :])
        return ("line", text)
