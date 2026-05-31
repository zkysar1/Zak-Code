"""Streaming TUI renderer for the CLI chat (rendering only).

This module consumes a stream of :class:`~zakcode.events.AgentEvent` objects and
renders them with ``rich``. It contains no agent logic, no network access, and no
provider imports, which keeps it fully unit-testable: feed a fake async generator
into :meth:`StreamRenderer.render` over a ``rich.console.Console`` backed by
``io.StringIO`` (``force_terminal=False``).

The subtle piece is *fence-safe* flushing of streamed text. Model output arrives
token by token, so naively writing every delta to a live console can render a
half-written fenced code block (an opening triple-backtick with no closing one
yet). :func:`split_on_safe_boundary` solves this: it only releases text up to the
last newline that is **not** inside an open code fence, keeping the remainder
buffered until the fence closes or the turn ends.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

from rich.console import Console

from zakcode.events import (
    AgentDone,
    AgentEvent,
    AgentStatus,
    AgentTextDelta,
    AgentToolCall,
    AgentToolResult,
    AgentUsage,
)
from zakcode.usage import Usage

_FENCE = "```"


def split_on_safe_boundary(text: str) -> tuple[str, str]:
    """Split ``text`` into ``(emit, keep)`` at a fence-safe newline boundary.

    ``emit`` is the prefix that is safe to write to a live console right now;
    ``keep`` is the remainder that must stay buffered (because it is either an
    incomplete final line, or it sits inside an open code fence).

    Rules:

    * A line whose *stripped* text starts with three backticks toggles the
      "inside a fence" state.
    * While a fence is **open**, nothing from the opening fence line onward may
      be emitted -- the whole fenced region is buffered until it closes.
    * Outside a fence, text is emitted up to and including the last newline; any
      trailing partial line (no terminating newline) is kept buffered so we
      never emit a half-written line.

    The function is pure: it does not mutate state, and ``emit + keep == text``
    for every input.
    """
    if not text:
        return "", ""

    in_fence = False
    safe_end = 0  # exclusive offset into ``text`` of the last safe-to-emit byte
    pos = 0
    length = len(text)

    while pos < length:
        nl = text.find("\n", pos)
        if nl == -1:
            # Trailing partial line (no newline). A partial line is never safe
            # to emit, and inside a fence it must stay buffered anyway. Stop.
            break

        line = text[pos:nl]
        if line.lstrip().startswith(_FENCE):
            if in_fence:
                # Closes the fence; up to and including this newline is safe.
                in_fence = False
                safe_end = nl + 1
            else:
                # Opens a fence; do NOT advance safe_end past this line's start.
                in_fence = True
        elif not in_fence:
            # Ordinary complete line outside any fence -> safe to emit.
            safe_end = nl + 1
        # else: complete line inside an open fence -> keep buffering.

        pos = nl + 1

    return text[:safe_end], text[safe_end:]


class StreamRenderer:
    """Render an :class:`~zakcode.events.AgentEvent` stream to a ``rich`` console.

    Streamed text is buffered and only flushed on fence-safe boundaries (see
    :func:`split_on_safe_boundary`), so a partially streamed code block never
    renders broken. Tool calls, tool results, and status messages render as
    concise dim lines. Usage is accumulated internally and summarized in a
    footer printed on :class:`~zakcode.events.AgentDone`.
    """

    def __init__(self, console: Console | None = None) -> None:
        self.console = console if console is not None else Console()
        self._text_buffer = ""
        self._usage = Usage()

    async def render(self, events: AsyncIterator[AgentEvent]) -> AgentDone | None:
        """Consume ``events``, render them, and return the final ``AgentDone``.

        Returns the :class:`~zakcode.events.AgentDone` event if one is seen,
        otherwise ``None`` (e.g. an empty or truncated stream). Any text still
        buffered at end of turn is flushed before the footer.
        """
        done: AgentDone | None = None

        async for event in events:
            if isinstance(event, AgentTextDelta):
                self._on_text_delta(event.text)
            elif isinstance(event, AgentToolCall):
                self._on_tool_call(event)
            elif isinstance(event, AgentToolResult):
                self._on_tool_result(event)
            elif isinstance(event, AgentStatus):
                self._on_status(event)
            elif isinstance(event, AgentUsage):
                self._on_usage(event)
            elif isinstance(event, AgentDone):
                done = event
                break  # terminal; flush + footer happen below

        self._flush_remaining_text()

        if done is not None:
            self._print_footer(done)
        return done

    # -- per-event handlers -------------------------------------------------

    def _on_text_delta(self, text: str) -> None:
        self._text_buffer += text
        emit, keep = split_on_safe_boundary(self._text_buffer)
        self._text_buffer = keep
        if emit:
            # ``end=""`` because deltas carry their own newlines; markup/
            # highlight off so raw model text is never reinterpreted.
            self.console.print(emit, end="", markup=False, highlight=False)

    def _on_tool_call(self, event: AgentToolCall) -> None:
        args = _format_args(event.arguments)
        self.console.print(f"tool {event.name}({args})", style="dim", highlight=False)

    def _on_tool_result(self, event: AgentToolResult) -> None:
        status = "err" if event.is_error else "ok"
        first_line = _first_line(event.output)
        suffix = f" {first_line}" if first_line else ""
        self.console.print(f"  -> {status}{suffix}", style="dim", highlight=False)

    def _on_status(self, event: AgentStatus) -> None:
        self.console.print(event.message, style="dim", highlight=False)

    def _on_usage(self, event: AgentUsage) -> None:
        self._usage = self._usage + event.usage

    # -- helpers ------------------------------------------------------------

    def _flush_remaining_text(self) -> None:
        if self._text_buffer:
            self.console.print(self._text_buffer, end="", markup=False, highlight=False)
            self._text_buffer = ""

    def _print_footer(self, done: AgentDone) -> None:
        # Prefer the authoritative per-turn usage on AgentDone; fall back to the
        # running total accumulated from AgentUsage events if it carries none.
        usage = done.usage if done.usage.total_tokens else self._usage
        parts = [f"{done.iterations} iter", f"{usage.total_tokens} tok"]
        if usage.cost_usd:
            parts.append(f"${usage.cost_usd:.4f}")
        self.console.print(f"· {' · '.join(parts)}", style="dim", highlight=False)


def _format_args(arguments: dict[str, object]) -> str:
    """Render tool args as a compact, single-line ``k=v,...`` string.

    Values are stringified, collapsed to their first line, and truncated so the
    tool-call line always stays on one row.
    """
    return ",".join(f"{key}={_abbrev_value(value)}" for key, value in arguments.items())


def _abbrev_value(value: object, *, limit: int = 40) -> str:
    text = str(value)
    if "\n" in text:
        text = text.splitlines()[0]
    if len(text) > limit:
        text = text[: limit - 1] + "…"
    return text


def _first_line(output: object, *, limit: int = 80) -> str:
    text = str(output).strip()
    if not text:
        return ""
    first = text.splitlines()[0]
    if len(first) > limit:
        first = first[: limit - 1] + "…"
    return first
