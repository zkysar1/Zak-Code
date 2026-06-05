"""Streaming TUI renderer for the CLI chat (rendering only).

Consumes a stream of :class:`~zakcode.events.AgentEvent` objects and renders them with
``rich`` in the "minimal-gutter" look: structure comes from a thin colored marker in a
2-column document margin plus one-blank-line rhythm, with ``dim`` reserved strictly for
chrome so the assistant's own words and the salient tool target are the brightest things
on screen. No agent logic, no network, no provider imports — feed a fake async generator
into :meth:`StreamRenderer.render` over a ``Console`` backed by ``io.StringIO`` to test it.

Two subtle pieces:

* *Fence-safe flushing* (:func:`split_on_safe_boundary`): streamed text is released only
  up to the last newline that is **not** inside an open code fence, so a half-written
  fenced block never renders.
* *Segment routing* (:func:`iter_segments`): a fence-safe ``emit`` is split into ordered
  prose / code runs; prose streams as bright text in the margin, while a completed fenced
  block is rendered once via ``rich.Syntax`` (highlighted), never half-drawn.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import cast

from rich.console import Console
from rich.padding import Padding
from rich.rule import Rule
from rich.syntax import Syntax
from rich.text import Text

from zakcode.cli._glyphs import resolve_glyphs
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

#: Max lines of shell/run output shown inline in a result block (then "... (+N more)").
_RUN_OUTPUT_LINES = 12

#: Map a tool name to the short verb shown in the gutter; unknown tools use their name.
_VERB_MAP = {
    "read_file": "read",
    "write_file": "write",
    "edit_file": "edit",
    "list_dir": "list",
    "glob": "glob",
    "grep": "grep",
    "bash": "run",
    "powershell": "run",
    "web_fetch": "fetch",
    "web_search": "search",
    "todo_write": "todo",
    "task": "task",
}


def split_on_safe_boundary(text: str) -> tuple[str, str]:
    """Split ``text`` into ``(emit, keep)`` at a fence-safe newline boundary.

    ``emit`` is the prefix safe to write to a live console now; ``keep`` is the
    remainder that must stay buffered (an incomplete final line, or text inside an
    open code fence). ``emit + keep == text`` always. A line whose stripped text
    starts with three backticks toggles the "inside a fence" state; while a fence is
    open nothing from the opening fence line onward is emitted.
    """
    if not text:
        return "", ""

    in_fence = False
    safe_end = 0
    pos = 0
    length = len(text)

    while pos < length:
        nl = text.find("\n", pos)
        if nl == -1:
            break

        line = text[pos:nl]
        if line.lstrip().startswith(_FENCE):
            if in_fence:
                in_fence = False
                safe_end = nl + 1
            else:
                in_fence = True
        elif not in_fence:
            safe_end = nl + 1

        pos = nl + 1

    return text[:safe_end], text[safe_end:]


def iter_segments(text: str) -> list[tuple[str, object]]:
    """Split a fence-safe ``text`` into ordered ``('prose', str)`` / ``('code', (lang, body))``.

    ``text`` comes from :func:`split_on_safe_boundary`, so every fence here is already
    closed — we only have to *classify* the lines. The trailing line terminator is
    dropped (so ``"a\\nb\\n"`` is two lines, not three) while genuine blank lines are
    preserved.
    """
    if not text:
        return []
    lines = text.split("\n")
    if lines and lines[-1] == "":
        lines.pop()

    segments: list[tuple[str, object]] = []
    prose: list[str] = []
    i = 0
    n = len(lines)
    while i < n:
        line = lines[i]
        if line.lstrip().startswith(_FENCE):
            if prose:
                segments.append(("prose", "\n".join(prose)))
                prose = []
            lang = line.lstrip()[len(_FENCE) :].strip()
            body: list[str] = []
            i += 1
            while i < n and not lines[i].lstrip().startswith(_FENCE):
                body.append(lines[i])
                i += 1
            i += 1  # skip the closing fence (or run off the end on an unclosed one)
            segments.append(("code", (lang, "\n".join(body))))
        else:
            prose.append(line)
            i += 1
    if prose:
        segments.append(("prose", "\n".join(prose)))
    return segments


class StreamRenderer:
    """Render an :class:`~zakcode.events.AgentEvent` stream to a ``rich`` console.

    Prose streams as bright text in the document margin (a ``·`` marker on the first
    row of a reply); a fenced code block renders once via ``rich.Syntax`` when it
    closes. Tool calls render as ``→ verb  target`` and their results as a status-
    colored ``✓/✗ name · summary`` line (a unified-diff preview when applicable). Usage
    is accumulated and summarized in a footer printed on :class:`AgentDone`. All glyphs
    resolve to cp1252-safe ASCII fallbacks when the console cannot encode them.
    """

    def __init__(self, console: Console | None = None) -> None:
        self.console = console if console is not None else Console()
        self._g = resolve_glyphs(self.console)
        self._text_buffer = ""
        self._usage = Usage()
        self._tool_names: dict[str, str] = {}
        self._assistant_marked = False

    async def render(self, events: AsyncIterator[AgentEvent]) -> AgentDone | None:
        """Consume ``events``, render them, and return the final ``AgentDone`` (or None)."""
        done: AgentDone | None = None
        self._assistant_marked = False

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
                break

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
            self._render_stream_text(emit)

    def _render_stream_text(self, text: str) -> None:
        for kind, payload in iter_segments(text):
            if kind == "code":
                lang, body = cast("tuple[str, str]", payload)
                self._print_code(lang, body)
            else:
                self._print_prose(cast("str", payload))

    def _print_prose(self, block: str) -> None:
        for line in block.split("\n"):
            if not self._assistant_marked:
                if line.strip() == "":
                    continue  # swallow leading blank lines before the first content
                self._assistant_marked = True
                marker = Text(self._g["dot"] + " ", style="assistant.marker")
                self.console.print(marker + _inline_md(line, self._g))
            elif line.strip() == "":
                self.console.print()
            else:
                self.console.print(Padding(_inline_md(line, self._g), (0, 0, 0, 2)))

    def _print_code(self, lang: str, body: str) -> None:
        if not body.strip():
            return
        syntax = Syntax(
            body,
            lang or "text",
            theme="ansi_dark",
            background_color="default",
            word_wrap=True,
        )
        self.console.print()
        self.console.print(Padding(syntax, (0, 0, 0, 4)))
        self.console.print()

    def _on_tool_call(self, event: AgentToolCall) -> None:
        verb, target = _format_tool_call(event.name, event.arguments)
        self._tool_names[event.id] = verb
        self.console.print()
        self.console.print(
            Padding(
                Text.assemble(
                    (self._g["arrow"] + " ", "tool.marker"),
                    (verb + "  ", "tool.verb"),
                    (target, "tool.target"),
                ),
                (0, 0, 0, 2),
            )
        )

    def _result_head(self, name: str, state: str, summary: str) -> Padding:
        glyph = self._g["fail"] if state == "err" else self._g["ok"]
        return Padding(
            Text.assemble(
                (glyph + " ", state),
                (name + " ", "tool.marker"),
                (self._g["dot"] + " ", "tool.marker"),
                (summary, "notice.dim"),
            ),
            (0, 0, 0, 4),
        )

    def _on_tool_result(self, event: AgentToolResult) -> None:
        name = self._tool_names.get(event.tool_use_id, "tool")
        state = "err" if event.is_error else "ok"
        output = str(event.output)

        if name == "run":
            # Shell output is the whole point — show the program's real stdout/stderr
            # (capped), not just its first line, so the user can see what actually ran.
            lines = output.splitlines()
            n = len(lines)
            if event.is_error:
                summary = "failed"
            else:
                summary = f"{n} line{'' if n == 1 else 's'}" if n else "no output"
            self.console.print(self._result_head(name, state, summary))
            preview = lines[:_RUN_OUTPUT_LINES]
            if preview:
                self.console.print(
                    Padding(Text("\n".join(preview), style="notice.dim"), (0, 0, 0, 6))
                )
            if n > _RUN_OUTPUT_LINES:
                more = n - _RUN_OUTPUT_LINES
                tail = f"{self._g['ellipsis']} (+{more} more line{'' if more == 1 else 's'})"
                self.console.print(Padding(Text(tail, style="notice.dim"), (0, 0, 0, 6)))
            return

        summary = _first_line(output) or ("error" if event.is_error else "ok")
        self.console.print(self._result_head(name, state, summary))
        diff = _diff_preview(output)
        if diff is not None:
            self.console.print(Padding(diff, (0, 0, 0, 6)))
        elif event.is_error:
            extra = "\n".join(output.splitlines()[1:6]).rstrip()
            if extra:
                self.console.print(Padding(Text(extra, overflow="fold"), (0, 0, 0, 6)))

    def _on_status(self, event: AgentStatus) -> None:
        self.console.print(
            Padding(
                Text.assemble((self._g["status"] + " ", "status"), (event.message, "status")),
                (0, 0, 0, 2),
            )
        )

    def _on_usage(self, event: AgentUsage) -> None:
        self._usage = self._usage + event.usage

    # -- helpers ------------------------------------------------------------

    def _flush_remaining_text(self) -> None:
        if self._text_buffer:
            self._render_stream_text(self._text_buffer)
            self._text_buffer = ""

    def _print_footer(self, done: AgentDone) -> None:
        usage = done.usage if done.usage.total_tokens else self._usage
        g = self._g
        line = (
            f"{g['dot']} {done.iterations} iter "
            f"{g['dot']} {_humanize_tokens(usage.total_tokens)} "
            f"{g['dot']} ${usage.cost_usd:.4f}"
        )
        self.console.print()
        self.console.print(Padding(Rule(characters=g["hline"], style="rule.line"), (0, 0, 0, 2)))
        self.console.print(Padding(Text(line, style="footer"), (0, 0, 0, 2)))


def _inline_md(line: str, glyphs: dict[str, str]) -> Text:
    """Render one prose line with lightweight inline markdown.

    Handles list bullets (``- ``/``* `` -> the bullet glyph), ATX headings (``##`` ->
    bold), ``**bold**`` and ``` `code` ```. Built by hand with ``Text.append`` so model
    text can never inject rich console markup (the markup=False safety we rely on).
    """
    stripped = line.lstrip(" ")
    lead = line[: len(line) - len(stripped)]
    out = Text(lead)
    if stripped[:2] in ("- ", "* "):
        out.append(glyphs["bullet"] + " ", style="dim")
        out.append_text(_inline_spans(stripped[2:]))
        return out
    if stripped.startswith("#"):
        body = _inline_spans(stripped.lstrip("#").lstrip())
        body.stylize("bold")
        out.append_text(body)
        return out
    out.append_text(_inline_spans(stripped))
    return out


def _inline_spans(text: str) -> Text:
    """Parse non-nested ``**bold**`` and ``` `code` ``` spans into a styled Text."""
    out = Text()
    i, n = 0, len(text)
    while i < n:
        if text.startswith("**", i):
            end = text.find("**", i + 2)
            if end != -1:
                out.append(text[i + 2 : end], style="bold")
                i = end + 2
                continue
        if text[i] == "`":
            end = text.find("`", i + 1)
            if end != -1:
                out.append(text[i + 1 : end], style="cyan")
                i = end + 1
                continue
        out.append(text[i])
        i += 1
    return out


def _format_tool_call(name: str, arguments: object) -> tuple[str, str]:
    """Map a tool call to ``(verb, target)`` for the gutter line."""
    verb = _VERB_MAP.get(name, name)
    return verb, _tool_target(arguments)


def _tool_target(arguments: object) -> str:
    """The single salient argument to show after the verb (path / command / pattern)."""
    if not isinstance(arguments, dict):
        return _abbrev_value(arguments)
    for key in ("command",):
        value = arguments.get(key)
        if isinstance(value, str) and value:
            return "$ " + _abbrev_value(value, limit=120)
    for key in ("path", "file_path", "pattern", "query", "url", "name"):
        value = arguments.get(key)
        if isinstance(value, str) and value:
            return _abbrev_value(value, limit=120)
    for value in arguments.values():
        if isinstance(value, str) and value:
            return _abbrev_value(value, limit=120)
    return _abbrev_value(arguments, limit=120)


def _diff_preview(output: str) -> Text | None:
    """A small colored unified-diff preview, or ``None`` if the output is not a diff."""
    lines = output.splitlines()
    diff_lines = [ln for ln in lines if ln[:1] in ("+", "-", "@")]
    if len(diff_lines) < 2:
        return None
    text = Text()
    for ln in diff_lines[:6]:
        if ln.startswith("+"):
            style = "diff.add"
        elif ln.startswith("-"):
            style = "diff.del"
        else:
            style = "diff.ctx"
        text.append(ln + "\n", style=style)
    text.rstrip()
    return text


def _abbrev_value(value: object, *, limit: int = 40) -> str:
    text = str(value)
    if "\n" in text:
        text = text.splitlines()[0]
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _first_line(output: object, *, limit: int = 80) -> str:
    text = str(output).strip()
    if not text:
        return ""
    first = text.splitlines()[0]
    return first if len(first) <= limit else first[: limit - 3] + "..."


def _humanize_tokens(n: int) -> str:
    if n < 1000:
        return f"{n} tok"
    return f"{n / 1000:.1f}k tok"
