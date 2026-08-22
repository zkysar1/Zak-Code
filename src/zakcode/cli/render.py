"""Streaming TUI renderer for the CLI chat (rendering only).

Consumes a stream of :class:`~zakcode.events.AgentEvent` objects and renders them
with ``rich`` in the Zak look: a two-level marker grammar (``●`` block / ``└``
receipt) on a fixed column grid, a continuous ``│`` rail binding every result body
to its block (red under a failed tool), synthesized receipts with durations from an
injectable monotonic clock, and a state-colored turn receipt::

    ● Read(src/config.py)
      └ 134 lines · 0.1s

    ● done · 3 iterations · 15.3k tokens · $0.023 · 41.2s

No agent logic, no network, no provider imports — feed a fake async generator into
:meth:`StreamRenderer.render` over a ``Console`` backed by ``io.StringIO`` to test
it (pass a fake ``clock`` for exact durations).

Spacing is a small state machine (see ``docs/UX.md``): every block opens with
:meth:`StreamRenderer._gap` (one blank, coalesced) and never prints a trailing
blank; a tool result directly after its own call line skips the gap so the group
stays contiguous; the rendered transcript never contains two consecutive blank
lines. The wait line lives in the REPL layer, never here.

Two subtle pieces:

* *Fence-safe flushing* (:func:`split_on_safe_boundary`): streamed text is released
  only up to the last newline that is **not** inside an open code fence, so a
  half-written fenced block never renders.
* *Segment routing* (:func:`iter_segments`): a fence-safe ``emit`` is split into
  ordered prose / code runs; prose streams line-at-a-time in the margin, while a
  completed fenced block is rendered once via ``rich.Syntax``, never half-drawn.
"""

from __future__ import annotations

import os
import time
from collections.abc import AsyncIterator, Callable
from typing import cast

from rich.console import Console
from rich.padding import Padding
from rich.syntax import Syntax
from rich.text import Text

from zakcode.cli._glyphs import _ASCII, _UNICODE, resolve_glyphs
from zakcode.cli._layout import block, rail
from zakcode.events import (
    AgentDone,
    AgentEvent,
    AgentStatus,
    AgentTextDelta,
    AgentThinkingDelta,
    AgentToolCall,
    AgentToolResult,
    AgentUsage,
)
from zakcode.usage import Usage

_FENCE = "```"


#: Run/Fetch output above this many lines is truncated head+tail (terminal
#: summaries like ``42 passed`` always survive).
_RUN_CAP = 12
_RUN_HEAD = 6
_RUN_TAIL = 4

#: Max diff lines shown in a result preview (then a dim-italic hidden-count row).
_DIFF_PREVIEW_LINES = 12

#: Max preview rows for Search / Glob results (then ``… +N more``).
_LIST_PREVIEW_LINES = 5

#: Max error-detail rail rows under a failed tool's receipt.
_ERROR_DETAIL_LINES = 8

#: Tool args middle-truncate at this many chars; paths keep their tail instead.
_ARG_LIMIT = 64

#: Map a tool name to the bold display name on the call line; unknown tools
#: title-case their parts (``some_tool`` -> ``SomeTool``).
_DISPLAY_NAME = {
    "read_file": "Read",
    "write_file": "Write",
    "edit_file": "Edit",
    "list_dir": "List",
    "glob": "Glob",
    "grep": "Search",
    "bash": "Run",
    "powershell": "Run",
    "web_fetch": "Fetch",
    "web_search": "WebSearch",
    "todo_write": "Todo",
    "update_plan": "Todo",
    "task": "Task",
}

#: Footer label per ``done.stop_reason`` (``{dash}`` resolves per console so the
#: outcome survives ASCII mode); unknown reasons print ``reason.replace("_", " ")``.
_STOP_LABEL = {
    "completed": "done",
    "max_iterations": "stopped early {dash} max iterations",
    "provider_error": "provider error",
    "doom_loop": "stopped {dash} repeating itself",
    "stuck": "stopped {dash} no progress",
    "recipe_stalled": "stopped {dash} recipe stalled",
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


def _env_glyphs() -> dict[str, str]:
    """Console-free glyph fallback for the exported helpers.

    The permission prompter calls :func:`display_call` without a renderer in hand;
    honor the forced-ASCII switch there. The encoding-probed fallback needs a real
    console and arrives via the ``glyphs`` argument instead.
    """
    return dict(_ASCII) if os.environ.get("ZAKCODE_ASCII") else dict(_UNICODE)


def _display_name(name: str) -> str:
    return _DISPLAY_NAME.get(name) or "".join(part.title() for part in name.split("_"))


def display_call(name: str, arguments: object, *, glyphs: dict[str, str] | None = None) -> Text:
    """The call headline: bold display name + dim parens + condensed argument.

    Shared by the renderer's tool-call line and the permission panel so the two can
    never drift. Commands get a dim ``$ `` prefix; args middle-truncate at 64 chars;
    paths truncate from the left so the filename survives.
    """
    g = glyphs or _env_glyphs()
    is_command, args = _condense_args(arguments, g)
    out = Text()
    out.append(_display_name(name), style="tool.name")
    out.append("(", style="tool.paren")
    if is_command:
        out.append("$ ", style="tool.paren")
    if args:
        out.append(args, style="tool.args")
    out.append(")", style="tool.paren")
    return out


def _condense_args(arguments: object, g: dict[str, str]) -> tuple[bool, str]:
    """Condense a tool's arguments to one display string (True marks a command)."""
    ellipsis = g["ellipsis"]
    if isinstance(arguments, dict):
        command = arguments.get("command")
        if isinstance(command, str) and command:
            return True, _squeeze_middle(_one_line(command), _ARG_LIMIT, ellipsis)
        pattern = arguments.get("pattern")
        if isinstance(pattern, str) and pattern:
            # Search/Glob calls read "pattern in scope" when a scope is given.
            scope = ""
            for key in ("glob", "path"):
                value = arguments.get(key)
                if isinstance(value, str) and value:
                    scope = value
                    break
            text = f"{pattern} in {scope}" if scope else pattern
            return False, _squeeze_middle(_one_line(text), _ARG_LIMIT, ellipsis)
        for key in ("path", "file_path"):
            value = arguments.get(key)
            if isinstance(value, str) and value:
                return False, _squeeze_path(value, _ARG_LIMIT, ellipsis)
        for key in ("query", "url", "name"):
            value = arguments.get(key)
            if isinstance(value, str) and value:
                return False, _squeeze_middle(_one_line(value), _ARG_LIMIT, ellipsis)
        for value in arguments.values():
            if isinstance(value, str) and value:
                return False, _squeeze_middle(_one_line(value), _ARG_LIMIT, ellipsis)
        return False, ""
    return False, _squeeze_middle(_one_line(str(arguments)), _ARG_LIMIT, ellipsis)


class StreamRenderer:
    """Render an :class:`~zakcode.events.AgentEvent` stream to a ``rich`` console.

    Prose streams line-at-a-time with one ``●`` per prose group (continuations keep
    an empty gutter); a fenced code block renders once via ``rich.Syntax`` when it
    closes. A tool call and its result form one contiguous block — the bold
    headline, a ``└`` receipt synthesized per tool (with a duration from the
    injectable ``clock``), and rail-bound preview rows. The turn ends in a
    state-colored ``●`` receipt whose body begins with the outcome word. All glyphs
    resolve to cp1252-safe ASCII fallbacks when the console cannot encode them.
    """

    def __init__(
        self, console: Console | None = None, clock: Callable[[], float] | None = None
    ) -> None:
        self.console = console if console is not None else Console()
        self._g = resolve_glyphs(self.console)
        self._clock: Callable[[], float] = clock if clock is not None else time.monotonic
        self._text_buffer = ""
        self._usage = Usage()
        #: tool_use_id -> display name (receipts are synthesized per display name).
        self._tool_names: dict[str, str] = {}
        #: tool_use_id -> clock() at the call line (keyed by id — parallel calls exist).
        self._tool_started: dict[str, float] = {}
        #: id of the most recently printed call line — only its own result may bind
        #: contiguously beneath it; any other result detaches and names its tool.
        self._last_call_id: str | None = None
        self._turn_start = 0.0
        self._assistant_marked = False
        #: has this turn already announced that the model is reasoning? A reasoning
        #: model emits hundreds of thinking deltas; the CLI announces the PHASE once
        #: and then stays quiet, rather than streaming the model's scratchpad into
        #: the transcript (the web client renders it incrementally in its own region).
        self._thinking_marked = False
        #: was the last physical line written blank (gaps coalesce through _gap).
        self._at_blank = False
        #: None | "prose" | "code" | "tool_call" | "tool" | "status".
        self._last_block: str | None = None
        #: The most recent turn's terminal event (set by render()); the REPL reads it for
        #: post-turn signals like the zakpick advisory. None before the first turn.
        self.last_done: AgentDone | None = None

    async def render(self, events: AsyncIterator[AgentEvent]) -> AgentDone | None:
        """Consume ``events``, render them, and return the final ``AgentDone`` (or None)."""
        done: AgentDone | None = None
        self._turn_start = self._clock()
        self._at_blank = False
        self._last_block = None
        self._assistant_marked = False
        self._thinking_marked = False

        async for event in events:
            if isinstance(event, AgentTextDelta):
                self._on_text_delta(event.text)
            elif isinstance(event, AgentThinkingDelta):
                self._on_thinking_delta()
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
        # Stash the terminal event so the REPL can read post-turn signals (the zakpick routing
        # fields for the advisory) without threading it back through _drive_stream's bool return.
        # The attribute doc lives at the __init__ declaration.
        self.last_done = done
        return done

    # -- the blank-line state machine -----------------------------------------

    def _gap(self) -> None:
        """Print one separating blank line iff the last line written was not blank."""
        if not self._at_blank:
            self.console.print()
            self._at_blank = True

    def _out(self, renderable: object) -> None:
        """Print a content row (the bottom line is no longer blank)."""
        self.console.print(renderable)
        self._at_blank = False

    # -- per-event handlers ----------------------------------------------------

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

    def _print_prose(self, prose: str) -> None:
        for line in prose.split("\n"):
            if line.strip() == "":
                # model blank lines never print directly; runs collapse to one gap
                self._gap()
                continue
            if not self._assistant_marked:
                self._gap()
                self._out(
                    block(
                        self.console,
                        _inline_md(line, self._g),
                        marker=self._g["marker"],
                        marker_style="assistant.marker",
                    )
                )
                self._assistant_marked = True
            else:
                if line.lstrip().startswith("#"):
                    self._gap()  # headings force a blank line above
                self._out(block(self.console, _inline_md(line, self._g)))
            self._last_block = "prose"

    def _print_code(self, lang: str, body: str) -> None:
        # Assistant code blocks are the assistant's voice, not a tool region: no
        # rail — a gap above, a dim language tag at col 4, the body at col 6.
        if not body.strip():
            return
        self._gap()
        if lang:
            self._out(
                block(
                    self.console,
                    Text(lang, style="code.tag"),
                    marker=self._g["dot"],
                    marker_style="code.tag",
                    indent=4,
                )
            )
        syntax = Syntax(
            body,
            lang or "text",
            theme="ansi_dark",
            background_color="default",
            word_wrap=True,
        )
        self._out(Padding(syntax, (0, 0, 0, 6)))
        self._last_block = "code"
        self._assistant_marked = False

    def _on_tool_call(self, event: AgentToolCall) -> None:
        # A pre-tool sentence without a trailing newline is still buffered; flush it
        # so prose always precedes its tool block (the stream's text content block is
        # complete once a tool call is emitted — no future delta can extend it).
        self._flush_remaining_text()
        self._gap()
        self._tool_names[event.id] = _display_name(event.name)
        self._tool_started[event.id] = self._clock()
        self._last_call_id = event.id
        self._out(
            block(
                self.console,
                display_call(event.name, event.arguments, glyphs=self._g),
                marker=self._g["marker_tool"],
                marker_style="tool.marker",
            )
        )
        self._last_block = "tool_call"
        self._assistant_marked = False

    def _on_tool_result(self, event: AgentToolResult) -> None:
        # Group binding: a result stays contiguous only under its OWN call line —
        # an interleaved/out-of-order result detaches and names its tool instead.
        attached = self._last_block == "tool_call" and event.tool_use_id == self._last_call_id
        if not attached:
            self._gap()
        self._last_call_id = None
        name = self._tool_names.pop(event.tool_use_id, "")
        started = self._tool_started.pop(event.tool_use_id, None)
        duration = "" if started is None else _fmt_duration(self._clock() - started)
        output = str(event.output)
        lines = output.splitlines()

        summary, rows = self._synthesize_result(name, lines, is_error=event.is_error)
        if not attached:
            head = Text.assemble(
                ((name or "Tool") + " ", "result.summary"),
                (self._g["dot"] + " ", "result.summary"),
            )
            head.append_text(summary)
            summary = head
        if duration:
            summary.append(f" {self._g['dot']} {duration}", style="result.summary")
        self._out(
            block(
                self.console,
                summary,
                marker=self._g["elbow"],
                marker_style="result.connector",
                indent=4,
            )
        )
        if rows:
            bar_style = "tool.bar.err" if event.is_error else "tool.bar"
            self._out(rail(self.console, rows, bar_style=bar_style))
        self._last_block = "tool"

    def _synthesize_result(
        self, name: str, lines: list[str], *, is_error: bool
    ) -> tuple[Text, list[Text]]:
        """The ``└`` receipt summary + the rail rows beneath it, per display name."""
        g = self._g
        n = len(lines)

        if is_error:
            # Errors restructure, not just recolor: ✗ first line on the receipt,
            # detail rows at full fg behind a (red) rail. Truncation always marked.
            first = lines[0] if lines else "error"
            summary = Text.assemble((g["fail"] + " ", "err"), (first, "err"))
            rows = [Text(ln, style="err.body") for ln in lines[1 : 1 + _ERROR_DETAIL_LINES]]
            hidden = len(lines) - 1 - _ERROR_DETAIL_LINES
            if hidden > 0:
                rows.append(
                    Text(f"{g['ellipsis']} +{_plural(hidden, 'line')}", style="result.more")
                )
            return summary, rows

        if name in ("Run", "Fetch"):
            label = _plural(n, "line") if n else "no output"
            if n > _RUN_CAP:
                hidden = n - _RUN_HEAD - _RUN_TAIL
                rows = [Text(ln, style="result.output") for ln in lines[:_RUN_HEAD]]
                rows.append(
                    Text(
                        f"{g['ellipsis']} +{_plural(hidden, 'line')} {g['ellipsis']}",
                        style="result.more",
                    )
                )
                rows += [Text(ln, style="result.output") for ln in lines[-_RUN_TAIL:]]
            else:
                rows = [Text(ln, style="result.output") for ln in lines]
            return Text(label, style="result.summary"), rows

        if name == "Search":
            summary = Text(_plural(n, "match", "matches"), style="result.summary")
            rows = [Text(ln, style="result.output") for ln in lines[:_LIST_PREVIEW_LINES]]
            if n > _LIST_PREVIEW_LINES:
                more = n - _LIST_PREVIEW_LINES
                rows.append(Text(f"{g['ellipsis']} +{more} more", style="result.more"))
            return summary, rows

        if name == "Glob":
            summary = Text(_plural(n, "file"), style="result.summary")
            rows = [Text(ln, style="result.output") for ln in lines[:_LIST_PREVIEW_LINES]]
            if n > _LIST_PREVIEW_LINES:
                more = n - _LIST_PREVIEW_LINES
                rows.append(Text(f"{g['ellipsis']} +{more} more", style="result.more"))
            return summary, rows

        if name == "Edit":
            diff_rows = _diff_preview("\n".join(lines), glyphs=g)
            if diff_rows is not None:
                adds = sum(1 for ln in lines if ln.startswith("+") and not ln.startswith("+++"))
                dels = sum(1 for ln in lines if ln.startswith("-") and not ln.startswith("---"))
                return Text(f"+{adds} -{dels}", style="result.summary"), diff_rows
            return Text(_plural(n, "line"), style="result.summary"), []

        if name == "Write":
            return Text("written", style="result.summary"), []

        if name == "Todo":
            items = [ln for ln in lines if ln.strip()]
            rows = [self._todo_row(ln) for ln in lines]
            return Text(_plural(len(items), "item"), style="result.summary"), rows

        # Read / List / unknown tools: a count, no preview.
        return Text(_plural(n, "line"), style="result.summary"), []

    def _todo_row(self, line: str) -> Text:
        # Format-tolerant: only "[x] " / "[ ] " prefixes glyph-map; others pass through.
        if line.startswith("[x] "):
            return Text.assemble(
                (self._g["todo_done"] + " ", "todo.done"), (line[4:], "result.output")
            )
        if line.startswith("[ ] "):
            return Text.assemble(
                (self._g["todo_open"] + " ", "todo.open"), (line[4:], "result.output")
            )
        return Text(line, style="result.output")

    def _on_thinking_delta(self) -> None:
        """Announce the reasoning PHASE once per turn, then stay silent.

        Takes no text on purpose. A reasoning model emits hundreds of thinking
        deltas over a phase that can run for minutes, and piping that scratchpad
        into a terminal transcript would bury the answer it precedes — and blur the
        one line this feature exists to keep sharp, that reasoning is not assistant
        output. What the CLI user actually needs is the difference between "hung"
        and "working", which one marker supplies; the web client, which has its own
        region to put it in, renders the content incrementally.
        """
        if self._thinking_marked:
            return
        self._thinking_marked = True
        self._flush_remaining_text()
        self._gap()
        self._out(
            block(
                self.console,
                Text("thinking…", style="status"),
                marker=self._g["dot"],
                marker_style="status",
            )
        )
        self._last_block = "status"
        self._assistant_marked = False

    def _on_status(self, event: AgentStatus) -> None:
        self._flush_remaining_text()  # buffered prose precedes the notice (see _on_tool_call)
        self._gap()
        self._out(
            block(
                self.console,
                Text(event.message, style="status"),
                marker=self._g["dot"],
                marker_style="status",
            )
        )
        self._last_block = "status"
        self._assistant_marked = False

    def _on_usage(self, event: AgentUsage) -> None:
        self._usage = self._usage + event.usage

    # -- helpers ----------------------------------------------------------------

    def _flush_remaining_text(self) -> None:
        if self._text_buffer:
            self._render_stream_text(self._text_buffer)
            self._text_buffer = ""

    def _print_footer(self, done: AgentDone) -> None:
        usage = done.usage if done.usage.total_tokens else self._usage
        g = self._g
        label, marker_style = self._stop_label(done)
        sep = f" {g['dot']} "
        body = Text(
            sep.join(
                (
                    label,
                    f"{done.iterations} iterations",
                    _humanize_tokens(usage.total_tokens),
                    _fmt_cost(usage.cost_usd),
                    _fmt_duration(self._clock() - self._turn_start),
                )
            ),
            style="footer",
        )
        self._gap()
        self._out(block(self.console, body, marker=g["marker"], marker_style=marker_style))

    def _stop_label(self, done: AgentDone) -> tuple[str, str]:
        """The footer's leading outcome word(s) + the marker's state style."""
        reason = done.stop_reason
        template = _STOP_LABEL.get(reason)
        label = (
            template.format(dash=self._g["dash"])
            if template is not None
            else reason.replace("_", " ")
        )
        if reason == "completed":
            return label, "ok"
        if reason == "provider_error":
            if done.error:
                label += f" {self._g['dash']} {done.error.splitlines()[0]}"
            return label, "err"
        return label, "warn"


def _inline_md(line: str, glyphs: dict[str, str]) -> Text:
    """Render one prose line with lightweight inline markdown.

    Handles list bullets (``- ``/``* `` -> the bullet glyph), ATX headings (``##`` ->
    ``md.h``), ``**bold**`` and ``` `code` ```. Built by hand with ``Text.append`` so
    model text can never inject rich console markup (the markup=False safety we rely
    on).
    """
    stripped = line.lstrip(" ")
    lead = line[: len(line) - len(stripped)]
    out = Text(lead)
    if stripped[:2] in ("- ", "* "):
        out.append(glyphs["bullet"] + " ", style="md.bullet")
        out.append_text(_inline_spans(stripped[2:]))
        return out
    if stripped.startswith("#"):
        body = _inline_spans(stripped.lstrip("#").lstrip())
        body.stylize("md.h")
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
                out.append(text[i + 1 : end], style="md.code")
                i = end + 1
                continue
        out.append(text[i])
        i += 1
    return out


def _diff_preview(output: str, *, glyphs: dict[str, str] | None = None) -> list[Text] | None:
    """Rail rows for a colored unified-diff preview, or ``None`` if not a diff.

    Gated on a real unified-diff SIGNATURE — an ``@@`` hunk header, or a ``--- ``/``+++ ``
    file-header pair — before colorizing. Without the gate, ordinary tool output whose
    lines merely begin with ``-``/``+`` (markdown bullets, an ``ls -l`` listing, a file
    read) was mis-painted as a red/green diff and truncated; the signature gate prevents
    that false positive. Truncation is always marked (``… +N lines``), never silent.
    """
    lines = output.splitlines()
    has_hunk = any(ln.startswith("@@") for ln in lines)
    has_file_headers = any(ln.startswith("--- ") for ln in lines) and any(
        ln.startswith("+++ ") for ln in lines
    )
    if not (has_hunk or has_file_headers):
        return None
    diff_lines = [ln for ln in lines if ln[:1] in ("+", "-", "@")]
    if len(diff_lines) < 2:
        return None
    g = glyphs or _env_glyphs()
    rows: list[Text] = []
    for ln in lines[:_DIFF_PREVIEW_LINES]:
        if ln.startswith(("@@", "---", "+++")):
            style = "diff.meta"
        elif ln.startswith("+"):
            style = "diff.add"
        elif ln.startswith("-"):
            style = "diff.del"
        else:
            style = "diff.ctx"
        # Style as a SPAN, not the Text's base style: the rail grid left-justifies
        # rows, and base-styled pad spaces would smear the add/del background band
        # past the text — the spec pins bands to text extent only.
        row = Text()
        row.append(ln, style=style)
        rows.append(row)
    if len(lines) > _DIFF_PREVIEW_LINES:
        hidden = len(lines) - _DIFF_PREVIEW_LINES
        rows.append(Text(f"{g['ellipsis']} +{_plural(hidden, 'line')}", style="result.more"))
    return rows


def _one_line(text: str) -> str:
    return text.splitlines()[0] if "\n" in text else text


def _squeeze_middle(text: str, limit: int, ellipsis: str) -> str:
    """Middle-truncate: queries/commands keep their head AND their tail."""
    if len(text) <= limit:
        return text
    head = (limit - 1) // 2
    tail = limit - 1 - head
    return text[:head] + ellipsis + text[-tail:]


def _squeeze_path(path: str, limit: int, ellipsis: str) -> str:
    """Left-truncate with a leading ellipsis so the filename survives."""
    if len(path) <= limit:
        return path
    return ellipsis + path[-(limit - 1) :]


def _plural(n: int, noun: str, plural: str | None = None) -> str:
    return f"{n} {noun if n == 1 else (plural or noun + 's')}"


def _fmt_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    return f"{int(seconds // 60)}m {int(seconds % 60)}s"


def _fmt_cost(cost: float) -> str:
    # :.4f with trailing zeros stripped down to a minimum of 2 decimals:
    # 0.0230 -> $0.023, 0.0004 -> $0.0004, 0.0 -> $0.00, 1.5 -> $1.50.
    whole, frac = f"{cost:.4f}".split(".")
    frac = frac.rstrip("0")
    if len(frac) < 2:
        frac = frac.ljust(2, "0")
    return f"${whole}.{frac}"


def _humanize_tokens(n: int) -> str:
    if n < 1000:
        return f"{n} tokens"
    return f"{n / 1000:.1f}k tokens"
