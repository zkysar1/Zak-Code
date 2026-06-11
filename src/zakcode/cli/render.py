"""Streaming TUI renderer for the CLI chat (rendering only).

Consumes a stream of :class:`~zakcode.events.AgentEvent` objects and renders them with
``rich`` in the "minimal-gutter" look: structure comes from a thin colored marker in a
2-column document margin plus one-blank-line rhythm, with ``dim`` reserved strictly for
chrome so the assistant's own words and the salient tool target are the brightest things
on screen. No agent logic, no network, no provider imports — feed a fake async generator
into :meth:`StreamRenderer.render` over a ``Console`` backed by ``io.StringIO`` to test it.

Spatial grammar (shared with the bundled web client — see ``docs/UX.md``):

* a tool call and its result form **one block**: ``→ verb  target`` with a
  ``└ ✓ summary`` connector line directly beneath, previews indented under that;
* one blank line separates blocks (prose run / tool block / status), never lines
  within a block, so related output reads as a unit;
* every turn closes with a one-line footer rule (``── 2 iter · 15.3k tok · $0.02 ──``);
* while the loop is waiting (model thinking, tool running) a transient spinner shows
  what it is waiting **for** — real terminals only, suspended by :func:`suspend_live`
  whenever something else (the permission prompter) needs the bottom line.

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
from rich.status import Status
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

#: One live spinner per console (keyed by ``id``), so anything about to take over the
#: terminal — the permission prompter, an input read — can stop it via suspend_live.
_LIVE_STATUS: dict[int, Status] = {}


def suspend_live(console: Console) -> None:
    """Stop the live spinner attached to ``console`` (no-op when none is active).

    The spinner is a :class:`rich.status.Status` whose refresh thread repaints the
    bottom line; anything that reads input or prints a prompt on the same console
    must call this first or the spinner draws over it. The renderer restarts the
    spinner on its next event, so callers never need to resume it.
    """
    status = _LIVE_STATUS.pop(id(console), None)
    if status is not None:
        status.stop()


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
    closes. A tool call and its result render as one block — ``→ verb  target`` with a
    ``└ ✓ summary`` connector beneath (a unified-diff or shell-output preview under
    that). Usage is accumulated and summarized in a one-line footer rule printed on
    :class:`AgentDone`. While the loop is waiting a transient spinner names what it is
    waiting for ("thinking", the running verb) — only on a real terminal, so hermetic
    StringIO tests never see it. All glyphs resolve to cp1252-safe ASCII fallbacks
    when the console cannot encode them.
    """

    def __init__(
        self, console: Console | None = None, *, live_feedback: bool | None = None
    ) -> None:
        self.console = console if console is not None else Console()
        self._g = resolve_glyphs(self.console)
        self._text_buffer = ""
        self._usage = Usage()
        self._tool_names: dict[str, str] = {}
        self._assistant_marked = False
        #: id of the most recent tool call printed — its result attaches without
        #: repeating the verb; an out-of-order result names its tool instead.
        self._last_call_id: str | None = None
        #: True right after a tool/status block: the next prose run opens with a
        #: blank line so blocks keep the one-blank-line rhythm.
        self._block_gap = False
        #: True when the last printed row was blank — separator blanks coalesce
        #: through :meth:`_blank` so adjacent blocks never stack two empty lines.
        self._just_blank = False
        self._live = self.console.is_terminal if live_feedback is None else live_feedback

    def _blank(self) -> None:
        """Print one separating blank line, coalescing with one just printed."""
        if not self._just_blank:
            self.console.print()
            self._just_blank = True

    def _out(self, renderable: object) -> None:
        """Print a content row (and remember the bottom row is no longer blank)."""
        self.console.print(renderable)
        self._just_blank = False

    async def render(self, events: AsyncIterator[AgentEvent]) -> AgentDone | None:
        """Consume ``events``, render them, and return the final ``AgentDone`` (or None)."""
        done: AgentDone | None = None
        self._assistant_marked = False

        self._spin("thinking")
        try:
            async for event in events:
                if isinstance(event, AgentTextDelta):
                    self._unspin()
                    self._on_text_delta(event.text)
                elif isinstance(event, AgentToolCall):
                    self._unspin()
                    verb = self._on_tool_call(event)
                    self._spin(verb)
                elif isinstance(event, AgentToolResult):
                    self._unspin()
                    self._on_tool_result(event)
                    self._spin("thinking")
                elif isinstance(event, AgentStatus):
                    self._unspin()
                    self._on_status(event)
                    self._spin("thinking")
                elif isinstance(event, AgentUsage):
                    self._on_usage(event)  # no console output — leave the spinner be
                elif isinstance(event, AgentDone):
                    done = event
                    break
        finally:
            self._unspin()

        self._flush_remaining_text()
        if done is not None:
            self._print_footer(done)
        return done

    # -- live wait feedback ---------------------------------------------------

    def _spin(self, label: str) -> None:
        """Show a transient spinner naming what the loop is waiting for."""
        if not self._live:
            return
        suspend_live(self.console)
        status = self.console.status(
            Text(label + self._g["ellipsis"], style="status"), spinner="dots"
        )
        status.start()
        _LIVE_STATUS[id(self.console)] = status

    def _unspin(self) -> None:
        suspend_live(self.console)

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
                self._open_block_gap()
                marker = Text(self._g["dot"] + " ", style="assistant.marker")
                self._out(Padding(marker + _inline_md(line, self._g), (0, 0, 0, 2)))
            elif line.strip() == "":
                self._blank()  # coalesced: runs of blank prose lines render as one
                self._block_gap = False  # the prose's own blank already separates blocks
            else:
                self._open_block_gap()
                self._out(Padding(_inline_md(line, self._g), (0, 0, 0, 4)))

    def _open_block_gap(self) -> None:
        """Print the one blank line that separates a new block from a tool/status block."""
        if self._block_gap:
            self._blank()
            self._block_gap = False

    def _print_code(self, lang: str, body: str) -> None:
        if not body.strip():
            return
        self._block_gap = False  # the code block prints its own surrounding blanks
        syntax = Syntax(
            body,
            lang or "text",
            theme="ansi_dark",
            background_color="default",
            word_wrap=True,
        )
        self._blank()
        self._out(Padding(syntax, (0, 0, 0, 4)))
        self._blank()

    def _on_tool_call(self, event: AgentToolCall) -> str:
        verb, target = _format_tool_call(event.name, event.arguments)
        self._tool_names[event.id] = verb
        self._last_call_id = event.id
        self._block_gap = False
        # A prose run that resumes after this block is a NEW block: give it its own
        # `·` marker rather than rendering it as an orphaned continuation.
        self._assistant_marked = False
        self._blank()
        self._out(
            Padding(
                Text.assemble(
                    (self._g["arrow"] + " ", "tool.marker"),
                    (verb + "  ", "tool.verb"),
                    (target, "tool.target"),
                ),
                (0, 0, 0, 2),
            )
        )
        return verb

    def _result_head(self, name: str | None, state: str, summary: str) -> Padding:
        """The ``└ ✓ summary`` connector line that attaches a result to its call.

        ``name`` is only shown when the result did not immediately follow its own
        call line (out-of-order results), so the common adjacent pair stays terse.
        """
        glyph = self._g["fail"] if state == "err" else self._g["ok"]
        line = Text.assemble(
            (self._g["branch"] + " ", "tool.connector"),
            (glyph + " ", state),
        )
        if name:
            line.append_text(Text(name + " " + self._g["dot"] + " ", style="tool.marker"))
        line.append_text(Text(summary, style="notice.dim"))
        return Padding(line, (0, 0, 0, 2))

    def _on_tool_result(self, event: AgentToolResult) -> None:
        name = self._tool_names.get(event.tool_use_id, "tool")
        attached = event.tool_use_id == self._last_call_id
        head_name = None if attached else name
        self._last_call_id = None
        self._block_gap = True  # whatever prints next starts a new block
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
            self._out(self._result_head(head_name, state, summary))
            preview = lines[:_RUN_OUTPUT_LINES]
            if preview:
                self._out(Padding(Text("\n".join(preview), style="notice.dim"), (0, 0, 0, 6)))
            if n > _RUN_OUTPUT_LINES:
                more = n - _RUN_OUTPUT_LINES
                tail = f"{self._g['ellipsis']} (+{more} more line{'' if more == 1 else 's'})"
                self._out(Padding(Text(tail, style="notice.dim"), (0, 0, 0, 6)))
            return

        diff = _diff_preview(output, glyphs=self._g)
        if diff is not None:
            # A diff result summarizes as change counts; the preview shows the lines.
            adds = sum(
                1 for ln in output.splitlines() if ln.startswith("+") and not ln.startswith("+++")
            )
            dels = sum(
                1 for ln in output.splitlines() if ln.startswith("-") and not ln.startswith("---")
            )
            summary = f"+{adds} -{dels}"
        else:
            summary = _first_line(output) or ("error" if event.is_error else "ok")
        self._out(self._result_head(head_name, state, summary))
        if diff is not None:
            self._out(Padding(diff, (0, 0, 0, 6)))
        elif event.is_error:
            extra = "\n".join(output.splitlines()[1:6]).rstrip()
            if extra:
                self._out(Padding(Text(extra, overflow="fold"), (0, 0, 0, 6)))

    def _on_status(self, event: AgentStatus) -> None:
        self._assistant_marked = False  # prose resuming after a notice is a new block
        self._open_block_gap()
        self._out(
            Padding(
                Text.assemble((self._g["status"] + " ", "status.glyph"), (event.message, "status")),
                (0, 0, 0, 2),
            )
        )
        self._block_gap = True

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
        self._blank()
        if done.stop_reason != "completed":
            # A turn that stopped early (stuck / doom_loop / max_iterations / ...) must
            # not end silently — the dim footer alone is easy to read past.
            self._out(
                Padding(
                    Text.assemble(
                        (g["bang"] + " ", "warn"),
                        (f"stopped early: {done.stop_reason}", "warn"),
                    ),
                    (0, 0, 0, 2),
                )
            )
        stats = (
            f"{done.iterations} iter "
            f"{g['dot']} {_humanize_tokens(usage.total_tokens)} "
            f"{g['dot']} ${usage.cost_usd:.4f}"
        )
        rule = Rule(
            Text(stats, style="footer"),
            characters=g["hline"],
            style="rule.line",
            align="left",
        )
        self.console.print(Padding(rule, (0, 0, 0, 2)))


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


#: Max diff lines shown in a result preview (then "… (+N more)").
_DIFF_PREVIEW_LINES = 6


def _diff_preview(output: str, *, glyphs: dict[str, str] | None = None) -> Text | None:
    """A small colored unified-diff preview, or ``None`` if the output is not a diff.

    Gated on a real unified-diff SIGNATURE — an ``@@`` hunk header, or a ``--- ``/``+++ ``
    file-header pair — before colorizing. Without the gate, ordinary tool output whose
    lines merely begin with ``-``/``+`` (markdown bullets, an ``ls -l`` listing, a file
    read) was mis-painted as a red/green diff and truncated; the signature gate prevents
    that false positive. Truncation is always marked (``… (+N more)``), never silent.
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
    text = Text()
    for ln in diff_lines[:_DIFF_PREVIEW_LINES]:
        if ln.startswith("+"):
            style = "diff.add"
        elif ln.startswith("-"):
            style = "diff.del"
        else:
            style = "diff.ctx"
        text.append(ln + "\n", style=style)
    if len(diff_lines) > _DIFF_PREVIEW_LINES:
        more = len(diff_lines) - _DIFF_PREVIEW_LINES
        ellipsis = (glyphs or {}).get("ellipsis", "...")
        text.append(f"{ellipsis} (+{more} more line{'' if more == 1 else 's'})", style="notice.dim")
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
