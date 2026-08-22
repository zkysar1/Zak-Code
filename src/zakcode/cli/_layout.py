"""Render-only layout primitives shared by the REPL and the stream renderer.

Common building blocks so every surface presents one consistent look: the
hanging-indent :func:`block` (the column grid's single primitive), the :func:`rail`
that binds result bodies to their tool block, the ceremonial :func:`panel` (welcome
box, permission panel — the only two boxes that ever exist), borderless aligned
key/value tables, headings, the input prompt, and the standard notice lines
(info / warn / error). Pure presentation over a rich Console — no agent logic
lives here.

Column grid: cols 0–1 are the document margin; block markers sit at col 2 with
their body at col 4; receipts (``└``) and rail rows (``│``) sit at col 4 with their
body at col 6. Wrapped text always lands under the body column, never under the
marker — the ragged left edge is structurally impossible.
"""

from __future__ import annotations

import os
from collections.abc import Sequence

from rich.box import ROUNDED
from rich.console import Console, RenderableType
from rich.padding import Padding
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from zakcode.cli._glyphs import resolve_glyphs


def margin(renderable: RenderableType, *, left: int = 2) -> Padding:
    """Wrap a renderable in the shared left document margin."""
    return Padding(renderable, (0, 0, 0, left))


def block(
    console: Console,
    body: RenderableType,
    *,
    marker: str = "",
    marker_style: str = "",
    indent: int = 2,
) -> Padding:
    """The hanging-indent primitive: a 2-wide gutter cell + a body cell.

    An empty ``marker`` renders an empty gutter — continuation lines of a prose
    group keep identical column math without repeating the group's marker.
    ``console`` is part of the shared primitive signature (its glyph-resolving
    siblings need it); the grid itself is console-independent.
    """
    grid = Table.grid(padding=0)
    grid.add_column(width=2)
    grid.add_column(overflow="fold")
    grid.add_row(Text(marker, style=marker_style) if marker else Text(""), body)
    return margin(grid, left=indent)


def rail(
    console: Console,
    lines: Sequence[Text | str],
    *,
    bar_style: str = "tool.bar",
) -> Padding:
    """A result region: every row is the ``│`` glyph at col 4 + its line at col 6.

    Blank source lines render as a bare-rail row, keeping the region continuous.
    Wrapped continuations of a long line hang under the body column (the rail is
    not repeated mid-wrap — previews are line-shaped).
    """
    g = resolve_glyphs(console)
    grid = Table.grid(padding=0)
    grid.add_column(width=2)
    grid.add_column(overflow="fold")
    for line in lines:
        body = line if isinstance(line, Text) else Text(line)
        grid.add_row(Text(g["bar"], style=bar_style), body)
    return margin(grid, left=4)


def panel(
    console: Console,
    title: str,
    body: RenderableType,
    *,
    border_style: str,
) -> Padding:
    """One of the two ceremonial boxes (welcome, permission), indented to col 2.

    Width floor 24 keeps degenerate consoles renderable; cap 60 keeps the box an
    object, not a region.
    """
    width = max(24, min(console.width - 4, 60))
    return margin(
        Panel(
            body,
            box=ROUNDED,
            title=title or None,
            title_align="left",
            border_style=border_style,
            width=width,
            padding=(1, 2),
        )
    )


def kv_table(rows: Sequence[tuple[str, str]], *, label_style: str = "banner.label") -> Table:
    """A borderless, aligned key/value table (dim labels, bright values)."""
    table = Table(show_header=False, box=None, padding=(0, 2), pad_edge=False)
    table.add_column(style=label_style, min_width=10, no_wrap=True)
    table.add_column(style="arg.value", overflow="fold")
    for key, value in rows:
        table.add_row(key, value)
    return table


def heading(text: str) -> Text:
    """A section heading in the banner accent style."""
    return Text(text, style="banner.title")


def prompt_str(console: Console) -> str:
    """The input-prompt string: 2-col margin + accent chevron + trailing space."""
    g = resolve_glyphs(console)
    return f"  [prompt.marker]{g['prompt']}[/prompt.marker] "


def _frame_width(console: Console) -> int:
    """Input-frame width: console width minus the 2-col margin, capped for readability."""
    return max(24, min(console.width - 4, 100))


_FRAME_OFF = {"0", "false", "off", "no", "hidden"}


def input_frame_enabled() -> bool:
    """Whether :func:`read_prompt` draws its labeled frame (default: yes).

    An embedding cockpit that provides its own input affordance — e.g. a tmux
    pane feeding this session via ``send-keys`` — sets ``ZAKCODE_INPUT_FRAME=off``
    so the screen offers exactly ONE labeled place to type (2026-08-22 operator
    feedback: zakcode's frame plus the cockpit's input box read as two competing
    message boxes). Only the chrome is suppressed; the read is unchanged, so
    typed or injected text still echoes on the tty. Any value outside the off
    set (unset included) keeps the frame.
    """
    return os.environ.get("ZAKCODE_INPUT_FRAME", "").strip().lower() not in _FRAME_OFF


def read_prompt(console: Console) -> str:
    """Print the turn seam, then read a line inside a visible input frame.

    The frame draws in three beats around ``console.input`` — top border with a
    ``your message`` label, a bordered prompt line, and a bottom border printed on
    Enter — so the reply point is findable on a busy transcript. A bare chevron on
    a quiet line was routinely missed by humans attaching mid-session (2026-08-22
    operator feedback). The middle line stays open on the right because the cursor
    owns it. Raises ``EOFError`` / ``KeyboardInterrupt`` exactly like
    ``console.input``; the bottom border still closes on interrupt (``finally``)
    so the frame never dangles open.

    When :func:`input_frame_enabled` is off, the read happens with no chrome at
    all — no border, no label, no chevron — after the same turn-seam blank line.
    """
    if not input_frame_enabled():
        console.print()
        return console.input("")
    g = resolve_glyphs(console)
    w = _frame_width(console)
    label = " your message "
    top = Text("  ")
    top.append(g["corner_tl"] + g["hline"], style="notice.dim")
    top.append(label, style="banner.label")
    top.append(g["hline"] * max(1, w - 4 - len(label)) + g["corner_tr"], style="notice.dim")
    bottom = Text("  ")
    bottom.append(g["corner_bl"] + g["hline"] * (w - 2) + g["corner_br"], style="notice.dim")
    console.print()
    console.print(top)
    try:
        return console.input(
            f"  [notice.dim]{g['bar']}[/notice.dim] [prompt.marker]{g['prompt']}[/prompt.marker] "
        )
    finally:
        console.print(bottom)


def notice_info(console: Console, msg: str) -> None:
    """A dim, low-key notice (``· session closed — goodbye``)."""
    g = resolve_glyphs(console)
    console.print()
    console.print(margin(Text.assemble((g["dot"] + " ", "notice.dim"), (msg, "notice.dim"))))


def notice_warn(console: Console, msg: str) -> None:
    """A visible (non-dim) warning notice — the REPL Ctrl-C interrupt line."""
    g = resolve_glyphs(console)
    console.print()
    console.print(block(console, Text(msg, style="warn"), marker=g["bang"], marker_style="warn"))


def notice_error(console: Console, label: str, detail: str = "") -> None:
    """An error notice: ``✗ label`` + detail lines behind a red rail, full fg."""
    g = resolve_glyphs(console)
    console.print()
    console.print(block(console, Text(label, style="err"), marker=g["fail"], marker_style="err"))
    if detail:
        rows = [Text(line, style="err.body", overflow="fold") for line in detail.splitlines()]
        console.print(rail(console, rows, bar_style="tool.bar.err"))
