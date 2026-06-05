"""Render-only layout primitives shared by the REPL and the stream renderer.

Common building blocks so every surface presents one consistent look: a 2-column
document margin, borderless aligned key/value tables, headings, the input prompt, and
the standard notice lines (info / warn / error). Pure presentation over a rich
Console — no agent logic lives here.
"""

from __future__ import annotations

from collections.abc import Sequence

from rich.console import Console, RenderableType
from rich.padding import Padding
from rich.table import Table
from rich.text import Text

from zakcode.cli._glyphs import resolve_glyphs


def margin(renderable: RenderableType, *, left: int = 2) -> Padding:
    """Wrap a renderable in the shared left document margin."""
    return Padding(renderable, (0, 0, 0, left))


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


def read_prompt(console: Console) -> str:
    """Print one leading blank line, then read a line of input at the prompt.

    Raises ``EOFError`` / ``KeyboardInterrupt`` exactly like ``console.input`` — the
    REPL catches those to exit cleanly.
    """
    console.print()
    return console.input(prompt_str(console))


def notice_info(console: Console, msg: str) -> None:
    """A dim, low-key notice (``· bye``, ``· Started a fresh session.``)."""
    g = resolve_glyphs(console)
    console.print()
    console.print(margin(Text.assemble((g["dot"] + " ", "notice.dim"), (msg, "notice.dim"))))


def notice_warn(console: Console, msg: str) -> None:
    """A visible (non-dim) warning notice, e.g. the interrupt line."""
    g = resolve_glyphs(console)
    console.print()
    console.print(margin(Text.assemble((g["bang"] + " ", "warn"), (msg, "warn"))))


def notice_error(console: Console, label: str, detail: str = "") -> None:
    """An error notice: a bold-red label line + an optional folded detail block."""
    g = resolve_glyphs(console)
    console.print()
    console.print(margin(Text.assemble((g["fail"] + " ", "err"), (label, "err"))))
    if detail:
        console.print(Padding(Text(detail, overflow="fold"), (0, 0, 0, 6)))
