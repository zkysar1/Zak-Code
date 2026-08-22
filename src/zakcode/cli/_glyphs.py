"""Terminal glyphs with cp1252-safe ASCII fallbacks (render-only support).

A Windows console on the legacy cp1252 code page cannot encode many glyphs a modern
TUI wants (``✦ ● └ │ ✓ ✗ …``); printing them raises ``UnicodeEncodeError`` or shows
mojibake. This module (a) best-effort upgrades the streams to UTF-8, (b) probes
whether the console can actually encode our glyph set, and (c) exposes one frozen
table resolved to unicode or ASCII accordingly. It is the single place a non-ASCII
glyph literal may appear — every other module pulls from :func:`resolve_glyphs`.

Every gutter-cell glyph (markers, ``elbow``, ``bar``, state marks) is exactly one
character in BOTH modes, so the col 2/4/6 column grid is identical under
``ZAKCODE_ASCII``. Multi-char fallbacks (``...``, ``--``) are inline-only — they
never occupy a gutter cell.
"""

from __future__ import annotations

import contextlib
import os
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rich.console import Console

#: Preferred glyphs.
_UNICODE: dict[str, str] = {
    "spark": "✦",
    "spark_soft": "✧",
    "marker": "●",
    "marker_tool": "●",
    "prompt": "›",
    "elbow": "└",
    "bar": "│",
    "ok": "✓",
    "fail": "✗",
    "bang": "!",
    "bullet": "•",
    "dot": "·",
    "dash": "—",
    "ellipsis": "…",
    "hline": "─",
    "add": "+",
    "del": "-",
    "todo_done": "✓",
    "todo_open": "○",
    "spin1": "·",
    "spin2": "✦",
    "spin3": "✶",
    "spin4": "✧",
    "corner_tl": "╭",
    "corner_tr": "╮",
    "corner_bl": "╰",
    "corner_br": "╯",
}
#: cp1252-safe fallbacks (used when the console cannot encode the unicode set).
#: ``marker_tool`` falls back to ``o`` (vs the prose marker's ``*``) so a
#: monochrome/NO_COLOR transcript still distinguishes prose from tools. The spinner
#: frames are always rendered via ``Text`` (never markup-parsed), so ``\\`` and
#: ``|`` are safe literals.
_ASCII: dict[str, str] = {
    "spark": "*",
    "spark_soft": "*",
    "marker": "*",
    "marker_tool": "o",
    "prompt": ">",
    "elbow": "\\",
    "bar": "|",
    "ok": "+",
    "fail": "x",
    "bang": "!",
    "bullet": "-",
    "dot": "-",
    "dash": "--",
    "ellipsis": "...",
    "hline": "-",
    "add": "+",
    "del": "-",
    "todo_done": "+",
    "todo_open": "o",
    "spin1": "-",
    "spin2": "\\",
    "spin3": "|",
    "spin4": "/",
    "corner_tl": "+",
    "corner_tr": "+",
    "corner_bl": "+",
    "corner_br": "+",
}

#: Every distinct non-ASCII glyph we might print — the encode probe target.
_PROBE = "".join(sorted({v for v in _UNICODE.values() if not v.isascii()}))


def enable_utf8() -> None:
    """Best-effort: reconfigure stdout/stderr to UTF-8 (modern Windows Terminal)."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            # best-effort; the encode probe in unicode_ok is the real gate
            with contextlib.suppress(Exception):
                reconfigure(encoding="utf-8")


def unicode_ok(console: Console) -> bool:
    """Whether ``console`` can encode our unicode glyphs (else ASCII fallback).

    ``ZAKCODE_ASCII`` forces ASCII regardless. Any encode failure -> ASCII.
    """
    if os.environ.get("ZAKCODE_ASCII"):
        return False
    file = getattr(console, "file", None)
    enc = getattr(file, "encoding", None) or getattr(console, "encoding", None) or ""
    try:
        _PROBE.encode(enc or "utf-8")
    except Exception:  # noqa: BLE001 — any encode failure -> ASCII
        return False
    return True


def resolve_glyphs(console: Console) -> dict[str, str]:
    """Return the glyph table resolved to unicode or ASCII for ``console``."""
    return dict(_UNICODE) if unicode_ok(console) else dict(_ASCII)
