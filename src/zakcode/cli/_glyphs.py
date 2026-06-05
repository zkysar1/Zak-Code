"""Terminal glyphs with cp1252-safe ASCII fallbacks (render-only support).

A Windows console on the legacy cp1252 code page cannot encode many glyphs a modern
TUI wants (``→ ✓ ✗ · …``); printing them raises ``UnicodeEncodeError`` or shows
mojibake. This module (a) best-effort upgrades the streams to UTF-8, (b) probes
whether the console can actually encode our glyph set, and (c) exposes one frozen
table resolved to unicode or ASCII accordingly. It is the single place a non-ASCII
glyph literal may appear — every other module pulls from :func:`resolve_glyphs`.
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
    "dot": "·",
    "arrow": "→",
    "prompt": "›",
    "ok": "✓",
    "fail": "✗",
    "bang": "!",
    "bullet": "•",
    "dash": "—",
    "ellipsis": "…",
    "status": "┄",
    "hline": "─",
    "add": "+",
    "del": "-",
}
#: cp1252-safe fallbacks (used when the console cannot encode the unicode set).
_ASCII: dict[str, str] = {
    "dot": "-",
    "arrow": "->",
    "prompt": ">",
    "ok": "[ok]",
    "fail": "[x]",
    "bang": "!",
    "bullet": "-",
    "dash": "--",
    "ellipsis": "...",
    "status": ".",
    "hline": "-",
    "add": "+",
    "del": "-",
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
