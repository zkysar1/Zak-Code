"""What the web page calls itself, written in as the server serves it (ADR-0350).

A deployment names itself in its own settings (``ZAKCODE_BRAND_NAME`` and its two siblings, set in
``~/.zakcode/.env``, the workspace's ``.env`` or the environment), never in the installed page.
``zakcode update`` replaces the install and nothing else, so an update keeps the name. The page
ships a slot wherever the name, the new chat's line or its suggestions appear, and
:func:`brand_page` fills each one, HTML-escaped, as the server fills the page's digest (ADR-0345).
With nothing set the page reads as it always has.

No FastAPI here: :mod:`zakcode.server` stays importable without the server extra.
"""

from __future__ import annotations

from html import escape

#: The page's own name, which it carries when a deployment sets none.
DEFAULT_NAME = "Zak Code"
#: The new chat's suggestions when a deployment sets none. Each fills the composer when clicked.
DEFAULT_SUGGESTIONS = (
    "explain this codebase",
    "find a bug and fix it",
    "run the tests and summarize",
)
#: The new chat's line when a deployment sets none, around the name it says can do the work.
_LINE = "Ask about this workspace — {} can read, edit, and run things."
#: The class the page's root carries under a deployment's own name. The page's display face is
#: cut to the letters of "Zak Code" (ADR-0221), so under it another name would set its other
#: letters in a fallback face; the page sets the whole name in that face instead.
OWN_BRAND_CLASS = "own-brand"

NAME_SLOT = b"__ZAKCODE_BRAND_NAME__"
LINE_SLOT = b"__ZAKCODE_BRAND_LINE__"
SUGGESTIONS_SLOT = b"__ZAKCODE_BRAND_SUGGESTIONS__"
CLASS_SLOT = b"__ZAKCODE_BRAND_CLASS__"


def brand_page(
    page: bytes,
    *,
    name: str | None = None,
    line: str | None = None,
    suggestions: list[str] | None = None,
) -> bytes:
    """``page`` with every brand slot filled.

    ``None`` or an empty string is the page's own: its name, its line and its three suggestions.
    A line left unset under a deployment's own name names that name, so a name set alone never
    leaves "Zak" on the page. ``[]`` shows no suggestions.
    """
    shown = name or DEFAULT_NAME
    own = shown != DEFAULT_NAME
    line = line or _LINE.format(shown if own else "Zak")
    picks = DEFAULT_SUGGESTIONS if suggestions is None else suggestions
    buttons = "".join(f'<button type="button" class="pill">{escape(s)}</button>' for s in picks)
    fills = {
        NAME_SLOT: escape(shown),
        LINE_SLOT: escape(line),
        SUGGESTIONS_SLOT: buttons,
        CLASS_SLOT: OWN_BRAND_CLASS if own else "",
    }
    for slot, text in fills.items():
        page = page.replace(slot, text.encode("utf-8"))
    return page
