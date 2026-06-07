"""A tiny, dependency-free HTML -> readable-text converter for ``web_fetch``.

Built on the stdlib ``html.parser`` (no BeautifulSoup / markdownify dependency, in keeping with
the project's dependency-light, clean-room values). It is deliberately modest: keep text, turn
headings into ``#`` lines and links into ``text (url)``, add line breaks at block boundaries, and
collapse runaway whitespace. The output is for a model to read, not to round-trip — and it is
defanged as UNTRUSTED by the caller before it ever reaches the prompt.

**Readability (cut the chrome), with a safety net so content is never lost:**

* ``script`` / ``style`` / ``head`` / ``noscript`` / embedded-media content is dropped outright
  (it is never page text).
* ``nav`` / ``footer`` / ``aside`` are *boilerplate*: dropped on a normal page, but kept as a
  last resort (see below) so a page whose real content happens to live there is never blanked.
* If the page marks its main content with ``<main>`` or ``<article>`` (most docs/CMS/blogs do),
  only text inside that region is returned — headers, sidebars, and mega-menus fall away.

``render`` picks the first non-empty of three tiers: (1) main/article content, (2) the whole body
minus boilerplate, (3) everything (boilerplate included). So the chrome-cut never reduces a page
to nothing — it only ever *prefers* the cleaner view when one exists.

Note on script/style stripping: like every HTML parser, ``html.parser`` ends a ``<script>`` /
``<style>`` CDATA region at the FIRST literal ``</script>`` / ``</style>``, so any text a page
places *after* an embedded close sentinel inside such an element will surface as page text —
exactly as a real browser would render it. This is content noise, not a security hole: the
result is defanged, so a leaked fragment cannot forge a tool frame or chat-template token.
"""

from __future__ import annotations

from html.parser import HTMLParser

#: Tags whose *content* is dropped outright and never recovered (never page text).
_DROP_TAGS = frozenset(
    {"script", "style", "head", "noscript", "svg", "template", "iframe", "canvas"}
)
#: Site-boilerplate containers: dropped on a normal page, but recoverable as a last resort
#: (render() falls back to including them rather than returning an empty page).
_BOILERPLATE_TAGS = frozenset({"nav", "footer", "aside"})
#: Semantic main-content landmarks. When present, only text inside them is kept.
_MAIN_TAGS = frozenset({"main", "article"})
#: Block-level tags: emit a line break around them so structure survives as newlines.
_BLOCK_TAGS = frozenset(
    {
        "p",
        "div",
        "section",
        "article",
        "header",
        "main",
        "li",
        "ul",
        "ol",
        "tr",
        "table",
        "blockquote",
        "pre",
        "br",
        "hr",
    }
)


def _is_heading(tag: str) -> bool:
    return len(tag) == 2 and tag[0] == "h" and tag[1].isdigit()


class _TextExtractor(HTMLParser):
    """Accumulate readable text, tagging each piece as in-main and/or boilerplate (for render)."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._parts: list[
            tuple[bool, bool, str]
        ] = []  # (in <main>/<article>, in boilerplate, text)
        self._drop_depth = 0  # inside script/style/etc. — content discarded, unrecoverable
        self._boiler_depth = 0  # inside nav/footer/aside — droppable but recoverable
        self._main_depth = 0
        self._href: str | None = None

    def _emit(self, text: str) -> None:
        self._parts.append((self._main_depth > 0, self._boiler_depth > 0, text))

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in _DROP_TAGS:
            self._drop_depth += 1
            return
        if self._drop_depth:
            return
        if tag in _BOILERPLATE_TAGS:
            self._boiler_depth += 1
        if tag in _MAIN_TAGS:
            self._main_depth += 1
        if _is_heading(tag):
            self._emit("\n\n" + "#" * int(tag[1]) + " ")
        elif tag == "a":
            self._href = dict(attrs).get("href")
        elif tag in _BLOCK_TAGS:
            self._emit("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in _DROP_TAGS:
            if self._drop_depth:
                self._drop_depth -= 1
            return
        if self._drop_depth:
            return
        if tag == "a" and self._href:
            href = self._href
            self._href = None
            # Only surface http(s) links; skip anchors/js/mailto noise.
            if href and href.startswith(("http://", "https://")):
                self._emit(f" ({href})")
        elif _is_heading(tag) or tag in _BLOCK_TAGS:
            self._emit("\n")
        if tag in _MAIN_TAGS and self._main_depth:
            self._main_depth -= 1
        if tag in _BOILERPLATE_TAGS and self._boiler_depth:
            self._boiler_depth -= 1

    def handle_data(self, data: str) -> None:
        if self._drop_depth == 0:
            self._emit(data)

    def render(self) -> str:
        # Three tiers, first non-empty wins: (1) main/article content, (2) body minus boilerplate,
        # (3) everything (boilerplate included) — so the chrome-cut never blanks a real page.
        main = "".join(t for in_main, boiler, t in self._parts if in_main and not boiler)
        if main.strip():
            raw = main
        else:
            body = "".join(t for _, boiler, t in self._parts if not boiler)
            raw = body if body.strip() else "".join(t for _, _, t in self._parts)
        # Collapse intra-line whitespace, then squeeze runs of blank lines to at most one.
        lines = [" ".join(line.split()) for line in raw.splitlines()]
        out: list[str] = []
        blanks = 0
        for line in lines:
            if line:
                out.append(line)
                blanks = 0
            else:
                blanks += 1
                if blanks <= 1:
                    out.append("")
        return "\n".join(out).strip()


def html_to_text(html: str) -> str:
    """Convert an HTML document to readable plain text (never raises)."""
    parser = _TextExtractor()
    try:
        parser.feed(html)
        parser.close()
    except Exception:  # noqa: BLE001 - malformed HTML must degrade, never crash the tool
        pass
    return parser.render()
