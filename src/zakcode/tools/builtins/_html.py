"""A tiny, dependency-free HTML -> readable-text converter for ``web_fetch``.

Built on the stdlib ``html.parser`` (no BeautifulSoup / markdownify dependency, in keeping with
the project's dependency-light, clean-room values). It is deliberately modest: drop
script/style/head/etc., keep text, turn headings into ``#`` lines and links into ``text (url)``,
add line breaks at block boundaries, and collapse runaway whitespace. The output is meant for a
model to read, not to round-trip — and it is defanged as UNTRUSTED by the caller before it ever
reaches the prompt.

Note on script/style stripping: like every HTML parser, ``html.parser`` ends a ``<script>`` /
``<style>`` CDATA region at the FIRST literal ``</script>`` / ``</style>``, so any text a page
places *after* an embedded close sentinel inside such an element will surface as page text —
exactly as a real browser would render it. This is content noise, not a security hole: the
result is defanged, so a leaked fragment cannot forge a tool frame or chat-template token.
"""

from __future__ import annotations

from html.parser import HTMLParser

#: Tags whose *content* is never text we want (scripts, styling, metadata, embedded media).
_SKIP_TAGS = frozenset(
    {"script", "style", "head", "noscript", "svg", "template", "iframe", "canvas"}
)
#: Block-level tags: emit a line break around them so structure survives as newlines.
_BLOCK_TAGS = frozenset(
    {
        "p",
        "div",
        "section",
        "article",
        "header",
        "footer",
        "main",
        "aside",
        "nav",
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


class _TextExtractor(HTMLParser):
    """Accumulate readable text from HTML events."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._parts: list[str] = []
        self._skip_depth = 0
        self._href: str | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in _SKIP_TAGS:
            self._skip_depth += 1
            return
        if self._skip_depth:
            return
        if len(tag) == 2 and tag[0] == "h" and tag[1].isdigit():
            self._parts.append("\n\n" + "#" * int(tag[1]) + " ")
        elif tag == "a":
            self._href = dict(attrs).get("href")
        elif tag in _BLOCK_TAGS:
            self._parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in _SKIP_TAGS:
            if self._skip_depth:
                self._skip_depth -= 1
            return
        if self._skip_depth:
            return
        if tag == "a" and self._href:
            href = self._href
            self._href = None
            # Only surface http(s) links; skip anchors/js/mailto noise.
            if href and href.startswith(("http://", "https://")):
                self._parts.append(f" ({href})")
        elif (len(tag) == 2 and tag[0] == "h" and tag[1].isdigit()) or tag in _BLOCK_TAGS:
            self._parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._skip_depth == 0:
            self._parts.append(data)

    def render(self) -> str:
        raw = "".join(self._parts)
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
