"""Cut the web client's display face: Fraunces, for the two "Zak Code" moments (ADR-0221).

The page carries its display face inline (the ``@font-face`` under ``:root`` in
src/zakcode/server/static/index.html), so the header wordmark and the empty state's name
draw in the Vinheim family's Fraunces on every machine, offline, with nothing fetched.

This script is how that face is made. The build and the tests never run it:

    pip install fonttools brotli        # tools for the cut only, never runtime deps
    python scripts/build_display_font.py

It fetches the Fraunces latin file from Google Fonts (weight axis 100-900, the other axes
pinned by Google). That is the file Vinheim's next/font build downloads: compared on
2026-09-23, every table but ``head`` is byte-identical. It keeps the glyphs in DRAWN and
AUTOHINT, writes the WOFF2 into the page's ``@font-face``, and prints the SHA-256 and the
character set that tests/test_ux_family.py pins.

AUTOHINT is not decoration. Fraunces carries no hinting, so FreeType (Chrome on Linux)
auto-hints it, and the auto-hinter measures these reference letters to place its
alignment zones. Cut them away and the same outlines land on a different pixel grid:
measured, a cut of DRAWN alone differed from the full font in 906 greyscale pixels of
seven isolated 96px letters, by up to 166 levels. With them the cut matches the full font
to the pixel at 600 and 700, at 18, 30 and 64px. The weight axis stays whole for the same
reason: limiting it to 600-700 re-rounds the outlines (1,562 differing pixels).

Fraunces is under the SIL Open Font License 1.1, Copyright 2020 The Fraunces Project
Authors (github.com/undercasetype/Fraunces). Every name record is kept, so the notice
travels inside the font as well as beside it in the page.
"""

from __future__ import annotations

import base64
import hashlib
import io
import re
import sys
import urllib.request
from pathlib import Path

from fontTools import subset
from fontTools.ttLib import TTFont

INDEX = Path(__file__).resolve().parents[1] / "src" / "zakcode" / "server" / "static" / "index.html"
CSS_URL = "https://fonts.googleapis.com/css2?family=Fraunces:wght@600;700"
#: A current browser's user agent: Google Fonts answers it with WOFF2.
USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/140.0 Safari/537.36"
)
#: What the page draws in the display face: the header wordmark and the empty state's name.
DRAWN = "Zak Code"
#: FreeType autofit's Latin blue-zone strings (afblue.dat): capital top, capital bottom,
#: small F top, small top and bottom, small descender.
AUTOHINT = "".join(("THEZOCQS", "HEZLOCUS", "fijkdbh", "xzroesc", "pqgjy"))
_SRC = re.compile(r'(src: url\(data:font/woff2;base64,)[A-Za-z0-9+/=]*(\) format\("woff2"\))')


def _get(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=30) as response:
        body: bytes = response.read()
    return body


def fetch() -> bytes:
    """The Fraunces latin WOFF2 Google Fonts serves for CSS_URL."""
    css = _get(CSS_URL).decode()
    # Google lists one @font-face per script subset, each under a /* <subset> */ comment.
    latin = css.split("/* latin */", 1)[1]
    url = re.search(r"url\((https://fonts\.gstatic\.com/[^)]+\.woff2)\)", latin)
    if url is None:
        # Seen once (2026-09-23), not reproduced: the next four requests answered WOFF2.
        raise SystemExit(
            f"Google Fonts answered without a latin WOFF2; run again. It sent:{latin[:300]}"
        )
    return _get(url.group(1))


def cut(font_file: bytes) -> bytes:
    """Keep DRAWN and AUTOHINT, every weight, every name record and every feature."""
    font = TTFont(io.BytesIO(font_file))
    options = subset.Options()
    options.flavor = "woff2"
    options.name_IDs = ["*"]  # the copyright (0) and the licence link (14) travel with it
    options.name_languages = ["*"]
    options.name_legacy = True
    options.layout_features = ["*"]  # kerning and every other feature the kept glyphs use
    options.notdef_outline = True
    subsetter = subset.Subsetter(options)
    subsetter.populate(text=DRAWN + AUTOHINT)
    subsetter.subset(font)
    out = io.BytesIO()
    font.flavor = "woff2"
    font.save(out)
    return out.getvalue()


def main() -> int:
    face = cut(fetch())
    encoded = base64.b64encode(face).decode()
    html = INDEX.read_text(encoding="utf-8")
    new, count = _SRC.subn(lambda m: m.group(1) + encoded + m.group(2), html)
    if count != 1:
        print(f"expected one display-face src in {INDEX}, found {count}", file=sys.stderr)
        return 1
    INDEX.write_text(new, encoding="utf-8")
    chars = "".join(sorted(chr(code) for code in TTFont(io.BytesIO(face)).getBestCmap()))
    print(f"wrote {len(face)} bytes into {INDEX.name}")
    print(f"DISPLAY_FACE_SHA256 = {hashlib.sha256(face).hexdigest()!r}")
    print(f"DISPLAY_FACE_CHARS = {chars!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
