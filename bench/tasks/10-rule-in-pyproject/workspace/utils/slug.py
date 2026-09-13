"""URL slugs."""

from __future__ import annotations

import re
import unicodedata

_NON_WORD = re.compile(r"[^a-z0-9]+")


def slugify(text: str, *, max_length: int = 60) -> str:
    """Lower-case ASCII slug of `text`, words joined by single hyphens.

    Accents are stripped, anything that is not a letter or digit becomes
    a hyphen, and the result is cut to `max_length` without a trailing
    hyphen. An input with no usable characters gives an empty string.
    """
    ascii_text = (
        unicodedata.normalize("NFKD", text)
        .encode("ascii", "ignore")
        .decode("ascii")
        .lower()
    )
    slug = _NON_WORD.sub("-", ascii_text).strip("-")
    return slug[:max_length].rstrip("-")
