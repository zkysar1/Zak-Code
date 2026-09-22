"""The Zak look wears the Vinheim family's colours, in both clients (ADR-0218).

docs/UX.md "The family" is the contract: the web client copies the family's tokens
verbatim, the terminal carries the family in its brand index, and the operator's orange
is the same colour on every door. These tests hold that contract without a browser or a
terminal — string inspection of the shipped page plus the theme objects themselves.
"""

from __future__ import annotations

import math
import re
from pathlib import Path

from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput
from rich.color import Color, ColorSystem
from rich.default_styles import DEFAULT_STYLES

from zakcode.cli._theme import ZAK_THEME
from zakcode.cli.saybox import SayBoxEditor

_STATIC = Path(__file__).resolve().parents[1] / "src" / "zakcode" / "server" / "static"
_INDEX = _STATIC / "index.html"

#: The Vinheim design system's tokens as shipped at vinheim.com (read 2026-09-22), keyed
#: by the web client's name for each. Change one only when the family changes.
FAMILY = {
    "--bg": "oklch(17% .030 258)",  # paper
    "--surface": "oklch(21% .032 258)",  # paper-2
    "--inset": "oklch(25% .034 258)",  # paper-3
    "--line": "oklch(34% .030 258)",  # line
    "--line-strong": "oklch(42% .034 258)",  # line-2
    "--fg": "oklch(96% .010 90)",  # ink
    "--muted": "oklch(66% .018 250)",  # ink-3
    "--brand": "oklch(80% .110 195)",  # accent
    "--brand-press": "oklch(72% .110 195)",  # accent-press
    "--on-brand": "oklch(20% .030 258)",  # on-accent
    "--focus": "oklch(86% .140 85)",  # focus
    "--ok": "oklch(78% .130 150)",  # success
    "--err": "oklch(70% .180 25)",  # danger
    "--warn": "oklch(82% .140 75)",  # warning
}


def _html() -> str:
    return _INDEX.read_text(encoding="utf-8")


def _root_tokens(html: str) -> dict[str, str]:
    block = re.search(r":root\s*\{(.*?)\n\s*\}", html, flags=re.S)
    assert block, "the page must declare its tokens in one :root block"
    body = re.sub(r"/\*.*?\*/", "", block.group(1), flags=re.S)
    return {name: value.strip() for name, value in re.findall(r"(--[a-z0-9-]+):\s*([^;]+);", body)}


def _oklch_to_srgb(lightness: float, chroma: float, hue: float) -> tuple[int, int, int]:
    """OKLCH -> 8-bit sRGB (Björn Ottosson's OKLab matrices)."""
    a, b = chroma * math.cos(math.radians(hue)), chroma * math.sin(math.radians(hue))
    l_ = lightness + 0.3963377774 * a + 0.2158037573 * b
    m_ = lightness - 0.1055613458 * a - 0.0638541728 * b
    s_ = lightness - 0.0894841775 * a - 1.2914855480 * b
    lin_l, lin_m, lin_s = l_**3, m_**3, s_**3
    linear = (
        4.0767416621 * lin_l - 3.3077115913 * lin_m + 0.2309699292 * lin_s,
        -1.2684380046 * lin_l + 2.6097574011 * lin_m - 0.3413193965 * lin_s,
        -0.0041960863 * lin_l - 0.7034186147 * lin_m + 1.7076147010 * lin_s,
    )

    def encode(x: float) -> int:
        x = min(1.0, max(0.0, x))
        return round(255 * (12.92 * x if x <= 0.0031308 else 1.055 * x ** (1 / 2.4) - 0.055))

    r, g, b_ = (encode(x) for x in linear)
    return r, g, b_


def _oklab(rgb: tuple[int, int, int]) -> tuple[float, float, float]:
    def decode(c: int) -> float:
        x = c / 255
        return x / 12.92 if x <= 0.04045 else ((x + 0.055) / 1.055) ** 2.4

    r, g, b = (decode(c) for c in rgb)
    lms = (
        0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b,
        0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b,
        0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b,
    )
    l_, m_, s_ = (math.copysign(abs(x) ** (1 / 3), x) for x in lms)
    return (
        0.2104542553 * l_ + 0.7936177850 * m_ - 0.0040720468 * s_,
        1.9779984951 * l_ - 2.4285922050 * m_ + 0.4505937099 * s_,
        0.0259040371 * l_ + 0.7827717662 * m_ - 0.8086757660 * s_,
    )


def _nearest_index(rgb: tuple[int, int, int]) -> int:
    """The xterm-256 index (16–255: the fixed cube and greys, never the themeable 0–15)
    perceptually nearest ``rgb``."""
    target = _oklab(rgb)

    def distance(index: int) -> float:
        triplet = Color.from_ansi(index).get_truecolor()
        return math.dist(target, _oklab((triplet.red, triplet.green, triplet.blue)))

    return min(range(16, 256), key=distance)


def _parse_oklch(value: str) -> tuple[float, float, float]:
    match = re.fullmatch(r"oklch\((\d+(?:\.\d+)?)%\s+(\d*\.\d+)\s+(\d+(?:\.\d+)?)\)", value)
    assert match, f"not an oklch() literal: {value!r}"
    return float(match.group(1)) / 100, float(match.group(2)), float(match.group(3))


# ── the web client ──────────────────────────────────────────────────────────


def test_web_tokens_are_the_family_tokens_verbatim() -> None:
    tokens = _root_tokens(_html())
    assert {name: tokens.get(name) for name in FAMILY} == FAMILY


def test_every_custom_property_the_page_reads_is_declared() -> None:
    # An undeclared var() is valid CSS that renders as nothing (or as its fallback),
    # silently — a renamed token leaves its readers painting the wrong thing.
    html = _html()
    declared = set(_root_tokens(html))
    read = set(re.findall(r"var\((--[a-z0-9-]+)", html))
    assert read, "positive control: the page reads its tokens through var()"
    assert read - declared == set()


def test_the_page_has_one_look() -> None:
    html = _html()
    assert "prefers-color-scheme" not in html  # no second theme to drift out of step
    assert "color-scheme: dark;" in html
    assert not re.search(r"#[0-9a-fA-F]{6}\b", html.replace(_root_block(html), ""))


def _root_block(html: str) -> str:
    block = re.search(r":root\s*\{.*?\n\s*\}", html, flags=re.S)
    assert block
    return block.group(0)


# ── the terminal ────────────────────────────────────────────────────────────


def test_terminal_brand_is_the_index_nearest_the_web_accent() -> None:
    # Derived, not restated: change the web accent without the terminal (or the reverse)
    # and this fails — the one cross-renderer colour link the family depends on.
    accent = _oklch_to_srgb(*_parse_oklch(_root_tokens(_html())["--brand"]))
    brand = ZAK_THEME.styles["brand"].color
    assert brand is not None and brand.number == _nearest_index(accent)
    for mark in ("assistant.marker", "spinner"):
        assert ZAK_THEME.styles[mark].color == brand, mark


def test_brand_marks_never_collide_with_inline_code_on_16_colours() -> None:
    brand, code = ZAK_THEME.styles["brand"].color, ZAK_THEME.styles["md.code"].color
    assert brand is not None and code is not None
    on_16 = ColorSystem.STANDARD
    assert brand.downgrade(on_16).number != code.downgrade(on_16).number


def test_no_theme_style_leans_on_dim() -> None:
    # tmux drops the dim attribute, so a dim style renders at full contrast in the
    # cockpit (ADR-0186): every quieter shade is a colour index instead. Only the
    # theme's OWN styles count — rich merges its defaults (``dim`` itself among them).
    ours = {
        name: style
        for name, style in ZAK_THEME.styles.items()
        if name not in DEFAULT_STYLES or style != DEFAULT_STYLES[name]
    }
    assert "brand" in ours and "footer" in ours  # positive control: the filter keeps ours
    assert sorted(name for name, style in ours.items() if style.dim) == []


def test_the_humans_orange_is_one_colour_on_every_door(tmp_path: Path) -> None:
    orange = ZAK_THEME.styles["user.marker"].color
    assert orange is not None
    triplet = orange.get_truecolor()
    assert ZAK_THEME.styles["prompt.marker"].color == orange
    with create_pipe_input() as pipe:
        editor = SayBoxEditor(
            tmp_path / ".say", tmp_path / ".interrupt", input=pipe, output=DummyOutput()
        )
        attrs = editor._session.style.get_attrs_for_style_str("class:prompt")
    assert attrs.color == f"{triplet.red:02x}{triplet.green:02x}{triplet.blue:02x}"
    assert attrs.bold
