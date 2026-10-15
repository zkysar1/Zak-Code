"""A malformed web page in a provider error is reduced in linear time, to the same words (ADR-0313).

ADR-0306 reduces an HTML page in an exception's message to its words with three lazy or greedy
patterns. Each rescans the rest of the page from every opening that never closes, so a page made
of unclosed openings took time quadratic in its length, and the error mapping is plain synchronous
code inside an async turn: one broken or hostile upstream page blocked the event loop that
``zakcode webapp`` serves every session from. Measured on main on 2026-10-15: ``<html>`` and 50,000
``<`` took 0.67 s here and 1.4 to 5.6 s in the review; ``<head`` 4,000 times took 0.24 s here and
0.5 to 1.6 s in the review. The pages below are larger, so the old code takes seconds, and the
bound is generous.

The reduction now does the same work in one pass, and its text is byte-identical to the patterns'
for every input: the reference below is main's code, and the comparison runs it over pages the
tests write and over generated fragments. Hermetic: no network.
"""

from __future__ import annotations

import html
import random
import re
import time

import pytest

import zakcode.providers.litellm_provider as lp
from zakcode.providers.base import ProviderUnavailable
from zakcode.providers.litellm_provider import LiteLLMProvider

#: Seconds an error mapping may take on any page below. The new code takes milliseconds; the
#: old took seconds on each of these pages.
_BOUND_S = 0.5

_ADVERSARIAL = {
    "unclosed angle brackets": "<html>" + "<" * 120_000,
    "unclosed heads": "<html>" + "<head" * 10_000,
    "unclosed scripts": "<html>" + "<script" * 10_000,
    "unclosed styles": "<html>" + "<style" * 10_000,
    "unclosed styles before a script's close": "<html>" + "<style" * 10_000 + "</script>",
}


def _vendor(name: str, message: str, status: int | None) -> Exception:
    """An exception shaped like litellm's: matched by class name, with a status."""
    return type(name, (Exception,), {"status_code": status})(message)


@pytest.mark.parametrize("page", list(_ADVERSARIAL.values()), ids=list(_ADVERSARIAL))
def test_a_malformed_page_is_mapped_within_the_bound(page: str) -> None:
    exc = _vendor("BadGatewayError", f"OpenAIException - {page}", 502)

    started = time.perf_counter()
    mapped = LiteLLMProvider._map_error(exc)
    elapsed = time.perf_counter() - started

    assert isinstance(mapped, ProviderUnavailable)
    assert elapsed < _BOUND_S


@pytest.mark.parametrize("opening", ["<head", "<script"])
def test_a_quarter_megabyte_of_unclosed_openings_is_reduced_within_the_bound(opening: str) -> None:
    """At this size, scanning for a closing tag from each opening that has none, even with a
    fast search, takes seconds: an opening after the last closing tag of its kind is skipped."""
    message = "OpenAIException - <html>" + opening * 50_000

    started = time.perf_counter()
    lp._page_text(message)
    elapsed = time.perf_counter() - started

    assert elapsed < _BOUND_S


# ── the same words as before ─────────────────────────────────────────────────────────────────

_REF_PAGE = re.compile(r"<(?:!doctype\s+)?html[\s>]", re.IGNORECASE)
_REF_HEAD = re.compile(r"<head\b.*?</head\s*>", re.IGNORECASE | re.DOTALL)
_REF_CODE = re.compile(r"<(script|style)\b.*?</\1\s*>", re.IGNORECASE | re.DOTALL)
_REF_TAG = re.compile(r"<[^>]*>")


def _reference(message: str) -> str:
    """``_page_text`` as ADR-0306 wrote it, with its three patterns."""
    found = _REF_PAGE.search(message)
    if found is None:
        return message
    page = _REF_CODE.sub(" ", message[found.start() :])
    text = " ".join(html.unescape(_REF_TAG.sub(" ", _REF_HEAD.sub(" ", page))).split())
    if not text:
        text = " ".join(html.unescape(_REF_TAG.sub(" ", page)).split())
    return f"{message[: found.start()].rstrip()} {text}".strip()


_PAGES = [
    "<html><head><title>504 Gateway Time-out</title></head><body><h1>504</h1></body></html>",
    "<html><head><title>504</title></head><body></body></html>",
    '<!DOCTYPE html><html lang="en"><head><title>504</title><style>b { c: d }</style></head>'
    "<body><script>var r = 1;</script><h1>504</h1><p>Try again &amp; wait</p></body></html>",
    "<HTML><HEAD><TITLE>Upper</TITLE></HEAD><BODY><SCRIPT>x</script>words</BODY></HTML>",
    "<html><head><title>no close<body>words</body></html>",
    "<html><header>kept</header><scripts>kept too</scripts><body>words</body></html>",
    "<html><style>a{content:'<script>'}</style><p>words</p><script>x</script></html>",
    "<html><script>var s = '</style>';</script><style>p{}</style><p>words</p></html>",
    "<html><head>one</head  ><head>two</head\n>three</html>",
    "<html><<b>bold</b> a < b and c > d <i",
    "<html>" + "<head" * 50 + "</head>" + "<script" * 50,
    "<html>" + "<" * 200 + ">" + "<" * 200,
    "<!doctype   html>&quot;quoted&quot; &#34;decimal&#34; &#x22;hex&#x22;",
    "no page at all, <tool_call> and all",
]


@pytest.mark.parametrize("page", _PAGES)
@pytest.mark.parametrize("prefix", ["", "OpenAIException - ", "  before  "])
def test_a_page_reads_as_the_patterns_read_it(prefix: str, page: str) -> None:
    message = prefix + page

    assert lp._page_text(message) == _reference(message)


_TOKENS = [
    "<html>",
    "<HTML ",
    "<!doctype html>",
    "<head>",
    "<head",
    "</head>",
    "</head  >",
    "<HEAD>",
    "</Head>",
    "<header>",
    "<script>",
    "<script",
    "</script>",
    "</script >",
    "<SCRIPT>",
    "</Script>",
    "<scripts>",
    "<style>",
    "<style",
    "</style>",
    "</STYLE\n>",
    "<title>",
    "</title>",
    "<p>",
    "</p>",
    "<b",
    "x>",
    "<",
    ">",
    "<<",
    ">>",
    "text",
    "more words",
    "&amp;",
    "&quot;",
    "&#34;",
    "&#x22;",
    "&lt;",
    " ",
    "\n",
    "\t",
]


def test_generated_fragments_read_as_the_patterns_read_them() -> None:
    rng = random.Random(20261015)
    for _ in range(3_000):
        message = "".join(rng.choice(_TOKENS) for _ in range(rng.randint(0, 40)))

        assert lp._page_text(message) == _reference(message), message
