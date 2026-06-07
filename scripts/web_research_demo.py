"""Live integration demo: drive Zak Code's web tools through a real research project.

Runs the REAL ``web_search`` / ``web_fetch`` builtins against the live web -- so it needs the
``[web]`` extra (``uv sync --extra web``) and network access. It is a dogfooding demo, NOT a unit
test (the hermetic tests live in ``tests/test_web_tools.py``).

It exercises both shapes the tools are meant to support:

* **Parallel fan-out** -- several ``web_search`` calls run concurrently via ``asyncio.gather``.
* **Sequential drill-down** -- a follow-up search is chosen *from* the first round's results.
* **Parallel fetch** -- the top sources are fetched concurrently with ``web_fetch``.

It prints what it found (with wall-clock timings that show the concurrency win) so a human or a
model can synthesize the answer. Run:  ``uv run python scripts/web_research_demo.py``
"""

from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path

from zakcode.config import load_settings
from zakcode.search import make_search_backend
from zakcode.tools.base import ToolContext
from zakcode.tools.builtins.web_fetch import WebFetchTool
from zakcode.tools.builtins.web_search import WebSearchTool

# The research project (a deliberate bit of dogfooding -- it validates the very SSRF design we
# just shipped against the wider world).
QUESTION = (
    "In 2026, what is the recommended way to defend an HTTP-fetch tool against SSRF -- "
    "is 'resolve the host then connect to the validated IP' the consensus, and what do "
    "OWASP and mature tools advise about DNS rebinding and the cloud-metadata endpoint?"
)
PARALLEL_QUERIES = [
    "OWASP SSRF prevention cheat sheet best practices",
    "DNS rebinding protection HTTP client pin resolved IP address",
    "Python httpx SSRF protection block private IP library",
    "cloud metadata 169.254.169.254 IMDS SSRF protection guidance",
]
#: Prefer authoritative sources when ranking which pages to actually fetch.
_PREFERRED = (
    "owasp.org",
    "cheatsheetseries.owasp.org",
    "portswigger.net",
    "github.com",
    "cloud.google.com",
    "docs.aws.amazon.com",
    "developer.mozilla.org",
)


def _hr(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def _utf8_stdout() -> None:
    """Windows consoles default to cp1252; fetched web content (and our arrows) are utf-8."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="replace")


def _rank_urls(urls: list[str], limit: int) -> list[str]:
    """Dedup (preserve order) and float preferred/authoritative domains to the front."""
    seen: set[str] = set()
    uniq: list[str] = []
    for u in urls:
        if u and u not in seen:
            seen.add(u)
            uniq.append(u)
    uniq.sort(key=lambda u: 0 if any(d in u for d in _PREFERRED) else 1)
    return uniq[:limit]


def _urls_from(result) -> list[str]:
    """Pull result URLs out of a web_search ToolResult's structured data."""
    if result is None or result.is_error or not result.data:
        return []
    return [r.get("url", "") for r in result.data.get("results", [])]


async def main() -> int:
    _utf8_stdout()
    settings = load_settings()
    ctx = ToolContext(workspace_root=Path.cwd())
    search = WebSearchTool(make_search_backend(settings))
    fetch = WebFetchTool(allowed_domains=settings.web_allowed_domains)

    _hr("RESEARCH QUESTION")
    print(QUESTION)
    print(
        f"\nbackend: {settings.search_backend}   "
        f"egress: {settings.web_allowed_domains or 'any public host'}"
    )

    # ── Round 1: PARALLEL searches ────────────────────────────────────────────────
    _hr(f"ROUND 1 -- {len(PARALLEL_QUERIES)} searches IN PARALLEL (asyncio.gather)")
    t0 = time.perf_counter()
    results = await asyncio.gather(
        *(search.execute({"query": q, "max_results": 5}, ctx) for q in PARALLEL_QUERIES)
    )
    parallel_elapsed = time.perf_counter() - t0

    all_urls: list[str] = []
    ok_count = 0
    for q, res in zip(PARALLEL_QUERIES, results, strict=True):
        status = "ERROR" if res.is_error else f"{res.data.get('count', 0)} hits"
        print(f"\n* [{status}] {q}")
        if res.is_error:
            print(f"    {res.output.strip()[:200]}")
            if res.fix:
                print(f"    fix: {res.fix}")
            continue
        ok_count += 1
        for r in (res.data or {}).get("results", [])[:3]:
            print(f"    - {r.get('title', '')[:70]}")
            print(f"      {r.get('url', '')}")
        all_urls.extend(_urls_from(res))
    print(
        f"\n-> {ok_count}/{len(PARALLEL_QUERIES)} searches succeeded in "
        f"{parallel_elapsed:.2f}s wall-clock (run concurrently)."
    )

    if not all_urls:
        print("\nNo results (the search backend may be rate-limiting). Stopping.")
        return 1

    # ── Round 2: SEQUENTIAL drill-down informed by round 1 ─────────────────────────
    _hr("ROUND 2 -- sequential follow-up (a focused query, then fetch the top sources)")
    followup = "OWASP SSRF prevention validate destination IP not hostname allowlist"
    print(f"follow-up query: {followup}")
    f_res = await search.execute({"query": followup, "max_results": 5}, ctx)
    if not f_res.is_error:
        all_urls.extend(_urls_from(f_res))
        for r in (f_res.data or {}).get("results", [])[:3]:
            print(f"    - {r.get('title', '')[:70]}\n      {r.get('url', '')}")

    # ── Fetch the top sources IN PARALLEL ──────────────────────────────────────────
    top = _rank_urls(all_urls, limit=4)
    _hr(f"FETCHING {len(top)} TOP SOURCES IN PARALLEL (web_fetch)")
    for u in top:
        print(f"  - {u}")
    t1 = time.perf_counter()
    pages = await asyncio.gather(*(fetch.execute({"url": u, "max_chars": 3000}, ctx) for u in top))
    fetch_elapsed = time.perf_counter() - t1
    print(
        f"\n-> fetched {sum(1 for p in pages if not p.is_error)}/{len(top)} pages in "
        f"{fetch_elapsed:.2f}s wall-clock (concurrent)."
    )

    _hr("EVIDENCE (extracted text, for synthesis)")
    for u, page in zip(top, pages, strict=True):
        print(f"\n----- {u} -----")
        if page.is_error:
            print(f"[fetch error] {page.output.strip()[:200]}")
            continue
        snippet = page.output.strip()
        print(snippet[:1400])
        if len(snippet) > 1400:
            print(f"... [{len(snippet)} chars total]")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
