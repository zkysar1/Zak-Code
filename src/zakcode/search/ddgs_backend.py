"""DuckDuckGo search backend (the zero-config default: free, no API key).

Wraps the ``ddgs`` library (the maintained successor to ``duckduckgo-search``). ``ddgs`` is
synchronous, so the blocking call runs in a worker thread. The import is local to
:meth:`search` so the package stays importable without the optional ``[web]`` dep.
"""

from __future__ import annotations

import asyncio

from zakcode._http import install_now_fix
from zakcode.search.base import BackendUnavailable, SearchBackend, SearchError, SearchItem

# Both [web] packages at once: search needs ddgs, and the natural next step (web_fetch on a
# result) needs httpx — one install covers the whole capability instead of failing twice.
_INSTALL_FIX = install_now_fix("ddgs", "httpx")

#: ``ddgs`` reports "every engine ran and none of them had anything" by RAISING, not by
#: returning an empty list: ``raise DDGSException(err or "No results found.")`` where ``err``
#: is ``None`` exactly when no engine errored. A real fault carries that error as the message
#: instead, and rate limits and timeouts raise dedicated SUBCLASSES — so the base type paired
#: with this exact sentinel is an unambiguous "searched fine, found nothing". Anything else
#: stays a fault, which is the safe direction: an unrecognised message degrades to today's
#: behaviour rather than swallowing a real failure as an empty result.
_NO_RESULTS_SENTINEL = "No results found."


class DuckDuckGoBackend(SearchBackend):
    """Free, no-key web search via DuckDuckGo (the default backend)."""

    name = "ddgs"

    def __init__(self, *, max_results_cap: int = 10) -> None:
        self._cap = max_results_cap

    async def search(self, query: str, *, max_results: int = 5) -> list[SearchItem]:
        try:
            from ddgs import DDGS
            from ddgs.exceptions import DDGSException
        except ImportError as exc:
            raise BackendUnavailable(
                "the 'ddgs' package is not installed", fix=_INSTALL_FIX
            ) from exc

        n = max(1, min(max_results, self._cap))

        def _run() -> list[dict]:
            with DDGS() as ddgs:
                return list(ddgs.text(query, max_results=n))

        try:
            raw = await asyncio.to_thread(_run)
        except Exception as exc:  # noqa: BLE001 - any ddgs/network error becomes a clean SearchError
            # A search that RAN and found nothing is not a failure. Reporting it as one told
            # the model the wrong thing ("may be rate-limiting; retry shortly") about the one
            # case where retrying is exactly wrong, and spent a tool error — which the stuck
            # ladder counts — on a working search (measured on the coach rig, 2026-09-10).
            # Returning the empty list hands it to web_search's own no-results branch, whose
            # advice is the right advice: try different or broader keywords.
            if type(exc) is DDGSException and str(exc) == _NO_RESULTS_SENTINEL:
                return []
            raise SearchError(
                f"DuckDuckGo search failed: {exc}",
                fix="DuckDuckGo may be rate-limiting; retry shortly, or set ZAKCODE_SEARCH_BACKEND",
            ) from exc

        items: list[SearchItem] = []
        for row in raw:
            if not isinstance(row, dict):
                continue
            items.append(
                SearchItem(
                    title=str(row.get("title") or ""),
                    url=str(row.get("href") or row.get("url") or ""),
                    snippet=str(row.get("body") or row.get("snippet") or ""),
                )
            )
        return items
