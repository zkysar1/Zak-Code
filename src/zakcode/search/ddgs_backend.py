"""DuckDuckGo search backend (the zero-config default: free, no API key).

Wraps the ``ddgs`` library (the maintained successor to ``duckduckgo-search``). ``ddgs`` is
synchronous, so the blocking call runs in a worker thread. The import is local to
:meth:`search` so the package stays importable without the optional ``[web]`` dep.
"""

from __future__ import annotations

import asyncio

from zakcode._http import pip_install_hint
from zakcode.search.base import BackendUnavailable, SearchBackend, SearchError, SearchItem

_INSTALL_FIX = f"the 'ddgs' search backend needs the ddgs package: {pip_install_hint('ddgs')}"


class DuckDuckGoBackend(SearchBackend):
    """Free, no-key web search via DuckDuckGo (the default backend)."""

    name = "ddgs"

    def __init__(self, *, max_results_cap: int = 10) -> None:
        self._cap = max_results_cap

    async def search(self, query: str, *, max_results: int = 5) -> list[SearchItem]:
        try:
            from ddgs import DDGS
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
