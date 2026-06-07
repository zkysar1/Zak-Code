"""Tavily search backend (LLM-optimized results; free tier, requires an API key).

Tavily is a plain JSON REST API, so this talks to it directly over ``httpx`` — no vendor SDK.
Per the project convention (config.py: provider keys are read from standard env vars, never
modeled in Settings), the key comes from ``TAVILY_API_KEY`` in the environment.
"""

from __future__ import annotations

import os

from zakcode._http import fetch_json_capped, load_httpx
from zakcode.search.base import (
    BackendNotConfigured,
    BackendUnavailable,
    SearchBackend,
    SearchError,
    SearchItem,
)

_TAVILY_URL = "https://api.tavily.com/search"


class TavilyBackend(SearchBackend):
    """Web search via the Tavily API (key from ``TAVILY_API_KEY``)."""

    name = "tavily"

    def __init__(self, *, api_key: str | None = None, timeout: float = 10.0) -> None:
        # Read the key at construction from the standard env var (not Settings — keeps secrets
        # off the config surface, like the LLM provider keys); explicit arg wins (for tests).
        self._api_key = api_key or os.getenv("TAVILY_API_KEY")
        self._timeout = timeout

    async def search(self, query: str, *, max_results: int = 5) -> list[SearchItem]:
        if not self._api_key:
            raise BackendNotConfigured(
                "TAVILY_API_KEY is not set",
                fix="get a free key at https://tavily.com and export TAVILY_API_KEY=...",
            )
        try:
            httpx = load_httpx()
        except ImportError as exc:
            raise BackendUnavailable(str(exc), fix="pip install 'zakcode[web]'") from exc

        n = max(1, max_results)
        payload = {"api_key": self._api_key, "query": query, "max_results": n}
        try:
            data = await fetch_json_capped(
                httpx, "POST", _TAVILY_URL, timeout=self._timeout, json=payload
            )
        except Exception as exc:  # noqa: BLE001 - any transport/HTTP error becomes a SearchError
            raise SearchError(f"Tavily search failed: {exc}") from exc

        results = data.get("results") if isinstance(data, dict) else None
        items: list[SearchItem] = []
        for row in results or []:
            if isinstance(row, dict):
                items.append(
                    SearchItem(
                        title=str(row.get("title") or ""),
                        url=str(row.get("url") or ""),
                        snippet=str(row.get("content") or ""),
                    )
                )
            # Honor the "up to max_results" contract even if the API over-returns.
            if len(items) >= n:
                break
        return items
