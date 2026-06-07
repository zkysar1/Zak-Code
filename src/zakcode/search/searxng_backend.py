"""SearXNG search backend (self-hosted metasearch; free, no key, you run the instance).

Queries a SearXNG instance's JSON API (``GET {base_url}/search?q=...&format=json``) over
``httpx``. The base URL comes from ``ZAKCODE_SEARXNG_URL``; the instance must have the ``json``
output format enabled in its settings.yml.
"""

from __future__ import annotations

import re

from zakcode._http import fetch_json_capped, load_httpx
from zakcode.search.base import (
    BackendNotConfigured,
    BackendUnavailable,
    SearchBackend,
    SearchError,
    SearchItem,
)

#: Mask ``//user:pass@`` userinfo wherever a URL appears inside a message (httpx exceptions embed
#: the full request URL in prose, so the bare-URL strip_url_credentials helper can't reach it).
_USERINFO_RE = re.compile(r"//[^/@\s]+@")


class SearxngBackend(SearchBackend):
    """Web search via a self-hosted SearXNG instance (URL from ``ZAKCODE_SEARXNG_URL``)."""

    name = "searxng"

    def __init__(self, *, base_url: str | None = None, timeout: float = 10.0) -> None:
        self._base_url = (base_url or "").rstrip("/")
        self._timeout = timeout

    async def search(self, query: str, *, max_results: int = 5) -> list[SearchItem]:
        if not self._base_url:
            raise BackendNotConfigured(
                "SearXNG base URL is not set",
                fix="point ZAKCODE_SEARXNG_URL at your SearXNG instance (with json format enabled)",
            )
        try:
            httpx = load_httpx()
        except ImportError as exc:
            raise BackendUnavailable(str(exc), fix="pip install 'zakcode[web]'") from exc

        params = {"q": query, "format": "json"}
        try:
            data = await fetch_json_capped(
                httpx, "GET", f"{self._base_url}/search", timeout=self._timeout, params=params
            )
        except Exception as exc:  # noqa: BLE001 - any transport/HTTP error becomes a SearchError
            # The base URL may embed credentials (basic-auth in front of a self-host); httpx
            # exceptions echo the full URL, so scrub userinfo before it reaches the model/CLI.
            raise SearchError(
                f"SearXNG search failed: {_USERINFO_RE.sub('//***@', str(exc))}",
                fix="check ZAKCODE_SEARXNG_URL and that the instance has the JSON format enabled",
            ) from exc

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
            if len(items) >= max(1, max_results):
                break
        return items
