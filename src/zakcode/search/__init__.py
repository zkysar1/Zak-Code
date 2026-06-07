"""Web-search backends + the config-driven factory that selects one.

``make_search_backend(settings)`` reads ``settings.search_backend`` (default ``ddgs``) and
returns the matching backend, lazy-importing only that backend's module — so an unselected
backend's optional deps are never imported. This mirrors how the Agent builds its LLM provider.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from zakcode.search.base import (
    BackendNotConfigured,
    BackendUnavailable,
    SearchBackend,
    SearchError,
    SearchItem,
)

if TYPE_CHECKING:
    from zakcode.config import Settings


def make_search_backend(settings: Settings | None = None) -> SearchBackend:
    """Build the search backend named by ``settings.search_backend`` (default ``ddgs``).

    ``None`` settings → the zero-config DuckDuckGo backend (no key, no setup), so callers that
    don't thread settings still get a working default.
    """
    backend = getattr(settings, "search_backend", "ddgs") or "ddgs"
    if backend == "tavily":
        from zakcode.search.tavily_backend import TavilyBackend

        return TavilyBackend()
    if backend == "searxng":
        from zakcode.search.searxng_backend import SearxngBackend

        return SearxngBackend(base_url=getattr(settings, "searxng_url", None))
    from zakcode.search.ddgs_backend import DuckDuckGoBackend

    return DuckDuckGoBackend()


__all__ = [
    "BackendNotConfigured",
    "BackendUnavailable",
    "SearchBackend",
    "SearchError",
    "SearchItem",
    "make_search_backend",
]
