"""Vendor-agnostic web-search backend abstraction.

A "mind" running on Zak Code reaches the web through a *swappable* search backend, the same
way model access goes through the provider layer — so switching from DuckDuckGo to Tavily or a
self-hosted SearXNG is a config change (``ZAKCODE_SEARCH_BACKEND``), never a code change. This
ABC stays LOCAL to ``zakcode`` (it is web/zakcode-specific, not a reusable vendor seam like the
LLM provider package), and is deliberately tiny: one ``search`` coroutine returning a common
result shape, so the ``web_search`` tool never knows which backend is active.

A concrete backend lazy-imports its transport (``ddgs`` / ``httpx``) only when it actually runs
a search, so importing this package never requires the optional ``[web]`` deps; a missing dep or
missing config surfaces as a :class:`SearchError` the tool renders as a clean, actionable error
(never an exception to the loop).
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from pydantic import BaseModel


class SearchItem(BaseModel):
    """One search hit: a title, its URL, and a short snippet (all may be empty)."""

    title: str = ""
    url: str = ""
    snippet: str = ""


class SearchError(Exception):
    """A search backend failed. Carries an optional ``fix`` (a concrete remedy the tool surfaces).

    Backends raise this (never let a transport/network exception escape) so the ``web_search``
    tool can turn any failure into a ``ToolResult.error`` with a next-step hint.
    """

    def __init__(self, message: str, *, fix: str | None = None) -> None:
        super().__init__(message)
        self.fix = fix


class BackendUnavailable(SearchError):
    """The backend's optional dependency (e.g. ``ddgs``/``httpx``) is not installed."""


class BackendNotConfigured(SearchError):
    """The backend needs configuration that is missing (an API key or a base URL)."""


class SearchBackend(ABC):
    """A web-search backend. Concrete backends set :attr:`name` and implement :meth:`search`."""

    #: Short identifier (matches the ``ZAKCODE_SEARCH_BACKEND`` value that selects it).
    name: str = "search"

    @abstractmethod
    async def search(self, query: str, *, max_results: int = 5) -> list[SearchItem]:
        """Run ``query`` and return up to ``max_results`` hits.

        MUST NOT raise transport/network exceptions — wrap any failure in :class:`SearchError`
        (or a subclass) so the calling tool can render a clean error with a remedy.
        """
        raise NotImplementedError
