"""``web_search`` — query the web via the configured, swappable search backend."""

from __future__ import annotations

from zakcode.config import PermissionTier
from zakcode.providers.text_tools import defang_untrusted
from zakcode.search.base import SearchBackend, SearchError, SearchItem
from zakcode.tools.base import (
    ConcurrencyClass,
    Tool,
    ToolContext,
    ToolResult,
    ToolSpec,
)

#: Hard cap on results regardless of what the model asks for (keeps context bounded).
_MAX_RESULTS = 10
_DEFAULT_RESULTS = 5


def _render(items: list[SearchItem]) -> str:
    """A compact, numbered, model-facing list. Title/url/snippet are UNTRUSTED -> all defanged
    (the URL too: it is the easiest field for a poisoned result to stuff a forged tool-frame /
    chat-template token into, and on the native path this is the only defang)."""
    blocks: list[str] = []
    for i, item in enumerate(items, 1):
        title = defang_untrusted(item.title) or "(no title)"
        url = defang_untrusted(item.url)
        snippet = defang_untrusted(item.snippet)
        line = f"{i}. {title}\n   {url}"
        if snippet:
            line += f"\n   {snippet}"
        blocks.append(line)
    return "\n".join(blocks)


class WebSearchTool(Tool):
    """Search the web and return ranked results (title, URL, snippet).

    The backend (DuckDuckGo / Tavily / SearXNG) is chosen by ``ZAKCODE_SEARCH_BACKEND`` and
    injected at construction, so the tool itself is backend-agnostic.
    """

    spec = ToolSpec(
        name="web_search",
        description=(
            "Search the web and get back a ranked list of results (title, URL, snippet). "
            "Use it to find pages, docs, or current information; follow up with web_fetch to "
            "read a result. Returns up to 10 results."
        ),
        parameters={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query.",
                },
                "max_results": {
                    "type": "integer",
                    "description": "How many results to return (1-10, default 5).",
                    "minimum": 1,
                    "maximum": _MAX_RESULTS,
                },
            },
            "required": ["query"],
        },
        required_permission=PermissionTier.READ_ONLY,
        concurrency=ConcurrencyClass.READ_ONLY_SAFE,
    )

    def __init__(self, backend: SearchBackend) -> None:
        self._backend = backend

    async def execute(self, args: dict, ctx: ToolContext) -> ToolResult:
        query = args.get("query")
        if not isinstance(query, str) or not query.strip():
            return ToolResult.error("'query' is required and must be a non-empty string.")

        n = args.get("max_results")
        if not isinstance(n, int) or isinstance(n, bool) or n <= 0:
            n = _DEFAULT_RESULTS
        n = min(n, _MAX_RESULTS)

        try:
            items = await self._backend.search(query.strip(), max_results=n)
        except SearchError as exc:
            return ToolResult.error(str(exc), data={"backend": self._backend.name}, fix=exc.fix)
        except Exception as exc:  # noqa: BLE001 - handlers must never raise
            return ToolResult.error(
                f"web_search failed: {exc}", data={"backend": self._backend.name}
            )

        # Defensively enforce the cap on the OUTPUT too: a backend that ignores max_results
        # (or a compromised endpoint) must not blow the context budget the tool relies on.
        items = items[:n]

        if not items:
            return ToolResult.ok(
                "(no results)",
                data={"backend": self._backend.name, "query": query, "count": 0, "results": []},
                hint="no hits — try different/broader keywords",
            )

        data = {
            "backend": self._backend.name,
            "query": query,
            "count": len(items),
            "results": [item.model_dump() for item in items],
        }
        return ToolResult.ok(
            _render(items),
            data=data,
            hint="read a result with web_fetch <url>",
        )
