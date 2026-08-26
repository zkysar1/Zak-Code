"""``web_search`` — query the web via the configured, swappable search backend.

A query LEAVES THE MACHINE (it goes to a third-party search engine), so this tool carries
privacy floors of its own (ADR-0019), mechanical where possible:

* **Length cap** — a real search query is a distilled question; long strings are how file
  contents, code, and private text get pasted into an external engine. Refused, never
  truncated (truncating would still send the head).
* **Saved-secret screen** — a query containing a saved secret VALUE (the shared
  :class:`SecretsProvider`'s scrub is the detector) is refused outright.
* **Credential-shape screen** — a query containing credential-shaped text (API keys, cloud
  tokens, PEM blocks, ``token=…`` assignments; :func:`zakcode.secrets.redact_secrets` is the
  detector) is refused outright.

The semantic half — no proprietary code, no client names, no personal data — cannot be
checked mechanically; it lives as hard rules in the tool description and the system prompt's
Safety section, where every model reads them.
"""

from __future__ import annotations

from zakcode.config import PermissionTier
from zakcode.providers.text_tools import defang_untrusted
from zakcode.search.base import SearchBackend, SearchError, SearchItem
from zakcode.secrets import redact_secrets
from zakcode.tools.base import (
    ConcurrencyClass,
    Tool,
    ToolContext,
    ToolResult,
    ToolSpec,
)
from zakcode.tools.builtins._secrets import SECRET_PLACEHOLDER_RE, SecretsProvider

#: Hard cap on results regardless of what the model asks for (keeps context bounded).
_MAX_RESULTS = 10
_DEFAULT_RESULTS = 5

#: Hard cap on query length (chars). Real distilled queries run well under this; longer
#: strings are pastes. Refuse-not-truncate: see the module docstring.
_MAX_QUERY_CHARS = 400


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
            "read a result. Returns up to 10 results. PRIVACY (hard rules): the query is sent "
            "to a third-party search engine — use only generic, public-vocabulary terms. "
            "Never include secrets or tokens, private or proprietary code, file contents, "
            "client or personal data, or internal hostnames/paths; search the generic form "
            "of an error instead of pasting it verbatim."
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

    def __init__(self, backend: SearchBackend, *, secrets: SecretsProvider | None = None) -> None:
        self._backend = backend
        # Always hold a provider (an empty one when unconfigured), mirroring web_fetch: the
        # saved-secret screen is then unconditional and inert-when-empty (scrub = identity).
        self._secrets = secrets or SecretsProvider(None)

    async def execute(self, args: dict, ctx: ToolContext) -> ToolResult:
        query = args.get("query")
        if not isinstance(query, str) or not query.strip():
            return ToolResult.error("'query' is required and must be a non-empty string.")
        query = query.strip()

        # Privacy floors (module docstring): the query leaves the machine, so screen it
        # BEFORE any backend sees it. Refusals name the rewrite, not just the rejection.
        if len(query) > _MAX_QUERY_CHARS:
            return ToolResult.error(
                f"query too long ({len(query)} chars > {_MAX_QUERY_CHARS}). A search query "
                "is a distilled question, not pasted content — long strings are how private "
                "text leaks to a third-party engine.",
                fix="rewrite the query as a short, generic phrasing of what you need to find",
            )
        if self._secrets.scrub(query) != query:
            return ToolResult.error(
                "the query contains a saved secret VALUE; queries go to a third-party "
                "search engine and must never carry secrets.",
                fix="rewrite the query without the secret (refer to secrets only by name)",
            )
        # Shape screen runs with {{secret:NAME}} placeholders stripped: the placeholder is
        # the SAFE form (a name, no value) yet its ``secret:NAME`` spelling matches the
        # redactor's key-value pattern — without the strip, the sanctioned form would be
        # the one refused.
        if redact_secrets(SECRET_PLACEHOLDER_RE.sub("", query))[1] > 0:
            return ToolResult.error(
                "the query contains credential-shaped text (an API key / token / key-value "
                "credential); queries go to a third-party search engine and must never "
                "carry secrets.",
                fix="rewrite the query without the credential material",
            )

        n = args.get("max_results")
        if not isinstance(n, int) or isinstance(n, bool) or n <= 0:
            n = _DEFAULT_RESULTS
        n = min(n, _MAX_RESULTS)

        try:
            items = await self._backend.search(query, max_results=n)
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
