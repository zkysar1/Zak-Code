"""The ``tool_search`` tool — surface lazily-registered tools on demand (M5).

When many tools are available (e.g. several MCP servers), exposing every schema to
the model bloats the prompt and degrades quality. So beyond a budget those tools are
registered *inactive* (hidden). ``tool_search`` lets the model find hidden tools by
keyword and **activate** them, so their schemas appear on the next turn — just-in-time
discovery instead of preloading.

The tool holds a reference to the live :class:`~zakcode.tools.base.ToolRegistry`
(the facade constructs it with the same registry the loop uses) and activates within
a budget so the exposed set stays small. It is itself always active, read-only, and
never raises.
"""

from __future__ import annotations

from typing import Any

from zakcode.config import PermissionTier
from zakcode.tools.base import (
    ConcurrencyClass,
    Tool,
    ToolContext,
    ToolRegistry,
    ToolResult,
    ToolSpec,
)

#: Default ceiling on how many MCP tools are exposed to the model at once. Built-in tools do
#: not count against it. They used to: the budget capped EVERY exposed tool, and once the
#: built-ins alone reached 25 (2026-09-10) there was never room, so the model was told a match
#: could not be surfaced "because the tool budget is full" and no MCP tool could ever be used.
DEFAULT_TOOL_BUDGET = 25


def _first_line(text: str) -> str:
    return text.strip().splitlines()[0] if text.strip() else ""


def _is_mcp(name: str) -> bool:
    """Whether ``name`` is an MCP tool's registry name (``mcp__<server>__<tool>``)."""
    return name.startswith("mcp__")


def _matches(query: str, name: str, description: str) -> bool:
    """Whether a tool matches ``query`` — any whitespace-separated term in name/desc."""
    haystack = f"{name}\n{description}".lower()
    terms = [t for t in query.lower().split() if t]
    if not terms:
        return False
    return any(term in haystack for term in terms)


class ToolSearchTool(Tool):
    """Search hidden tools by keyword and activate the matches (within a budget)."""

    spec = ToolSpec(
        name="tool_search",
        description=(
            "Search for additional tools by name or keyword and make them available. "
            "Some tools (documents, PDFs and images, and tools from connected MCP servers) "
            "are hidden to keep the toolset small; call this with a tool's name or a query "
            "describing what you need (e.g. 'create_docx' or 'github issues') to surface and "
            "activate matching tools. They become callable on the next step."
        ),
        parameters={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Keywords describing the capability you need.",
                }
            },
            "required": ["query"],
        },
        required_permission=PermissionTier.READ_ONLY,
        concurrency=ConcurrencyClass.READ_ONLY_SAFE,
    )

    def __init__(self, registry: ToolRegistry, *, budget: int = DEFAULT_TOOL_BUDGET) -> None:
        self._registry = registry
        self._budget = budget

    async def execute(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        query = args.get("query")
        if not isinstance(query, str) or not query.strip():
            return ToolResult.error("'query' is required and must be a non-empty string.")

        active = set(self._registry.active_names())
        # Candidate = registered, not currently exposed, AND not blocked by the operator's
        # tool-exposure filter (Step 4) — surfacing a filtered-out tool would activate it,
        # waste a budget slot, and tell the model it is "now callable" when definitions() still
        # hides it and the execution seam rejects it. Never offer what can't actually be used.
        candidates = [
            n
            for n in self._registry.names()
            if n not in active and self._registry.exposure_allows(n)
        ]
        matched: list[str] = []
        for name in candidates:
            tool = self._registry.get(name)
            if tool is not None and _matches(query, name, tool.spec.description):
                matched.append(name)

        if not matched:
            return ToolResult.ok(
                f"No additional tools matched {query!r}. "
                f"{len(active)} tool(s) are already available."
            )

        # The budget is for MCP tools only. A hidden built-in (ADR-0228) loads whatever the
        # budget, which is 0 in a session without MCP; there are only a handful of them.
        builtin = [n for n in matched if not _is_mcp(n)]
        mcp = [n for n in matched if _is_mcp(n)]
        # The budget counts the MCP tools already exposed; built-ins never use it up.
        available = max(0, self._budget - sum(1 for n in active if _is_mcp(n)))
        # If the budget is full, make room by evicting previously-surfaced MCP tools
        # that aren't part of this match — so the model is never wedged, unable to
        # reach a needed tool. Builtins (no ``mcp__`` prefix) are never evicted.
        evicted: list[str] = []
        need = len(mcp) - available
        if need > 0:
            evictable = [n for n in active if _is_mcp(n) and n not in matched]
            for name in evictable[:need]:
                self._registry.deactivate(name)
                evicted.append(name)
            available += len(evicted)

        activated = builtin + mcp[:available]
        deferred = mcp[available:]
        for name in activated:
            self._registry.activate(name)

        lines: list[str] = []
        if activated:
            lines.append(f"Activated {len(activated)} tool(s) (now callable):")
            for name in activated:
                tool = self._registry.get(name)
                desc = _first_line(tool.spec.description) if tool is not None else ""
                lines.append(f"  - {name}: {desc}" if desc else f"  - {name}")
        if evicted:
            lines.append(
                f"(made room by hiding {len(evicted)} previously-surfaced tool(s); "
                "search again to bring them back)"
            )
        if deferred:
            lines.append(
                f"{len(deferred)} more matched, but at most {self._budget} MCP tools can be "
                "available at once; finish with the active tools or refine your query."
            )
        return ToolResult.ok(
            "\n".join(lines),
            data={"activated": activated, "deferred": deferred, "evicted": evicted},
        )


__all__ = ["ToolSearchTool", "DEFAULT_TOOL_BUDGET"]
