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

#: Default ceiling on how many tools are exposed (active) to the model at once.
DEFAULT_TOOL_BUDGET = 25


def _first_line(text: str) -> str:
    return text.strip().splitlines()[0] if text.strip() else ""


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
            "Search for additional tools by keyword and make them available. Many "
            "tools (e.g. from connected MCP servers) are hidden by default to keep the "
            "toolset small; call this with a query describing what you need (e.g. "
            "'github issues' or 'database query') to surface and activate matching "
            "tools. They become callable on the next step."
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
        # Candidate = registered but not currently exposed.
        candidates = [n for n in self._registry.names() if n not in active]
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

        available = max(0, self._budget - len(active))
        activated = matched[:available]
        deferred = matched[available:]
        for name in activated:
            self._registry.activate(name)

        lines: list[str] = []
        if activated:
            lines.append(f"Activated {len(activated)} tool(s) (now callable):")
            for name in activated:
                tool = self._registry.get(name)
                desc = _first_line(tool.spec.description) if tool is not None else ""
                lines.append(f"  - {name}: {desc}" if desc else f"  - {name}")
        if deferred:
            lines.append(
                f"{len(deferred)} more matched but the tool budget ({self._budget}) is full; "
                "narrow your query or finish with the active tools."
            )
        return ToolResult.ok("\n".join(lines), data={"activated": activated, "deferred": deferred})


__all__ = ["ToolSearchTool", "DEFAULT_TOOL_BUDGET"]
