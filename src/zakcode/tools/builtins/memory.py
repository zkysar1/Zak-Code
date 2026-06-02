"""The ``remember`` and ``recall`` tools — model-driven cross-session memory.

These give the model an explicit way to persist a durable note and to look one up,
on top of a :class:`~zakcode.memory.MemoryProvider`. Each tool holds the provider
the facade constructed (mirroring how ``tool_search`` holds the live registry), so
the agent loop and tools share one store. ``remember`` is a write (gated at
``WORKSPACE_WRITE`` and never run in parallel, since it mutates the store);
``recall`` is read-only.

Automatic recall (relevant memories surfaced without an explicit call) is handled
separately by :class:`~zakcode.memory.MemoryRecallHook`; these tools are the
deliberate, model-initiated path.
"""

from __future__ import annotations

from typing import Any

from zakcode.config import PermissionTier
from zakcode.memory import MemoryProvider
from zakcode.secrets import redact_secrets
from zakcode.tools.base import ConcurrencyClass, Tool, ToolContext, ToolResult, ToolSpec


class RememberTool(Tool):
    """Persist a durable memory for future sessions."""

    spec = ToolSpec(
        name="remember",
        description=(
            "Save a durable note to cross-session memory so it can be recalled in "
            "future sessions (e.g. a user preference, a project fact, or a lesson "
            "learned). Use sparingly for things genuinely worth persisting."
        ),
        parameters={
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": (
                        "The fact or note to remember (a short, self-contained sentence)."
                    ),
                },
                "kind": {
                    "type": "string",
                    "description": "Optional category, e.g. 'preference', 'fact', 'lesson'.",
                },
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional keywords to aid later recall.",
                },
            },
            "required": ["text"],
        },
        required_permission=PermissionTier.WORKSPACE_WRITE,
        concurrency=ConcurrencyClass.NEVER_PARALLEL,
    )

    def __init__(self, provider: MemoryProvider, *, source: str = "") -> None:
        self._provider = provider
        self._source = source

    async def execute(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        text = args.get("text")
        if not isinstance(text, str) or not text.strip():
            return ToolResult.error("'text' is required and must be a non-empty string.")
        kind = args.get("kind")
        kind = kind if isinstance(kind, str) and kind.strip() else "note"
        raw_tags = args.get("tags")
        tags = [t for t in raw_tags if isinstance(t, str)] if isinstance(raw_tags, list) else []
        # Never persist credential-shaped text into a long-lived, re-injected store
        # (docs/GUARDRAILS.md §6). Scrub before storing and tell the model it happened.
        scrubbed, redactions = redact_secrets(text.strip())
        try:
            record = self._provider.add(scrubbed, kind=kind, tags=tags, source=self._source)
        except Exception as exc:  # noqa: BLE001 — a store failure is a tool error, not a crash
            return ToolResult.error(f"failed to store memory: {type(exc).__name__}: {exc}")
        note = f" Redacted {redactions} suspected secret(s) before storing." if redactions else ""
        return ToolResult.ok(
            f"Remembered (id={record.id}).{note}",
            data={"id": record.id, "kind": record.kind, "tags": record.tags},
        )


class RecallTool(Tool):
    """Search cross-session memory for relevant notes."""

    spec = ToolSpec(
        name="recall",
        description=(
            "Search cross-session memory for notes relevant to a query and return the "
            "best matches. Use this to check whether something was learned or decided "
            "in a previous session."
        ),
        parameters={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "What to look for (keywords or a short phrase).",
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of memories to return (default 5).",
                },
            },
            "required": ["query"],
        },
        required_permission=PermissionTier.READ_ONLY,
        concurrency=ConcurrencyClass.READ_ONLY_SAFE,
    )

    def __init__(self, provider: MemoryProvider, *, default_limit: int = 5) -> None:
        self._provider = provider
        self._default_limit = default_limit

    async def execute(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        query = args.get("query")
        if not isinstance(query, str) or not query.strip():
            return ToolResult.error("'query' is required and must be a non-empty string.")
        raw_limit = args.get("limit")
        limit = (
            raw_limit if isinstance(raw_limit, int) and not isinstance(raw_limit, bool) else None
        )
        limit = limit if limit and limit > 0 else self._default_limit
        try:
            records = self._provider.search(query.strip(), limit=limit)
        except Exception as exc:  # noqa: BLE001 — a store failure is a tool error, not a crash
            return ToolResult.error(f"failed to search memory: {type(exc).__name__}: {exc}")
        if not records:
            return ToolResult.ok("No relevant memories found.", data={"count": 0, "matches": []})
        lines = [f"{len(records)} memory(ies) found:"]
        for r in records:
            lines.append(f"- [{r.kind}] {r.text}")
        return ToolResult.ok(
            "\n".join(lines),
            data={
                "count": len(records),
                "matches": [{"id": r.id, "text": r.text, "kind": r.kind} for r in records],
            },
        )


__all__ = ["RememberTool", "RecallTool"]
