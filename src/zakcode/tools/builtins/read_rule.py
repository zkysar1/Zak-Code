"""The ``read_rule`` tool — fetch a project rule's full body by name, on demand.

The companion to lean rules (Vinheim Lever A). In lean mode the system prompt carries
only a one-line-per-rule INDEX (:meth:`zakcode.rules.RuleRegistry.render_index`), not
every rule's full body — so a rules-heavy "mind" stops paying every body on every cached
turn. This tool is how the model acts on that index: name a listed rule and its full text
comes back as the tool RESULT, which the model then applies.

Why a registry-backed tool rather than "just read the file":

* **Sandbox-independent.** Rules are discovered from bundled / user / project roots; the
  bundled and user roots sit OUTSIDE the workspace, so the sandboxed file-read tool cannot
  reach them. The :class:`~zakcode.rules.RuleRegistry` already holds every rule's content in
  memory (read at discovery), so resolving by name returns the body regardless of where the
  file lives.
* **By name, not path.** The index lists names; the model names a rule, no path bookkeeping.

Read-only: it only reads guidance the operator already authored into the prompt's cached
tier (in full-render mode) — so it is ``READ_ONLY`` tier and never prompts. Any *writes* the
rule's guidance calls for go through the ordinary file tools and their own permission gates.
"""

from __future__ import annotations

from typing import Any

from zakcode.config import PermissionTier
from zakcode.rules import RuleRegistry
from zakcode.tools.base import ConcurrencyClass, Tool, ToolContext, ToolResult, ToolSpec


class ReadRuleTool(Tool):
    """Load a discovered rule's full body by name and return it for the model to apply."""

    spec = ToolSpec(
        name="read_rule",
        description=(
            "Load a project rule's full text by name and apply it. The rules index in your "
            "context lists each rule by name with a one-line summary; call this when a listed "
            "rule looks relevant to the current step to get its full guidance."
        ),
        parameters={
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": (
                        "The rule name to load, exactly as it appears in the rules index."
                    ),
                },
            },
            "required": ["name"],
        },
        required_permission=PermissionTier.READ_ONLY,
        # A pure read of reference guidance — side-effect-free and fan-out friendly (the model
        # may pull several relevant rules at once). Unlike use_skill (NEVER_PARALLEL, because a
        # skill body is a procedure that steers control flow), a rule is reference text.
        concurrency=ConcurrencyClass.READ_ONLY_SAFE,
    )

    def __init__(self, registry: RuleRegistry) -> None:
        self._registry = registry

    async def execute(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        name = args.get("name")
        if not isinstance(name, str) or not name.strip():
            return ToolResult.error("'name' is required and must be a non-empty string.")
        name = name.strip()
        rule = self._registry.get(name)
        if rule is None:
            available = ", ".join(self._registry.names()) or "(none discovered)"
            return ToolResult.error(
                f"no rule named {name!r}.",
                fix=f"Use one of the available rule names: {available}.",
            )
        return ToolResult.ok(
            rule.content,
            data={"rule": rule.name},
            hint="Apply this rule's guidance to the current step.",
        )


__all__ = ["ReadRuleTool"]
