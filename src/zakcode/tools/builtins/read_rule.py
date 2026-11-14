"""The ``read_rule`` tool — fetch ONE rule body by name (Vinheim Lever A, chunk 2).

This is the retrieval half of ``lean_rules``. Chunk 1 (commit 5e9e98f) added
:meth:`~zakcode.rules.RuleRegistry.render_index`, which puts every rule's *name + summary +
path* in the cached prefix instead of every rule's full *body*. On a rules-heavy "mind" that
is both far smaller and strictly more complete — the full render is bounded by
``MAX_RULES_TOTAL_CHARS`` and silently drops rules past the budget, while the index lists all
of them. But an index is only worth having if fetching a body is cheap and unambiguous, and
until now retrieval fell back to the generic ``read_file`` tool with the path from the index.

A named ``read_rule(name)`` removes the path-resolution step and, more importantly, makes the
affordance explicit *in the tool schema itself* — an expressive signature guides correct use
without needing examples. It also decouples retrieval from the on-disk layout: the model asks
for a rule by the name it was shown, and the registry resolves it.

Read-only by construction: the tool reads from the already-discovered in-memory
:class:`~zakcode.rules.RuleRegistry` and touches no filesystem path the model supplies, so it
cannot be steered into reading an arbitrary file. ``READ_ONLY`` tier, never prompts.

The registry arrives on the :class:`ToolContext`; it is ``None`` when rules are disabled, and
the tool degrades to a clean error rather than raising.
"""

from __future__ import annotations

from typing import Any

from zakcode.config import PermissionTier
from zakcode.tools.base import ConcurrencyClass, Tool, ToolContext, ToolResult, ToolSpec

#: Cap on a single returned body. Rules are already bounded per-file at discovery
#: (``MAX_RULE_FILE_CHARS``); this is a second belt so one pathological rule cannot dominate a
#: turn's context. Truncation is announced in the output rather than silent.
MAX_RULE_BODY_CHARS = 16000

#: The answer to a repeat of a rule whose text is still in the conversation (ADR-0372), given at
#: most once per rule while it stays there: a read after it returns the body.
ALREADY_READ = (
    "[already read] Rule {name!r} is in your context from an earlier read_rule call, unchanged. "
    "Apply it from there. If you need the text itself, read it again: a read after this line "
    "returns it in full."
)


class ReadRuleTool(Tool):
    """Return one rule's full body by name, from the session's discovered rule registry."""

    spec = ToolSpec(
        name="read_rule",
        description=(
            "Read one project rule's full text by name, for detail its one-line summary in "
            "the rules index does not give. Apply a rule's summary directly; call this only "
            "when you need that detail, and at most once per rule: the text then stays in "
            "your context. Pinned rules are already in your prompt in full. Use the name "
            "exactly as it appears in the index."
        ),
        parameters={
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": (
                        "The rule name to read, exactly as it appears in the rules index."
                    ),
                },
            },
            "required": ["name"],
        },
        required_permission=PermissionTier.READ_ONLY,
        # A pure in-memory lookup with no side effects — safe to fan out alongside other reads
        # when the model wants several rules at once (the common case after scanning the index).
        concurrency=ConcurrencyClass.READ_ONLY_SAFE,
    )

    async def execute(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        registry = getattr(ctx, "rule_registry", None)
        if registry is None:
            return ToolResult.error(
                "rules are not enabled in this session, so read_rule is unavailable."
            )
        name = args.get("name")
        if not isinstance(name, str) or not name.strip():
            return ToolResult.error("'name' is required and must be a non-empty string.")
        name = name.strip()

        rule = registry.get(name)
        if rule is None:
            # Case-insensitive second pass: the model is copying a name out of prose, so a
            # capitalisation slip should not read as "no such rule" when one plainly exists.
            lowered = name.lower()
            for candidate in registry.names():
                if candidate.lower() == lowered:
                    rule = registry.get(candidate)
                    name = candidate
                    break

        if rule is None:
            available = ", ".join(registry.names()) or "(none discovered)"
            return ToolResult.error(
                f"no rule named {name!r}.",
                fix=f"Use one of the available rules: {available}.",
            )

        body = (rule.content or "").strip()
        if not body:
            return ToolResult.error(
                f"rule {name!r} exists but its body is empty.",
                fix="The rule file may be a stub; proceed without it.",
            )
        if len(body) > MAX_RULE_BODY_CHARS:
            body = body[:MAX_RULE_BODY_CHARS] + f"\n\n[truncated at {MAX_RULE_BODY_CHARS} chars]"
        # The closing line makes the text self-delimiting: a rule whose last lines were deleted on
        # disk is a prefix of the older text, and without it the check below read that as unchanged.
        text = f"# {name}\n{body}\n[end of rule {name!r}]"
        pointer = ALREADY_READ.format(name=name)
        in_context = getattr(ctx, "in_context", None)
        # Once per rule: a session that asks again after the pointer wants the text, and answering
        # it with the pointer again kept it asking until the doom-loop guard refused (ADR-0372).
        if in_context is not None and in_context(text) and not in_context(pointer):
            return ToolResult.ok(pointer)
        return ToolResult.ok(text, verbatim=True)  # instructions, never clamped
