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

import re
from collections.abc import Callable
from typing import Any

from zakcode.config import PermissionTier
from zakcode.tools.base import ConcurrencyClass, Tool, ToolContext, ToolResult, ToolSpec

#: Cap on a single returned body. Rules are already bounded per-file at discovery
#: (``MAX_RULE_FILE_CHARS``); this is a second belt so one pathological rule cannot dominate a
#: turn's context. Truncation is announced in the output rather than silent.
MAX_RULE_BODY_CHARS = 16000

#: The answer to EVERY repeat of a rule whose text is still in the conversation (ADR-0372,
#: ADR-0379). Numbered, so the model sees how often it has asked for the same text; it names the
#: plan's current step and restates the rule's summary. The text comes back with ``refresh=true``,
#: or for an ask right after this line (ADR-0381). ADR-0372 gave this line once and then the body,
#: and the body was asked for again within minutes: measured 2026-11-10 on ten worker sessions,
#: one read three rules nine times each in 21 minutes with one plan update and no other tool
#: between them.
ALREADY_READ = (
    "[already read; ask {n}] Rule {name!r} is in your context from an earlier read_rule call, "
    "unchanged; this is ask {n} for it. Apply the text you have{step}. Reading it again changes "
    "nothing; call read_rule with refresh=true only if you need its exact wording.{summary}"
)

#: How far back the ask count is read (ADR-0379); past it, the pointer keeps this number.
_MAX_COUNTED_ASKS = 99

#: The lead on the text sent again for an ask that comes right after that rule's own pointer, with
#: no other tool result between (ADR-0381): the pointer did not land. ADR-0379 answered it with the
#: next pointer, and in a trial two sessions asked again and again until the exact-repeat guard
#: refused them; ADR-0372's text at that ask is the measured baseline that the guard refused less.
ASKED_AGAIN = (
    "[asked again right after its pointer] The full text of rule {name!r} follows, unchanged. "
    "Apply it to your current step; the next ask for it gets a pointer."
)


def _right_after_its_pointer(ctx: ToolContext, name: str) -> bool:
    """True when the conversation's most recent tool result is a pointer for rule ``name``."""
    last = getattr(ctx, "last_tool_output", None)
    previous = last() if last is not None else None
    head = r"\[already read; ask [0-9]+\] Rule " + re.escape(repr(name)) + " "
    return previous is not None and re.match(head, previous) is not None


def _pointer_head(name: str, n: int) -> str:
    """How ask ``n``'s pointer for rule ``name`` begins: the count reads these back."""
    return f"[already read; ask {n}] Rule {name!r} "


def _pointer(rule: Any, name: str, ctx: ToolContext, in_context: Callable[[str], bool]) -> str:
    """The answer to a repeat: ask N, where N - 1 of this rule's pointers are still in the
    conversation. A conversation read, as ADR-0372's check is, so a compaction or an elision
    that takes the pointers away starts the count over with nothing to reset."""
    n = 1
    while n <= _MAX_COUNTED_ASKS and in_context(_pointer_head(name, n)):
        n += 1
    network = getattr(ctx, "task_network", None)
    current = network.current() if network is not None else None
    step = f" to your current step, {current.title!r}" if current is not None else ""
    summary = rule.summary()
    return ALREADY_READ.format(
        n=n, name=name, step=step, summary=f" Its summary: {summary}" if summary else ""
    )


class ReadRuleTool(Tool):
    """Return one rule's full body by name, from the session's discovered rule registry."""

    spec = ToolSpec(
        name="read_rule",
        description=(
            "Read one project rule's full text by name. Your context lists the available rules "
            "as an index (name + one-line summary); call this the first time a rule's summary "
            "looks relevant, then apply the rule you get back. A rule's text stays in your "
            "context and does not change: read each rule once, and when its summary looks "
            "relevant again, at a new plan step too, apply the text you already have. Use the "
            "name exactly as it appears in that index."
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
                "refresh": {
                    "type": "boolean",
                    "description": (
                        "true returns the full text again although it is still in your "
                        "context. Only for when you need its exact wording: the text does not "
                        "change between reads."
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
        in_context = getattr(ctx, "in_context", None)
        # A repeat while the text is in the conversation gets the numbered pointer (ADR-0379); the
        # body comes back when asked for on purpose, or for an ask right after this rule's own
        # pointer, which did not land (ADR-0381).
        if in_context is not None and args.get("refresh") is not True and in_context(text):
            if _right_after_its_pointer(ctx, name):
                return ToolResult.ok(f"{ASKED_AGAIN.format(name=name)}\n{text}", verbatim=True)
            return ToolResult.ok(_pointer(rule, name, ctx, in_context))
        return ToolResult.ok(text, verbatim=True)  # instructions, never clamped
