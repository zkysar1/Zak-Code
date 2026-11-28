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

import difflib
import re
from collections.abc import Callable
from typing import Any

from zakcode.config import PermissionTier
from zakcode.tools.base import (
    RULE_POINTER,
    ConcurrencyClass,
    Tool,
    ToolContext,
    ToolResult,
    ToolSpec,
)

#: Cap on a single returned body. Rules are already bounded per-file at discovery
#: (``MAX_RULE_FILE_CHARS``); this is a second belt so one pathological rule cannot dominate a
#: turn's context. Truncation is announced in the output rather than silent.
MAX_RULE_BODY_CHARS = 16000

#: The answer to EVERY repeat of a rule whose text is still in the conversation (ADR-0372,
#: ADR-0379). Numbered, so the model sees how often it has asked for the same text; it names the
#: plan's current step and restates the rule's summary. The text comes back with ``refresh=true``;
#: an ask right after this line gets :data:`JUST_ASKED` (ADR-0387). ADR-0372 gave this line once
#: and then the body, and the body was asked for again within minutes: measured 2026-11-10 on ten
#: worker sessions, one read three rules nine times each in 21 minutes with one plan update and no
#: other tool between them.
ALREADY_READ = (
    "[already read; ask {n}] Rule {name!r} is in your context from an earlier read_rule call, "
    "unchanged; this is ask {n} for it. Apply the text you have{step}. Reading it again changes "
    "nothing; call read_rule with refresh=true only if you need its exact wording.{summary}"
)

#: How far back the ask count is read (ADR-0379); past it, the pointer keeps this number.
_MAX_COUNTED_ASKS = 99

#: How alike (difflib ratio of the lowercased names) a rule name must be to a name that matches no
#: rule to be offered first in the not-found answer (ADR-0380). The skill catalog's suggest line
#: (ADR-0040): rule names are hyphenated phrases that share connective words, so at the tool
#: registry's 0.6 unrelated names already score as neighbours.
_CLOSEST_RULE_RATIO = 0.72

#: The answer to an ask that comes right after that rule's own pointer line, with no other tool
#: result between (ADR-0387). ADR-0381 sent the text again here. Measured 2026-11-17 on two worker
#: sessions: the same rule was asked again at once after 4 of the 7 texts sent again, against 8 of
#: 17 pointers and 1 of 4 exact-repeat refusals, and those 7 texts were 25,877 of the 39,711
#: characters of rule text sent while the model already held it. So this line says where the text
#: is and that the ask changed nothing; it opens like the pointer, so the count goes on.
JUST_ASKED = (
    "[already read; ask {n}] Rule {name!r} was answered by the tool result just before this call, "
    'and its text has not changed: it is in your context, from "# {name}" to '
    '"[end of rule {name!r}]". This ask changed nothing. Apply that text{step} and go on with '
    "your work; refresh=true returns it again only if you need its exact wording."
)

#: The answer to a plain ask for a rule pinned whole in the system prompt (``alwaysApply``,
#: ADR-0105, ADR-0387). Before, the first ask in each window sent the text the prompt already
#: held, and later ones alternated pointer and text: measured 2026-11-17 on one worker session,
#: three pinned rules sent 13,834 such characters, and the rule its host pinned for being read
#: most was asked seven times in three minutes. Opens like the pointer, so the count goes on.
PINNED = (
    "[already read; ask {n}] Rule {name!r} is pinned: its full text is in your system prompt, "
    'under the heading "## {name}", and it is always in force. Apply it{step}; refresh=true '
    "returns it here only if you need it as a tool result."
)


def _right_after_its_pointer(ctx: ToolContext, name: str) -> bool:
    """True when the conversation's most recent tool result is a pointer line for rule ``name``:
    the plain one, :data:`JUST_ASKED` or :data:`PINNED`, which all open the same way."""
    last = getattr(ctx, "last_tool_output", None)
    previous = last() if last is not None else None
    head = r"\[already read; ask [0-9]+\] Rule " + re.escape(repr(name)) + " "
    return previous is not None and re.match(head, previous) is not None


def _pointer_head(name: str, n: int) -> str:
    """How ask ``n``'s pointer for rule ``name`` begins: the count reads these back."""
    return f"[already read; ask {n}] Rule {name!r} "


def rule_text(name: str, rule: Any) -> str | None:
    """The text ``read_rule`` returns for ``rule`` under ``name``, or ``None`` for an empty body.

    One function renders it for the tool and for the rules a compaction carries (ADR-0386), so a
    carried text is the very string the pointer's check (ADR-0372) looks for."""
    body = (rule.content or "").strip()
    if not body:
        return None
    if len(body) > MAX_RULE_BODY_CHARS:
        body = body[:MAX_RULE_BODY_CHARS] + f"\n\n[truncated at {MAX_RULE_BODY_CHARS} chars]"
    # The closing line makes the text self-delimiting: a rule whose last lines were deleted on
    # disk is a prefix of the older text, and without it the check below read that as unchanged.
    return f"# {name}\n{body}\n[end of rule {name!r}]"


def _pointer(
    rule: Any,
    name: str,
    ctx: ToolContext,
    in_context: Callable[[str], bool] | None,
    template: str = ALREADY_READ,
) -> str:
    """The answer to a repeat: ask N, where N - 1 of this rule's pointers are still in the
    conversation. A conversation read, as ADR-0372's check is, so a compaction or an elision
    that takes the pointers away starts the count over with nothing to reset. ``template`` is
    the plain pointer, :data:`JUST_ASKED` or :data:`PINNED` (ADR-0387)."""
    n = 1
    while n <= _MAX_COUNTED_ASKS and in_context is not None and in_context(_pointer_head(name, n)):
        n += 1
    network = getattr(ctx, "task_network", None)
    current = network.current() if network is not None else None
    step = f" to your current step, {current.title!r}" if current is not None else ""
    summary = rule.summary()
    return template.format(
        n=n, name=name, step=step, summary=f" Its summary: {summary}" if summary else ""
    )


def _pinned_in_prompt(ctx: ToolContext, rule: Any) -> bool:
    """True when the system prompt holds this rule's whole text: pinned (ADR-0105), and its block
    in the prompt's rules uncut, so neither the per-rule cap nor the rules budget took any of it."""
    in_prompt = getattr(ctx, "in_prompt", None)
    body = (rule.content or "").strip()
    if not getattr(rule, "always_apply", False) or not body or in_prompt is None:
        return False
    return bool(in_prompt(f"## {rule.name}\n{body}"))


def _find_rule(registry: Any, name: str) -> tuple[str, Any] | None:
    """The rule ``name`` names, with the name it is registered under, or ``None``.

    Case-insensitive second pass: the model is copying a name out of prose, so a capitalisation
    slip should not read as "no such rule" when one plainly exists."""
    rule = registry.get(name)
    if rule is not None:
        return name, rule
    lowered = name.lower()
    for candidate in registry.names():
        if candidate.lower() == lowered:
            rule = registry.get(candidate)
            return (candidate, rule) if rule is not None else None
    return None


def _pointer_template(
    ctx: ToolContext, rule: Any, name: str, text: str, refresh: bool
) -> str | None:
    """The pointer that answers this ask, or ``None`` when the rule's text itself is sent.

    ONE decision, read by :meth:`ReadRuleTool.execute` and by :func:`would_point`, so the
    lookup-only bound (ADR-0399) never refuses an ask the tool would have answered with text."""
    if refresh:
        return None
    # A rule pinned whole in the system prompt is already there: say where, never send it again
    # unless asked for on purpose (ADR-0387).
    if _pinned_in_prompt(ctx, rule):
        return PINNED
    # A repeat while the text is in the conversation gets the numbered pointer (ADR-0379); one
    # right after this rule's own pointer gets the firmer line that says where the text is, not
    # the text again (ADR-0387). The body comes back only when asked for on purpose.
    # The text a compaction carried sits in a system message, not in a tool result, so the first
    # ask after it gets the text and later ones the pointer (ADR-0417). A context without the
    # narrower check keeps the ADR-0372 behaviour.
    held = getattr(ctx, "in_tool_results", None) or getattr(ctx, "in_context", None)
    if held is not None and held(text):
        return JUST_ASKED if _right_after_its_pointer(ctx, name) else ALREADY_READ
    return None


def would_point(args: dict[str, Any], ctx: ToolContext) -> bool:
    """True when ``read_rule`` would answer ``args`` with a pointer and send no rule text.

    The decision :meth:`ReadRuleTool.execute` makes, without its one side effect (the ask count a
    compaction reads, ADR-0386), so the loop can ask it of a call it has not run (ADR-0399).
    False for every other answer: a first read, ``refresh=true``, a name that matches no rule, an
    empty body, rules disabled."""
    registry = getattr(ctx, "rule_registry", None)
    name = args.get("name")
    if registry is None or not isinstance(name, str) or not name.strip():
        return False
    found = _find_rule(registry, name.strip())
    if found is None:
        return False
    name, rule = found
    text = rule_text(name, rule)
    refresh = args.get("refresh") is True
    return text is not None and _pointer_template(ctx, rule, name, text, refresh) is not None


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

        found = _find_rule(registry, name)
        if found is None:
            names = registry.names()
            available = ", ".join(names) or "(none discovered)"
            # The closest one or two names lead (ADR-0380): with only the full list, a session asked
            # again and again for a name a few letters off a rule the list held.
            by_lower = {candidate.lower(): candidate for candidate in names}
            close = difflib.get_close_matches(
                name.lower(), list(by_lower), n=2, cutoff=_CLOSEST_RULE_RATIO
            )
            lead = (
                f"Did you mean {' or '.join(repr(by_lower[c]) for c in close)}? " if close else ""
            )
            return ToolResult.error(
                f"no rule named {name!r}.",
                fix=f"{lead}Use one of the available rules: {available}.",
            )

        name, rule = found
        text = rule_text(name, rule)
        if text is None:
            return ToolResult.error(
                f"rule {name!r} exists but its body is empty.",
                fix="The rule file may be a stub; proceed without it.",
            )
        # Every ask for a rule that exists is counted, pointers included: a compaction carries
        # the most-asked texts across it (ADR-0386).
        note = getattr(ctx, "note_rule_read", None)
        if note is not None:
            note(name)
        template = _pointer_template(ctx, rule, name, text, args.get("refresh") is True)
        if template is not None:
            # Flagged, so the stuck ladder can count a run of asks answered this way (ADR-0399).
            pointer = _pointer(rule, name, ctx, getattr(ctx, "in_context", None), template)
            return ToolResult.ok(pointer, data={RULE_POINTER: True})
        return ToolResult.ok(text, verbatim=True)  # instructions, never clamped
