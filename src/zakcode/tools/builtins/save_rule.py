"""The ``save_rule`` tool — let the model record a rule it learned by experience.

The write counterpart to ``read_rule``. Until now the rules lane was read-only from a
turn's side: ``read_rule`` fetches a rule body by name, but nothing let a turn WRITE one,
so the only way a project rule could change was an out-of-band human edit of the rules
directory. That made "a changed world rule is learned through experience rather than
pushed" unachievable by construction — an agent could perceive that a rule had changed and
had nowhere to put what it learned.

The gap was not "no durable write exists". ``save_skill`` is durable and turn-callable, but
it is PROCEDURE-shaped: name / description / body / tools resolving to an *invocable*
skill, read back by ``use_skill``, which expects to EXECUTE what it reads. A knowledge
record executed as a procedure is a real failure mode, not a hypothetical, so routing
learned facts through that lane would overload it. Rules are the fact-shaped store, and
they already have a reader; this adds the missing half rather than a new store.

Like ``save_skill``, the tool only *stores*. A newly-authored rule becomes available on the
**next** session: rule discovery and the rendered rules index are cache-stable per session,
so mutating the live registry would change the prompt's cached prefix mid-session — the
exact invariant ``save_skill`` defers for. This is a deliberate constraint, not an
oversight, and the tool description says so to the model.

It writes into the project rules dir (``<workspace>/.zakcode/rules``) the facade passes in;
``zakcode.rules.save_rule`` validates the name so the path can never escape that directory.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from zakcode.config import PermissionTier
from zakcode.rules import RuleError, save_rule
from zakcode.tools.base import ConcurrencyClass, Tool, ToolContext, ToolResult, ToolSpec


class SaveRuleTool(Tool):
    """Persist a rule the model learned, for future sessions."""

    spec = ToolSpec(
        name="save_rule",
        description=(
            "Record a durable project rule you have learned, so it applies in future "
            "sessions. Use this when you discover how this project actually works — a "
            "convention, a constraint, a corrected assumption — and a future session would "
            "get it wrong without being told. Provide a kebab-case name, a one-line "
            "description (this is what future sessions see in the rules index), and the "
            "markdown body. Read it back with read_rule. The rule takes effect in the next "
            "session, not the current one. This is for facts about the project; use "
            "save_skill instead for a repeatable procedure you want to invoke."
        ),
        parameters={
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Kebab-case rule name, e.g. 'deploy-target-is-dev'.",
                },
                "description": {
                    "type": "string",
                    "description": (
                        "One line, shown in the rules index — make it specific enough that a "
                        "future session can tell whether the rule is relevant without "
                        "reading the body."
                    ),
                },
                "body": {
                    "type": "string",
                    "description": "The markdown rule text, including why it is true.",
                },
                "overwrite": {
                    "type": "boolean",
                    "description": "Replace an existing rule of the same name (default false).",
                },
            },
            "required": ["name", "description", "body"],
        },
        required_permission=PermissionTier.WORKSPACE_WRITE,
        concurrency=ConcurrencyClass.NEVER_PARALLEL,
    )

    def __init__(self, rules_dir: str | Path) -> None:
        self._rules_dir = Path(rules_dir)

    async def execute(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        name = args.get("name")
        description = args.get("description")
        body = args.get("body")
        if not isinstance(name, str) or not name.strip():
            return ToolResult.error("'name' is required and must be a non-empty string.")
        if not isinstance(body, str) or not body.strip():
            return ToolResult.error("'body' is required and must be a non-empty string.")
        # A rule with no description still loads, but it reaches future sessions as a blank
        # line in the index — present and unfindable. Require it rather than defaulting to
        # "" the way save_skill can, because for a rule the index line IS the discovery path.
        if not isinstance(description, str) or not description.strip():
            return ToolResult.error(
                "'description' is required and must be a non-empty string.",
                fix="It is the one line future sessions see in the rules index.",
            )
        overwrite = bool(args.get("overwrite", False))
        try:
            path = save_rule(
                name.strip(),
                description.strip(),
                body,
                rules_dir=self._rules_dir,
                overwrite=overwrite,
            )
        except RuleError as exc:
            return ToolResult.error(str(exc))
        except OSError as exc:
            return ToolResult.error(f"failed to write rule: {exc}")
        return ToolResult.ok(
            f"Saved rule {name.strip()!r} to {path}. It applies from the next session; "
            f"read it back with read_rule once it is discovered."
        )
