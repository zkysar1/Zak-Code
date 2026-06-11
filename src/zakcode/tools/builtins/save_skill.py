"""The ``save_skill`` tool — let the model author a reusable skill.

After working out a repeatable procedure, the model can persist it as a ``SKILL.md``
so the capability is available in future sessions — the model-driven half of the
"skill extraction" idea (the framework-driven half is a self-learning layer writing
the same files). The tool only *stores*; it makes no decision about when a skill is
worth creating, and a newly-authored skill becomes available on the **next** session
(discovery + the skills catalog are cache-stable per session, so the prompt's cached
prefix is never mutated mid-session).

It writes into the project skills dir (``<workspace>/.zakcode/skills``) the facade
passes in; ``zakcode.skills.save_skill`` validates the name so the path can never
escape that directory.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from zakcode.config import PermissionTier
from zakcode.skills import SkillError, save_skill
from zakcode.tools.base import ConcurrencyClass, Tool, ToolContext, ToolResult, ToolSpec


class SaveSkillTool(Tool):
    """Persist a new skill (SKILL.md) for future sessions."""

    spec = ToolSpec(
        name="save_skill",
        description=(
            "Save a reusable skill so a procedure you worked out is available in "
            "future sessions. Provide a kebab-case name, a one-line description of "
            "what it does and when to use it, and the markdown body (the steps). The "
            "skill becomes available in the next session, not the current one."
        ),
        parameters={
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Kebab-case skill name, e.g. 'release-checklist'.",
                },
                "description": {
                    "type": "string",
                    "description": "One line: what the skill does and when to use it.",
                },
                "body": {
                    "type": "string",
                    "description": "The markdown instructions the skill runs when invoked.",
                },
                "allowed_tools": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional advisory list of tools the skill uses.",
                },
                "extras": {
                    "type": "object",
                    "description": (
                        "Optional extra frontmatter (string or list-of-string values) "
                        "preserved verbatim — e.g. a host framework's cognitive metadata "
                        "like minimum_mode or companion_scripts."
                    ),
                },
                "overwrite": {
                    "type": "boolean",
                    "description": "Replace an existing skill of the same name (default false).",
                },
            },
            "required": ["name", "description", "body"],
        },
        required_permission=PermissionTier.WORKSPACE_WRITE,
        concurrency=ConcurrencyClass.NEVER_PARALLEL,
    )

    def __init__(self, skills_dir: str | Path) -> None:
        self._skills_dir = Path(skills_dir)

    async def execute(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        name = args.get("name")
        description = args.get("description")
        body = args.get("body")
        if not isinstance(name, str) or not name.strip():
            return ToolResult.error("'name' is required and must be a non-empty string.")
        if not isinstance(body, str) or not body.strip():
            return ToolResult.error("'body' is required and must be a non-empty string.")
        description = description if isinstance(description, str) else ""
        raw_tools = args.get("allowed_tools")
        allowed_tools = (
            [t for t in raw_tools if isinstance(t, str)] if isinstance(raw_tools, list) else None
        )
        overwrite = bool(args.get("overwrite", False))
        raw_extras = args.get("extras")
        extras: dict[str, str | list[str]] | None = None
        if isinstance(raw_extras, dict):
            # Keep only round-trippable shapes (str, or list of str); junk is dropped
            # rather than erroring so a slightly-off model call still saves the skill.
            extras = {}
            for key, value in raw_extras.items():
                if not isinstance(key, str) or not key.strip():
                    continue
                if (
                    isinstance(value, str)
                    or isinstance(value, list)
                    and all(isinstance(v, str) for v in value)
                ):
                    extras[key.strip()] = value
        try:
            path = save_skill(
                name.strip(),
                description.strip(),
                body,
                skills_dir=self._skills_dir,
                allowed_tools=allowed_tools,
                extras=extras,
                overwrite=overwrite,
            )
        except SkillError as exc:
            return ToolResult.error(str(exc))
        except OSError as exc:
            return ToolResult.error(f"failed to write skill: {exc}")
        return ToolResult.ok(
            f"Saved skill {name.strip()!r} to {path}. It will be available next session.",
            data={"name": name.strip(), "path": str(path)},
        )


__all__ = ["SaveSkillTool"]
