"""The ``use_skill`` tool — let the MODEL invoke a discovered skill (M7).

Discovery surfaces an L0 catalog (name + description) in the system prompt; this tool is how
the model acts on it: name a skill and its full L1 instructions come back as the tool RESULT,
which the model then follows. It is the model-facing counterpart to the human CLI ``/<name>``
path — the "future model-facing tool" :meth:`zakcode.Agent.invoke_skill` anticipates — and the
basis for skill *chaining*: a skill body whose step says "now use the X skill" is carried out by
the model calling ``use_skill`` again.

Two deliberate properties:

* **Result, not session surgery.** The body is returned as the tool's ``output`` (the natural
  place fresh context attaches), NOT injected as a user message mid-turn — so model-driven
  invocation can never reorder the assistant/tool-result exchange the loop is assembling.
* **Read-only.** The tool only reads a skill file and fires the observe-only ``ON_SKILL_SELECTED``
  signal; any *writes* the instructions call for go through the ordinary file tools and their own
  permission gates. So it is ``READ_ONLY`` tier and never prompts.

The :class:`~zakcode.tools.base.SkillResolver` is supplied on the :class:`ToolContext`; it is
``None`` when skills are disabled (or for a sub-agent), and the tool degrades to a clean error.
"""

from __future__ import annotations

from typing import Any

from zakcode.config import PermissionTier
from zakcode.tools.base import ConcurrencyClass, Tool, ToolContext, ToolResult, ToolSpec


class UseSkillTool(Tool):
    """Load a discovered skill's instructions by name and return them for the model to follow."""

    spec = ToolSpec(
        name="use_skill",
        description=(
            "Load a skill's full step-by-step instructions by name and follow them. Call this "
            "when one of the skills listed in your context fits the task. The skill's body is "
            "returned as this tool's result — act on it. Skills can chain: if a skill's steps "
            "tell you to use another skill, call use_skill again with that name."
        ),
        parameters={
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "The skill name to load, exactly as it appears in the catalog.",
                },
            },
            "required": ["name"],
        },
        required_permission=PermissionTier.READ_ONLY,
        # Loading instructions changes the turn's control flow; it is not a fan-out-friendly
        # read, and the model chains skills sequentially anyway. Run it on its own.
        concurrency=ConcurrencyClass.NEVER_PARALLEL,
    )

    async def execute(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        resolver = ctx.skill_resolver
        if resolver is None:
            return ToolResult.error(
                "skills are not enabled in this session, so use_skill is unavailable."
            )
        name = args.get("name")
        if not isinstance(name, str) or not name.strip():
            return ToolResult.error("'name' is required and must be a non-empty string.")
        name = name.strip()
        load = await resolver.load(name)
        if not load.found:
            available = ", ".join(resolver.names()) or "(none discovered)"
            return ToolResult.error(
                f"no skill named {name!r}.",
                fix=f"Use one of the available skills: {available}.",
            )
        if load.denied_reason:
            # Policy refusal (e.g. the per-turn skill-invocation budget) — distinct from a load
            # failure: the skill exists and is readable, but invoking it now is not allowed.
            return ToolResult.error(
                load.denied_reason,
                fix="Finish the work with the skills already loaded; do not invoke more.",
            )
        if load.error or load.body is None:
            return ToolResult.error(
                f"skill {name!r} could not be loaded: {load.error or 'unreadable'}."
            )
        return ToolResult.ok(
            load.body,
            data={"skill": load.name},
            hint=(
                "Follow these skill instructions now. If a step says to use another skill, "
                "call use_skill with that name."
            ),
        )


__all__ = ["UseSkillTool"]
