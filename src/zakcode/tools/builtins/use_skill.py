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

from pathlib import Path
from typing import Any

from zakcode.config import PermissionTier
from zakcode.tools.base import ConcurrencyClass, Tool, ToolContext, ToolResult, ToolSpec
from zakcode.tools.builtins._suggest import _display

#: Sibling entries named in the skill-directory footer before "+N more" (ADR-0044).
_MAX_SKILL_FILES = 12


def skill_directory_line(skill_md: str | None, workspace_root: Path) -> str:
    """One line naming the skill's directory and what sits beside its SKILL.md ('' if unknown).

    Field incident 2026-08-27: asked whether ``google-drive-list`` was a skill, the model
    answered from the body it had loaded — "it is a python file, not a skill" — and never
    listed the directory that body lives in. The footer puts the directory in the tool result
    so "what IS this skill" is answered by the load itself, not by memory.
    """
    if not skill_md:
        return ""
    skill_dir = Path(skill_md).parent
    try:
        entries = sorted(
            (p for p in skill_dir.iterdir() if p.name != "SKILL.md" and not p.name.startswith(".")),
            key=lambda p: p.name,
        )
    except OSError:
        return ""
    shown = [f"{p.name}/" if p.is_dir() else p.name for p in entries[:_MAX_SKILL_FILES]]
    extra = len(entries) - len(shown)
    if shown:
        listed = ", ".join(shown) + (f", +{extra} more" if extra > 0 else "")
    else:
        listed = "(only SKILL.md)"
    return (
        f"[skill directory] {_display(skill_dir, Path(workspace_root))}: {listed}. "
        "The skill IS this directory; read_file its scripts before describing what it is or does."
    )


#: Bodies at or above this size get the DECOMPOSE hint instead of the plain follow hint
#: (ADR-0027). A small model cannot hold a wall of instructions as working state — field
#: incident: a 2,776-line skill body was followed for two steps and then narrated instead
#: of executed. Decomposing into the plan converts instructions into checked-off steps
#: that survive compaction AND the seam clamp; the risk asymmetry favors decomposing
#: (worst case is a few extra plan steps), so the threshold is deliberately low.
_DECOMPOSE_HINT_MIN_CHARS = 2_000


class UseSkillTool(Tool):
    """Load a discovered skill's instructions by name and return them for the model to follow."""

    spec = ToolSpec(
        name="use_skill",
        description=(
            "Load a skill's full step-by-step instructions by name and follow them. Call this "
            "when one of the skills listed in your context fits the task. The skill's body is "
            "returned as this tool's result — act on it. For a LONG skill, first decompose its "
            "steps into your plan with update_plan, then execute the plan. Skills can chain: if "
            "a skill's steps tell you to use another skill, call use_skill again with that name."
        ),
        parameters={
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "The skill name to load, exactly as it appears in the catalog.",
                },
                "args": {
                    "type": "string",
                    "description": (
                        "Optional arguments for the skill (e.g. a sub-command like 'loop'). "
                        "Surfaced to you alongside the skill's instructions."
                    ),
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
        skill_args = args.get("args", "")
        if not isinstance(skill_args, str):
            skill_args = ""
        # Pass the invoking turn's prompt so the selection signal is attributed to THIS caller
        # (a sub-agent's task, not the parent's originating turn).
        load = await resolver.load(name, query=ctx.caller_query, args=skill_args)
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
        footer = skill_directory_line(load.path, ctx.workspace_root)
        output = f"{load.body}\n\n{footer}" if footer else load.body
        if len(load.body) >= _DECOMPOSE_HINT_MIN_CHARS:
            # The decompose rail (ADR-0027): a long body is a plan waiting to happen, not
            # working state to hold in the model's head. Fired at the exact moment the
            # body arrives — the one point where the model has both the instructions and
            # its task context in hand.
            return ToolResult.ok(
                output,
                data={"skill": load.name, "decompose": True},
                hint=(
                    "These instructions are long — do not try to hold them all in your "
                    "head. FIRST decompose them: with update_plan, record the concrete "
                    "steps THIS request needs (fold in the context you already have), "
                    "then execute the steps in order, marking each done as you finish. "
                    "If a step says to use another skill, call use_skill with that name."
                ),
            )
        return ToolResult.ok(
            output,
            data={"skill": load.name},
            hint=(
                "Follow these skill instructions now. If a step says to use another skill, "
                "call use_skill with that name."
            ),
        )


__all__ = ["UseSkillTool", "skill_directory_line"]
