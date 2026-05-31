"""Sub-agent delegation (M4): run an isolated child agent, return a condensed summary.

A *sub-agent* is a lightweight child :class:`~zakcode.agent.loop.AgentLoop` the parent
spawns to handle a self-contained subtask. The three isolation properties that make
delegation safe and cheap:

* **Fresh history** — the child runs on a brand-new :class:`~zakcode.session.store.Session`
  (empty message list), so the parent's conversation never bleeds into the child's context
  and vice-versa. The child sees only the prompt it was handed.
* **Filtered tools** — the child gets a :meth:`ToolRegistry.subset` exposing only the tools
  its :class:`SubAgentDefinition` allows (the schema-level mechanism behind Plan Mode's
  write-tool absence, and behind giving a researcher read-only tools).
* **Shared budget** — every child draws from the *same* :class:`IterationBudget` as the
  parent, so the whole delegation tree's iteration count is bounded by one pool rather than
  multiplying per-agent caps. Spawning also counts against the budget's child cap.

The child returns a small :class:`SubAgentResult` (its final assistant text, not its raw
transcript), so the parent's context stays compact regardless of how much work the child did.

This module is pure orchestration over the existing frozen contracts (loop, registry, budget,
session) — it adds no new agent behavior and imports no vendor SDK.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field

from zakcode.agent.budget import IterationBudget
from zakcode.agent.loop import AgentLoop
from zakcode.agent.prompt import SystemPromptBuilder
from zakcode.config import Settings
from zakcode.hooks import HookManager
from zakcode.permissions import PermissionPolicy
from zakcode.providers.base import Provider
from zakcode.session.store import Session
from zakcode.tools.base import ToolRegistry
from zakcode.usage import Usage


class SubAgentDefinition(BaseModel):
    """A named sub-agent *type* the parent may delegate to.

    ``allowed_tools=None`` inherits the parent's full toolset; a list restricts the
    child to exactly those tools (canonical names or aliases). ``system_suffix`` is
    appended to the child's system prompt to specialize its behavior (e.g. a planner
    that is told to produce a plan rather than edit files).
    """

    name: str
    description: str = ""
    allowed_tools: list[str] | None = None
    system_suffix: str | None = None


class SubAgentResult(BaseModel):
    """The condensed handoff a sub-agent returns to its parent.

    ``summary`` is the child's final assistant text — deliberately *not* its raw
    transcript — so the parent's context does not balloon with the child's
    intermediate reasoning and tool chatter.
    """

    name: str
    summary: str = ""
    stop_reason: str = "completed"
    iterations: int = 0
    usage: Usage = Field(default_factory=Usage)


#: A general-purpose sub-agent with the parent's full toolset (the default delegate).
GENERAL_PURPOSE = SubAgentDefinition(
    name="general-purpose",
    description="A capable agent with the full toolset for self-contained subtasks.",
)


class SubAgentRunner:
    """Spawns isolated child agents that share the parent's iteration budget.

    Constructed once with the parent's collaborators (provider, base registry,
    settings, permission policy, hooks) and the shared budget; :meth:`run` then
    launches one child per call. The runner is deliberately stateless beyond its
    injected collaborators, so a parent can fan out several children from it.
    """

    def __init__(
        self,
        *,
        provider: Provider,
        registry: ToolRegistry,
        settings: Settings,
        budget: IterationBudget,
        permission_policy: PermissionPolicy | None = None,
        hook_manager: HookManager | None = None,
        workspace_root: Path | None = None,
    ) -> None:
        self.provider = provider
        self.registry = registry
        self.settings = settings
        self.budget = budget
        self.permission_policy = permission_policy
        self.hook_manager = hook_manager
        self.workspace_root = workspace_root or settings.workspace_root

    def child_registry(self, definition: SubAgentDefinition) -> ToolRegistry:
        """The tool registry a child of ``definition`` will see (full, or a subset)."""
        if definition.allowed_tools is None:
            return self.registry
        return self.registry.subset(definition.allowed_tools)

    def prompt_builder_for(self, definition: SubAgentDefinition) -> SystemPromptBuilder:
        """The system-prompt builder for a child (specialized iff a suffix is set)."""
        return SystemPromptBuilder(extra_instructions=definition.system_suffix)

    async def run(
        self,
        definition: SubAgentDefinition,
        prompt: str,
        *,
        depth: int = 1,
    ) -> SubAgentResult:
        """Run one sub-agent to completion and return its condensed summary.

        Counts the spawn against the shared budget's child cap (raising
        :class:`~zakcode.agent.budget.ChildLimitExceeded` past it), builds an
        isolated child loop (fresh session, filtered tools, shared budget, same
        permission gate + hooks as the parent), runs a single turn, and condenses
        the child's assistant output into a :class:`SubAgentResult`.
        """
        # Count this child against the shared cap before doing any work (raises if
        # the cap is already reached — the parent/task tool decides how to surface it).
        self.budget.register_child()

        registry = self.child_registry(definition)
        session = Session(cwd=str(self.workspace_root), model=self.settings.default_model)
        loop = AgentLoop(
            self.provider,
            registry,
            session,
            prompt_builder=self.prompt_builder_for(definition),
            settings=self.settings,
            permission_policy=self.permission_policy,
            hook_manager=self.hook_manager,
            budget=self.budget,
            workspace_root=self.workspace_root,
        )
        result = await loop.arun_turn(prompt)
        summary = "\n".join(m.text for m in result.assistant_messages if m.text).strip()
        return SubAgentResult(
            name=definition.name,
            summary=summary,
            stop_reason=result.stop_reason,
            iterations=result.iterations,
            usage=result.usage,
        )


__all__ = [
    "SubAgentDefinition",
    "SubAgentResult",
    "SubAgentRunner",
    "GENERAL_PURPOSE",
]
