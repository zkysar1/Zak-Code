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

from collections.abc import Callable, Sequence
from pathlib import Path

from pydantic import BaseModel, Field

from zakcode.agent.budget import IterationBudget
from zakcode.agent.loop import AgentLoop, TurnResult
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
    that is told to produce a plan rather than edit files). ``model`` optionally routes
    this sub-agent type to a DIFFERENT model than the parent's (e.g. a cheap/local model
    for a read-only planner); ``None`` uses the parent's ``default_model``.
    """

    name: str
    description: str = ""
    allowed_tools: list[str] | None = None
    system_suffix: str | None = None
    model: str | None = None


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


#: The read-only built-in tools (those that never mutate the workspace). The planner
#: (Plan Mode) is restricted to exactly these, so the write tools (write_file /
#: edit_file / bash) are ABSENT from its tool schema entirely: the model is never
#: offered them, rather than being offered them and refused at runtime. This is the
#: structural, schema-level guarantee behind "a planner literally cannot edit" (see
#: docs/ROADMAP.md M4 exit criteria).
READ_ONLY_TOOLS = ["read_file", "list_dir", "glob", "grep"]

#: A general-purpose sub-agent with the parent's full toolset (the default delegate).
GENERAL_PURPOSE = SubAgentDefinition(
    name="general-purpose",
    description="A capable agent with the full toolset for self-contained subtasks.",
)

#: The planner used by Plan Mode: read-only tools only, instructed to investigate and
#: return a step-by-step plan rather than make changes. Because its registry is a
#: subset over READ_ONLY_TOOLS, every write tool is absent from the schema it is given.
PLAN = SubAgentDefinition(
    name="plan",
    description="A read-only planner: investigates and returns a step-by-step plan, never edits.",
    allowed_tools=READ_ONLY_TOOLS,
    system_suffix=(
        "You are in PLAN MODE. You have read-only tools only and cannot modify the "
        "workspace. Investigate the request, then return a clear, ordered, step-by-step "
        "plan describing exactly what changes should be made and why. Do not attempt to "
        "make changes yourself; produce the plan as your final answer."
    ),
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
        extra_workspace_roots: Sequence[Path] | None = None,
        rules: str | None = None,
        provider_for: Callable[[str | None], Provider] | None = None,
    ) -> None:
        self.provider = provider
        self.registry = registry
        self.settings = settings
        self.budget = budget
        self.permission_policy = permission_policy
        self.hook_manager = hook_manager
        # Resolves a sub-agent's optional ``model`` override to a provider (e.g. the parent
        # Agent's model-routing seam). ``None`` here — or a definition with no ``model`` — means
        # every child uses the parent ``provider``, so the default delegation path is unchanged.
        self._provider_for = provider_for
        self.workspace_root = workspace_root or settings.workspace_root
        # The parent's multi-root sandbox, so a child gets the SAME roots (not a narrower
        # one) and a delegated --skill-dir-granted path isn't wrongly rejected. (audit4 #4)
        self.extra_workspace_roots = list(extra_workspace_roots or [])
        # The parent's rendered always-on rules, threaded into every child's prompt so
        # delegated work runs under the same standing guidance as the parent.
        self.rules = rules

    def child_registry(self, definition: SubAgentDefinition) -> ToolRegistry:
        """The tool registry a child of ``definition`` will see (full, or a subset)."""
        if definition.allowed_tools is None:
            return self.registry
        return self.registry.subset(definition.allowed_tools)

    def prompt_builder_for(self, definition: SubAgentDefinition) -> SystemPromptBuilder:
        """The system-prompt builder for a child (specialized iff a suffix is set).

        The parent's always-on rules are carried through so a sub-agent obeys the
        same operator guidance; the definition's ``system_suffix`` (e.g. Plan Mode)
        coexists with them in the stable tier.
        """
        return SystemPromptBuilder(extra_instructions=definition.system_suffix, rules=self.rules)

    async def run(self, definition: SubAgentDefinition, prompt: str) -> SubAgentResult:
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
        # Route this sub-agent type to its own model when the definition names one AND a
        # resolver is wired (production); otherwise the parent provider/model is used — so the
        # default path, and any path with an injected provider (tests), is unchanged.
        child_provider = self.provider
        child_model = self.settings.default_model
        if definition.model and self._provider_for is not None:
            routed = self._provider_for(definition.model)
            # Only adopt the role-model label when it actually routes to a DISTINCT provider. If
            # the resolver short-circuits to the parent (an injected provider, or model ==
            # default), keep the default label so Session.model never disagrees with the
            # provider that will actually run. (review: Session.model honesty)
            if routed is not self.provider:
                child_provider = routed
                child_model = definition.model
        session = Session(cwd=str(self.workspace_root), model=child_model)
        # A child gets its OWN permission view (same mode + blocklist, isolated session
        # grants) so a child's "allow for session" cannot bleed into the parent/siblings.
        child_policy = (
            self.permission_policy.child_view() if self.permission_policy is not None else None
        )
        loop = AgentLoop(
            child_provider,
            registry,
            session,
            prompt_builder=self.prompt_builder_for(definition),
            settings=self.settings,
            permission_policy=child_policy,
            hook_manager=self.hook_manager,
            budget=self.budget,
            workspace_root=self.workspace_root,
            extra_workspace_roots=self.extra_workspace_roots,  # same sandbox as the parent
            # Write-grounding + the verify-before-finish gate are always-on in AgentLoop, so
            # delegated create-and-run work is grounded/gated automatically — nothing to thread.
        )
        try:
            result = await loop.arun_turn(prompt)
        finally:
            # The child loop is discarded here; release its egress-proxy listener (a no-op when
            # off) so a delegation tree doesn't leak one listener per child on a long-lived loop.
            await loop.aclose()
        summary = self._summarize(result)
        return SubAgentResult(
            name=definition.name,
            summary=summary,
            stop_reason=result.stop_reason,
            iterations=result.iterations,
            usage=result.usage,
        )

    @staticmethod
    def _summarize(result: TurnResult) -> str:
        """Condense a child turn into a handoff string for the parent.

        Prefers the child's final assistant text. When there is none (e.g. the
        child stopped mid-tool-loop at the shared budget), it falls back to the
        last tool result, then to a terse status line — so the parent is never
        handed an empty summary for a child that actually did work.
        """
        text = "\n".join(m.text for m in result.assistant_messages if m.text).strip()
        if text:
            return text
        if result.tool_results:
            last = result.tool_results[-1].output.strip()
            if last:
                return last
        return (
            f"(sub-agent produced no final text; stop_reason={result.stop_reason}, "
            f"iterations={result.iterations})"
        )


class SubAgentManager:
    """A concrete :class:`~zakcode.tools.base.SubAgentSpawner` over a runner + type table.

    Maps named sub-agent *types* (a registry of :class:`SubAgentDefinition`) to the
    :class:`SubAgentRunner` that launches them. This is the object the ``Agent``
    facade places on the loop so the ``task`` tool can delegate. One named type is
    the explicit default (returned by :meth:`default_type`), so the ``task`` tool
    never has to guess from list ordering.
    """

    def __init__(
        self,
        runner: SubAgentRunner,
        definitions: list[SubAgentDefinition],
        *,
        default: str,
    ) -> None:
        if not definitions:
            raise ValueError("a SubAgentManager needs at least one sub-agent definition")
        self._runner = runner
        self._defs: dict[str, SubAgentDefinition] = {d.name: d for d in definitions}
        if default not in self._defs:
            raise ValueError(f"default type {default!r} is not among the definitions")
        self._default = default

    async def spawn(self, *, type_name: str, prompt: str) -> SubAgentResult:
        definition = self._defs.get(type_name)
        if definition is None:
            raise KeyError(f"unknown sub-agent type {type_name!r}")
        return await self._runner.run(definition, prompt)

    def available_types(self) -> list[str]:
        return list(self._defs)

    def default_type(self) -> str:
        return self._default


__all__ = [
    "SubAgentDefinition",
    "SubAgentResult",
    "SubAgentRunner",
    "SubAgentManager",
    "GENERAL_PURPOSE",
    "PLAN",
    "READ_ONLY_TOOLS",
]
