"""Zak Code — a clean-room, vendor-agnostic, API-first agentic coding tool.

The public entry point is :class:`Agent`: a thin facade that wires settings, a
provider, the built-in tool registry, a session, and the ReAct
:class:`~zakcode.agent.loop.AgentLoop` together, then exposes ``run_turn`` /
``arun_turn``.

Vendor isolation is structural, not import-order based: ``litellm`` is imported
only under :mod:`zakcode.providers` (the one place allowed to touch a vendor SDK).
Nothing here triggers network activity at import time — only an actual
``run_turn`` call reaches the model.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path
from typing import TYPE_CHECKING, Any

from zakcode.agent.budget import IterationBudget
from zakcode.agent.loop import AgentLoop, TurnResult
from zakcode.agent.prompt import SystemPromptBuilder
from zakcode.config import Settings, load_settings
from zakcode.events import AgentEvent
from zakcode.hooks import HookManager
from zakcode.messages import Message
from zakcode.permissions import PermissionPolicy, PermissionPrompter
from zakcode.session.store import Session, SessionStore
from zakcode.tools.builtins.default_registry import default_registry
from zakcode.version import __version__

if TYPE_CHECKING:
    # Type-only imports for MCP annotations. Kept out of the runtime import graph so
    # importing ``zakcode`` never pulls in the MCP subsystem; the concrete imports
    # happen inside ``__init__`` only when ``enable_mcp=True``.
    from zakcode.mcp.config import McpServerConfig
    from zakcode.mcp.manager import DiscoveryReport, ExtensionManager

__all__ = ["Agent", "AgentLoop", "IterationBudget", "Message", "TurnResult", "__version__"]


class Agent:
    """High-level facade over the agent loop.

    Construct with defaults (settings come from env / ``.env``) or pass an
    explicit :class:`Settings`, :class:`Session`, or :class:`SessionStore`.
    Keyword overrides are forwarded to :func:`~zakcode.config.load_settings`.

    Pass ``enable_subagents=True`` to expose the ``task`` delegation tool and a
    shared iteration budget (off by default). Pass ``enable_mcp=True`` to wire
    configured MCP servers; their tools register on ``await connect_mcp()``.
    """

    def __init__(
        self,
        *,
        settings: Settings | None = None,
        session: Session | None = None,
        session_store: SessionStore | None = None,
        prompt_builder: SystemPromptBuilder | None = None,
        prompter: PermissionPrompter | None = None,
        permission_policy: PermissionPolicy | None = None,
        hook_manager: HookManager | None = None,
        budget: IterationBudget | None = None,
        enable_subagents: bool = False,
        enable_mcp: bool = False,
        mcp_servers: list[McpServerConfig] | None = None,
        mcp_command_allowlist: list[str] | None = None,
        **setting_overrides: Any,
    ) -> None:
        from zakcode.providers.litellm_provider import LiteLLMProvider

        self.settings = settings or load_settings(**setting_overrides)
        self.provider = LiteLLMProvider(self.settings)
        self.registry = default_registry()
        self.store = session_store
        self.session = session or Session(
            cwd=str(self.settings.workspace_root),
            model=self.settings.default_model,
        )
        # Deny-first by construction: the facade always builds a permission policy
        # from settings.permission_mode (default 'ask'). An interactive client may
        # pass a ``prompter`` so escalations can be approved; with none, 'ask'
        # fails closed (writes/shell denied) — safe for non-interactive use.
        self.permission_policy = permission_policy or PermissionPolicy(
            self.settings.permission_mode, prompter=prompter
        )
        self.hook_manager = hook_manager or HookManager()

        # Delegation (M4), opt-in. When enabled, the parent gets the ``task`` tool and
        # a shared :class:`IterationBudget`, and a :class:`SubAgentManager` is placed
        # on the loop so ``task`` can launch sub-agents. Sub-agents are built (in the
        # runner) with a task-FREE registry and NO spawner, so one-level nesting is
        # structural: a child can neither see nor call ``task``. Disabled by default,
        # so an ordinary ``Agent`` is byte-for-byte unchanged.
        shared_budget = budget
        spawner = None
        if enable_subagents:
            from zakcode.agent.subagent import (
                GENERAL_PURPOSE,
                PLAN,
                SubAgentManager,
                SubAgentRunner,
            )
            from zakcode.tools.builtins.task import TaskTool

            shared_budget = budget or IterationBudget(self.settings.max_iterations)
            runner = SubAgentRunner(
                provider=self.provider,
                registry=default_registry(),  # task-free registry for children
                settings=self.settings,
                budget=shared_budget,
                permission_policy=self.permission_policy,
                hook_manager=self.hook_manager,
                workspace_root=self.settings.workspace_root,
            )
            # general-purpose (full toolset) + plan (read-only planner whose registry
            # subset omits write tools, so Plan Mode is schema-enforced).
            spawner = SubAgentManager(runner, [GENERAL_PURPOSE, PLAN], default=GENERAL_PURPOSE.name)
            self.registry.register(TaskTool())

        # MCP (M5), opt-in. Build (but do NOT start) a client per configured server;
        # __init__ stays side-effect-free. The servers are spawned and their tools
        # discovered into ``self.registry`` only when the caller awaits ``connect_mcp``.
        self.extension_manager: ExtensionManager | None = None
        self.mcp_config_errors: dict[str, str] = {}
        self.mcp_report: DiscoveryReport | None = None
        if enable_mcp:
            from zakcode.mcp.config import discover_config
            from zakcode.mcp.manager import build_extension_manager

            servers = (
                mcp_servers
                if mcp_servers is not None
                else discover_config(self.settings.workspace_root)
            )
            self.extension_manager, self.mcp_config_errors = build_extension_manager(
                servers, allowlist=mcp_command_allowlist
            )

        self.loop = AgentLoop(
            self.provider,
            self.registry,
            self.session,
            prompt_builder=prompt_builder,
            settings=self.settings,
            store=self.store,
            workspace_root=self.settings.workspace_root,
            permission_policy=self.permission_policy,
            hook_manager=self.hook_manager,
            budget=shared_budget,
            spawner=spawner,
        )

    async def arun_turn(self, user_text: str) -> TurnResult:
        """Run one user turn asynchronously."""
        return await self.loop.arun_turn(user_text)

    def run_turn(self, user_text: str) -> TurnResult:
        """Run one user turn synchronously (wraps :meth:`arun_turn`)."""
        return self.loop.run_turn(user_text)

    def astream_turn(self, user_text: str) -> AsyncIterator[AgentEvent]:
        """Stream one user turn as a sequence of :class:`~zakcode.events.AgentEvent`.

        Returns the loop's async iterator directly (no ``await`` needed to obtain
        it), so callers can ``async for event in agent.astream_turn(text)``. This
        is the incremental counterpart to :meth:`run_turn` / :meth:`arun_turn`.
        """
        return self.loop.astream_turn(user_text)

    async def connect_mcp(self) -> DiscoveryReport | None:
        """Spawn configured MCP servers and register their tools into the registry.

        Returns the discovery report (registered qualified tool names + per-server
        failures), or ``None`` if MCP was not enabled. Call once before the first
        turn; a failed server is reported in the result, never fatal. Tools register
        into the same ``self.registry`` the loop already holds, so they become
        available to the next turn with no further wiring.
        """
        if self.extension_manager is None:
            return None
        self.mcp_report = await self.extension_manager.discover_into(self.registry)
        return self.mcp_report

    async def aclose_mcp(self) -> None:
        """Close any open MCP server connections (best-effort)."""
        if self.extension_manager is not None:
            await self.extension_manager.aclose()

    @classmethod
    def for_workspace(cls, path: str | Path, **setting_overrides: Any) -> Agent:
        """Construct an :class:`Agent` pinned to ``path`` as the workspace root."""
        return cls(workspace_root=Path(path), **setting_overrides)
