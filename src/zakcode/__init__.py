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
from typing import Any

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

__all__ = ["Agent", "AgentLoop", "IterationBudget", "Message", "TurnResult", "__version__"]


class Agent:
    """High-level facade over the agent loop.

    Construct with defaults (settings come from env / ``.env``) or pass an
    explicit :class:`Settings`, :class:`Session`, or :class:`SessionStore`.
    Keyword overrides are forwarded to :func:`~zakcode.config.load_settings`.
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
            budget=budget,
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

    @classmethod
    def for_workspace(cls, path: str | Path, **setting_overrides: Any) -> Agent:
        """Construct an :class:`Agent` pinned to ``path`` as the workspace root."""
        return cls(workspace_root=Path(path), **setting_overrides)
