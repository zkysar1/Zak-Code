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

from collections.abc import AsyncIterator, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

from zakcode.agent.budget import IterationBudget
from zakcode.agent.compact import Compactor
from zakcode.agent.loop import AgentLoop, TurnResult
from zakcode.agent.prompt import SystemPromptBuilder
from zakcode.config import Settings, load_settings
from zakcode.events import AgentEvent
from zakcode.hooks import HookManager
from zakcode.messages import Message
from zakcode.permissions import PermissionPolicy, PermissionPrompter
from zakcode.providers.base import Provider
from zakcode.session.store import Session, SessionStore
from zakcode.tools.builtins.default_registry import default_registry
from zakcode.version import __version__

if TYPE_CHECKING:
    # Type-only imports for MCP/plugin annotations. Kept out of the runtime import
    # graph so importing ``zakcode`` never pulls in those subsystems; the concrete
    # imports happen inside ``__init__`` only when the feature is enabled.
    from zakcode.mcp.config import McpServerConfig
    from zakcode.mcp.manager import DiscoveryReport, ExtensionManager
    from zakcode.memory import MemoryProvider
    from zakcode.plugins import PluginLoadReport
    from zakcode.rules import RuleRegistry
    from zakcode.skills import SkillRegistry

__all__ = ["Agent", "AgentLoop", "IterationBudget", "Message", "TurnResult", "__version__"]


# Provider prefixes whose *native* (function-calling) tool path is unreliable via
# litellm: they advertise tool support (``litellm.supports_function_calling`` returns
# True) yet native calls commonly come back empty for local models — verified with
# ``ollama_chat/qwen2.5:3b``, which never calls a tool natively — while the text
# protocol drives tools reliably. So in ``auto`` mode these are routed to the text
# protocol. Explicit ``tool_calling_mode="native"`` still forces native. This vendor
# knowledge lives in the application layer, never in the vendor-agnostic
# zds-llm-provider wrapper (which must stay free of any "ollama" special-casing).
_TEXT_TOOL_PROTOCOL_PREFIXES = frozenset({"ollama", "ollama_chat"})


def _resolve_tool_calling_mode(mode: str, model: str) -> str:
    """Resolve the effective tool-calling mode for ``model``.

    Only ``"auto"`` is adjusted: for a provider whose native tool path is unreliable
    (see :data:`_TEXT_TOOL_PROTOCOL_PREFIXES`) it becomes ``"text"``; otherwise
    ``"auto"`` is preserved. Explicit ``"native"`` / ``"text"`` pass through unchanged.
    """
    if mode != "auto":
        return mode
    prefix = model.split("/", 1)[0] if "/" in model else model
    return "text" if prefix in _TEXT_TOOL_PROTOCOL_PREFIXES else "auto"


def _find_repo_root(start: Path) -> Path | None:
    """Walk up from ``start`` to find the nearest directory containing ``.git``."""
    current = start.resolve()
    for parent in [current, *current.parents]:
        if (parent / ".git").exists():
            return parent
    return None


def _parse_local_paths_conf(conf_path: Path) -> list[Path]:
    """Parse a claude-mind-style ``local-paths.conf`` and return the external paths.

    The file is a simple ``KEY=VALUE`` format (one per line, no quoting); this
    function extracts ``WORLD_PATH`` and ``META_PATH`` values when present.
    """
    paths: list[Path] = []
    if not conf_path.is_file():
        return paths
    try:
        for raw_line in conf_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip()
            if key in ("WORLD_PATH", "META_PATH") and value:
                p = Path(value)
                if p.is_absolute() and p.is_dir():
                    paths.append(p)
    except OSError:
        pass
    return paths


def _infer_roots_from_skill_dir(skill_dir: Path) -> list[Path]:
    """Infer extra workspace roots from a ``--skill-dir`` path.

    1. Find the skill directory's owning git repo root (if any) and add it.
    2. Look for ``agents/*/local-paths.conf`` under that repo root and parse
       ``WORLD_PATH`` / ``META_PATH`` from each conf file found.

    Returns a deduplicated list of existing directories.
    """
    roots: list[Path] = []
    seen: set[Path] = set()

    repo_root = _find_repo_root(skill_dir)
    if repo_root is not None:
        resolved = repo_root.resolve()
        if resolved not in seen:
            roots.append(resolved)
            seen.add(resolved)

        # Scan for agent local-paths.conf files under the repo root.
        agents_dir = repo_root / "agents"
        if agents_dir.is_dir():
            for child in agents_dir.iterdir():
                conf = child / "local-paths.conf"
                for p in _parse_local_paths_conf(conf):
                    rp = p.resolve()
                    if rp not in seen:
                        roots.append(rp)
                        seen.add(rp)

    return roots


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
        provider: Provider | None = None,
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
        mcp_tool_budget: int | None = None,
        enable_plugins: bool = False,
        trusted_plugins: list[str] | None = None,
        enable_skills: bool = False,
        extra_skill_dirs: Sequence[str | Path] | None = None,
        extra_workspace_roots: Sequence[str | Path] | None = None,
        enable_rules: bool = False,
        enable_memory: bool = False,
        memory_provider: MemoryProvider | None = None,
        enable_compaction: bool = False,
        **setting_overrides: Any,
    ) -> None:
        self.settings = settings or load_settings(**setting_overrides)
        # The provider is normally built from settings (litellm — the one vendor seam),
        # but a caller may inject any ``Provider`` (the eval harness drives the loop with
        # a no-network ScriptedProvider this way). Importing litellm lazily keeps the
        # vendor SDK out of the import graph when an explicit provider is supplied.
        if provider is not None:
            self.provider = provider
        else:
            from zakcode.providers.litellm_provider import LiteLLMProvider
            from zakcode.providers.text_tools import TextToolCallingProvider

            # Wrap the vendor provider so tool-less (or unreliable-native) models still
            # get tool-calling via a text protocol (a no-op passthrough in "native"
            # mode). In "auto", Ollama models are routed to the text protocol because
            # their native path is unreliable via litellm — see
            # _resolve_tool_calling_mode and zakcode.providers.text_tools.
            self.provider = TextToolCallingProvider(
                LiteLLMProvider(self.settings),
                mode=_resolve_tool_calling_mode(
                    self.settings.tool_calling_mode, self.settings.default_model
                ),
            )
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
        from zakcode.permissions import compile_deny_patterns

        self.permission_policy = permission_policy or PermissionPolicy(
            self.settings.permission_mode,
            prompter=prompter,
            extra_dangerous_patterns=compile_deny_patterns(self.settings.denied_commands),
        )
        self.hook_manager = hook_manager or HookManager()
        # One-shot guard so aclose() (and its SESSION_END encode step) runs at most once.
        self._closed = False
        # Slash-command registry (M6) — plugins register commands here; clients
        # (the CLI) consult it for any slash command they do not handle themselves.
        from zakcode.commands import CommandRegistry

        self.command_registry = CommandRegistry()

        # Stable-tier prompt content (skills catalog + always-on rules), discovered
        # BEFORE delegation so sub-agents can inherit the same rules. Both are opt-in;
        # a bad skill/rule file is recorded, never fatal.
        self.skill_registry: SkillRegistry | None = None
        self.skill_errors: dict[str, str] = {}
        skills_catalog = ""
        if enable_skills:
            from zakcode.skills import discover_skills, project_skills_dir
            from zakcode.tools.builtins.save_skill import SaveSkillTool

            self.skill_registry, self.skill_errors = discover_skills(
                self.settings.workspace_root, extra_skill_dirs=extra_skill_dirs
            )
            skills_catalog = self.skill_registry.render_catalog()
            # Model-driven skill authoring (persisted to the project skills dir; the
            # new skill is discovered next session). save_skill validates the name so
            # the write can never escape that directory.
            self.registry.register(SaveSkillTool(project_skills_dir(self.settings.workspace_root)))

        # Rules: always-on guidance (bundled + user + project, incl. .claude/rules for
        # Claude-Code/Claude-Mind compatibility) rendered into the cacheable tier.
        self.rule_registry: RuleRegistry | None = None
        self.rule_errors: dict[str, str] = {}
        rules_text = ""
        if enable_rules:
            from zakcode.rules import discover_rules

            self.rule_registry, self.rule_errors = discover_rules(self.settings.workspace_root)
            rules_text = self.rule_registry.render()

        # Wire the discovered content into the prompt builder. With no injected
        # builder, construct one; with an injected builder, fill any empty stable-tier
        # slot so enable_skills/enable_rules are never silently no-ops.
        if prompt_builder is None and (skills_catalog or rules_text):
            prompt_builder = SystemPromptBuilder(
                extra_instructions=skills_catalog or None,
                rules=rules_text or None,
            )
        elif prompt_builder is not None:
            if skills_catalog and not prompt_builder.extra_instructions:
                prompt_builder.extra_instructions = skills_catalog
            if rules_text and not prompt_builder.rules:
                prompt_builder.rules = rules_text

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
                rules=rules_text or None,  # sub-agents inherit the parent's always-on rules
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
        self._mcp_tool_budget = 0
        if enable_mcp:
            from zakcode.mcp.config import discover_config
            from zakcode.mcp.manager import build_extension_manager
            from zakcode.tools.builtins.tool_search import DEFAULT_TOOL_BUDGET, ToolSearchTool

            self._mcp_tool_budget = (
                mcp_tool_budget if mcp_tool_budget is not None else DEFAULT_TOOL_BUDGET
            )
            servers = (
                mcp_servers
                if mcp_servers is not None
                else discover_config(self.settings.workspace_root)
            )
            self.extension_manager, self.mcp_config_errors = build_extension_manager(
                servers, allowlist=mcp_command_allowlist
            )
            # tool_search lets the model surface MCP tools that the budget kept hidden;
            # it holds the live registry so activations are visible to the next turn.
            self.registry.register(ToolSearchTool(self.registry, budget=self._mcp_tool_budget))

        # Plugins (M6), opt-in. Discover plugins (project + user dirs + entry points)
        # and run each trusted+enabled one's register(ctx) against the live
        # registry/hooks/commands. Trust is explicit: a plugin runs only if its
        # manifest is trusted, which ``trusted_plugins`` (an allowlist) grants by name.
        # Discovery + load are synchronous and side-effect-light (importing a module +
        # calling register), so they run here; a bad plugin is recorded, never fatal.
        self.plugin_report: PluginLoadReport | None = None
        self.plugin_discovery_errors: dict[str, str] = {}
        if enable_plugins:
            from zakcode.plugins import PluginManager
            from zakcode.plugins.discovery import discover_plugins

            trusted = set(trusted_plugins or [])
            found, self.plugin_discovery_errors = discover_plugins(
                self.settings.workspace_root, trusted_names=trusted
            )
            manager = PluginManager()
            for plugin in found:
                manager.add(plugin)
            self.plugin_report = manager.load_into(
                registry=self.registry,
                hook_manager=self.hook_manager,
                command_registry=self.command_registry,
                settings=self.settings,
            )

        # Cross-session memory, opt-in. Wire a MemoryProvider (default: a local
        # SQLite/FTS5 store), register the remember/recall tools, and — unless recall
        # is disabled — add a PRE_LLM_CALL hook that injects relevant memories each
        # turn (fenced as untrusted by the loop). Substrate only: WHAT to remember is
        # the model's / an integrating framework's choice, not the store's.
        self.memory: MemoryProvider | None = None
        if enable_memory:
            from zakcode.memory import MemoryRecallHook
            from zakcode.memory.sqlite_store import SqliteMemoryProvider
            from zakcode.tools.builtins.memory import RecallTool, RememberTool

            # Default the store to <workspace>/.zakcode/memory.db — per-project memory
            # (no cross-project bleed; isolated under a tmp workspace in tests). An
            # explicit memory_db_path (or ZAKCODE_MEMORY_DB_PATH) overrides it.
            db_path = self.settings.memory_db_path or str(
                Path(self.settings.workspace_root) / ".zakcode" / "memory.db"
            )
            self.memory = memory_provider or SqliteMemoryProvider(db_path)
            self.registry.register(RememberTool(self.memory, source=self.session.id))
            self.registry.register(
                RecallTool(self.memory, default_limit=self.settings.memory_recall_limit)
            )
            if self.settings.memory_recall_limit > 0:
                self.hook_manager.register_context(
                    MemoryRecallHook(
                        self.memory,
                        limit=self.settings.memory_recall_limit,
                        min_overlap=self.settings.memory_recall_min_overlap,
                    )
                )

        # Compaction (M8), opt-in. When enabled, the loop auto-compacts the session
        # before a turn once it exceeds the provider's context-window threshold.
        self.compactor: Compactor | None = None
        if enable_compaction:
            self.compactor = Compactor()

        # Multi-root sandbox (M-3): compute extra workspace roots from explicit
        # args plus auto-detected roots from --skill-dir (the skill directory's
        # owning repo root, and any external paths declared in its local-paths.conf).
        computed_extra_roots: list[Path] = []
        if extra_workspace_roots:
            computed_extra_roots.extend(Path(r) for r in extra_workspace_roots)
        if extra_skill_dirs:
            for sd in extra_skill_dirs:
                computed_extra_roots.extend(_infer_roots_from_skill_dir(Path(sd)))

        self.loop = AgentLoop(
            self.provider,
            self.registry,
            self.session,
            prompt_builder=prompt_builder,
            settings=self.settings,
            store=self.store,
            workspace_root=self.settings.workspace_root,
            extra_workspace_roots=computed_extra_roots,
            permission_policy=self.permission_policy,
            hook_manager=self.hook_manager,
            budget=shared_budget,
            spawner=spawner,
            compactor=self.compactor,
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
        # ``tool_search`` is already active and counts toward the exposed budget, so
        # MCP tools beyond the budget register hidden and are surfaced on demand.
        self.mcp_report = await self.extension_manager.discover_into(
            self.registry, budget=self._mcp_tool_budget or None
        )
        return self.mcp_report

    async def aclose_mcp(self) -> None:
        """Close any open MCP server connections (best-effort)."""
        if self.extension_manager is not None:
            await self.extension_manager.aclose()

    async def aclose(self) -> None:
        """End the session: fire ``SESSION_END`` then release resources.

        Fires the ``SESSION_END`` lifecycle hook (a host's encode/serialize step),
        closes the memory store, and closes any MCP connections. Best-effort and
        idempotent — a second call is a no-op (so the encode step never double-runs),
        and every step is isolated so a failing one never blocks the rest.
        """
        import contextlib

        from zakcode.hooks import HookEvent, LifecyclePayload

        if self._closed:
            return
        self._closed = True

        if self.hook_manager.has_lifecycle_hooks(HookEvent.SESSION_END):
            with contextlib.suppress(Exception):
                await self.hook_manager.fire(
                    LifecyclePayload(
                        event=HookEvent.SESSION_END,
                        session_id=self.session.id,
                        cwd=str(self.settings.workspace_root),
                    )
                )
        if self.memory is not None:
            with contextlib.suppress(Exception):
                self.memory.close()
        await self.aclose_mcp()

    @classmethod
    def for_workspace(cls, path: str | Path, **setting_overrides: Any) -> Agent:
        """Construct an :class:`Agent` pinned to ``path`` as the workspace root."""
        return cls(workspace_root=Path(path), **setting_overrides)
