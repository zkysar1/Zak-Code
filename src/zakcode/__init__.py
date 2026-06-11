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

import contextlib
import logging
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from zakcode.agent.budget import IterationBudget
from zakcode.agent.compact import Compactor
from zakcode.agent.loop import AgentLoop, TurnResult
from zakcode.agent.prompt import SystemPromptBuilder
from zakcode.config import Settings, load_settings
from zakcode.events import AgentEvent
from zakcode.hooks import HookEvent, HookManager, LifecyclePayload
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

__all__ = [
    "Agent",
    "AgentLoop",
    "IterationBudget",
    "Message",
    "SkillInvocation",
    "TurnResult",
    "__version__",
]

logger = logging.getLogger(__name__)

# Library logging etiquette (PR-4 review): a NullHandler on the package root so an
# application that configures no logging sees NOTHING on stderr — Python's lastResort
# handler would otherwise print every ``zakcode.*`` WARNING (e.g. permission denials,
# which the CLI already renders in its own UI). Operators opt in by configuring
# handlers/levels for the ``zakcode`` hierarchy; the library never configures global
# logging and never silences anyone else's.
logging.getLogger("zakcode").addHandler(logging.NullHandler())


# Provider prefixes whose *native* (function-calling) tool path is unreliable via
# litellm: they advertise tool support (``litellm.supports_function_calling`` returns
# True) yet native calls commonly come back empty for local models — verified with
# ``ollama_chat/qwen2.5:3b``, which never calls a tool natively — while the text
# protocol drives tools reliably. So in ``auto`` mode these are routed to the text
# protocol. Explicit ``tool_calling_mode="native"`` still forces native. This vendor
# knowledge lives in the application layer, never in the vendor-agnostic
# text-tool wrapper (which must stay free of any "ollama" special-casing).
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


@dataclass(frozen=True)
class SkillInvocation:
    """Outcome of :meth:`Agent.invoke_skill`.

    ``invoked`` is True iff ``name`` resolved to a discovered skill (so the caller treats it
    as handled, not an unknown command); ``error`` is set iff the skill's body failed to load.
    """

    invoked: bool
    name: str = ""
    error: str | None = None


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
        enable_identity: bool = True,
        identity: str | None = None,
        enable_memory: bool = False,
        memory_provider: MemoryProvider | None = None,
        enable_compaction: bool = False,
        enable_settings_hooks: bool = False,
        agent_identity_dir: str | Path | None = None,
        **setting_overrides: Any,
    ) -> None:
        self.settings = settings or load_settings(**setting_overrides)
        # The provider is normally built from settings (litellm — the one vendor seam),
        # but a caller may inject any ``Provider`` (the eval harness drives the loop with
        # a no-network ScriptedProvider this way). Importing litellm lazily keeps the
        # vendor SDK out of the import graph when an explicit provider is supplied.
        # Whether the provider was injected (so per-role model routing can't rebuild it from
        # settings — an injected provider is used for every role) vs built from settings.
        self._provider_injected = provider is not None
        # Cache of role providers built for a non-default model, keyed by model string, so
        # spawning N sub-agents on the same role model doesn't rebuild the litellm wrapper N×.
        self._provider_cache: dict[str, Provider] = {}
        if provider is not None:
            self.provider = provider
        else:
            self.provider = self._build_provider(self.settings.default_model)
        self.registry = default_registry(self.settings)
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
            # Opt-in egress gate: confirm every web_fetch before it reaches the network.
            confirm_tools={"web_fetch"} if self.settings.web_fetch_confirm else None,
            # Per-tool trust overrides (audit P0-2b / D12) — validated at Settings load.
            tool_mode_overrides=dict(self.settings.tool_trust_overrides),
        )
        # Rehydrate operator grants persisted with the session (audit P0-2d / D12 / Q5).
        # Honored only when the active mode is at least as loose as the grant-time mode;
        # a fresh session carries no grants, so this is a no-op there.
        self.permission_policy.restore_grants(self.session.permission_grants)
        self.hook_manager = hook_manager or HookManager()
        # settings.json hook ingestion (PR-T5).  TE-R3(3): when a caller passes
        # BOTH enable_settings_hooks=True AND a programmatic hook_manager, the
        # settings.json specs are APPENDED to the existing manager's shell_hooks.
        if enable_settings_hooks:
            from zakcode.hooks.settings_loader import load_settings_hooks

            _specs, _errs = load_settings_hooks(
                self.settings.workspace_root,
                permission_mode=str(self.settings.permission_mode),
            )
            for _key, _err in _errs.items():
                logger.warning("settings.json hook %s: %s", _key, _err)
            if _specs:
                if hook_manager is not None:
                    # TE-R3(3): append, don't replace.
                    self.hook_manager.shell_hooks.extend(_specs)
                else:
                    self.hook_manager = HookManager(shell_hooks=_specs)
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

        # Operator identity (self.md), loaded ONCE here so it is cache-stable for the session.
        # An explicit ``identity`` arg wins; else discover it from the workspace when enabled
        # (the default). It is operator-authored = TRUSTED, so it is injected UN-fenced (unlike
        # memory recall); never wire a request-supplied identity through this path.
        self.identity: str | None = None
        self.identity_error: str | None = None
        if identity is not None:
            self.identity = identity
        elif enable_identity:
            from zakcode.identity import load_identity

            self.identity, self.identity_error = load_identity(
                self.settings.workspace_root, agent_identity_dir=agent_identity_dir
            )
            # An operator-authored self.md that fails to load (unreadable, or empty after
            # frontmatter) is a silent footgun: the intended persona is gone with no signal.
            # Log it like rule-discovery failures so it surfaces; clients (e.g. the CLI banner)
            # can also read ``identity_error``. A missing self.md is not an error (stays None).
            if self.identity_error:
                logger.warning("operator identity (self.md) not loaded: %s", self.identity_error)

        # Wire the discovered content into the prompt builder. With no injected
        # builder, construct one; with an injected builder, fill any empty stable-tier
        # slot so enable_identity/enable_skills/enable_rules are never silently no-ops.
        # NOTE: the construction condition MUST include self.identity, or a workspace with
        # ONLY a self.md (no rules/skills) would leave prompt_builder=None and the loop's
        # default builder would ignore the loaded identity.
        if prompt_builder is None and (skills_catalog or rules_text or self.identity):
            prompt_builder = SystemPromptBuilder(
                identity=self.identity,
                extra_instructions=skills_catalog or None,
                rules=rules_text or None,
            )
        elif prompt_builder is not None:
            if self.identity and not prompt_builder.identity:
                prompt_builder.identity = self.identity
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
        # Multi-root sandbox (M-3): compute extra workspace roots from explicit args plus
        # auto-detected roots from --skill-dir. Computed HERE (before the sub-agent runner)
        # so the SAME sandbox is threaded into both the parent loop AND every child loop —
        # else a delegated --skill-dir-granted path would hit PathEscapeError. (audit4 #4)
        computed_extra_roots: list[Path] = []
        if extra_workspace_roots:
            computed_extra_roots.extend(Path(r) for r in extra_workspace_roots)
        if extra_skill_dirs:
            for sd in extra_skill_dirs:
                computed_extra_roots.extend(_infer_roots_from_skill_dir(Path(sd)))

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
                registry=default_registry(self.settings),  # task-free registry for children
                settings=self.settings,
                budget=shared_budget,
                permission_policy=self.permission_policy,
                hook_manager=self.hook_manager,
                workspace_root=self.settings.workspace_root,
                extra_workspace_roots=computed_extra_roots,  # same sandbox as the parent
                rules=rules_text or None,  # sub-agents inherit the parent's always-on rules
                provider_for=self._provider_for,  # per-role model routing (model_roles)
            )
            # general-purpose (full toolset) + plan (read-only planner whose registry
            # subset omits write tools, so Plan Mode is schema-enforced). Apply optional
            # per-role model overrides: model_roles['subagent'] -> the general delegate,
            # model_roles['planner'] -> the plan sub-agent. model_copy(model=None) leaves the
            # model unset (use default_model), so an empty model_roles is the unchanged default.
            roles = self.settings.model_roles
            general_def = GENERAL_PURPOSE.model_copy(update={"model": roles.get("subagent")})
            plan_def = PLAN.model_copy(update={"model": roles.get("planner")})
            spawner = SubAgentManager(runner, [general_def, plan_def], default=general_def.name)
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
            # Same store the recall hook + remember/recall tools use, so harness-authored
            # recovery lessons (research R1) are recalled next session. None when memory is off.
            memory_provider=self.memory,
            # Per-role model routing: a cheaper/local model for compaction summaries when the
            # mind configures model_roles['summarizer']; None = use the generator's provider.
            summarizer_provider=(
                self._provider_for(self.settings.model_roles["summarizer"])
                if "summarizer" in self.settings.model_roles
                else None
            ),
        )

    def _build_provider(self, model: str) -> Provider:
        """Build a settings-based provider for ``model`` (litellm wrapped in the text-tool
        protocol) — the same construction used for the default model and for per-role overrides.
        """
        from zakcode.providers.litellm_provider import LiteLLMProvider
        from zakcode.providers.text_tools import TextToolCallingProvider

        if model == self.settings.default_model:
            role_settings = self.settings
        else:
            update: dict[str, object] = {"default_model": model}
            # api_base/api_key are ENDPOINT-specific. Don't carry the default model's custom
            # endpoint onto a routed model on a DIFFERENT backend — e.g. an ollama_chat/* role
            # would otherwise be sent to the default's OpenAI gateway. (review: cross-backend
            # endpoint bleed) Same backend keeps sharing the endpoint (correct).
            default_backend = self.settings.default_model.split("/", 1)[0].lower()
            if model.split("/", 1)[0].lower() != default_backend:
                update["api_base"] = None
                update["api_key"] = None
            role_settings = self.settings.model_copy(update=update)
        return TextToolCallingProvider(
            LiteLLMProvider(role_settings),
            mode=_resolve_tool_calling_mode(role_settings.tool_calling_mode, model),
            single_tool_per_turn=True,
        )

    def _provider_for(self, model: str | None) -> Provider:
        """Resolve a per-role model string to a :class:`Provider` (the model-routing seam).

        Returns the default ``self.provider`` for ``None``, the default model, or when the
        provider was injected (an injected test/eval provider can't be rebuilt per model, so
        every role uses it). Otherwise builds — and caches — a provider for that model.
        """
        if not model or model == self.settings.default_model or self._provider_injected:
            return self.provider
        if model not in self._provider_cache:
            self._provider_cache[model] = self._build_provider(model)
        return self._provider_cache[model]

    async def invoke_skill(self, name: str) -> SkillInvocation:
        """Load a discovered skill's body into the session and emit the selection signal.

        The single CORE entry point for "use this skill", so every client (the CLI ``/<skill>``
        path today; a future server route or a model-facing tool) injects the body identically
        AND fires the observe-only :attr:`~zakcode.hooks.HookEvent.ON_SKILL_SELECTED` lifecycle
        hook. That hook is the seam a learning "mind" records ``(query -> skill)`` from to learn
        habitual skill preferences — the substrate emits the signal; choosing/learning is the
        mind's job. Never raises: a missing/unreadable skill file is a UX result, not a crash.
        """
        registry = getattr(self, "skill_registry", None)
        skill = registry.get(name) if registry is not None else None
        if skill is None:
            return SkillInvocation(invoked=False)
        # Capture the triggering context (the user's last natural-language turn) BEFORE the
        # skill body is injected, so a learner can associate the query with the chosen skill.
        query = self._recent_user_text()
        try:
            body = skill.body()  # L1 read lazily; the file may have changed/vanished
        except Exception as exc:  # noqa: BLE001 — a bad skill file is a UX error, not a crash
            return SkillInvocation(invoked=True, name=skill.name, error=str(exc))
        from zakcode.providers.text_tools import defang_untrusted

        # File-authored body folded into a TRUSTED user message; defang protocol/template
        # sentinels so a skill file can't forge a frame in text mode.
        self.session.add_message(Message.user(f"[skill: {skill.name}]\n{defang_untrusted(body)}"))
        await self._emit_skill_selected(skill.name, query)
        return SkillInvocation(invoked=True, name=skill.name)

    def _recent_user_text(self) -> str:
        """The most recent user message text (the query that motivated a skill), or ''."""
        for message in reversed(self.session.messages):
            if message.role == "user" and message.text:
                return message.text
        return ""

    async def _emit_skill_selected(self, skill_name: str, query: str) -> None:
        """Fire the observe-only ON_SKILL_SELECTED lifecycle hook (cheap + failure-isolated)."""
        with contextlib.suppress(Exception):
            await self.hook_manager.fire(
                LifecyclePayload(
                    event=HookEvent.ON_SKILL_SELECTED,
                    session_id=self.session.id,
                    cwd=str(self.settings.workspace_root),
                    data={"skill": skill_name, "query": query, "source": "command"},
                )
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
        with contextlib.suppress(Exception):
            await self.loop.aclose()  # tear down the egress-proxy listener (no-op when off)
        await self.aclose_mcp()

    @classmethod
    def for_workspace(cls, path: str | Path, **setting_overrides: Any) -> Agent:
        """Construct an :class:`Agent` pinned to ``path`` as the workspace root."""
        return cls(workspace_root=Path(path), **setting_overrides)
