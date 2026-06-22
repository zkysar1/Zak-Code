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

import asyncio
import contextlib
import logging
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass
from importlib.metadata import version as _pkg_version
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from zakcode.agent.budget import IterationBudget
from zakcode.agent.compact import Compactor
from zakcode.agent.loop import AgentLoop, TurnResult
from zakcode.agent.prompt import SystemPromptBuilder
from zakcode.config import Settings, load_settings
from zakcode.events import AgentEvent
from zakcode.hooks import HookEvent, HookManager, LifecyclePayload
from zakcode.messages import Message
from zakcode.permissions import PermissionPolicy, PermissionPrompter
from zakcode.providers.base import Provider, ProviderError
from zakcode.providers.resolve import AUTO_SENTINEL, ZAKPICK_SENTINEL, ResolvedModel
from zakcode.session.store import Session, SessionStore
from zakcode.tools.base import SkillLoad, SkillResolver
from zakcode.tools.builtins.default_registry import default_registry

__version__: str = _pkg_version("zakcode")

if TYPE_CHECKING:
    # Type-only imports for MCP/plugin annotations. Kept out of the runtime import
    # graph so importing ``zakcode`` never pulls in those subsystems; the concrete
    # imports happen inside ``__init__`` only when the feature is enabled.
    from zakcode.mcp.config import McpServerConfig
    from zakcode.mcp.manager import DiscoveryReport, ExtensionManager
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


class _SkillToolResolver:
    """Adapts the Agent's skill registry + selection signal into the
    :class:`~zakcode.tools.base.SkillResolver` the ``use_skill`` tool calls.

    Holds the ``Agent`` by reference so :meth:`names`/:meth:`load` always see the live registry,
    and routes :meth:`load` through the SAME core the CLI ``/<name>`` path uses
    (:meth:`Agent._load_skill_body`) — so a model-driven invocation reads/defangs the body and
    fires ``ON_SKILL_SELECTED`` identically, only with ``source="tool"``.
    """

    def __init__(self, agent: Agent) -> None:
        self._agent = agent

    def names(self) -> list[str]:
        registry = self._agent.skill_registry
        return registry.names() if registry is not None else []

    async def load(self, name: str, *, query: str = "") -> SkillLoad:
        # ``query`` is the INVOKING turn's prompt (a sub-agent's task, not the parent's), so the
        # signal is attributed to the actual caller even though the resolver is the parent's.
        return await self._agent._load_skill_body(name, source="tool", query=query)


#: Turn stop-reasons that count as a STALL — the agent didn't cleanly finish. Seam B's best-of-N
#: retry fires only on these (never on ``completed`` / ``budget_exhausted``).
_STALL_STOPS = frozenset(
    {"doom_loop", "stuck", "recipe_stalled", "verification_failed", "max_iterations"}
)


class Agent:
    """High-level facade over the agent loop.

    Construct with defaults (settings come from env / ``.env``) or pass an
    explicit :class:`Settings`, :class:`Session`, or :class:`SessionStore`.
    Keyword overrides are forwarded to :func:`~zakcode.config.load_settings`.

    Pass ``enable_subagents=True`` to expose the ``task`` delegation tool and a
    shared iteration budget (off by default). Pass ``enable_mcp=True`` to wire
    configured MCP servers; their tools register on ``await connect_mcp()``.
    Pass ``enable_context_gathering=True`` for a deterministic within-session
    context gatherer that injects relevant workspace context every turn (off by
    default; see :mod:`zakcode.context`). Set ``context_classifier="model"`` to
    rank that context with one cheap model call per turn (fail-soft to the
    heuristic), and ``context_signal_log=<path>`` to append each turn's
    offered-vs-used relevance signal as JSONL (training data for a learned ranker).
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
        enable_compaction: bool = False,
        enable_context_gathering: bool = False,
        context_classifier: str = "heuristic",
        context_signal_log: str | None = None,
        enable_settings_hooks: bool | None = None,
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
        # default_model == "auto" engages the availability resolver (PKG-AUTO): resolve
        # to a concrete model BEFORE anything reads default_model (provider build,
        # session record, role routing). Skipped when a provider is injected — hermetic
        # callers must never trigger a network probe. Resolution failure is LOUD
        # (ModelResolutionError with a per-source diagnosis).
        self.model_resolution: ResolvedModel | None = None
        # zakpick task-category routing (default_model="zakpick"): route each call site to the
        # model the user assigned to its category (or the built-in default). Resolve a concrete
        # STARTUP model now — the deep_code category's model, the safe/capable one — so provider
        # build / session record / count_tokens have a real model; per-turn routing then
        # downgrades easy turns. No availability probing: the user's assignment is used directly,
        # and a bad/missing key fails at call time like any provider error (fallback_model
        # applies). Gate all zakpick behavior on ``self._zakpick`` — default_model is rewritten
        # to the startup model below.
        self._zakpick = False
        if not self._provider_injected and self.settings.default_model == AUTO_SENTINEL:
            from zakcode.providers.resolve import resolver_for

            self.model_resolution = resolver_for(self.settings).resolve(require_tools=True)
            self.settings = self.settings.model_copy(
                update={"default_model": self.model_resolution.model}
            )
            logger.info(
                "auto model resolution: %s (%s)",
                self.model_resolution.model,
                self.model_resolution.reason,
            )
        elif not self._provider_injected and self.settings.default_model == ZAKPICK_SENTINEL:
            from zakcode.providers.routing import model_for_category

            self._zakpick = True
            startup_model = model_for_category("deep_code", self.settings)
            self.settings = self.settings.model_copy(update={"default_model": startup_model})
            logger.info("zakpick: routing per task category; startup model %s", startup_model)
        # Cache of role providers built for a non-default model, keyed by model string, so
        # spawning N sub-agents on the same role model doesn't rebuild the litellm wrapper N×.
        self._provider_cache: dict[str, Provider] = {}
        if provider is not None:
            self.provider = provider
        else:
            self.provider = self._build_provider(self.settings.default_model)
        #: The model currently driving the main loop (failover may move it off default).
        self._active_model = self.settings.default_model
        self.registry = default_registry(self.settings)
        # Per-task tool-exposure filter (self-remediation Step 4): narrow the model-facing
        # toolset to the operator's allow/deny globs before any (possibly untrusted) content
        # enters context. Applied lazily at definitions()-time, so it also covers MCP tools
        # discovered later via connect_mcp(). Empty lists (the default) = no restriction.
        self.registry.set_exposure_filter(
            allow=list(self.settings.tool_exposure_allow),
            deny=list(self.settings.tool_exposure_deny),
        )
        self.store = session_store
        self.session = session or Session(
            cwd=str(self.settings.workspace_root),
            model=self.settings.default_model,
        )
        # Deny-first by construction: the facade always builds a permission policy
        # from settings.permission_mode (default 'ask'). An interactive client may
        # pass a ``prompter`` so escalations can be approved; with none, 'ask'
        # fails closed (writes/shell denied) — safe for non-interactive use.
        from zakcode.deps_gate import read_declared_packages
        from zakcode.permissions import compile_deny_patterns, compile_protected_paths

        # Declared-dependency gate (self-remediation Step 1): when enabled, give the policy a
        # lazy reader of the workspace's declared package set. It is invoked only when a command
        # actually names an install, and re-reads each time so a package added mid-session is
        # recognised. ``None`` (gate off) leaves the policy's pure matrix unchanged.
        workspace_root = self.settings.workspace_root
        declared_packages = (
            (lambda: read_declared_packages(workspace_root))
            if self.settings.dependency_gate
            else None
        )
        self.permission_policy = permission_policy or PermissionPolicy(
            self.settings.permission_mode,
            prompter=prompter,
            extra_dangerous_patterns=compile_deny_patterns(self.settings.denied_commands),
            # Opt-in egress gate: confirm every web_fetch before it reaches the network.
            confirm_tools={"web_fetch"} if self.settings.web_fetch_confirm else None,
            # Per-tool trust overrides (audit P0-2b / D12) — validated at Settings load.
            tool_mode_overrides=dict(self.settings.tool_trust_overrides),
            declared_packages=declared_packages,
            # Protected-path floor extras (self-remediation Step 2): operator-added patterns
            # appended to the built-in .git/.env/venv/config floor.
            extra_protected_paths=compile_protected_paths(self.settings.protected_paths),
        )
        # Rehydrate operator grants persisted with the session (audit P0-2d / D12 / Q5).
        # Honored only when the active mode is at least as loose as the grant-time mode;
        # a fresh session carries no grants, so this is a no-op there.
        self.permission_policy.restore_grants(self.session.permission_grants)
        self.hook_manager = hook_manager or HookManager()
        # settings.json hook ingestion (PR-T5).  TE-R3(3): when a caller passes
        # BOTH enable_settings_hooks=True AND a programmatic hook_manager, the
        # settings.json specs are APPENDED to the existing manager's shell_hooks.
        # None defers to Settings.settings_hooks (ZAKCODE_SETTINGS_HOOKS); an explicit
        # True/False from the host wins either way.
        if (
            enable_settings_hooks
            if enable_settings_hooks is not None
            else self.settings.settings_hooks
        ):
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
        #: Model-driven skill invocations (the use_skill tool) — per-turn (reset each top-level
        #: turn; the bound for ``skill_invocation_budget``) and session-cumulative (for /skills).
        #: Shared across the whole turn-tree: a sub-agent's use_skill goes through this Agent's
        #: resolver, so the counters cover the parent + its sub-agents. Always initialized so the
        #: per-turn reset is safe even when skills are disabled.
        self._skill_invocations_this_turn = 0
        self._skill_invocations_total = 0
        skill_resolver: SkillResolver | None = None
        skills_catalog = ""
        if enable_skills:
            from zakcode.skills import discover_skills, project_skills_dir
            from zakcode.tools.builtins.save_skill import SaveSkillTool
            from zakcode.tools.builtins.use_skill import UseSkillTool

            self.skill_registry, self.skill_errors = discover_skills(
                self.settings.workspace_root, extra_skill_dirs=extra_skill_dirs
            )
            skills_catalog = self.skill_registry.render_catalog()
            # Model-driven skill authoring (persisted to the project skills dir; the
            # new skill is discovered next session). save_skill validates the name so
            # the write can never escape that directory.
            self.registry.register(SaveSkillTool(project_skills_dir(self.settings.workspace_root)))
            # Model-facing skill INVOCATION: the use_skill tool loads a skill's instructions by
            # name (and lets skills chain). It reads the resolver off the ToolContext, which the
            # loop is handed below; only registered when skills are on, so the default tool
            # surface is unchanged. (Gated identically to the catalog so the two stay consistent.)
            self.registry.register(UseSkillTool())
            skill_resolver = _SkillToolResolver(self)

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
        # untrusted tool output or host-injected context); never wire a request-supplied identity
        # through this path.
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

        # Build the shared budget when delegation is on (its original reason) OR when an
        # optional cost/token ceiling is configured (parity #4) — so a single non-delegating
        # agent still gets a cost cap. None otherwise (the per-turn iteration cap is the only
        # bound), keeping an ordinary ceiling-free Agent byte-for-byte unchanged.
        shared_budget = budget
        if shared_budget is None and (
            enable_subagents
            or self.settings.max_cost_usd is not None
            or self.settings.max_tokens is not None
        ):
            shared_budget = IterationBudget(
                self.settings.max_iterations,
                max_cost_usd=self.settings.max_cost_usd,
                max_tokens=self.settings.max_tokens,
            )
        # Held so the deep_think sampler can fold its extra calls into the same turn-tree budget.
        self._shared_budget = shared_budget
        spawner = None
        if enable_subagents:
            from zakcode.agent.subagent import (
                GENERAL_PURPOSE,
                PLAN,
                SubAgentManager,
                SubAgentRunner,
            )
            from zakcode.tools.builtins.task import TaskTool

            # Guaranteed non-None: the construction above runs whenever enable_subagents.
            assert shared_budget is not None
            # Task-free registry for children (omits the ``task`` tool to bar delegation
            # recursion). It must carry the SAME tool-exposure filter (Step 4): a tool the
            # operator denied for the session must stay hidden from a delegated sub-agent too
            # (a sub-agent reads untrusted content as well), else a `bash`-denied session could
            # spawn a child that still has `bash`. (subset() also copies the filter, covering the
            # allowed_tools path; this covers the full-registry path.)
            child_registry = default_registry(self.settings)
            child_registry.set_exposure_filter(
                allow=list(self.settings.tool_exposure_allow),
                deny=list(self.settings.tool_exposure_deny),
            )
            # Give the general-purpose delegate the same skill-invocation surface as the parent:
            # use_skill on the child registry + the resolver below. (Plan Mode's subset omits it,
            # so a read-only planner still cannot chain skills.) Gated on enable_skills via the
            # registry's presence, so a skills-off session adds nothing to the child surface.
            if self.skill_registry is not None:
                from zakcode.tools.builtins.use_skill import UseSkillTool

                child_registry.register(UseSkillTool())
            runner = SubAgentRunner(
                provider=self.provider,
                registry=child_registry,
                settings=self.settings,
                budget=shared_budget,
                permission_policy=self.permission_policy,
                hook_manager=self.hook_manager,
                workspace_root=self.settings.workspace_root,
                extra_workspace_roots=computed_extra_roots,  # same sandbox as the parent
                rules=rules_text or None,  # sub-agents inherit the parent's always-on rules
                provider_for=self._provider_for,  # per-role model routing (model_roles)
                # zakpick: a definition that names a CATEGORY but no model routes by task. Only
                # wired under zakpick (None otherwise), so the default delegation path is unchanged.
                provider_for_task=(self._provider_pair_for_task if self._zakpick else None),
                # The parent's skill resolver — so a delegated general-purpose agent can invoke
                # (and chain) the same skills, drawing from the shared per-turn skill budget.
                skill_resolver=skill_resolver,
            )
            # general-purpose (full toolset) + plan (read-only planner whose registry
            # subset omits write tools, so Plan Mode is schema-enforced). Apply optional
            # per-role model overrides: model_roles['subagent'] -> the general delegate,
            # model_roles['planner'] -> the plan sub-agent. model_copy(model=None) leaves the
            # model unset (use default_model), so an empty model_roles is the unchanged default.
            # The zakpick CATEGORY is set on each definition (delegate/plan); it only routes when
            # no model override is set AND the runner has the provider_for_task hook (zakpick on),
            # so model_roles always wins over the category default.
            roles = self.settings.model_roles
            general_def = GENERAL_PURPOSE.model_copy(
                update={"model": roles.get("subagent"), "category": "delegate"}
            )
            plan_def = PLAN.model_copy(update={"model": roles.get("planner"), "category": "plan"})
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

        # Cross-session MEMORY is NOT a harness concern (see docs/PERSISTENCE-BOUNDARY.md): the
        # substrate records the transcript (SessionStore, for /resume) and exposes generic seams —
        # register_context (PRE_LLM_CALL injection), register_lifecycle, register_turn_end, and the
        # tool registry — that claude-mind (or any framework) attaches its own recall/remember to.
        # The harness ships no store, no recall, no remember/recall tools.

        # An opt-in convenience ON that same generic seam: a deterministic, within-session
        # context gatherer (current workspace/session only — NOT the cross-session memory the
        # boundary removed). It runs every turn so context-gathering can't be silently skipped;
        # relevance ranking is swappable behind context.RelevanceClassifier (a zero-model
        # heuristic by default). Off by default — the clean substrate ships no gatherer.
        if enable_context_gathering:
            from zakcode.context import default_gatherer

            if context_classifier == "model":
                from zakcode.context import SmallModelClassifier

                # The cheap relevance call rides the context seam; SmallModelClassifier exposes
                # on_usage for callers that want to account the (small) spend.
                gatherer = default_gatherer(SmallModelClassifier(self.provider))
            else:
                gatherer = default_gatherer()
            self.hook_manager.register_context(gatherer)
            if context_signal_log:
                from zakcode.context import SignalLogger

                self.hook_manager.register_turn_end(
                    SignalLogger(gatherer, self.session, context_signal_log).on_turn_end
                )

        # Compaction (M8), opt-in. When enabled, the loop auto-compacts the session
        # before a turn once it exceeds the provider's context-window threshold.
        self.compactor: Compactor | None = None
        if enable_compaction:
            self.compactor = Compactor()

        # Compaction summarizer routing. Precedence: an explicit model_roles['summarizer']
        # wins; else under zakpick the "summarize" category routes to a cheap model; else None
        # (use the generator's provider). So model_roles still overrides the zakpick default.
        if "summarizer" in self.settings.model_roles:
            summarizer_provider = self._provider_for(self.settings.model_roles["summarizer"])
        elif self._zakpick:
            summarizer_provider = self._resolve_task_provider("summarize")[0]
        else:
            summarizer_provider = None

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
            summarizer_provider=summarizer_provider,
            # Runtime model failover (PKG-AUTO): once per turn, on a non-rate-limit
            # provider failure, the loop may swap to a new provider we build here.
            # None when the provider was injected (it can't be rebuilt from settings).
            model_failover=(None if self._provider_injected else self._model_failover),
            # zakpick per-turn main-model routing (None unless zakpick is on, so the legacy
            # single-provider path is byte-identical). The cheap→capable "escalation" is the soft
            # quick_code→deep_code latch in the loop — it only ever switches between the two coder
            # models the user configured, never a substitute Zak Code chose.
            main_provider_for=(self._main_provider_for if self._zakpick else None),
            # zakpick base-difficulty router: a cheap classify-model call judges each turn's SCOPE
            # (quick vs deep) instead of message length, so a terse-but-large request ("build a pdf
            # reader") reaches the capable coder instead of stalling the cheap one. Wired only for a
            # real (rebuildable) zakpick provider; None elsewhere keeps the length heuristic.
            difficulty_classifier=(
                self._classify_difficulty
                if (self._zakpick and not self._provider_injected)
                else None
            ),
            # deep_think's model access: sample the strongest configured model (zakpick deep_code,
            # else default_model). Always wired for a real Agent; the tool no-ops on a bare loop.
            sampler=self._deep_think_sample,
            # use_skill's loader: resolve a skill name -> instructions and fire the selection
            # signal. None unless enable_skills, so the use_skill tool (also only registered then)
            # has its seam exactly when skills are on.
            skill_resolver=skill_resolver,
            # TURN_END veto seam (T2/T3/T4): 0 (the default) disables the gate.
            turn_end_veto_budget=self.settings.turn_end_veto_budget,
            # Completion-review gate: 0 (the default) disables it; when >0, a code-changing turn
            # is sent back that many times to verify it satisfied the request before finishing.
            completion_review_attempts=self.settings.completion_review_attempts,
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

    def _model_failover(self, exc: ProviderError) -> tuple[Provider, str] | None:
        """Pick a replacement model after a provider failure (the loop's failover seam).

        Precedence (D19): an explicitly configured ``fallback_model`` is the override
        of the auto chain — it is tried first (if not already active); otherwise, when
        the session started from ``default_model="auto"``, auto resolution re-runs
        WITHOUT the probe cache (re-probe on failure) and with the failed model
        excluded. Returns ``(new_provider, description)`` or ``None`` when there is
        nowhere better to go (the loop then ends the turn as provider_error).
        """
        failed = self._active_model
        fallback = self.settings.fallback_model
        if fallback and fallback != failed:
            new_model, reason = fallback, "fallback_model (explicit override)"
        elif self.model_resolution is not None:
            # Availability re-resolution is the "auto" recovery path. zakpick never lands here
            # (its model_resolution stays None): the user assigned a concrete model per category,
            # so failover is the explicit fallback_model above or nothing — Zak Code never picks
            # a substitute the user didn't choose.
            from zakcode.providers.resolve import ModelResolutionError, resolver_for

            try:
                resolved = resolver_for(self.settings, use_cache=False).resolve(
                    require_tools=True, exclude={failed}
                )
            except ModelResolutionError as resolution_exc:
                logger.warning("auto re-resolution found no alternative: %s", resolution_exc)
                return None
            new_model, reason = resolved.model, resolved.reason
        else:
            return None
        logger.warning("model failover: %s -> %s (%s) after: %s", failed, new_model, reason, exc)
        provider = self._provider_for(new_model)
        self._active_model = new_model
        return provider, f"{failed} -> {new_model} ({reason})"

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

    # ── zakpick task-category routing seams (active only when default_model="zakpick") ──────

    def _resolve_task_provider(self, category: str) -> tuple[Provider, str]:
        """Resolve a zakpick task ``category`` to ``(provider, litellm model string)``.

        Returns the current provider/active model when zakpick is off or a provider was injected
        (an injected provider can't be rebuilt per model). Otherwise looks up the user's
        assignment (or the built-in Groq default) for the category and reuses ``_provider_for``
        so the provider cache and cross-backend endpoint guard apply unchanged. No availability
        probing — the user's model is used directly; if it fails at call time, ``fallback_model``
        handles it like any other provider error.
        """
        if not self._zakpick or self._provider_injected:
            return self.provider, self._active_model
        from zakcode.providers.routing import model_for_category

        model = model_for_category(category, self.settings)
        return self._provider_for(model), model

    def _provider_pair_for_task(self, category: str) -> tuple[Provider, str]:
        """``(provider, model)`` for a category-routed sub-agent (plan / delegate)."""
        return self._resolve_task_provider(category)

    def _main_provider_for(self, category: str) -> Provider:
        """The loop's main-turn router: select the generator's provider for ``category`` and
        update the active-model bookkeeping (so a later failover excludes the right model).

        The loop calls this ONLY when the classified category changes, so updating
        ``_active_model`` here is always a real selection — never a spurious reset that would
        fight an in-progress failover swap.
        """
        provider, model = self._resolve_task_provider(category)
        self._active_model = model
        return provider

    async def _classify_difficulty(
        self, user_text: str, context_frac: float
    ) -> Literal["quick_code", "deep_code"]:
        """zakpick base-difficulty router (the activated ``classify`` category): judge the
        request's SCOPE with a cheap classify-model call, not its LENGTH — a terse "build a pdf
        reader and maker" is a deep task no character count reveals. Returns ``"quick_code"`` or
        ``"deep_code"``.

        Only the AMBIGUOUS short case (where the length heuristic would say quick) consults the
        model; a long request or an already-large context fast-paths to ``deep_code`` with NO
        call. Any failure — provider error, or output that never validated — FAILS UP to
        ``deep_code``, so Zak Code never guesses "quick" from length. The cheap call's spend is
        accounted on the session + shared budget like any other model call (tagged to the classify
        model), so it is visible in ``/cost`` and bounded by the turn-tree budget — never hidden.
        """
        from zakcode.providers.routing import (
            DIFFICULTY_SCHEMA,
            difficulty_system_prompt,
            parse_difficulty,
            should_consult_classifier,
        )
        from zakcode.providers.structured import (
            StructuredValidationError,
            coerce_structured,
            make_response_format,
        )

        if not should_consult_classifier(user_text, context_frac):
            return "deep_code"  # long / large-context -> capable coder, no model call
        provider, model = self._resolve_task_provider("classify")
        try:
            # json_OBJECT mode (native JSON), NOT json_schema: litellm implements a json_schema
            # response_format on Groq via FUNCTION CALLING, and Groq's open models (incl. the
            # cheap llama-3.1-8b classifier) flakily emit a malformed tool call the provider
            # rejects (tool_use_failed) — the very unreliability zakpick routes around. Plain JSON
            # mode sidesteps the tool path; the trivial {"difficulty": ...} object is validated
            # locally against DIFFICULTY_SCHEMA below.
            result = await provider.acomplete(
                [Message.user(user_text)],
                system=difficulty_system_prompt(),
                response_format=make_response_format(None),
                temperature=0.0,
            )
        except ProviderError:
            return "deep_code"  # classifier unavailable -> fail UP to the reliable coder
        with contextlib.suppress(Exception):  # accounting must never break routing
            self.session.add_usage(result.usage, model=model)
            if self._shared_budget is not None:
                self._shared_budget.add_usage(result.usage.cost_usd, result.usage.total_tokens)
        try:
            data = coerce_structured(result.text, schema=DIFFICULTY_SCHEMA)
        except StructuredValidationError:
            return "deep_code"  # output was not schema-valid JSON -> fail UP
        return parse_difficulty(data)

    async def _deep_think_sample(
        self, prompt: str, *, system: str | None = None, temperature: float = 0.0
    ) -> str:
        """The ``deep_think`` :class:`~zakcode.tools.base.Sampler`: one completion on the agent's
        strongest model (zakpick ``deep_code``, else ``default_model``), with the spend accounted.

        Routes through ``_resolve_task_provider`` so under zakpick a deliberation uses the user's
        capable coder even when the current turn is on the cheap one. The usage is tagged for the
        ``/cost`` per-model breakdown and folded into the shared turn-tree budget, so a
        deliberation's cost is visible and bounded like any other model call — never hidden.
        """
        provider, _model = self._resolve_task_provider("deep_code")
        result = await provider.acomplete(
            [Message.user(prompt)], system=system, temperature=temperature
        )
        with contextlib.suppress(Exception):  # accounting must never break the deliberation
            self.session.add_usage(result.usage, model=provider.model_id())
            if self._shared_budget is not None:
                self._shared_budget.add_usage(result.usage.cost_usd, result.usage.total_tokens)
        return result.text

    async def _load_skill_body(
        self, name: str, *, source: str, query: str | None = None
    ) -> SkillLoad:
        """Resolve a skill, read+defang its L1 body, and fire the selection signal — the CORE
        shared by both invocation paths (the CLI ``/<name>`` and the model's ``use_skill`` tool).

        Does NOT mutate the session: it returns the body so each path delivers it the right way
        — the CLI folds it into a user message, the tool returns it as the tool result. ``query``
        is the triggering turn's prompt the ``ON_SKILL_SELECTED`` hook records so a learner can
        associate ``(query -> skill)``; the ``use_skill`` tool passes the INVOKING turn's prompt
        (so a sub-agent attributes the skill to ITS task, not the parent's), and a falsy ``query``
        (the CLI path, or a bare context) falls back to this session's recent user text. Fires the
        observe-only :attr:`~zakcode.hooks.HookEvent.ON_SKILL_SELECTED` hook with ``source``. Never
        raises: a missing skill yields ``found=False``; an unreadable one yields ``error`` set.

        Budget: a model-driven (``source="tool"``) invocation draws from the per-turn skill
        budget (:attr:`Settings.skill_invocation_budget`); over the cap it returns a
        ``denied_reason`` (no body, no signal) to stop a runaway/cyclic chain. A human
        ``/<name>`` (``source="command"``) is operator-controlled and never throttled.
        """
        registry = getattr(self, "skill_registry", None)
        skill = registry.get(name) if registry is not None else None
        if skill is None:
            return SkillLoad(found=False, name=name)
        if source == "tool":
            budget = self.settings.skill_invocation_budget
            if budget > 0 and self._skill_invocations_this_turn >= budget:
                return SkillLoad(
                    found=True,
                    name=skill.name,
                    denied_reason=(
                        f"skill invocation budget exhausted ({budget} this turn); "
                        "finish with the skills already loaded"
                    ),
                )
        # The invoking turn's prompt (the tool passes it; a sub-agent's differs from the parent's);
        # fall back to this session's recent user text for the CLI path or a bare/empty caller.
        query = query or self._recent_user_text()
        try:
            body = skill.body()  # L1 read lazily; the file may have changed/vanished
        except Exception as exc:  # noqa: BLE001 — a bad skill file is a UX error, not a crash
            return SkillLoad(found=True, name=skill.name, error=str(exc))
        from zakcode.providers.text_tools import defang_untrusted

        if source == "tool":  # count only model-driven loads that actually inject a body
            self._skill_invocations_this_turn += 1
            self._skill_invocations_total += 1
            logger.info(
                "skill %r invoked via use_skill (%d this turn, %d this session)",
                skill.name,
                self._skill_invocations_this_turn,
                self._skill_invocations_total,
            )
        # Defang protocol/template sentinels so a file-authored body can't forge a frame in
        # text mode; the body is preserved verbatim otherwise (defang never deletes content).
        await self._emit_skill_selected(skill.name, query, source=source)
        return SkillLoad(found=True, name=skill.name, body=defang_untrusted(body))

    async def invoke_skill(self, name: str) -> SkillInvocation:
        """Load a discovered skill's body into the session and emit the selection signal.

        The CLI ``/<skill>`` entry point: it folds the body into a TRUSTED user message so the
        next turn applies the skill. Shares :meth:`_load_skill_body` with the model-facing
        ``use_skill`` tool (which instead returns the body as its tool result), so both paths
        read, defang, and fire ``ON_SKILL_SELECTED`` identically. Never raises: a
        missing/unreadable skill file is a UX result, not a crash.
        """
        load = await self._load_skill_body(name, source="command")
        if not load.found:
            return SkillInvocation(invoked=False)
        if load.error or load.body is None:
            return SkillInvocation(invoked=True, name=load.name, error=load.error)
        self.session.add_message(Message.user(f"[skill: {load.name}]\n{load.body}"))
        return SkillInvocation(invoked=True, name=load.name)

    @property
    def skill_invocations_this_session(self) -> int:
        """How many times the model has invoked a skill via ``use_skill`` this session (across
        the whole turn-tree). Surfaced by ``/skills`` so skill usage's cost is visible."""
        return self._skill_invocations_total

    def _recent_user_text(self) -> str:
        """The most recent user message text (the query that motivated a skill), or ''."""
        for message in reversed(self.session.messages):
            if message.role == "user" and message.text:
                return message.text
        return ""

    async def _emit_skill_selected(
        self, skill_name: str, query: str, *, source: str = "command"
    ) -> None:
        """Fire the observe-only ON_SKILL_SELECTED lifecycle hook (cheap + failure-isolated).

        ``source`` records HOW the skill was chosen (``"command"`` = the human ``/<name>`` path,
        ``"tool"`` = the model's ``use_skill`` call) so a learning mind can weight model-driven
        vs operator-driven selections.
        """
        with contextlib.suppress(Exception):
            await self.hook_manager.fire(
                LifecyclePayload(
                    event=HookEvent.ON_SKILL_SELECTED,
                    session_id=self.session.id,
                    cwd=str(self.settings.workspace_root),
                    data={"skill": skill_name, "query": query, "source": source},
                )
            )

    async def arun_turn(self, user_text: str) -> TurnResult:
        """Run one user turn asynchronously.

        Seam B: when the turn STALLS and ``best_of_attempts > 1`` with a ``verify_command`` set, fan
        out best-of-N isolated retries and adopt (diff-apply) the first that verifies.
        """
        self._skill_invocations_this_turn = 0  # new top-level turn: refill the skills budget
        result = await self.loop.arun_turn(user_text)
        if (
            self.settings.best_of_attempts > 1
            and self.settings.verify_command
            and result.stop_reason in _STALL_STOPS
        ):
            from zakcode.agent.best_of_attempts import run_best_of_attempts

            result = await run_best_of_attempts(self, user_text, result)
        return result

    def run_turn(self, user_text: str) -> TurnResult:
        """Run one user turn synchronously (wraps :meth:`arun_turn`, so seam B applies).

        Refuses to run if an event loop is already active; await ``arun_turn`` from async code.
        """
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.arun_turn(user_text))
        raise RuntimeError(
            "run_turn() cannot be called from a running event loop; await arun_turn() instead."
        )

    def astream_turn(self, user_text: str) -> AsyncIterator[AgentEvent]:
        """Stream one user turn as a sequence of :class:`~zakcode.events.AgentEvent`.

        Returns the loop's async iterator directly (no ``await`` needed to obtain
        it), so callers can ``async for event in agent.astream_turn(text)``. This
        is the incremental counterpart to :meth:`run_turn` / :meth:`arun_turn`.
        """
        self._skill_invocations_this_turn = 0  # new top-level turn: refill the skills budget
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

        Fires the ``SESSION_END`` lifecycle hook (a host's encode/serialize step) and
        closes any MCP connections. Best-effort and idempotent — a second call is a no-op
        (so the encode step never double-runs), and every step is isolated so a failing one
        never blocks the rest.
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
                        data={
                            "trigger": "session_end",
                            "session_summary": self._session_summary(),
                        },
                    )
                )
        with contextlib.suppress(Exception):
            await self.loop.aclose()  # tear down the egress-proxy listener (no-op when off)
        await self.aclose_mcp()

    def _session_summary(self) -> dict[str, Any]:
        """Build a session-summary dict for lifecycle hook payloads (PR-T7)."""
        return {
            "session_id": self.session.id,
            "message_count": len(self.session.messages),
            "total_usage": self.session.cumulative_usage().model_dump(),
            "created_at": self.session.created_at,
        }

    @classmethod
    def for_workspace(cls, path: str | Path, **setting_overrides: Any) -> Agent:
        """Construct an :class:`Agent` pinned to ``path`` as the workspace root."""
        return cls(workspace_root=Path(path), **setting_overrides)
