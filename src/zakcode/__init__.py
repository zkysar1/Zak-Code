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
from typing import TYPE_CHECKING, Any

from zakcode.agent.budget import IterationBudget
from zakcode.agent.compact import Compactor
from zakcode.agent.loop import _MIN_ANSWER_ROOM, AgentLoop, TurnResult, _composed_skill_name
from zakcode.agent.prompt import SystemPromptBuilder
from zakcode.config import Settings, load_settings
from zakcode.events import AgentEvent
from zakcode.hooks import HookEvent, HookManager, LifecyclePayload
from zakcode.messages import Message
from zakcode.permissions import PermissionPolicy, PermissionPrompter
from zakcode.providers.base import Provider, ProviderError, UnknownContextWindow, WindowResolution
from zakcode.providers.resolve import AUTO_SENTINEL, ZAKPICK_SENTINEL, ResolvedModel
from zakcode.providers.routing import DifficultyVerdict
from zakcode.session.store import Session, SessionStore
from zakcode.skills.fit import SkillFit, measure_skill_fit
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
    from zakcode.status_line import StatusLineSpec

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


def _mind_external_roots(repo_root: Path) -> list[Path]:
    """External world/meta roots a Mind repo declares via ``agents/*/local-paths.conf``.

    Returns a deduplicated list of existing absolute directories.
    """
    roots: list[Path] = []
    seen: set[Path] = set()
    agents_dir = repo_root / "agents"
    if agents_dir.is_dir():
        for child in sorted(agents_dir.iterdir()):
            conf = child / "local-paths.conf"
            for p in _parse_local_paths_conf(conf):
                rp = p.resolve()
                if rp not in seen:
                    roots.append(rp)
                    seen.add(rp)
    return roots


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

        for rp in _mind_external_roots(repo_root):
            if rp not in seen:
                roots.append(rp)
                seen.add(rp)

    return roots


@dataclass(frozen=True)
class SkillInvocation:
    """Outcome of :meth:`Agent.compose_skill_turn` / :meth:`Agent.invoke_skill`.

    ``invoked`` is True iff ``name`` resolved to a discovered skill (so the caller treats it
    as handled, not an unknown command); ``error`` is set iff the skill's body failed to load.
    """

    invoked: bool
    name: str = ""
    error: str | None = None
    #: Set when a discovered skill exists but this invocation path is refused — e.g. a
    #: ``user-invocable: false`` skill typed as a human ``/<name>`` command (it runs internally,
    #: reached only by another skill chaining to it). Distinct from ``error`` (a load failure).
    denied_reason: str | None = None
    #: Typo tolerance (ADR-0040). ``corrected_from`` is the token the operator actually typed
    #: when the catalog held exactly one near-identical skill name and the harness ran that
    #: skill instead (``/enocde-session`` → ``encode-session``); the caller renders it so
    #: the correction is visible. ``suggestions`` carries the near names when NOTHING ran —
    #: an ambiguous or weak match is a did-you-mean, never a guess on the operator's behalf.
    corrected_from: str | None = None
    suggestions: tuple[str, ...] = ()
    #: The composed turn text when the load succeeded: Claude Code's command-expansion
    #: frame (``<command-message>``/``<command-name>``, plus ``<command-args>`` when trailing
    #: text was given) followed by the body. The frame is the INVOCATION-PROVENANCE signal —
    #: it tells the model a HUMAN typed this slash command, so skills whose own rules say
    #: "user-invocable only / the model must not invoke this" execute instead of refusing.
    #: :meth:`Agent.compose_skill_turn` leaves delivery to the caller — the CLI runs it as
    #: THIS turn's user message (Claude Code slash semantics); :meth:`Agent.invoke_skill`
    #: has already folded it into the session when this is set.
    turn_text: str | None = None


#: Typo tolerance thresholds (ADR-0040), difflib ratios over the skill catalog's names. A
#: name below SUGGEST is noise, not a neighbour; a UNIQUE neighbour at or above AUTOCORRECT
#: runs (``enocde-session`` scores 0.93 against ``encode-session``); two neighbours, or one
#: between the two lines, are offered back as did-you-mean. Only skills are ever corrected —
#: the REPL's own commands are suggestion-only, because a typo must not run ``/clear``.
_SKILL_SUGGEST_RATIO = 0.72
_SKILL_AUTOCORRECT_RATIO = 0.8


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

    async def load(self, name: str, *, query: str = "", args: str = "") -> SkillLoad:
        # ``query`` is the INVOKING turn's prompt (a sub-agent's task, not the parent's), so the
        # signal is attributed to the actual caller even though the resolver is the parent's.
        return await self._agent._load_skill_body(name, source="tool", query=query, args=args)

    def body(self, name: str) -> str | None:
        # The whole body, defanged like a load but with none of the ceremony (ADR-0067): the
        # loop seeds a skeleton and pages sections from it after a load delivered page 1 only.
        registry = self._agent.skill_registry
        skill = registry.resolve(name) if registry is not None else None
        if skill is None:
            return None
        try:
            text = skill.body()
        except Exception:  # noqa: BLE001 — an unreadable skill is the load's problem to report
            return None
        from zakcode.providers.text_tools import defang_untrusted

        return defang_untrusted(text)


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
    offered-vs-used relevance signal as JSONL (``context_signal_judge=True`` labels
    "used" with a cheap-model judge that also catches in-context use). Point
    ``context_classifier_weights`` at a model trained from that log
    (``zakcode.context.train_relevance``) to rank with it (fail-soft to the heuristic).
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
        # SDK-only hard per-turn iteration bound (tests, evals, embedders). None/0 =
        # unlimited — the only product behavior; there is no operator config for this
        # (ZAKCODE_MAX_ITERATIONS removed 2026-08-25, no-knobs ruling).
        max_iterations: int | None = None,
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
        lean_rules: bool | None = None,
        enable_identity: bool = True,
        identity: str | None = None,
        enable_compaction: bool = False,
        enable_context_gathering: bool = False,
        context_classifier: str = "heuristic",
        context_signal_log: str | None = None,
        context_classifier_weights: str | None = None,
        context_signal_judge: bool = False,
        enable_status_line: bool | None = None,
        enable_output_style: bool | None = None,
        agent_identity_dir: str | Path | None = None,
        **setting_overrides: Any,
    ) -> None:
        self.settings = settings or load_settings(**setting_overrides)
        # BYOK (g-369-11): a member's own provider key, saved in this environment's vault,
        # overlays the deployment's before anything reads a key. Placed HERE, immediately
        # after settings resolve and before model resolution / provider build, because the
        # availability resolver runs ONLY for default_model == "auto" — wiring it there
        # would have made BYOK silently inert for zakpick and for an explicit model.
        # No-op without a vault (see apply_vault_provider_keys).
        from zakcode.providers.resolve import apply_vault_provider_keys

        apply_vault_provider_keys(self.settings)
        # SDK-only per-turn iteration bound (see the kwarg comment); normalized so 0 and
        # None both mean unlimited everywhere downstream (loop + shared budget).
        self._max_iterations = max_iterations or 0
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
            from zakcode.providers.routing import model_spec_for_category

            self._zakpick = True
            startup_spec = model_spec_for_category("deep_code", self.settings)
            startup_model = startup_spec.litellm_string
            # The window travels WITH the model (ADR-0066): the startup model's entry
            # carries its own context_window, and the copied settings must say so too, or
            # the default provider would be built against Settings.context_window — which
            # describes a concrete ZAKCODE_DEFAULT_MODEL, not this category's model.
            self.settings = self.settings.model_copy(
                update={
                    "default_model": startup_model,
                    "context_window": startup_spec.context_window,
                }
            )
            logger.info("zakpick: routing per task category; startup model %s", startup_model)
        #: Every effective model's context window and its source (ADR-0066), by label —
        #: filled by ``_assert_context_windows`` for the info panel and the skill-fit check.
        self.context_windows: dict[str, WindowResolution] = {}
        #: Loud-but-not-fatal window findings (a server declaring a different window than
        #: the config), printed red at startup by the CLI.
        self.window_warnings: list[str] = []
        # Cost guarantee, checked once the sentinels above have resolved to concrete models so
        # every destination is knowable. Placed BEFORE the first provider is built (and long
        # before any completion), and safe to run after auto-resolution because the resolver
        # only makes read-only /v1/models probes — no billable call has happened yet.
        if not self._provider_injected:
            self._assert_local_only()
            self._assert_context_windows()
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
        # A RESUMED/injected session carries its OWN persisted cwd; realign it to THIS run's
        # workspace so every cwd-sensitive surface (tools, rules, TURN_END/Stop hooks) agrees on one
        # directory instead of splitting between the session's old dir and the active workspace.
        if session is not None and self.session.cwd != str(self.settings.workspace_root):
            logger.info(
                "resumed session %r recorded cwd %s; realigning to active workspace %s",
                self.session.id,
                self.session.cwd,
                self.settings.workspace_root,
            )
        self.session.cwd = str(self.settings.workspace_root)
        # Deny-first by construction: the facade always builds a permission policy
        # from settings.permission_mode (default 'ask'). An interactive client may
        # pass a ``prompter`` so escalations can be approved; with none, 'ask'
        # fails closed (writes/shell denied) — safe for non-interactive use.
        from zakcode.deps_gate import harness_declared_packages, read_declared_packages
        from zakcode.permissions import compile_deny_patterns, compile_protected_paths

        # Declared-dependency gate (self-remediation Step 1): when enabled, give the policy a
        # lazy reader of the workspace's declared package set. It is invoked only when a command
        # actually names an install, and re-reads each time so a package added mid-session is
        # recognised. ``None`` (gate off) leaves the policy's pure matrix unchanged.
        # The harness's OWN declared packages are unioned in (ADR-0019) so the self-service
        # remedy for a missing optional dep — which pip_install_hint aims at zakcode's own
        # interpreter — is not refused by the very gate that guards the workspace.
        workspace_root = self.settings.workspace_root
        declared_packages = (
            (lambda: read_declared_packages(workspace_root) | harness_declared_packages())
            if self.settings.dependency_gate
            else None
        )
        # Claude Code permissions.{allow,deny,ask} ingestion — UNCONDITIONAL since ADR-0029, the
        # same "declared config is live" posture as settings hooks (ADR-0025). The workspace's
        # ``permissions`` block IS the authorization posture: with the built-in agent-config
        # protected class removed, these ingested rules are the ONLY authority over ``.claude/``
        # (a framework that wants its config protected declares the denies here). Skipped only
        # when an explicit permission_policy was injected — an injected policy is the caller's
        # full authority. Gestures translate into the SAME tighten-only seams the operator
        # settings use, and UNION them: ingested deny command/path patterns are appended to the
        # operator's (the floor only grows), and ingested per-tool modes are laid down FIRST so
        # the operator's tool_trust_overrides (higher-authority local config) win on any
        # conflict. The always-on catastrophic + protected-path floor runs before any allow.
        denied_command_regexes = list(self.settings.denied_commands)
        protected_path_regexes = list(self.settings.protected_paths)
        write_only_path_regexes: list[str] = []
        ingested_tool_modes: dict[str, str] = {}
        ingested_denied_tools: set[str] = set()
        if permission_policy is None:
            from zakcode.permissions_settings import load_settings_permissions

            _ingested, _perm_errs = load_settings_permissions(workspace_root)
            for _key, _err in _perm_errs.items():
                logger.warning("settings.json permission %s: %s", _key, _err)
            # Union, tighten-only: ingested deny patterns extend the operator's; ingested per-tool
            # modes go under the operator's (operator wins a conflict). Read-denies join the
            # operator's strict (read+write) pool; Edit/Write-denies compile write-only below
            # (ADR-0030 — an ingested Edit deny must not block reading the path).
            denied_command_regexes.extend(_ingested.denied_command_regexes)
            protected_path_regexes.extend(_ingested.protected_path_regexes)
            write_only_path_regexes = list(_ingested.protected_path_regexes_write_only)
            ingested_tool_modes = dict(_ingested.tool_mode_overrides)
            # Whole-tool deny gestures → a tier-independent unconditional deny (binds read-only
            # tools too, which a mode override cannot). The operator cannot loosen these.
            ingested_denied_tools = set(_ingested.denied_tools)
        # Operator tool_trust_overrides overlay the ingested ones (operator is the trusted local
        # authority); the merged map feeds tool_mode_overrides below.
        merged_tool_modes = {**ingested_tool_modes, **dict(self.settings.tool_trust_overrides)}
        self.permission_policy = permission_policy or PermissionPolicy(
            self.settings.permission_mode,
            prompter=prompter,
            extra_dangerous_patterns=compile_deny_patterns(denied_command_regexes),
            # Opt-in egress gate: confirm every web_fetch before it reaches the network.
            confirm_tools={"web_fetch"} if self.settings.web_fetch_confirm else None,
            # Per-tool trust overrides (audit P0-2b / D12) — validated at Settings load — merged
            # with any ingested CC bare-tool deny/allow gestures (operator wins a conflict).
            tool_mode_overrides=merged_tool_modes,
            declared_packages=declared_packages,
            # Protected-path floor extras (self-remediation Step 2): operator-added patterns and
            # ingested Read-denies bind reads AND writes; ingested Edit/Write-denies compile
            # write-only (ADR-0030), all appended to the built-in .git/.env/venv floor.
            extra_protected_paths=(
                compile_protected_paths(protected_path_regexes)
                + compile_protected_paths(write_only_path_regexes, write_only=True)
            ),
            # Ingested whole-tool CC deny gestures — denied unconditionally, regardless of tier.
            extra_denied_tools=ingested_denied_tools,
            # Relative path arguments resolve against the workspace before the protected-path
            # scan (ADR-0031) so a ``*/``-prefixed deny glob binds a relative spelling too.
            workspace_root=workspace_root,
        )
        # Rehydrate operator grants persisted with the session (audit P0-2d / D12 / Q5).
        # Honored only when the active mode is at least as loose as the grant-time mode;
        # a fresh session carries no grants, so this is a no-op there.
        self.permission_policy.restore_grants(self.session.permission_grants)
        self.hook_manager = hook_manager or HookManager()
        # settings.json hook ingestion (PR-T5; UNCONDITIONAL since ADR-0025). A workspace's
        # declared hooks always load — no adoption flag, no folder-trust prompt, no env
        # toggle. Hooks are the workspace's own committed automation, and a framework whose
        # protections ride on them must never silently run unprotected (field incident: a
        # Mind deployment ran with ZERO of its 43 gates because the adoption ask was never
        # answered). The security floor is not a flag and is unchanged: every command is
        # danger-scanned (hard-denied in autonomous mode) and provider keys are scrubbed
        # from hook children. TE-R3(3): with a programmatic hook_manager, the settings.json
        # specs are APPENDED to its shell_hooks.
        from zakcode.hooks.settings_loader import load_settings_hooks

        _specs, _errs = load_settings_hooks(
            self.settings.workspace_root,
            permission_mode=str(self.settings.permission_mode),
        )
        for _key, _err in _errs.items():
            logger.warning("settings.json hook %s: %s", _key, _err)
        if _specs:
            self.hook_manager.shell_hooks.extend(_specs)
        # Claude Code statusLine support (cosmetic; opt-in). When on, load the configured
        # statusLine command from settings.json now (one read at construction, danger-scanned
        # + provider-key-scrubbed like a hook) and stash it for a client (the CLI) to render
        # after each turn. None defers to Settings.status_line; an explicit True/False wins.
        # The spec is None when nothing is configured or it was denied — a client just renders
        # no line. This NEVER touches the loop: a status line is decoration a client repaints.
        self.status_line_enabled: bool = (
            enable_status_line if enable_status_line is not None else self.settings.status_line
        )
        self.status_line_spec: StatusLineSpec | None = None
        if self.status_line_enabled:
            from zakcode.status_line import load_status_line_spec

            self.status_line_spec, _sl_err = load_status_line_spec(
                self.settings.workspace_root,
                permission_mode=str(self.settings.permission_mode),
            )
            if _sl_err:
                logger.warning("statusLine: %s", _sl_err)
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
        #: Per-turn reload dedup: skill name -> sha1 of the body already injected THIS turn.
        #: A same-turn use_skill of an unchanged skill returns a short pointer instead of the
        #: full body (measured 2026-08-25: one turn loaded the same ~1,200-line skill three
        #: times). Compaction fires only at turn START, so the earlier body is still in
        #: context for the whole turn; the dict resets with the invocation counter.
        self._skills_loaded_this_turn: dict[str, str] = {}
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
        # ``lean_rules`` (Vinheim Lever A) swaps the full-body render for a compact
        # one-line-per-rule INDEX, so a rules-heavy "mind" stops paying every rule's full
        # body on every cached turn — the model reads a rule's body on demand instead.
        # Default False keeps the always-on render byte-for-byte unchanged.
        self.rule_registry: RuleRegistry | None = None
        self.rule_errors: dict[str, str] = {}
        rules_text = ""
        if enable_rules:
            from zakcode.rules import discover_rules

            self.rule_registry, self.rule_errors = discover_rules(self.settings.workspace_root)
            # ``None`` defers to Settings.lean_rules (ZAKCODE_LEAN_RULES); an explicit
            # True/False from the host wins — the same deferral shape as
            # enable_status_line / enable_output_style above.
            # Before g-016-86 this was a hard ``False`` default, so the documented env
            # var reached the Agent through server/app.py ONLY: CLI, library and bench
            # constructions silently took the full render, and an A/B driven by the env
            # var returned byte-identical arms. Settings.lean_rules still defaults to
            # False, so an operator who sets nothing sees no change.
            use_lean = self.settings.lean_rules if lean_rules is None else lean_rules
            rules_text = (
                self.rule_registry.render_index() if use_lean else self.rule_registry.render()
            )
            # Vinheim Lever A chunk 2 (g-016-82): the retrieval half of the lean path. The
            # index names every rule but carries no bodies, so the model needs a cheap,
            # unambiguous way to fetch one — register ``read_rule`` whenever rules are on.
            # Registered for BOTH renders on purpose: under the full render the index header
            # is absent, but a rule dropped past MAX_RULES_TOTAL_CHARS is still reachable by
            # name, which is exactly the completeness gap the full render otherwise has.
            from zakcode.tools.builtins.read_rule import ReadRuleTool

            self.registry.register(ReadRuleTool())

        # Claude Code output style (opt-in): the active outputStyle's body, folded into the
        # SAME stable tier as rules so it shapes generation and stays cache-safe. Loaded here
        # (before delegation) so sub-agents inherit the same style. None defers to
        # Settings.output_style (ZAKCODE_OUTPUT_STYLE); an explicit True/False from the host
        # wins. Fully defensive: a missing/unknown style is recorded, never fatal, and leaves
        # output_style_text empty so the prompt is byte-identical to today.
        self.output_style_error: str | None = None
        output_style_text = ""
        if enable_output_style if enable_output_style is not None else self.settings.output_style:
            from zakcode.output_styles import load_active_output_style

            _style_block, self.output_style_error = load_active_output_style(
                self.settings.workspace_root
            )
            output_style_text = _style_block or ""
            if self.output_style_error and _style_block is None:
                # A configured-but-unloadable style (bad name/file) is a quiet footgun: the
                # selected voice is silently absent. Log it; a clean "unconfigured" reason
                # (the common no-op) is not worth a warning, so only log when something was
                # actually attempted and failed (handled inside load_active_output_style for
                # parse/read errors; this covers the host-facing surface).
                logger.debug("output style not injected: %s", self.output_style_error)

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
        if prompt_builder is None and (
            skills_catalog or rules_text or output_style_text or self.identity
        ):
            prompt_builder = SystemPromptBuilder(
                identity=self.identity,
                extra_instructions=skills_catalog or None,
                rules=rules_text or None,
                output_style=output_style_text or None,
            )
        elif prompt_builder is not None:
            if self.identity and not prompt_builder.identity:
                prompt_builder.identity = self.identity
            if skills_catalog:
                # Append (don't conditionally set) so an injected prompt_builder with its own
                # extra_instructions can't silently drop the skills catalog.
                prompt_builder.extra_instructions = (
                    f"{prompt_builder.extra_instructions}\n\n{skills_catalog}"
                    if prompt_builder.extra_instructions
                    else skills_catalog
                )
            if rules_text and not prompt_builder.rules:
                prompt_builder.rules = rules_text
            # Fill an injected builder's empty output-style slot so enable_output_style is
            # never a silent no-op (mirrors the rules slot above).
            if output_style_text and not prompt_builder.output_style:
                prompt_builder.output_style = output_style_text

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
        # A Mind workspace declares its EXTERNAL world/meta homes in
        # agents/*/local-paths.conf — the same inference --skill-dir repos already
        # get. Without this, file tools refuse the real world ("resolves outside
        # the workspace root") and a relative Write("world/…") lands in a stray
        # world/ INSIDE the repo (measured on serene, 2026-08-25: scripts written
        # to a divergent copy the framework never reads). Structural, no config.
        computed_extra_roots.extend(_mind_external_roots(Path(self.settings.workspace_root)))

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
                self._max_iterations,
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
                trace_session=self.session.id,  # children trace under this session's directory
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

            if context_classifier not in ("heuristic", "model"):
                logger.warning(
                    "context_classifier=%r is unrecognized (use 'heuristic' or 'model'); "
                    "falling back to the heuristic ranker",
                    context_classifier,
                )

            def _cheap_ctx_provider() -> Provider:
                # A cheap model for the every-turn context calls (the relevance classifier and the
                # used-judge): an explicit role wins, else zakpick's "classify" category, else the
                # generator's own provider.
                if "context_classifier" in self.settings.model_roles:
                    return self._provider_for(self.settings.model_roles["context_classifier"])
                if self._zakpick:
                    return self._resolve_task_provider("classify")[0]
                return self.provider

            if context_classifier_weights:
                from zakcode.context import RelevanceModel, TrainedClassifier

                try:
                    gatherer = default_gatherer(
                        TrainedClassifier(RelevanceModel.load(context_classifier_weights))
                    )
                except Exception:  # noqa: BLE001 - a missing/bad weights file falls back to heuristic
                    gatherer = default_gatherer()
            elif context_classifier == "model":
                from zakcode.context import SmallModelClassifier

                clf_provider = _cheap_ctx_provider()
                gatherer = default_gatherer(
                    SmallModelClassifier(
                        clf_provider,
                        on_usage=lambda u: self.session.add_usage(u, model=clf_provider.model_id()),
                    )
                )
            else:
                gatherer = default_gatherer()
            self.hook_manager.register_context(gatherer)
            if context_signal_log:
                from zakcode.context import SignalLogger

                # The `used` label defaults to the cheap reference proxy; context_signal_judge
                # upgrades it to a cheap-model judge that also catches IN-CONTEXT use (the gatherer
                # injects file content, so the model can consume it without naming the file -> the
                # proxy false-negatives).
                if context_signal_judge:
                    from zakcode.context import ModelUsedDetector

                    jp = _cheap_ctx_provider()
                    signal = SignalLogger(
                        gatherer,
                        self.session,
                        context_signal_log,
                        used_detector=ModelUsedDetector(
                            jp, on_usage=lambda u: self.session.add_usage(u, model=jp.model_id())
                        ),
                    )
                else:
                    signal = SignalLogger(gatherer, self.session, context_signal_log)
                self.hook_manager.register_turn_end_observer(signal.on_turn_end)

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
            # read_rule's source: the discovered rule registry, so the model can fetch a rule
            # body by name. None unless enable_rules (same shape as skill_resolver above), so
            # the tool's seam exists exactly when rules are on.
            rule_registry=self.rule_registry,
            # TURN_END veto seam (T2/T3/T4): structurally ALWAYS ON for the main loop —
            # a Stop hook registered by the workspace's adopted hooks fires at every
            # vetoable turn end. Sub-agent loops never set this (their completions
            # return to the parent). No knob (2026-08-25 no-knobs ruling).
            turn_end_vetoable=True,
            # SDK-only per-turn bound; 0 = unlimited (the only product behavior).
            max_iterations=self._max_iterations,
            # Completion-review gate: 0 (the default) disables it; when >0, a code-changing turn
            # is sent back that many times to verify it satisfied the request before finishing.
            completion_review_attempts=self.settings.completion_review_attempts,
            # A Stop-hook veto opens a fresh turn for per-turn skill state (ADR-0048): the
            # loop calls this the moment a TURN_END hook vetoes, so the veto's mandated
            # re-entry (a skill loaded earlier in the same turn) gets its body again.
            turn_end_veto_reset=self._begin_skill_turn,
            # Mid-turn say delivery (ADR-0051): the MAIN loop polls the workspace say
            # inbox at every iteration boundary, so a message sent while a turn runs
            # reaches the model without waiting for a turn boundary that a
            # perpetual-loop deployment never produces. Sub-agent loops never set this
            # (they would steal the user's message into a child conversation). No knob.
            consume_say_inbox=True,
            # A say that is a typed ``/<skill> [args]`` RUNS the skill mid-turn (ADR-0073)
            # through the same composition the REPL uses for a typed slash — one way.
            compose_skill=self.compose_skill_turn,
        )

    def _begin_skill_turn(self) -> None:
        """Open a fresh turn for per-turn skill state.

        Refills the use_skill invocation budget and forgets which bodies are "already
        loaded" (the reload dedup). Called at every top-level turn start and — ADR-0048 —
        on every TURN_END veto: a vetoed stop re-enters the loop INSIDE the same turn while
        the hook is telling the model to do new work, and a perpetual-loop framework runs
        its whole autonomous session that way (one ``/start``, then vetoes without end).
        Its mandated re-entry is a skill the model already loaded; a pointer instead of
        the body is a dead loop (measured 2026-08-26: four vetoes, four pointers, ~29h dark).
        """
        self._skill_invocations_this_turn = 0
        self._skills_loaded_this_turn.clear()

    def _register_composed_skill(self, user_text: str) -> None:
        """Count a typed ``/<skill>`` turn's body as loaded for the reload dedup (ADR-0063).

        The command path never registered its load, so a ``use_skill`` of the very skill
        the turn is running came back as the whole body again. Measured 2026-08-28 (coach,
        zc-03): an empty completion inside ``/start`` drew the skill nudge, the model
        answered ``use_skill start``, and 65 KB of instructions it already held landed a
        second time. Runs AFTER :meth:`_begin_skill_turn` at every top-level turn start, on
        the same digest the dedup compares, so that re-invocation now gets the pointer. A
        TURN_END veto still clears it (ADR-0048: a vetoed stop is a fresh skill turn).
        """
        name = _composed_skill_name(user_text)
        if name is None:
            return
        registry = getattr(self, "skill_registry", None)
        skill = registry.resolve(name) if registry is not None else None
        if skill is None:
            return
        try:
            body = skill.body()
        except Exception:  # noqa: BLE001 — an unreadable skill is the load's problem, not ours
            return
        import hashlib

        self._skills_loaded_this_turn[skill.name] = hashlib.sha1(
            body.encode("utf-8", errors="replace")
        ).hexdigest()

    def _assert_local_only(self) -> None:
        """Refuse to start when ``local_only`` is set but a configured model is metered.

        The ANTICIPATION half of the cost guarantee (rb-605). It cannot be the whole
        guarantee — it only sees the config it knows to enumerate, so a route added later
        would bypass it; that is what the fail-closed check in
        ``LiteLLMProvider._build_kwargs`` is for. What this half buys is a failure at
        STARTUP, naming every offender at once, instead of a refusal mid-turn on whichever
        one happened to be reached first.

        It enumerates EFFECTIVE models, not configured ones — the distinction matters and is
        the whole reason this is not a two-line check. Under zakpick, a category the user
        never overrode still routes: it falls through to ``DEFAULT_CATEGORY_MODELS``, which
        is Groq/OpenAI. So an operator who sets ``local_only`` and points only ``deep_code``
        at their pod has FIVE categories still aimed at metered APIs, and checking only
        ``zakpick_models`` would report a clean config for exactly the setup most likely to
        spend money by surprise.
        """
        if not self.settings.local_only:
            return
        from zakcode.providers.endpoints import (
            LocalOnlyViolation,
            classify_destination,
            is_sentinel,
        )
        from zakcode.providers.routing import ZAKPICK_CATEGORIES, model_for_category

        api_base = self.settings.api_base
        local_api_bases = list(getattr(self.settings, "local_api_bases", []) or [])
        offenders: list[str] = []

        def check(label: str, model: str | None) -> None:
            # Sentinels name no destination; by this point default_model is already resolved
            # to a concrete model, and anything still a sentinel is not a real call target.
            if not model or is_sentinel(model):
                return
            ok, reason = classify_destination(model, api_base, local_api_bases)
            if not ok:
                offenders.append(f"  - {label}: {reason}")

        check("default_model", self.settings.default_model)
        check("fallback_model", self.settings.fallback_model)
        if self._zakpick:
            for category in sorted(ZAKPICK_CATEGORIES):
                check(f"zakpick category '{category}'", model_for_category(category, self.settings))
        for role, role_model in sorted((self.settings.model_roles or {}).items()):
            check(f"model_roles['{role}']", role_model)

        if offenders:
            raise LocalOnlyViolation(
                "local_only is set, but these configured models would reach a metered API:\n"
                + "\n".join(offenders)
                + "\n\nFix by pointing them at a self-hosted endpoint (set ZAKCODE_API_BASE and "
                "use an openai/<model> name) or an ollama_chat/<model>, or unset "
                "ZAKCODE_LOCAL_ONLY to allow paid calls."
            )

    def _assert_context_windows(self) -> None:
        """Refuse to start when any effective model has no known context window (ADR-0066),
        naming every offender at once; report (loudly, not fatally) any server whose
        ``/models`` listing declares a different window than the one in force.

        The same enumeration as ``_assert_local_only`` — EFFECTIVE models, so a zakpick
        category the operator never overrode is checked against its built-in default too.
        Each model's window resolves the one way the provider itself will resolve it (the
        model's entry, else the registry); this sweep exists so the failure happens at
        startup with the whole list, instead of on whichever model a turn reached first.
        The server listing is asked once per api_base and only ever to CHECK.
        """
        from zakcode.providers.litellm_provider import (
            resolve_context_window,
            unknown_window_message,
        )
        from zakcode.providers.routing import effective_model_entries

        settings = self.settings
        local_api_bases = list(getattr(settings, "local_api_bases", []) or [])
        listing_cache: dict[str, object] = {}
        offenders: list[str] = []
        warned: set[tuple[str, int | None, int | None]] = set()
        for label, model, declared in effective_model_entries(settings, zakpick=self._zakpick):
            resolution = resolve_context_window(
                model,
                declared,
                api_base=settings.api_base,
                api_key=settings.api_key,
                local_only=settings.local_only,
                local_api_bases=local_api_bases,
                verify=True,
                listing_cache=listing_cache,
            )
            self.context_windows[label] = resolution
            if resolution.source == "sentinel":
                continue
            if resolution.window is None:
                offenders.append(
                    f"  - {label}: {unknown_window_message(model, resolution, settings.api_base)}"
                )
            elif resolution.mismatch:
                # One warning per (model, config, served) — under zakpick the startup model
                # is listed twice (as default_model and as its category).
                key = (model, resolution.window, resolution.served)
                if key in warned:
                    continue
                warned.add(key)
                warning = (
                    f"context window check: {label} ({model}) is configured at "
                    f"{resolution.window:,} tokens but the server at {settings.api_base} "
                    f"declares {resolution.served:,} — the configured number stays in force; "
                    "fix whichever one is wrong."
                )
                self.window_warnings.append(warning)
                logger.warning(warning)
        if offenders:
            raise UnknownContextWindow(
                "these configured models have no known context window:\n" + "\n".join(offenders)
            )

    def skill_fit_report(self) -> list[SkillFit]:
        """Every discovered skill measured against the smallest effective window (ADR-0066):
        the startup half of the loud block — see :func:`zakcode.skills.fit.measure_skill_fit`.
        Empty when there are no skills or no window is known (an injected provider)."""
        if self.skill_registry is None:
            return []
        windows = [r.window for r in self.context_windows.values() if r.window]
        if not windows:
            return []
        try:
            system_tokens = self.provider.count_tokens([], system=self.loop._build_system())
        except Exception:  # noqa: BLE001 — a fit report is advisory; never block startup on it
            system_tokens = 0
        reserve = self.provider.capabilities().max_output or _MIN_ANSWER_ROOM
        from zakcode.tasks import skill_pages

        def count(text: str) -> int:
            return self.provider.count_tokens([Message.user(text)])

        # What the model holds at once: a paged skill's largest page (ADR-0067), else the body.
        skills: list[tuple[str, str]] = []
        paged: set[str] = set()
        for skill in (self.skill_registry.get(n) for n in self.skill_registry.names()):
            if skill is None:
                continue
            try:
                body = skill.body()
            except Exception:  # noqa: BLE001 — an unreadable skill is not a fit finding
                continue
            pages = skill_pages(body, skill=skill.name)
            if pages is None:
                skills.append((skill.name, body))
                continue
            paged.add(skill.name)
            units = [pages.first(), *(pages.render(i) for i in range(2, pages.count + 1))]
            try:
                skills.append((skill.name, max(units, key=count)))
            except Exception:  # noqa: BLE001 — measured below by the same counter; skip here
                continue
        return measure_skill_fit(
            skills,
            count_tokens=count,
            window=min(windows),
            system_tokens=system_tokens,
            reserve=reserve,
            paged=paged,
        )

    def _build_provider(
        self,
        model: str,
        *,
        extra_body: dict[str, object] | None = None,
        context_window: int | None = None,
    ) -> Provider:
        """Build a settings-based provider for ``model`` (litellm wrapped in the text-tool
        protocol) — the same construction used for the default model and for per-role overrides.

        ``extra_body`` (a zakpick category's thinking flag) is MERGED OVER the configured
        ``Settings.extra_body`` rather than replacing it, so a global knob and a per-category
        one compose instead of one silently erasing the other. ``context_window`` is the
        routed model's own declared window (ADR-0066) and REPLACES the settings' value,
        which describes the default model only.
        """
        from zakcode.providers.endpoints import model_uses_generic_endpoint
        from zakcode.providers.litellm_provider import LiteLLMProvider
        from zakcode.providers.text_tools import TextToolCallingProvider

        if model == self.settings.default_model and not extra_body and context_window is None:
            role_settings = self.settings
        else:
            if context_window is None and model == self.settings.default_model:
                # A variant of the default model (a per-category thinking flag) is the same
                # model, so the settings' window still describes it.
                context_window = self.settings.context_window
            update: dict[str, object] = {"default_model": model, "context_window": context_window}
            # api_base/api_key are ENDPOINT-specific. Don't carry the configured custom
            # endpoint onto a routed model that litellm sends somewhere else — an
            # ollama_chat/* or groq/* role would otherwise be handed the OpenAI-compatible
            # gateway. (review: cross-backend endpoint bleed)
            #
            # The test is "does this model use the generic endpoint?", NOT "is it the same
            # backend as default_model?". The old same-backend comparison broke on the
            # ROUTING SENTINELS: default_model="zakpick" (or "auto") names no backend, so
            # splitting it yields the literal "zakpick", which equals no real prefix — the
            # comparison was therefore false for EVERY routed model and the configured
            # api_base was dropped on every zakpick call, including the openai/* categories
            # that exist precisely to reach the self-hosted server. Measured 2026-08-17.
            #
            # Using the same predicate the request builder uses keeps the two from drifting:
            # a model that WILL be given api_base at call time is exactly the model that
            # should keep it here.
            if not model_uses_generic_endpoint(model):
                update["api_base"] = None
                update["api_key"] = None
            if extra_body:
                update["extra_body"] = {**(self.settings.extra_body or {}), **extra_body}
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

    def _provider_for(
        self,
        model: str | None,
        *,
        extra_body: dict[str, object] | None = None,
        context_window: int | None = None,
    ) -> Provider:
        """Resolve a per-role model string to a :class:`Provider` (the model-routing seam).

        Returns the default ``self.provider`` for ``None``, the default model, or when the
        provider was injected (an injected test/eval provider can't be rebuilt per model, so
        every role uses it). Otherwise builds — and caches — a provider for that model.

        ``extra_body`` carries per-assignment request-body knobs (today: a zakpick category's
        ``thinking`` flag). It participates in the CACHE KEY, which it must: two categories
        can name the SAME model and want different thinking, and a model-only key would hand
        the second one the first one's provider and silently apply the wrong setting.
        ``context_window`` (the category entry's declared window, ADR-0066) is keyed the
        same way for the same reason.
        """
        if not model or self._provider_injected:
            return self.provider
        if model == self.settings.default_model and not extra_body:
            return self.provider
        key = model
        if extra_body or context_window is not None:
            key = f"{model}\x00{sorted((extra_body or {}).items())!r}\x00{context_window}"
        if key not in self._provider_cache:
            self._provider_cache[key] = self._build_provider(
                model, extra_body=extra_body, context_window=context_window
            )
        return self._provider_cache[key]

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
        from zakcode.providers.routing import model_spec_for_category

        # Read the SPEC, not just the model string: the assignment also carries this
        # category's thinking preference, which is the whole point of setting it per
        # category (reasoning tokens bill against max_tokens, so classify/summarize want
        # it off and deep_code wants it on).
        spec = model_spec_for_category(category, self.settings)
        model = spec.litellm_string
        return (
            self._provider_for(
                model, extra_body=spec.extra_body, context_window=spec.context_window
            ),
            model,
        )

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

    async def _classify_difficulty(self, user_text: str, context_frac: float) -> DifficultyVerdict:
        """zakpick base-difficulty router (the activated ``classify`` category): judge the
        request's SCOPE with a cheap classify-model call, not its LENGTH — a terse "build a pdf
        reader and maker" is a deep task no character count reveals. Returns a
        :class:`~zakcode.providers.routing.DifficultyVerdict`: ``"quick_code"`` or
        ``"deep_code"``, plus (ADR-0035) the catalogued skill the request is asking to RUN when
        the workspace has skills — the loop seeds a plan step and arms the coverage backstop
        for it, so a request that names its skill in prose ("finish forging this skill") is
        held to it exactly like one that typed the ``/slash``.

        Only the AMBIGUOUS short case (where the length heuristic would say quick) consults the
        model; a long request or an already-large context fast-paths to ``deep_code`` with NO
        call. Any failure — provider error, or output that never validated — FAILS UP to
        ``deep_code`` (and no skill), so Zak Code never guesses "quick" from length. The cheap
        call's spend is accounted on the session + shared budget like any other model call
        (tagged to the classify model), so it is visible in ``/cost`` and bounded by the
        turn-tree budget — never hidden.
        """
        from zakcode.providers.routing import (
            DIFFICULTY_SCHEMA,
            difficulty_system_prompt,
            implied_skill_anchored,
            parse_verdict,
            should_consult_classifier,
        )
        from zakcode.providers.structured import (
            StructuredValidationError,
            coerce_structured,
            make_response_format,
        )

        if not should_consult_classifier(user_text, context_frac):
            return DifficultyVerdict("deep_code")  # long / large-context -> capable coder, no call
        provider, model = self._resolve_task_provider("classify")
        skills = self.skill_registry.catalog() if self.skill_registry is not None else []
        try:
            # json_OBJECT mode (native JSON), NOT json_schema: litellm implements a json_schema
            # response_format on Groq via FUNCTION CALLING, and Groq's open models (incl. the
            # cheap llama-3.1-8b classifier) flakily emit a malformed tool call the provider
            # rejects (tool_use_failed) — the very unreliability zakpick routes around. Plain JSON
            # mode sidesteps the tool path; the trivial {"difficulty": ...} object is validated
            # locally against DIFFICULTY_SCHEMA below.
            result = await provider.acomplete(
                [Message.user(user_text)],
                system=difficulty_system_prompt(skills),
                response_format=make_response_format(None),
                temperature=0.0,
            )
        except ProviderError:
            return DifficultyVerdict("deep_code")  # classifier unavailable -> fail UP
        with contextlib.suppress(Exception):  # accounting must never break routing
            self.session.add_usage(result.usage, model=model)
            if self._shared_budget is not None:
                self._shared_budget.add_usage(result.usage.cost_usd, result.usage.total_tokens)
        try:
            data = coerce_structured(result.text, schema=DIFFICULTY_SCHEMA)
        except StructuredValidationError:
            return DifficultyVerdict("deep_code")  # output was not schema-valid JSON -> fail UP
        verdict = parse_verdict(data, known=[name for name, _desc in skills])
        if verdict.skill is not None:
            # ADR-0036: the deterministic floor under "never guess" — a skill the request is
            # asking to RUN is named or described in the request's own words; no shared
            # content word means a topic match, and the skill is dropped (category kept).
            description = next((d for n, d in skills if n == verdict.skill), "")
            if not implied_skill_anchored(user_text, verdict.skill, description):
                return DifficultyVerdict(verdict.category)
        return verdict

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
        self, name: str, *, source: str, query: str | None = None, args: str = ""
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
        # resolve() (not get()): match the skill's name OR its ``triggers:`` frontmatter, so a
        # skill ``foo`` with ``triggers: ["/start"]`` is reachable as ``/start``. Name wins.
        skill = registry.resolve(name) if registry is not None else None
        if skill is None:
            return SkillLoad(found=False, name=name)
        # Honor Claude Code's ``user-invocable: false``: such a skill is internal — reached by
        # another skill chaining to it (or the model's use_skill), never by a human typing
        # ``/<name>``. Refuse only the human COMMAND path; the model's tool path may still chain.
        not_user_invocable = (
            str(skill.frontmatter.extras.get("user_invocable", "")).strip().lower() == "false"
        )
        if source == "command" and not_user_invocable:
            return SkillLoad(
                found=True,
                name=skill.name,
                denied_reason=(
                    f"{skill.name} is not user-invocable — it runs internally "
                    "(another skill invokes it), so it can't be started by typing it."
                ),
            )
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

        if source == "tool":
            # Per-turn reload dedup: the SAME unchanged body already injected this turn is
            # not re-injected — a short pointer back to it is returned instead (args still
            # surfaced below so a sub-command chain like `tree add` -> `tree read` works).
            # Costs no invocation budget and fires no selection signal: nothing new loaded.
            import hashlib

            digest = hashlib.sha1(body.encode("utf-8", errors="replace")).hexdigest()
            if self._skills_loaded_this_turn.get(skill.name) == digest:
                # A paged skill (ADR-0067) is re-delivered at its CURRENT section — the one
                # recovery a model that lost the page (compaction, a long detour) needs —
                # instead of a bare pointer to text that may no longer be in context.
                loop = getattr(self, "loop", None)
                page = loop.current_skill_page(skill.name) if loop is not None else None
                if page is not None:
                    pointer = (
                        f"[already loaded] Skill {skill.name!r} is running this turn, delivered "
                        "one section at a time; here is the CURRENT section again. Continue "
                        f"from where you are in it.\n\n{page}"
                    )
                else:
                    pointer = (
                        f"[already loaded] The full instructions for skill {skill.name!r} are "
                        "already in your context THIS turn — the /command you were given, or "
                        "an earlier use_skill call — unchanged. Continue those instructions "
                        "from where you are; do not reload them."
                    )
                if args.strip():
                    pointer = f"[arguments: {defang_untrusted(args.strip())}]\n\n{pointer}"
                logger.info("skill %r use_skill deduped (already loaded this turn)", skill.name)
                return SkillLoad(found=True, name=skill.name, body=pointer)
            self._skills_loaded_this_turn[skill.name] = digest
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
        rendered = defang_untrusted(body)
        if args.strip() and source == "tool":
            # use_skill arguments (use_skill(name, args="loop")): surfaced to the model ahead of
            # the body so a skill whose steps branch on an argument (a sub-command like `loop`)
            # can see it. A presentation frame the model reads — NOT a trust boundary: defang
            # only neutralizes tool-call sentinels, not brackets, and body + args share the same
            # untrusted tier the model already consumes. The human ``/<name>`` path does NOT get
            # this frame: :meth:`compose_skill_turn` wraps args in ``<command-args>`` inside the
            # command-expansion frame instead, and the two shapes staying DISTINCT is what lets
            # the model tell a user-typed slash from a model-chained load (provenance).
            rendered = f"[arguments: {defang_untrusted(args.strip())}]\n\n{rendered}"
        return SkillLoad(found=True, name=skill.name, body=rendered, path=str(skill.path))

    async def compose_skill_turn(
        self, name: str, args: str = "", *, fuzzy: bool = True
    ) -> SkillInvocation:
        """Resolve a skill for the human ``/<name>`` path and return the turn text to run.

        Claude Code slash semantics: typing ``/<skill> [args]`` RUNS the skill now — the
        command-expansion frame (``<command-message>``/``<command-name>``/``<command-args>``)
        plus the body IS the turn's user message. The frame is invocation provenance: it is
        how the model knows a HUMAN typed the slash, so a skill whose own rules restrict it
        to user invocation executes instead of refusing. This method does the loading half
        without mutating the session, so the caller hands ``turn_text`` to its normal turn
        entry (``astream_turn`` / ``arun_turn``) and renders it like any other turn. Shares
        :meth:`_load_skill_body` with the model-facing ``use_skill`` tool and
        :meth:`invoke_skill`, so every path reads, defangs, and fires ``ON_SKILL_SELECTED``
        identically. Never raises: a missing/unreadable skill file is a UX result, not a crash.
        """
        load = await self._load_skill_body(name, source="command", args=args)
        corrected_from: str | None = None
        if not load.found:
            # Typo tolerance (ADR-0040): ``/enocde-session`` is not "unsupported" when the
            # catalog holds exactly one near-identical name — run it, and say so. Anything
            # less certain (two neighbours, a weak match) is handed back as a did-you-mean;
            # the harness never guesses a command on the operator's behalf. ``fuzzy=False``
            # is the exact-only probe a caller uses before trying its own command tables.
            candidates = self.closest_skill_names(name) if fuzzy else []
            if len(candidates) == 1 and candidates[0][1] >= _SKILL_AUTOCORRECT_RATIO:
                corrected_from = name.lstrip("/").strip()
                load = await self._load_skill_body(candidates[0][0], source="command", args=args)
            if not load.found:
                return SkillInvocation(
                    invoked=False, name=name, suggestions=tuple(c for c, _ in candidates)
                )
        if load.denied_reason:
            # Discovered but refused (e.g. user-invocable: false typed as /<name>): handled, but
            # not loaded — surfaced to the operator with the session left untouched.
            return SkillInvocation(
                invoked=True,
                name=load.name,
                denied_reason=load.denied_reason,
                corrected_from=corrected_from,
            )
        if load.error or load.body is None:
            return SkillInvocation(
                invoked=True, name=load.name, error=load.error, corrected_from=corrected_from
            )
        from zakcode.providers.text_tools import defang_untrusted

        # Claude Code's command-expansion frame — the INVOCATION-PROVENANCE signal. A skill
        # body alone cannot tell the model WHO invoked it, and frameworks (claude-mind) ship
        # skills whose own rules forbid model self-invocation ("Claude MUST NOT invoke
        # /start"); without this frame a model obeying those rules refuses the human's own
        # keystroke (live 2026-08-19: `/start sera` answered "user-only command, run it
        # yourself in the terminal" — from the terminal). The frame echoes what the USER
        # TYPED (`name`), which under `triggers:` routing may differ from the resolved
        # skill (`load.name`); the system-prompt skills section states the contract
        # (:meth:`zakcode.skills.SkillRegistry.render_catalog`). Only a frame at the very
        # START of a user message carries this meaning — a body-embedded lookalike is just
        # text. use_skill loads stay `[arguments: …]`-framed; the asymmetry IS the signal.
        # …unless the harness corrected a typo: then the frame carries the skill the operator
        # MEANT, because the mistyped token is not a command and must not reach the model as one.
        typed = load.name if corrected_from else (name.lstrip("/").strip() or load.name)
        frame = [
            f"<command-message>{typed} is running</command-message>",
            f"<command-name>/{typed}</command-name>",
        ]
        if args.strip():
            frame.append(f"<command-args>{defang_untrusted(args.strip())}</command-args>")
        # A sectioned skill is paged through the plan (ADR-0067): the turn text carries the
        # front matter and section 1; the loop seeds every section from the whole body and
        # hands over the next one as update_plan marks the previous done — the same delivery
        # the use_skill door gets, so both doors run a long skill one section at a time.
        from zakcode.tasks import skill_pages

        pages = skill_pages(load.body, skill=load.name)
        body_text = pages.first() if pages is not None else load.body
        return SkillInvocation(
            invoked=True,
            name=load.name,
            turn_text="\n".join(frame) + f"\n\n{body_text}",
            corrected_from=corrected_from,
        )

    def closest_skill_names(self, token: str, *, limit: int = 3) -> list[tuple[str, float]]:
        """Skill names near ``token`` (a mistyped ``/<name>``), best first, with similarity.

        difflib over the catalog's names; below :data:`_SKILL_SUGGEST_RATIO` a name is noise,
        not a neighbour (ADR-0040). Never raises — an agent without a registry has none.
        """
        import difflib

        registry = getattr(self, "skill_registry", None)
        needle = token.lstrip("/").strip().lower()
        if registry is None or not needle:
            return []
        scored: list[tuple[str, float]] = []
        for candidate in registry.names():
            ratio = difflib.SequenceMatcher(None, needle, candidate.lower()).ratio()
            if ratio >= _SKILL_SUGGEST_RATIO:
                scored.append((candidate, ratio))
        scored.sort(key=lambda item: (-item[1], item[0]))
        return scored[:limit]

    async def invoke_skill(self, name: str, args: str = "") -> SkillInvocation:
        """Load a discovered skill's body into the session and emit the selection signal.

        The DEFERRED variant of :meth:`compose_skill_turn`: it folds the body into a TRUSTED
        user message so the NEXT turn applies the skill, without running a turn itself —
        for embedding hosts that stage context ahead of a run. The CLI ``/<name>`` path uses
        :meth:`compose_skill_turn` and runs the skill immediately (Claude Code parity).
        Never raises: a missing/unreadable skill file is a UX result, not a crash.
        """
        result = await self.compose_skill_turn(name, args)
        if result.turn_text is not None:
            self.session.add_message(Message.user(result.turn_text))
        return result

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
        self._begin_skill_turn()  # new top-level turn: refill the skills budget, forget loads
        self._register_composed_skill(user_text)  # …except the skill this turn IS (ADR-0063)
        if self._shared_budget is not None:
            self._shared_budget.reset()  # the pool is per-TURN-tree, not per-Agent
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
        self._begin_skill_turn()  # new top-level turn: refill the skills budget, forget loads
        self._register_composed_skill(user_text)  # …except the skill this turn IS (ADR-0063)
        if self._shared_budget is not None:
            self._shared_budget.reset()  # the pool is per-TURN-tree, not per-Agent
        return self.loop.astream_turn(user_text)

    def inject_user_line(self, text: str) -> None:
        """Hand a line typed at THIS agent's own REPL to its running turn (ADR-0078).

        Delivered at the next iteration boundary exactly like a say — same frame, same
        step-seam hold, same typed-``/skill`` dispatch — but in-process, so it can only
        reach this agent. The workspace say inbox stays the door for producers outside
        the process; several sessions may share one workspace, and that slot cannot
        tell which of them a keystroke was meant for.
        """
        self.loop.inject_user_line(text)

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
