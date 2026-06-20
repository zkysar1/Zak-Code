"""Runtime configuration for Zak Code.

Values are resolved (highest precedence first) from explicit overrides, environment
variables prefixed ``ZAKCODE_``, a local ``.env`` file, then defaults.

Provider API keys are intentionally **not** modeled here: litellm reads them from their
standard environment variables (e.g. ``OPENAI_API_KEY``, ``GROQ_API_KEY``). This keeps
secrets out of our config surface — see ``docs/GUARDRAILS.md``. So that a key placed in
the project ``.env`` actually reaches litellm, :func:`load_settings` first exports the
local ``.env`` into the process environment (existing variables always win) — without
that step pydantic-settings would extract only the ``ZAKCODE_*`` entries and silently
ignore provider keys in the file.

Note: the primary model field is named ``default_model`` (not ``model``) because
pydantic reserves the bare name ``model``.
"""

from __future__ import annotations

import json
import os
from enum import IntEnum
from pathlib import Path
from typing import Annotated, Literal

from dotenv import dotenv_values, load_dotenv
from pydantic import Field, ValidationInfo, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

from zakcode.providers.routing import ZAKPICK_CATEGORIES, ZakpickModel


class PermissionTier(IntEnum):
    """Ordered privilege levels a tool can require.

    Higher value = more dangerous. Ordering matters: a tool is authorized when the active
    permission level satisfies (>=) the tool's required tier. M0 only *records* each
    tool's required tier; the enforcing policy lands in M2 (see ``docs/ROADMAP.md``).
    """

    READ_ONLY = 0
    WORKSPACE_WRITE = 1
    DANGER_FULL_ACCESS = 2


class Settings(BaseSettings):
    """Typed configuration for the Zak Code core engine."""

    model_config = SettingsConfigDict(
        env_prefix="ZAKCODE_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── Model / provider (vendor-agnostic via litellm) ──────────────────────
    # ``"auto"`` engages the availability resolver (PKG-AUTO): local Ollama if up,
    # else the first viable external per ``auto_model_preference`` — see
    # ``zakcode.providers.resolve``. The default stays an explicit model for now;
    # flipping the out-of-the-box default to ``auto`` is a one-line decision later.
    default_model: str = Field(
        default="ollama_chat/llama3.1",
        description=(
            "Primary litellm model string (e.g. 'ollama_chat/llama3.1', 'openai/gpt-4o'), "
            "'auto' to resolve by availability at startup, or 'zakpick' to route each prompt to "
            "the model you assigned to its task category (see 'zakpick_models' and "
            "zakcode.providers.routing)."
        ),
    )
    fallback_model: str | None = Field(
        default=None,
        description=(
            "Optional model to switch to when the primary call errors. With "
            "default_model='auto' this is the EXPLICIT override of the auto chain: "
            "runtime failover uses it (once per turn) before re-running auto resolution."
        ),
    )
    # External-provider order the 'auto' resolver tries after local (D19). Names map to
    # known sources in zakcode.providers.resolve. Comma/space/JSON list from the env.
    auto_model_preference: Annotated[list[str], NoDecode] = Field(
        default_factory=lambda: ["groq", "openai", "anthropic"],
        description="External provider order for default_model='auto' (after local).",
    )
    # Optional per-ROLE model overrides so a mind can route cheap/local models to cheap roles
    # and reserve the capable model for generation (the "three specialized models" pattern).
    # Recognized keys: 'planner' (the plan sub-agent), 'subagent' (other sub-agents),
    # 'summarizer' (compaction), 'judge' (the quality engine's judges/scorers). The main agent
    # loop (the generator) always uses default_model.
    # Empty (default) = every role uses default_model. From an env var: a JSON object, e.g.
    # ZAKCODE_MODEL_ROLES={"planner":"ollama_chat/qwen2.5:3b"} (JSON is the only env form).
    model_roles: dict[str, str] = Field(
        default_factory=dict,
        description="Per-role model overrides (keys: planner | subagent | summarizer | judge).",
    )
    # ── Quality engine (increment 6) — wire the small-model fan-out engine into the loop. All OFF
    # by default, so the default path is byte-identical; each seam is bounded (best_of_attempts and
    # the per-turn budget), fail-safe, and uses model_roles['judge'] for judging.
    quality_gate: bool = Field(
        default=False,
        description="Seam A: after the verifier passes, score the result and refine if it falls "
        "short — runs ALONGSIDE the binary completion critic. Off = today's behavior.",
    )
    quality_gate_threshold: float = Field(
        default=0.8, ge=0.0, le=1.0, description="Seam A ship threshold (overall rubric score)."
    )
    quality_gate_dimensions: dict[str, str] | None = Field(
        default=None,
        description="Seam A rubric (dim -> what to assess); None = a built-in code rubric.",
    )
    best_of_attempts: int = Field(
        default=1,
        ge=1,
        description="Seam B: N attempts at a stalled step to select among (1 = off).",
    )
    # Per-CATEGORY model assignments for ``default_model="zakpick"`` (inert otherwise) — the
    # zakpick interface. Each value is a {model, source} pair (source defaults to "groq"); the
    # user parks a model on each task category (keys: quick_code | deep_code | summarize | plan |
    # delegate | classify). Unset categories use the built-in Groq defaults (see
    # ``zakcode.providers.routing.DEFAULT_CATEGORY_MODELS``), so zakpick works out of the box.
    # Zak Code never picks a model the user didn't assign and owns no local/cloud tradeoff — a
    # slow local model is slow; a failing cloud model uses ``fallback_model`` like any other.
    # From an env var: a JSON object, e.g.
    # ZAKCODE_ZAKPICK_MODELS={"deep_code":{"model":"qwen3:32b","source":"local"}}.
    zakpick_models: dict[str, ZakpickModel] = Field(
        default_factory=dict,
        description=(
            "Per-category model assignments for default_model='zakpick' (JSON; keys "
            "quick_code | deep_code | summarize | plan | delegate | classify; each value "
            "{model, source}, source defaults to 'groq'). Unset categories use Groq defaults."
        ),
    )
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    # How tools are offered to the model. ``auto`` (default) uses native
    # function-calling when the model supports it and transparently falls back to a
    # text protocol when it does not (so tool-less local models still work). In
    # ``auto``, Ollama models (``ollama``/``ollama_chat``) are routed to the text
    # protocol regardless, because their native tool path is unreliable via litellm
    # (it returns empty responses for common local models like qwen2.5:3b); use
    # ``native`` to force native there. ``native`` forces native only; ``text`` forces
    # the text protocol for any backend. See ``zakcode.providers.text_tools`` and
    # ``zakcode._resolve_tool_calling_mode``.
    tool_calling_mode: str = Field(
        default="auto",
        description="How tools reach the model: auto | native | text.",
    )
    # NOTE (autonomous-by-design): the small-model reliability scaffolding is NOT
    # configurable — there is one way of doing things, and the agent steers itself:
    #   * write-grounding (read a written file back) is always on (no-ops without a write);
    #   * the verify-before-finish "Recipe Cursor" gate self-arms when the model writes a
    #     runnable script this turn, always extracts a stated expected-output literal
    #     (high-precision), and lets the harness run the file to verify it whenever that run
    #     would auto-allow without a prompt;
    #   * one tool call per turn is always enforced on the text protocol.
    # These were once Settings flags (verify_writes / recipe_mode / recipe_attempt_cap /
    # recipe_acceptance_compare / recipe_harness_run / single_tool_per_turn); they were
    # removed in favor of observed-signal autonomy. ``tool_calling_mode`` is kept because
    # ``auto`` already self-resolves by provider; native/text remain a debug override.

    # ── Local (Ollama) ──────────────────────────────────────────────────────
    ollama_base_url: str = Field(default="http://localhost:11434")

    # ── Generic endpoint override (any OpenAI-compatible server) ─────────────
    # Lets you point at a local llama.cpp / llama-cpp-python / vLLM / LM Studio server
    # (or any OpenAI-compatible gateway) by config alone — e.g. set
    # ZAKCODE_DEFAULT_MODEL=openai/qwen2.5-coder and ZAKCODE_API_BASE=http://127.0.0.1:8000/v1.
    # When unset, the provider derives a base only for Ollama (from ollama_base_url).
    api_base: str | None = Field(
        default=None,
        description="Override the provider endpoint URL (any OpenAI-compatible server).",
    )
    # Many local servers ignore the key but litellm's openai route still requires one to be
    # present; this is a non-secret placeholder knob, not a vault. Real cloud keys still come
    # from the standard env vars (e.g. OPENAI_API_KEY) — see the module docstring.
    #
    # ``exclude=True`` keeps this field out of every ``model_dump()`` (e.g. the server's
    # ``GET /config``) so it can never leak by serialization, while attribute access
    # (``settings.api_key``, used by the provider) still works. CONVENTION: any future
    # secret-bearing Settings field MUST set ``exclude=True`` for the same reason.
    api_key: str | None = Field(
        default=None,
        exclude=True,
        description="Optional API key to pass through (e.g. a dummy for a local server).",
    )

    # ── Agent behavior ──────────────────────────────────────────────────────
    max_iterations: int = Field(
        default=50, ge=1, description="Hard cap on agent-loop iterations per turn."
    )
    # Per-turn ceiling on MODEL-driven skill invocations (the use_skill tool), shared across a
    # turn and its whole sub-agent tree (one counter, reset per top-level turn). Bounds a runaway
    # or cyclic skill chain (A -> B -> A) more tightly than max_iterations. A human /<name>
    # invocation is operator-controlled and never throttled. 0 (default) = unlimited, so behavior
    # is unchanged.
    skill_invocation_budget: int = Field(
        default=0, ge=0, description="Max model-driven use_skill invocations per turn (0 = off)."
    )
    # Fold the workspace README.md into the agent's project-context block (alongside the discovered
    # AGENTS.md / CLAUDE.md / ZAK.md guides). The guides are always loaded; this toggles ONLY the
    # README, which is human-facing and can be large — content/total caps still bound it. Default
    # on (the README is usually the best available project overview).
    context_include_readme: bool = Field(
        default=True, description="Fold the workspace README.md into the agent's project context."
    )
    # Optional cost/token ceilings (parity #4), bounding CUMULATIVE spend across a turn and
    # its whole sub-agent delegation tree (one shared budget). A completed call's actuals are
    # folded in post-completion; once a ceiling is crossed the turn stops with
    # stop_reason="budget_exhausted". None (default) = that ceiling is unbounded, so behavior
    # is unchanged. These are spend guards, not per-call input caps.
    max_cost_usd: float | None = Field(
        default=None,
        ge=0,
        description="Stop the turn-tree once cumulative model cost (USD) reaches this; None = off.",
    )
    max_tokens: int | None = Field(
        default=None,
        ge=1,
        description="Stop the turn-tree once cumulative total tokens reach this; None = off.",
    )
    # Bounded retry for RATE-LIMITED provider calls only (a 429 is the one failure
    # class where waiting is the documented remedy). Other provider errors are never
    # retried — they end the turn gracefully with stop_reason="provider_error".
    # This is THE retry mechanism: litellm's own ``num_retries`` stays 0 so the two
    # layers can never compound (see the Provider ABC docstring on retry layering).
    provider_max_retries: int = Field(
        default=3,
        ge=0,
        description="Retries (with backoff) after a rate-limited provider call; 0 disables.",
    )
    # TURN_END veto seam (T2/T3): how many times per turn a TURN_END hook may veto a
    # vetoable stop (completed / doom_loop / stuck) and re-enter the loop with its
    # continuation prompt. 0 (default) disables the gate — no hook fires at turn end.
    # max_iterations / provider_error / recipe_stalled are never vetoable.
    turn_end_veto_budget: int = Field(
        default=0,
        ge=0,
        description="Max TURN_END hook vetoes per turn (Stop-hook seam); 0 disables.",
    )

    # Completion-review gate: when a turn CHANGED code (wrote a runnable file) and the agent tries
    # to finish, send it back this many times to re-read the request and verify EVERY requirement
    # against what is actually on disk — and finish any abandoned/failed operation — before
    # completing. Bounded so it converges (an unbounded "don't finish until perfect" loops forever
    # on a model that can't reach perfect). 0 (default) disables it; behavior is unchanged.
    completion_review_attempts: int = Field(
        default=0,
        ge=0,
        description="Self-review rounds before a code-changing turn may finish; 0 disables.",
    )
    trace_dir: str | None = Field(
        default=None,
        description="If set, write a structured per-turn JSONL decision trace to this directory "
        "(observability); unset disables. Env: ZAKCODE_TRACE_DIR.",
    )
    permission_mode: str = Field(
        default="ask",
        description="One of: ask | acceptEdits | allow | autonomous | deny.",
    )
    # Per-tool trust overrides (audit P0-2b / D12): judge the NAMED tool under a
    # different permission mode than the session's — both directions (loosen bash to
    # 'allow', or tighten web_fetch to 'ask'). In an autonomous session the
    # dangerous-command floor stays a hard deny regardless of any per-tool loosening.
    # From an env var: JSON only, e.g.
    # ZAKCODE_TOOL_TRUST_OVERRIDES={"bash":"allow","web_fetch":"ask"}.
    tool_trust_overrides: dict[str, str] = Field(
        default_factory=dict,
        description="Per-tool permission-mode overrides (tool name -> mode).",
    )
    # Subprocess key hygiene (PR #3 review / RISKS): by default, provider API keys
    # (OPENAI_API_KEY and anything matching *_API_KEY) are scrubbed from the
    # environment handed to bash/powershell children, so agent-run scripts cannot
    # read the host's model credentials. Set true only when a workflow's scripts
    # legitimately call a provider themselves.
    subprocess_inherit_provider_keys: bool = Field(
        default=False,
        description="Let bash/powershell children inherit provider API keys (default: scrubbed).",
    )
    # Declared-vs-undeclared dependency gate (self-remediation Step 1; docs/SELF-REMEDIATION.md).
    # When on (default), a shell command that installs a package the project's manifests/lockfile
    # do not declare (pip/uv/poetry/npm/pnpm/yarn) is escalated to a confirmation prompt — and is a
    # hard DENY in ``autonomous`` mode, where there is no prompt and no execution sandbox yet. This
    # closes the typosquatting / malicious-dependency / injection-escalation vector while letting
    # the agent freely (re)install what the project already vouches for, ``uv sync``, ``npm ci``,
    # and editable/local installs. Tighten-only: it can never loosen a verdict. Set false to
    # disable the gate entirely (installs are then governed by mode + the dangerous blocklist only).
    dependency_gate: bool = Field(
        default=True,
        description="Escalate/deny installs of packages not declared in the project's manifests.",
    )
    denied_commands: Annotated[list[str], NoDecode] = Field(
        default_factory=list,
        description=(
            "Extra operator deny regexes (case-insensitive) appended to the built-in "
            "dangerous-command blocklist; they only ever tighten the verdict. From an env "
            "var: one regex per line (regexes may contain commas/spaces), or a JSON array."
        ),
    )
    protected_paths: Annotated[list[str], NoDecode] = Field(
        default_factory=list,
        description=(
            "Extra operator protected-path regexes appended to the built-in protected-path "
            "floor (a write matching one escalates, and hard-denies in autonomous, even under a "
            "grant). Tighten-only. From an env var: one regex per line, or a JSON array."
        ),
    )
    # Project-verifier gate (R1): a shell command that proves the workspace is still healthy
    # (e.g. ``uv run poe check``, ``pytest -q``, ``npm test``). When set, a turn that CHANGED code
    # may not finish ``completed`` until this command has run successfully — the harness runs it
    # itself when it would auto-allow (allow/autonomous modes or a prior grant), else it nudges the
    # model to run it; after a bounded number of attempts a still-failing turn ends
    # ``verification_failed`` (degraded). Deliberately domain-AGNOSTIC: the engine never guesses
    # the command — an operator/mind/skill provides it. Unset (default) = no project gate (the
    # always-on recipe gate that verifies a freshly written script still applies).
    verify_command: str | None = Field(
        default=None,
        description=(
            "Shell command that verifies the workspace (tests/lint); gates completion after edits."
        ),
    )
    # Plan-first gate (R5, opt-in, OFF by default). When true, the harness will not run a MUTATING
    # tool (write/edit/shell) until the model has laid out a plan with update_plan — "plan before
    # you act", the harness-enforced-planning pole. Read-only investigation is never gated, and the
    # gate is bounded (after a few nudges it lets the action through, fail-open) so it can never
    # deadlock. Off by default because forcing a plan on trivial turns is counterproductive; an
    # operator/mind that wants the discipline opts in.
    require_plan: bool = Field(
        default=False,
        description="Require a plan (update_plan) before the first mutating tool runs in a turn.",
    )
    # Per-task tool-exposure filter (self-remediation Step 4): narrow which tools are advertised
    # to the model (and invocable) to a task-appropriate subset — the most effective single
    # prompt-injection defense (a tool that's never offered can't be hijacked by injected
    # content). Glob patterns over canonical tool names; ``deny`` wins over ``allow``. Exposure
    # only — never loosens the permission gate. From an env var: comma/space-separated globs or a
    # JSON array (e.g. ZAKCODE_TOOL_EXPOSURE_DENY=bash,powershell,mcp__*).
    tool_exposure_allow: Annotated[list[str], NoDecode] = Field(
        default_factory=list,
        description=(
            "If non-empty, ONLY tools matching these globs (canonical names) are exposed to the "
            "model. Empty = no allow restriction."
        ),
    )
    tool_exposure_deny: Annotated[list[str], NoDecode] = Field(
        default_factory=list,
        description="Tool-name globs never exposed to the model (wins over the allow list).",
    )
    workspace_root: Path = Field(
        default_factory=Path.cwd, description="Root directory the agent operates within."
    )

    # ── Server (HTTP) auth + multi-tenant hardening ──────────────────────────
    # When set, the FastAPI server (``zakcode serve``) requires every request to carry
    # ``Authorization: Bearer <auth_token>``; ``GET /health`` is the only exemption. Browsers
    # (which cannot set a handshake Authorization header) authenticate the WebSocket via the
    # ``Sec-WebSocket-Protocol: bearer, <token>`` subprotocol — NOT a ``?token=`` query param,
    # which would land in access logs. When unset, the server is unauthenticated and ``serve``
    # refuses to bind a non-loopback host without an explicit ``--insecure`` opt-in.
    # ``exclude=True`` keeps the token out of every ``model_dump()`` (e.g. ``GET /config``).
    # NOTE: a token used for browser WebSocket auth must contain NO commas or whitespace — the
    # ``Sec-WebSocket-Protocol`` header is a comma-separated list, so an embedded comma is
    # ambiguous and the handshake is rejected (the HTTP ``Authorization`` path is unaffected).
    # Generate one with e.g. ``secrets.token_urlsafe()``.
    auth_token: str | None = Field(
        default=None,
        exclude=True,
        description="Bearer token required by the HTTP server when set (else unauthenticated).",
    )
    # Optional allowlist for the per-request ``model`` override on /chat, /chat/stream and
    # /complete. When non-empty, a request whose ``model`` is not listed is rejected (400),
    # so a client cannot route prompts/cost to an arbitrary provider the host has creds for.
    # Empty (default) = no restriction. From an env var: comma- or whitespace-separated model
    # strings (e.g. ``ZAKCODE_ALLOWED_MODELS=openai/gpt-4o,ollama_chat/qwen2.5:3b``) or a JSON
    # array. (``NoDecode`` + the validator below; a bare ``list[str]`` would demand JSON and a
    # plain value would crash settings load.)
    allowed_models: Annotated[list[str], NoDecode] = Field(
        default_factory=list,
        description="If non-empty, the only model strings a request may override to.",
    )
    # Egress allowlist for the web_fetch tool. Empty (default) = web_fetch may reach any PUBLIC
    # host (the SSRF guard still blocks loopback/private/metadata). When non-empty, web_fetch is
    # restricted to these domains and their subdomains — the named hardening for the public-egress
    # exfil residual (see docs/RISKS.md). Comma/space/JSON list from the env, like allowed_models.
    web_allowed_domains: Annotated[list[str], NoDecode] = Field(
        default_factory=list,
        description="If non-empty, web_fetch may only reach these domains (and their subdomains).",
    )
    # Per-call confirmation gate for web_fetch egress (the other named hardening for the
    # public-egress residual). When true, every web_fetch is escalated to a confirmation prompt
    # (session-grantable, like a write); in ``deny`` mode it is blocked outright. Default off.
    web_fetch_confirm: bool = Field(
        default=False,
        description="Require operator confirmation before each web_fetch (egress gate).",
    )
    # Network-egress sandbox (opt-in). When ``egress_proxy`` is true, subprocess tools (bash /
    # powershell) are pointed at a localhost domain-allowlisting proxy via HTTP(S)_PROXY, so their
    # outbound HTTP/HTTPS to ``egress_allowed_domains`` (+ subdomains) on the standard web ports
    # (80/443) is allowed and everything else is refused. An empty list with the proxy on = deny
    # all subprocess egress. Best-effort (cooperating clients only — a process that ignores the
    # proxy env can still egress directly); see docs/RISKS.md. The agent's own model calls are NOT
    # proxied.
    egress_proxy: bool = Field(
        default=False,
        description="Route subprocess egress through a localhost domain-allowlisting proxy.",
    )
    egress_allowed_domains: Annotated[list[str], NoDecode] = Field(
        default_factory=list,
        description="Domains (+ subdomains) the egress proxy permits; empty = deny all egress.",
    )

    @field_validator(
        "denied_commands",
        "protected_paths",
        "allowed_models",
        "tool_exposure_allow",
        "tool_exposure_deny",
        "web_allowed_domains",
        "egress_allowed_domains",
        "auto_model_preference",
        mode="before",
    )
    @classmethod
    def _parse_list_from_env(cls, value: object, info: ValidationInfo) -> object:
        """Accept a list, a JSON array string, or a plain delimited string from an env var.

        ``NoDecode`` hands us the raw env string (pydantic would otherwise require JSON and
        crash on a plain value). A list passes through unchanged (programmatic construction).
        ``denied_commands`` holds regexes that may contain commas/spaces, so it splits on
        NEWLINES only; ``allowed_models`` holds simple ids, so it splits on commas/whitespace.
        A value that looks like a JSON array is parsed as one for either field.
        """
        if not isinstance(value, str):
            return value
        text = value.strip()
        if not text:
            return []
        if text.startswith("["):
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError:
                parsed = None
            if isinstance(parsed, list):
                return parsed
        if info.field_name in ("denied_commands", "protected_paths"):
            return [line.strip() for line in text.splitlines() if line.strip()]
        return [part.strip() for part in text.replace(",", " ").split() if part.strip()]

    @field_validator("tool_trust_overrides", mode="after")
    @classmethod
    def _check_tool_trust_overrides(cls, value: dict[str, str]) -> dict[str, str]:
        """Reject unknown mode values so a typo (e.g. 'alow') fails fast at load,
        rather than silently resolving to 'ask' (PermissionMode.parse's fallback) at
        authorization time and quietly changing the tool's posture. The set is kept
        as literals (not imported from zakcode.permissions) to avoid a config↔
        permissions import cycle; PermissionMode.parse accepts the same spellings.
        """
        recognized = {"deny", "ask", "acceptedits", "allow", "autonomous"}
        for tool, mode in value.items():
            normalized = str(mode).strip().lower().replace("-", "").replace("_", "")
            if normalized not in recognized:
                raise ValueError(
                    f"unrecognized permission mode {mode!r} for tool {tool!r}; "
                    "expected one of: deny | ask | acceptEdits | allow | autonomous"
                )
        return value

    @field_validator("model_roles", mode="after")
    @classmethod
    def _check_model_roles(cls, value: dict[str, str]) -> dict[str, str]:
        """Reject unrecognized role keys so a typo (e.g. 'planer') fails fast at load rather
        than silently disabling the intended routing (the misspelled key would just be a no-op).
        """
        recognized = {"planner", "subagent", "summarizer", "judge"}
        unknown = sorted(set(value) - recognized)
        if unknown:
            raise ValueError(
                f"unrecognized model_roles key(s): {unknown}; recognized roles are "
                f"{sorted(recognized)}"
            )
        return value

    @field_validator("zakpick_models", mode="after")
    @classmethod
    def _check_zakpick_models(cls, value: dict[str, ZakpickModel]) -> dict[str, ZakpickModel]:
        """Reject unrecognized category keys so a typo (e.g. 'planr') fails fast at load rather
        than silently leaving that category on its default."""
        unknown = sorted(set(value) - ZAKPICK_CATEGORIES)
        if unknown:
            raise ValueError(
                f"unrecognized zakpick_models key(s): {unknown}; recognized categories are "
                f"{sorted(ZAKPICK_CATEGORIES)}"
            )
        return value

    # ── Workspace settings.json hook ingestion (PR-T5; opt-in) ───────────────
    settings_hooks: bool = Field(
        default=False,
        description=(
            "Load shell hooks from <workspace>/.claude/settings.json and "
            ".zakcode/settings.json (ZAKCODE_SETTINGS_HOOKS=true). Off by default: "
            "workspaces configured for other hook runtimes (e.g. Claude Code) would "
            "otherwise have those hooks half-fire here with a different stdin schema. "
            "Hosts can force the behavior per-Agent via "
            "Agent(enable_settings_hooks=True/False)."
        ),
    )

    # ── Web tools (web_search / web_fetch) ───────────────────────────────────
    # Which vendor-agnostic search backend `web_search` uses. ``ddgs`` (the default) is free and
    # needs no key; ``tavily`` reads TAVILY_API_KEY from the env (like other provider keys — never
    # modeled here); ``searxng`` queries the instance at ``searxng_url``. Switching backend is a
    # config change, never a code change. (web_fetch needs no backend — it is plain HTTP.)
    search_backend: Literal["ddgs", "tavily", "searxng"] = Field(
        default="ddgs",
        description="web_search backend: ddgs (default, no key) | tavily | searxng.",
    )
    searxng_url: str | None = Field(
        default=None,
        description="Base URL of a self-hosted SearXNG instance (when search_backend=searxng).",
    )

    @property
    def provider(self) -> str:
        """The provider prefix of the configured model (text before the first '/')."""
        model = self.default_model
        return model.split("/", 1)[0] if "/" in model else model


def zakcode_home() -> Path:
    """The per-user Zak Code config home: ``~/.zakcode`` (D20, issue #14).

    ``%USERPROFILE%\\.zakcode`` on Windows — the ``~/.claude`` precedent. The
    ``ZAKCODE_HOME`` env var overrides the directory (tests / portable installs).
    This is a **config home only**: workspace discovery never looks here, and it
    must never be treated as ``workspace_root``.
    """
    override = os.environ.get("ZAKCODE_HOME", "").strip()
    return Path(override) if override else Path.home() / ".zakcode"


#: Where each env var visible to settings came from — refreshed by load_settings()
#: and read by env_source(). Best-effort within one process: dotenv exports stick
#: in os.environ, so names we exported are remembered (with their source) in
#: _DOTENV_EXPORTED to keep them from masquerading as real environment later.
_ENV_SOURCES: dict[str, str] = {}
_DOTENV_EXPORTED: dict[str, str] = {}


def env_source(name: str) -> str:
    """Provenance of env var ``name``: where its resolved value came from.

    One of ``"env"`` (the real process environment), ``"workspace .env"``,
    ``"user .env"`` (``~/.zakcode/.env``), or ``"not set"``. This is the debugging
    story for "why is it using that key on this machine" — surfaced in the
    ``zakcode info`` key panel. Accurate after a ``load_settings()`` call.
    """
    if name in _ENV_SOURCES:
        return _ENV_SOURCES[name]
    if name in os.environ:
        return _DOTENV_EXPORTED.get(name, "env")
    return "not set"


def load_settings(**overrides: object) -> Settings:
    """Load settings, applying any explicit keyword overrides.

    Precedence (lowest → highest): built-in defaults → ``~/.zakcode/.env`` (the
    per-user config home) → workspace ``.env`` (CWD-relative, matching
    pydantic-settings' ``env_file``) → the real process environment → explicit
    overrides. Closer to the invocation wins. Mechanically: the workspace file is
    exported into ``os.environ`` first, THEN the user file, both with
    ``override=False`` — so a workspace value shadows a user-home value and a real
    environment variable always wins. This is what lets provider keys
    (``OPENAI_API_KEY``, ``GROQ_API_KEY``, …) live in either ``.env``: litellm
    reads them from ``os.environ``, never from our settings. A missing file is a
    silent no-op.
    """
    user_env = zakcode_home() / ".env"
    workspace_vals = {k: v for k, v in dotenv_values(".env").items() if v is not None}
    user_vals = {k: v for k, v in dotenv_values(user_env).items() if v is not None}
    _ENV_SOURCES.clear()
    for name in set(workspace_vals) | set(user_vals):
        if name in os.environ and name not in _DOTENV_EXPORTED:
            _ENV_SOURCES[name] = "env"
        elif name in workspace_vals:
            _ENV_SOURCES[name] = "workspace .env"
        else:
            _ENV_SOURCES[name] = "user .env"
    before = set(os.environ)
    load_dotenv(".env", override=False)
    load_dotenv(user_env, override=False)
    for name in set(os.environ) - before:
        _DOTENV_EXPORTED[name] = _ENV_SOURCES.get(name, "workspace .env")
    return Settings(**overrides)  # type: ignore[arg-type]


__all__ = ["PermissionTier", "Settings", "env_source", "load_settings", "zakcode_home"]
