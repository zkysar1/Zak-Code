"""Runtime configuration for Zak Code.

Values are resolved (highest precedence first) from explicit overrides, environment
variables prefixed ``ZAKCODE_``, a local ``.env`` file, then defaults.

Provider API keys are intentionally **not** modeled here: litellm reads them from their
standard environment variables (e.g. ``OPENAI_API_KEY``). This keeps secrets out of our
config surface — see ``docs/GUARDRAILS.md``.

Note: the primary model field is named ``default_model`` (not ``model``) because
pydantic reserves the bare name ``model``.
"""

from __future__ import annotations

import json
from enum import IntEnum
from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field, ValidationInfo, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict


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
    default_model: str = Field(
        default="ollama_chat/llama3.1",
        description="Primary litellm model string, e.g. 'ollama_chat/llama3.1' or 'openai/gpt-4o'.",
    )
    fallback_model: str | None = Field(
        default=None,
        description="Optional model to retry with if the primary call errors.",
    )
    # Optional per-ROLE model overrides so a mind can route cheap/local models to cheap roles
    # and reserve the capable model for generation (the "three specialized models" pattern).
    # Recognized keys: 'planner' (the plan sub-agent), 'subagent' (other sub-agents),
    # 'summarizer' (compaction). The main agent loop (the generator) always uses default_model.
    # Empty (default) = every role uses default_model. From an env var: a JSON object, e.g.
    # ZAKCODE_MODEL_ROLES={"planner":"ollama_chat/qwen2.5:3b"} (JSON is the only env form).
    model_roles: dict[str, str] = Field(
        default_factory=dict,
        description="Per-role model overrides (keys: planner | subagent | summarizer).",
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
    permission_mode: str = Field(
        default="ask", description="One of: ask | acceptEdits | allow | deny."
    )
    denied_commands: Annotated[list[str], NoDecode] = Field(
        default_factory=list,
        description=(
            "Extra operator deny regexes (case-insensitive) appended to the built-in "
            "dangerous-command blocklist; they only ever tighten the verdict. From an env "
            "var: one regex per line (regexes may contain commas/spaces), or a JSON array."
        ),
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

    @field_validator("denied_commands", "allowed_models", "web_allowed_domains", mode="before")
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
        if info.field_name == "denied_commands":
            return [line.strip() for line in text.splitlines() if line.strip()]
        return [part.strip() for part in text.replace(",", " ").split() if part.strip()]

    @field_validator("model_roles", mode="after")
    @classmethod
    def _check_model_roles(cls, value: dict[str, str]) -> dict[str, str]:
        """Reject unrecognized role keys so a typo (e.g. 'planer') fails fast at load rather
        than silently disabling the intended routing (the misspelled key would just be a no-op).
        """
        recognized = {"planner", "subagent", "summarizer"}
        unknown = sorted(set(value) - recognized)
        if unknown:
            raise ValueError(
                f"unrecognized model_roles key(s): {unknown}; recognized roles are "
                f"{sorted(recognized)}"
            )
        return value

    # ── Cross-session memory (opt-in via Agent(enable_memory=True)) ──────────
    memory_db_path: str | None = Field(
        default=None,
        description=(
            "Path to the cross-session memory SQLite DB. When unset, the Agent "
            "defaults it to <workspace>/.zakcode/memory.db (per-project memory)."
        ),
    )
    memory_recall_limit: int = Field(
        default=5, ge=0, description="How many memories the recall hook injects per turn."
    )
    memory_recall_min_overlap: int = Field(
        default=1,
        ge=0,
        description=(
            "Relevance floor for auto-recall: a memory is injected only if it shares at "
            "least this many distinctive (non-stopword) words with the user's turn — so a "
            "memory that merely matched a common word like 'the' is dropped. 0 disables "
            "the floor (inject every search match). Corpus-size-independent (unlike a raw "
            "bm25 score, which collapses to ~0 in a small store)."
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


def load_settings(**overrides: object) -> Settings:
    """Load settings from env/.env, applying any explicit keyword overrides."""
    return Settings(**overrides)  # type: ignore[arg-type]


__all__ = ["PermissionTier", "Settings", "load_settings"]
