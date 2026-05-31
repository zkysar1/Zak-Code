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

from enum import IntEnum
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


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
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)

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
    workspace_root: Path = Field(
        default_factory=Path.cwd, description="Root directory the agent operates within."
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
