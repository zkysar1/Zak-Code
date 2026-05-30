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

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


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
