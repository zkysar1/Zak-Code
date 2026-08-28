"""Smoke tests for the scaffold: version, config defaults, CLI wiring, secret safety."""

from __future__ import annotations

import os

import pytest
from typer.testing import CliRunner

from zakcode import __version__
from zakcode.cli import app, build_info_lines
from zakcode.config import Settings


def test_version_string() -> None:
    assert isinstance(__version__, str)
    assert __version__.count(".") >= 2


def test_settings_defaults(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    # Hermetic on purpose: importing litellm ANYWHERE in the suite exports the
    # workspace .env into os.environ at import time (dotenv's find_dotenv walks
    # up from site-packages — inside this repo — to the repo root; probed
    # 2026-08-28). Without this scrub, the "defaults" test asserts whatever the
    # developer's .env says, on every box with a populated .env, while CI stays
    # green because its checkout has none.
    for k in [k for k in os.environ if k.startswith("ZAKCODE_")]:
        monkeypatch.delenv(k, raising=False)
    monkeypatch.chdir(tmp_path)  # Settings(env_file=".env") is CWD-relative
    settings = Settings()
    assert settings.default_model
    assert settings.provider == settings.default_model.split("/", 1)[0]
    assert not hasattr(settings, "max_iterations")  # no knob — unlimited is the only behavior
    # None = send no temperature; every backend runs at its own default (ADR-0018).
    assert settings.temperature is None


def test_cli_version_command() -> None:
    result = CliRunner().invoke(app, ["version"])
    assert result.exit_code == 0
    assert __version__ in result.stdout


def test_cli_info_command_runs() -> None:
    result = CliRunner().invoke(app, ["info"])
    assert result.exit_code == 0


def test_info_reports_key_presence_not_value(monkeypatch: pytest.MonkeyPatch) -> None:
    import zakcode.config as cfg

    # Isolate from any dotenv exports earlier tests made in this process.
    monkeypatch.setattr(cfg, "_ENV_SOURCES", {})
    monkeypatch.setattr(cfg, "_DOTENV_EXPORTED", {})
    secret = "sk-this-value-must-never-be-displayed"
    monkeypatch.setenv("OPENAI_API_KEY", secret)
    rows = dict(build_info_lines(Settings()))
    assert rows["OPENAI_API_KEY"] == "set (env)"  # provenance named, value never shown
    assert all(secret not in value for value in rows.values())
