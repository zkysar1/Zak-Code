"""Smoke tests for the scaffold: version, config defaults, CLI wiring, secret safety."""

from __future__ import annotations

import pytest
from typer.testing import CliRunner

from zakcode import __version__
from zakcode.cli import app, build_info_lines
from zakcode.config import Settings


def test_version_string() -> None:
    assert isinstance(__version__, str)
    assert __version__.count(".") >= 2


def test_settings_defaults() -> None:
    settings = Settings()
    assert settings.default_model
    assert settings.provider == settings.default_model.split("/", 1)[0]
    assert not hasattr(settings, "max_iterations")  # no knob — unlimited is the only behavior
    assert 0.0 <= settings.temperature <= 2.0


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
