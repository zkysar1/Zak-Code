"""The per-user config home ``~/.zakcode`` (D20, issue #14).

Covers the precedence matrix (user-only / workspace-only / both / env-wins), the
``ZAKCODE_HOME`` override, the home-path shape, the guard that the config home is
never picked up as a workspace root, and key-provenance reporting (``env_source``).

Hygiene note: ``load_dotenv`` exports stick in ``os.environ`` for the process, so
every test runs under an autouse fixture that removes the vars it uses afterward
and restores the module-level provenance state.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

import zakcode.config as cfg
from zakcode.config import env_source, load_settings, zakcode_home

#: The settings var the precedence matrix drives (obscure → unlikely in a real env).
_VAR = "ZAKCODE_SEARXNG_URL"
#: Fake provider-key names for provenance tests (never real keys).
_KEYS = ["FAKE_USER_KEY", "FAKE_WS_KEY", "FAKE_ENV_KEY", "FAKE_ABSENT_KEY"]


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """Run each test in an empty tmp cwd with a tmp config home; undo env exports."""
    monkeypatch.chdir(tmp_path)
    home = tmp_path / "confighome"
    home.mkdir()
    monkeypatch.setenv("ZAKCODE_HOME", str(home))
    for name in (_VAR, *_KEYS):
        monkeypatch.delenv(name, raising=False)
    saved_sources = dict(cfg._ENV_SOURCES)
    saved_exported = dict(cfg._DOTENV_EXPORTED)
    yield home
    for name in (_VAR, *_KEYS):
        os.environ.pop(name, None)
    cfg._ENV_SOURCES.clear()
    cfg._ENV_SOURCES.update(saved_sources)
    cfg._DOTENV_EXPORTED.clear()
    cfg._DOTENV_EXPORTED.update(saved_exported)


def _write(path: Path, **vars_: str) -> None:
    path.write_text("".join(f"{k}={v}\n" for k, v in vars_.items()), encoding="utf-8")


# ── the precedence matrix ─────────────────────────────────────────────────────


def test_user_env_applies_when_no_workspace_env(_clean_env: Path) -> None:
    _write(_clean_env / ".env", **{_VAR: "http://from-user.example"})
    settings = load_settings()
    assert settings.searxng_url == "http://from-user.example"


def test_workspace_env_still_works_alone(_clean_env: Path, tmp_path: Path) -> None:
    _write(tmp_path / ".env", **{_VAR: "http://from-workspace.example"})
    settings = load_settings()
    assert settings.searxng_url == "http://from-workspace.example"


def test_workspace_env_wins_over_user_env(_clean_env: Path, tmp_path: Path) -> None:
    # Closer to the invocation wins: workspace shadows the user config home.
    _write(_clean_env / ".env", **{_VAR: "http://from-user.example"})
    _write(tmp_path / ".env", **{_VAR: "http://from-workspace.example"})
    settings = load_settings()
    assert settings.searxng_url == "http://from-workspace.example"


def test_process_env_wins_over_both(
    _clean_env: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write(_clean_env / ".env", **{_VAR: "http://from-user.example"})
    _write(tmp_path / ".env", **{_VAR: "http://from-workspace.example"})
    monkeypatch.setenv(_VAR, "http://from-env.example")
    settings = load_settings()
    assert settings.searxng_url == "http://from-env.example"


# ── the home itself ───────────────────────────────────────────────────────────


def test_zakcode_home_honors_override(_clean_env: Path) -> None:
    assert zakcode_home() == _clean_env


def test_zakcode_home_default_shape(monkeypatch: pytest.MonkeyPatch) -> None:
    # Without the override: ~/.zakcode (= %USERPROFILE%\.zakcode on Windows).
    monkeypatch.delenv("ZAKCODE_HOME", raising=False)
    home = zakcode_home()
    assert home.name == ".zakcode"
    assert home.parent == Path.home()
    if sys.platform == "win32":
        assert str(home).startswith(os.environ["USERPROFILE"])


def test_config_home_is_never_the_workspace_root(_clean_env: Path, tmp_path: Path) -> None:
    # ~/.zakcode is a CONFIG home only: loading user config must not change
    # workspace discovery — the workspace root stays the invocation cwd.
    _write(_clean_env / ".env", **{_VAR: "http://from-user.example"})
    settings = load_settings()
    assert Path(settings.workspace_root).resolve() == tmp_path.resolve()
    assert Path(settings.workspace_root).resolve() != _clean_env.resolve()


def test_missing_user_env_is_a_silent_noop(_clean_env: Path) -> None:
    settings = load_settings()  # no .env anywhere
    assert settings.searxng_url is None


# ── provenance (the key panel's "why is it using that key" story) ─────────────


def test_env_source_names_each_origin(
    _clean_env: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write(_clean_env / ".env", FAKE_USER_KEY="u", FAKE_ENV_KEY="ignored")
    _write(tmp_path / ".env", FAKE_WS_KEY="w")
    monkeypatch.setenv("FAKE_ENV_KEY", "from-real-env")
    load_settings()
    assert env_source("FAKE_USER_KEY") == "user .env"
    assert env_source("FAKE_WS_KEY") == "workspace .env"
    assert env_source("FAKE_ENV_KEY") == "env"  # real environment wins + is named
    assert env_source("FAKE_ABSENT_KEY") == "not set"
    # The user-file key was exported so litellm-style consumers can read it…
    assert os.environ["FAKE_USER_KEY"] == "u"
    # …and the real env value was not overridden.
    assert os.environ["FAKE_ENV_KEY"] == "from-real-env"


def test_workspace_shadows_user_in_provenance(_clean_env: Path, tmp_path: Path) -> None:
    _write(_clean_env / ".env", FAKE_WS_KEY="user-value")
    _write(tmp_path / ".env", FAKE_WS_KEY="workspace-value")
    load_settings()
    assert env_source("FAKE_WS_KEY") == "workspace .env"
    assert os.environ["FAKE_WS_KEY"] == "workspace-value"
