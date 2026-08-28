"""Tests for the chat-agent builder + plugin-command dispatch wiring (M6-6).

These cover the two M6 fresh-eyes MAJORs: the CLI now (1) dispatches plugin-
registered slash commands via ``agent.command_registry`` and (2) trusts plugins
named in ``ZAKCODE_TRUSTED_PLUGINS`` so they are actually runnable from the CLI.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from rich.console import Console

from zakcode.cli import ConsolePermissionPrompter, _build_chat_agent

_PLUGIN_BODY = """\
from zakcode.commands import CommandResult


def register(ctx):
    ctx.register_command("hello", lambda arg: CommandResult.ok(f"hi {arg}"), description="greet")
"""


def _write_plugin(workspace: Path, name: str) -> None:
    pdir = workspace / ".zakcode" / "plugins" / name
    pdir.mkdir(parents=True)
    (pdir / "plugin.json").write_text(json.dumps({"name": name}), encoding="utf-8")
    (pdir / "plugin.py").write_text(_PLUGIN_BODY, encoding="utf-8")


def _overrides(tmp_path: Path) -> dict[str, object]:
    return {
        "default_model": "scripted/test",
        "context_window": 8192,
        "workspace_root": str(tmp_path),
    }


def test_builder_enables_features(tmp_path: Path) -> None:
    agent = _build_chat_agent(ConsolePermissionPrompter(Console()), _overrides(tmp_path))
    # All three interactive features are on, and the command registry exists.
    assert agent.loop.spawner is not None  # sub-agents
    assert agent.extension_manager is not None  # MCP
    assert agent.plugin_report is not None  # plugins discovered
    assert agent.command_registry is not None
    # Rules are on by default in chat. (Cross-session memory is the Mind's job, not the
    # harness's — a default chat agent registers no remember/recall tool.)
    assert agent.rule_registry is not None
    assert agent.registry.get("remember") is None
    assert agent.registry.get("recall") is None


def test_no_rules_flag_disables_rules(tmp_path: Path) -> None:
    agent = _build_chat_agent(
        ConsolePermissionPrompter(Console()), _overrides(tmp_path), enable_rules=False
    )
    assert agent.rule_registry is None


def test_session_flag_resumes_a_saved_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # CLI-08: `-s <id>` resumes a persisted session (it was a dead flag -- store never wired).
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    from zakcode.messages import Message
    from zakcode.session.store import Session, SessionStore

    saved = Session(id="resume-me", cwd=str(tmp_path), model="scripted/test")
    saved.add_message(Message.user("remember this line"))
    SessionStore().save(saved)

    agent = _build_chat_agent(
        ConsolePermissionPrompter(Console()), _overrides(tmp_path), session_id="resume-me"
    )
    assert agent.session.id == "resume-me"  # the named session, not a fresh one
    assert len(agent.session.messages) >= 1  # its transcript came back
    assert agent.store is not None  # and turns now persist


def test_session_flag_unknown_id_raises_session_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # CLI-08: a bad `-s` id raises SessionError (chat() turns it into a clean message, not a crash).
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    from zakcode.session.store import SessionError

    with pytest.raises(SessionError):
        _build_chat_agent(
            ConsolePermissionPrompter(Console()), _overrides(tmp_path), session_id="no-such-session"
        )


def test_resumed_session_cwd_realigns_to_workspace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # CLI-08/Phase 4: a session resumed in a DIFFERENT workspace has its cwd realigned to the active
    # workspace, so TURN_END/Stop hooks don't run in the session's stale recorded directory.
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    from zakcode.messages import Message
    from zakcode.session.store import Session, SessionStore

    old_dir = tmp_path / "old_project"
    new_dir = tmp_path / "new_project"
    old_dir.mkdir()
    new_dir.mkdir()
    saved = Session(id="movers", cwd=str(old_dir), model="scripted/test")
    saved.add_message(Message.user("from the old dir"))
    SessionStore().save(saved)

    agent = _build_chat_agent(
        ConsolePermissionPrompter(Console()),
        {"default_model": "scripted/test", "context_window": 8192, "workspace_root": str(new_dir)},
        session_id="movers",
    )
    assert agent.session.id == "movers"  # the session was resumed
    assert agent.session.cwd != str(old_dir)  # realigned away from the stale dir
    assert Path(agent.session.cwd).resolve() == new_dir.resolve()  # to the active workspace


def test_untrusted_plugin_not_runnable_by_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_plugin(tmp_path, "p")
    monkeypatch.delenv("ZAKCODE_TRUSTED_PLUGINS", raising=False)
    agent = _build_chat_agent(ConsolePermissionPrompter(Console()), _overrides(tmp_path))
    assert agent.plugin_report is not None
    assert "p" in agent.plugin_report.skipped  # untrusted → not loaded
    assert not agent.command_registry.has("hello")


def test_trusted_plugin_command_is_dispatchable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_plugin(tmp_path, "p")
    monkeypatch.setenv("ZAKCODE_TRUSTED_PLUGINS", "p")
    agent = _build_chat_agent(ConsolePermissionPrompter(Console()), _overrides(tmp_path))
    assert agent.plugin_report is not None
    assert agent.plugin_report.loaded == ["p"]
    # The exact path the CLI now uses: command_registry.run(name, arg).
    result = agent.command_registry.run("hello", "world")
    assert result is not None
    assert result.is_error is False
    assert result.output == "hi world"


def test_trusted_plugins_env_is_comma_split(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_plugin(tmp_path, "a")
    _write_plugin(tmp_path, "b")
    monkeypatch.setenv("ZAKCODE_TRUSTED_PLUGINS", " a , b ")  # whitespace tolerated
    agent = _build_chat_agent(ConsolePermissionPrompter(Console()), _overrides(tmp_path))
    assert agent.plugin_report is not None
    # comma-split + whitespace-trim ⇒ both names are TRUSTED, so neither is skipped as
    # untrusted. (Both ship the same demo command, so only the first fully loads — that
    # command-name collision is incidental; this test is about parsing the env var.)
    assert "a" not in agent.plugin_report.skipped
    assert "b" not in agent.plugin_report.skipped


def test_unknown_command_returns_none(tmp_path: Path) -> None:
    agent = _build_chat_agent(ConsolePermissionPrompter(Console()), _overrides(tmp_path))
    # An unregistered slash command falls through (None) so the CLI prints its notice.
    assert agent.command_registry.run("nope", "") is None
