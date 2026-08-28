"""Tests for Claude Code ``statusLine`` support (``zakcode.status_line``).

The contract: a workspace's ``statusLine`` command (from ``.claude/settings.json`` /
``settings.local.json``) is run after a turn with a CC-shaped status JSON on stdin, and
its stdout's first line is the rendered status line. It is GENERIC (no plug-in named),
COSMETIC, and FAIL-SAFE — any error/timeout/non-zero exit yields no line, never a raise.

The command is exercised with tiny throwaway Python scripts invoked via ``sys.executable``
so the tests are hermetic and cross-platform (no real shell), mirroring ``test_hooks.py``.
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path

import pytest

from zakcode.config import Settings
from zakcode.status_line import (
    DEFAULT_STATUS_LINE_TIMEOUT,
    StatusLineSpec,
    build_status_input,
    load_status_line_spec,
    render_status_line,
)


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _settings_with_status_line(workspace: Path, command: str, *, local: bool = False) -> Path:
    """Write a ``.claude/settings.json`` (or settings.local.json) with a statusLine command."""
    name = "settings.local.json" if local else "settings.json"
    return _write(
        workspace / ".claude" / name,
        json.dumps({"statusLine": {"type": "command", "command": command}}),
    )


def _spec(
    tmp_path: Path, name: str, body: str, *, timeout: float = 5.0, drop_env=None
) -> StatusLineSpec:
    """A StatusLineSpec that runs a tiny Python script (bypasses settings discovery)."""
    path = tmp_path / name
    path.write_text(body, encoding="utf-8")
    return StatusLineSpec(
        command=[sys.executable, str(path)], timeout=timeout, drop_env=drop_env or []
    )


def _status(tmp_path: Path):
    return build_status_input(
        session_id="sess-123",
        cwd=str(tmp_path),
        model_id="openai/gpt-4o",
        cost_usd=0.0123,
        total_tokens=4242,
        prompt_tokens=4000,
        completion_tokens=242,
        version="9.9.9",
    )


# ── the status JSON shape (the stdin contract) ────────────────────────────────


def test_status_input_carries_cc_shape() -> None:
    # The documented Claude Code statusLine input shape: top-level hook_event_name /
    # session_id / cwd / transcript_path / version, and nested model / workspace / cost.
    status = build_status_input(
        session_id="s1",
        cwd="/work",
        model_id="openai/gpt-4o",
        cost_usd=0.5,
        total_tokens=100,
    )
    doc = json.loads(status.model_dump_json(by_alias=True))
    assert doc["hook_event_name"] == "Status"
    assert doc["session_id"] == "s1"
    assert doc["cwd"] == "/work"
    assert doc["model"]["id"] == "openai/gpt-4o"
    assert doc["model"]["display_name"] == "openai/gpt-4o"  # defaults to the id
    assert doc["workspace"]["current_dir"] == "/work"
    assert doc["workspace"]["project_dir"] == "/work"
    assert doc["cost"]["total_cost_usd"] == 0.5
    assert doc["cost"]["total_tokens"] == 100


async def test_command_receives_model_cwd_and_cost_on_stdin(tmp_path: Path) -> None:
    # The status JSON fed on stdin carries the expected fields (model, cwd, cost) — proven by a
    # script that reads stdin and echoes them back as the status line.
    body = (
        "import json,sys\n"
        "d=json.load(sys.stdin)\n"
        "print(f\"{d['model']['id']} @ {d['cwd']} ${d['cost']['total_cost_usd']}\")\n"
    )
    spec = _spec(tmp_path, "echo.py", body)
    line = await render_status_line(spec, _status(tmp_path))
    assert line == f"openai/gpt-4o @ {tmp_path} $0.0123"


async def test_stdout_first_line_is_captured_and_returned(tmp_path: Path) -> None:
    # A configured statusLine command's stdout is captured; only the first non-empty line,
    # trimmed, becomes the status line (the strip is one line).
    spec = _spec(tmp_path, "multi.py", "print('  CTX: 42%  ')\nprint('second line ignored')")
    line = await render_status_line(spec, _status(tmp_path))
    assert line == "CTX: 42%"


async def test_blank_stdout_yields_no_line(tmp_path: Path) -> None:
    # A command that prints only whitespace/blank lines produces no status line (None), not "".
    spec = _spec(tmp_path, "blank.py", "print('')\nprint('   ')")
    assert await render_status_line(spec, _status(tmp_path)) is None


# ── fail-safe: a status line NEVER raises and NEVER breaks a turn ──────────────


async def test_missing_command_returns_none(tmp_path: Path) -> None:
    # A spec whose executable does not exist must fail-safe to None (spawn failure), not raise.
    spec = StatusLineSpec(command=[str(tmp_path / "does-not-exist-xyz")], timeout=5.0)
    assert await render_status_line(spec, _status(tmp_path)) is None


async def test_empty_command_returns_none(tmp_path: Path) -> None:
    spec = StatusLineSpec(command=[], timeout=5.0)
    assert await render_status_line(spec, _status(tmp_path)) is None


async def test_crashing_command_returns_none_without_raising(tmp_path: Path) -> None:
    # A status script that raises (non-zero exit) yields no line — the exception is the script's,
    # the host must not propagate it.
    spec = _spec(tmp_path, "boom.py", "raise SystemError('kaboom')")
    assert await render_status_line(spec, _status(tmp_path)) is None


async def test_nonzero_exit_returns_none(tmp_path: Path) -> None:
    # Even with output on stdout, a non-zero exit means "no line" (fail quiet).
    spec = _spec(tmp_path, "exit1.py", "import sys; print('ignored'); sys.exit(1)")
    assert await render_status_line(spec, _status(tmp_path)) is None


async def test_timeout_returns_none_without_raising(tmp_path: Path) -> None:
    # A slow status script is killed at the timeout and yields None — never stalls a turn.
    spec = _spec(tmp_path, "slow.py", "import time; time.sleep(30); print('too late')", timeout=0.5)
    start = time.monotonic()
    assert await render_status_line(spec, _status(tmp_path)) is None
    assert time.monotonic() - start < 15  # killed promptly, not after the full 30s sleep


async def test_cancellation_tears_down_and_propagates(tmp_path: Path) -> None:
    # Cancelling a turn mid-render tears the child down and re-raises CancelledError promptly —
    # the one thing render_status_line deliberately re-raises (a cancelled turn must unwind).
    spec = _spec(tmp_path, "slow2.py", "import time; time.sleep(30)", timeout=30)
    task = asyncio.ensure_future(render_status_line(spec, _status(tmp_path)))
    await asyncio.sleep(0.5)  # let the child spawn
    task.cancel()
    start = time.monotonic()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert time.monotonic() - start < 15  # torn down promptly


# ── settings discovery (the .claude/settings.json contract) ───────────────────


def test_load_status_line_spec_from_settings_json(tmp_path: Path) -> None:
    _settings_with_status_line(tmp_path, "bash my-status.sh")
    spec, error = load_status_line_spec(tmp_path)
    assert spec is not None
    assert error is None
    assert "my-status.sh" in " ".join(spec.command)
    assert spec.timeout == DEFAULT_STATUS_LINE_TIMEOUT


def test_no_status_line_configured_yields_none(tmp_path: Path) -> None:
    # A workspace with settings but no statusLine key → (None, None), the common case.
    _write(tmp_path / ".claude" / "settings.json", json.dumps({"hooks": {}}))
    spec, error = load_status_line_spec(tmp_path)
    assert spec is None and error is None


def test_settings_local_json_overrides_shared(tmp_path: Path) -> None:
    # Claude Code's per-machine settings.local.json wins over the shared settings.json for a
    # non-hook key like statusLine (the local override replaces the shared one).
    _settings_with_status_line(tmp_path, "bash shared.sh")
    _settings_with_status_line(tmp_path, "bash local.sh", local=True)
    spec, _error = load_status_line_spec(tmp_path)
    assert spec is not None
    assert "local.sh" in " ".join(spec.command)
    assert "shared.sh" not in " ".join(spec.command)


def test_claude_project_dir_is_substituted(tmp_path: Path) -> None:
    # CC status scripts reference themselves via $CLAUDE_PROJECT_DIR; a faithful host substitutes
    # the workspace root (the same gesture as the hook loader).
    _settings_with_status_line(tmp_path, "bash $CLAUDE_PROJECT_DIR/s.sh")
    spec, _error = load_status_line_spec(tmp_path)
    assert spec is not None
    joined = " ".join(spec.command)
    assert tmp_path.as_posix() in joined
    assert "$CLAUDE_PROJECT_DIR" not in joined


def test_unsupported_type_is_rejected(tmp_path: Path) -> None:
    # Only type:"command" is supported; another type yields no spec plus a reason (degrade loudly).
    _write(
        tmp_path / ".claude" / "settings.json",
        json.dumps({"statusLine": {"type": "static", "command": "x"}}),
    )
    spec, error = load_status_line_spec(tmp_path)
    assert spec is None
    assert error is not None and "static" in error


def test_dangerous_command_denied_in_autonomous(tmp_path: Path) -> None:
    # The dangerous-pattern scan applies to a statusLine command like every settings.json shell
    # command: a catastrophic command is hard-denied in autonomous mode (no spec).
    _settings_with_status_line(tmp_path, "rm -rf /")
    spec, error = load_status_line_spec(tmp_path, permission_mode="autonomous")
    assert spec is None
    assert error is not None and "DENIED" in error


def test_dangerous_command_warns_outside_autonomous(tmp_path: Path) -> None:
    # Outside autonomous mode the dangerous command is registered WITH A WARNING (parity with the
    # hook loader's warn-and-continue), so the operator is told but not silently blocked.
    _settings_with_status_line(tmp_path, "rm -rf /")
    spec, error = load_status_line_spec(tmp_path, permission_mode="ask")
    assert spec is not None
    assert error is not None and "WARNING" in error


def test_malformed_settings_file_does_not_raise(tmp_path: Path) -> None:
    # A corrupt settings.json yields (None, reason) — never a crash (fail-safe discovery).
    _write(tmp_path / ".claude" / "settings.json", "{ not json")
    spec, error = load_status_line_spec(tmp_path)
    assert spec is None
    assert error is not None and "parse error" in error


# ── environment hygiene (provider keys scrubbed from the subprocess) ──────────


def test_discovery_populates_provider_key_scrub_list(tmp_path: Path, monkeypatch) -> None:
    # A statusLine spec read from settings carries the provider-key scrub list (like every
    # workspace-sourced shell command), so render_status_line drops those keys from the child env.
    monkeypatch.setenv("OPENAI_API_KEY", "sk-should-be-hidden")
    _settings_with_status_line(tmp_path, "bash my-status.sh")
    spec, _error = load_status_line_spec(tmp_path)
    assert spec is not None
    assert "OPENAI_API_KEY" in spec.drop_env


async def test_provider_keys_are_scrubbed_from_the_command_env(tmp_path: Path, monkeypatch) -> None:
    # The cosmetic subprocess must not be able to read the host's model credentials: a key named
    # in the spec's drop_env (what discovery populates) is absent from the env the command sees.
    monkeypatch.setenv("OPENAI_API_KEY", "sk-should-be-hidden")
    body = (
        "import os\n"
        "print('OPENAI_API_KEY=' + ('PRESENT' if os.environ.get('OPENAI_API_KEY') else 'ABSENT'))\n"
    )
    spec = _spec(tmp_path, "leak.py", body, drop_env=["OPENAI_API_KEY"])
    line = await render_status_line(spec, _status(tmp_path))
    assert line == "OPENAI_API_KEY=ABSENT"


async def test_claude_project_dir_is_set_in_the_command_env(tmp_path: Path) -> None:
    # A status script resolves its paths through CLAUDE_PROJECT_DIR (CC's env var); the host sets
    # it to the workspace root (forward-slash form) for the child.
    body = "import os\nprint('CPD=' + os.environ.get('CLAUDE_PROJECT_DIR', 'MISSING'))\n"
    spec = _spec(tmp_path, "cpd.py", body)
    line = await render_status_line(spec, _status(tmp_path))
    assert line == f"CPD={tmp_path.as_posix()}"


# ── config / Agent wiring (off by default) ────────────────────────────────────


def test_status_line_off_by_default_in_config() -> None:
    assert Settings().status_line is False


def test_agent_does_not_load_status_line_when_off(tmp_path: Path) -> None:
    # Off by default: even with a statusLine configured, an Agent that did not opt in never loads
    # the spec — so the command can never run.
    from zakcode import Agent

    _settings_with_status_line(tmp_path, "bash my-status.sh")
    agent = Agent(
        settings=Settings(
            default_model="scripted/test", context_window=8192, workspace_root=tmp_path
        )
    )
    assert agent.status_line_enabled is False
    assert agent.status_line_spec is None


def test_agent_loads_status_line_when_enabled(tmp_path: Path) -> None:
    # With the per-Agent opt-in, the configured statusLine spec is loaded and stashed for a client
    # to render (the env-var path, Settings.status_line, is the other way in).
    from zakcode import Agent

    _settings_with_status_line(tmp_path, "bash my-status.sh")
    agent = Agent(
        settings=Settings(
            default_model="scripted/test", context_window=8192, workspace_root=tmp_path
        ),
        enable_status_line=True,
    )
    assert agent.status_line_enabled is True
    assert agent.status_line_spec is not None
    assert "my-status.sh" in " ".join(agent.status_line_spec.command)


def test_agent_status_line_enabled_via_settings_flag(tmp_path: Path) -> None:
    # The Settings.status_line flag (env ZAKCODE_STATUS_LINE) enables it without a per-Agent arg.
    from zakcode import Agent

    _settings_with_status_line(tmp_path, "bash my-status.sh")
    agent = Agent(
        settings=Settings(
            default_model="scripted/test",
            context_window=8192,
            workspace_root=tmp_path,
            status_line=True,
        ),
    )
    assert agent.status_line_enabled is True
    assert agent.status_line_spec is not None
