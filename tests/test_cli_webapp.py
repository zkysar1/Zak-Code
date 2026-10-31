"""Tests for the `zakcode webapp` CLI command (workspace pointer wiring).

Hermetic: ``uvicorn.Server`` and ``create_app`` are monkeypatched so nothing binds a port
or builds a real app — the tests assert that --workspace threads the workspace root
into create_app, that the bounded-run callback can bring the server down, and that a
newer build installed while serving is exec'd into with the chat kept (ADR-0333).
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("uvicorn")

from typer.testing import CliRunner  # noqa: E402

from zakcode.cli import _exec_argv, _webapp_restart_args, app  # noqa: E402
from zakcode.server.app import NewBuild  # noqa: E402

runner = CliRunner()


def _patch(monkeypatch: pytest.MonkeyPatch, captured: dict[str, Any]) -> None:
    import uvicorn

    import zakcode.server.app as server_app

    def fake_create_app(*, settings: Any = None, **kw: Any) -> object:
        captured["settings"] = settings
        captured["on_run_end"] = kw.get("on_run_end")
        captured["new_session"] = kw.get("new_session")
        captured["on_new_build"] = kw.get("on_new_build")
        captured["resume_turn"] = kw.get("resume_turn")
        return object()  # dummy ASGI app; the server is stubbed so it is never served

    class _FakeServer:
        """Stands in for uvicorn's Server: records itself, never binds a port.

        ``serve`` constructs a Server rather than calling ``uvicorn.run`` so a bounded
        run can ask it to exit (ADR-0039), so THIS is the seam the tests must stub —
        patching ``uvicorn.run`` would no longer intercept anything and the test would
        really serve.
        """

        def __init__(self, config: Any) -> None:
            self.config = config
            self.should_exit = False
            captured["server"] = self

        def run(self) -> None:
            captured["ran"] = True
            build = captured.get("installed_while_serving")
            if build is not None:  # the app's build follower, as it fires while serving
                asyncio.run(captured["on_new_build"](build))

    monkeypatch.setattr(server_app, "create_app", fake_create_app)
    monkeypatch.setattr(uvicorn, "Server", _FakeServer)


def test_webapp_workspace_threads_workspace_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, Any] = {}
    _patch(monkeypatch, captured)

    result = runner.invoke(app, ["webapp", "--workspace", str(tmp_path)])

    assert result.exit_code == 0, result.stdout
    assert captured["settings"] is not None
    assert str(captured["settings"].workspace_root) == str(tmp_path)


def test_webapp_without_workspace_uses_configured_default(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}
    _patch(monkeypatch, captured)

    result = runner.invoke(app, ["webapp"])

    assert result.exit_code == 0, result.stdout
    # create_app() is called with no settings (server resolves them from env/.env).
    assert captured["settings"] is None


@pytest.mark.parametrize(
    ("flags", "opens_a_new_chat"),
    [
        (["--new-session"], True),
        (["--new-session", "--workspace", "{tmp}"], True),
        ([], True),
        (["--workspace", "{tmp}"], True),
        (["--no-new-session"], False),
    ],
    ids=[
        "new-session",
        "new-session-with-workspace",
        "default",
        "workspace-only",
        "no-new-session",
    ],
)
def test_webapp_new_session_reaches_the_app(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    flags: list[str],
    opens_a_new_chat: bool,
) -> None:
    """Launch starts a fresh chat by default (ADR-0319); ``--no-new-session`` resumes
    the workspace's current chat, as a restart must."""
    captured: dict[str, Any] = {}
    _patch(monkeypatch, captured)

    result = runner.invoke(app, ["webapp", *(f.replace("{tmp}", str(tmp_path)) for f in flags)])

    assert result.exit_code == 0, result.stdout
    assert captured["new_session"] is opens_a_new_chat


# ── non-loopback bind guard (unauthenticated exposure) ─────────────────────────────


def test_webapp_refuses_non_loopback_without_auth(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}
    _patch(monkeypatch, captured)
    monkeypatch.delenv("ZAKCODE_AUTH_TOKEN", raising=False)

    result = runner.invoke(app, ["webapp", "--host", "0.0.0.0"])

    assert result.exit_code == 1
    assert "refusing" in result.stdout.lower()
    assert "settings" not in captured  # never reached create_app/uvicorn.run


def test_webapp_non_loopback_allowed_with_insecure(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}
    _patch(monkeypatch, captured)
    monkeypatch.delenv("ZAKCODE_AUTH_TOKEN", raising=False)

    result = runner.invoke(app, ["webapp", "--host", "0.0.0.0", "--insecure"])

    assert result.exit_code == 0, result.stdout
    assert "settings" in captured  # the guard let it through to create_app


def test_webapp_non_loopback_allowed_with_auth_token(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}
    _patch(monkeypatch, captured)
    monkeypatch.setenv("ZAKCODE_AUTH_TOKEN", "a-token")

    result = runner.invoke(app, ["webapp", "--host", "0.0.0.0"])

    assert result.exit_code == 0, result.stdout
    assert "settings" in captured


# ── bounded runs: the callback has to actually stop the server (ADR-0039) ──────────


def test_run_end_callback_asks_the_server_to_exit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The cap only saves money if the vessel STOPS.

    Stopping the turn loop while uvicorn keeps serving leaves the host idle and still
    billing — the cap would buy nothing. This pins the wiring end to end: serve hands
    create_app an ``on_run_end``, and awaiting it sets ``should_exit`` on the very
    server that was started.
    """
    import asyncio

    captured: dict[str, Any] = {}
    _patch(monkeypatch, captured)

    result = runner.invoke(app, ["webapp", "--workspace", str(tmp_path)])
    assert result.exit_code == 0, result.stdout
    assert captured["ran"] is True

    on_run_end = captured["on_run_end"]
    assert on_run_end is not None, "serve must hand create_app a run-end callback"
    assert captured["server"].should_exit is False  # control: not stopped by merely serving

    asyncio.run(on_run_end("duration_cap"))
    assert captured["server"].should_exit is True


# ── the served process runs on the stdlib event loop (ADR-0197) ────────────────────


def test_the_served_process_runs_on_the_stdlib_event_loop(monkeypatch: pytest.MonkeyPatch) -> None:
    """uvicorn's "auto" picks uvloop wherever it is installed, and ``uvicorn[standard]``
    installs it on every non-Windows box. Under uvloop a child's own ends of its three
    streams stay open in every descendant, whatever that descendant redirects, so a hook or
    a shell command that leaves anything running is never seen to finish: a SessionStart
    hook that spawned a daemon took 95.0 s served and 0.7 s from the CLI (2026-09-18). The
    served path runs on the loop the CLI and this suite run on.
    """
    captured: dict[str, Any] = {}
    _patch(monkeypatch, captured)

    result = runner.invoke(app, ["webapp"])

    assert result.exit_code == 0, result.stdout
    assert captured["server"].config.loop == "asyncio"


# ── a newer build installed while serving (ADR-0333) ───────────────────────────────

_HAND_OFF = ("ZAKCODE_RESTARTED_INTO", "ZAKCODE_RESTART_CONTINUATION", "ZAKCODE_RESTART_BOUNDARY")


def _clean_hand_off(monkeypatch: pytest.MonkeyPatch) -> None:
    """Start with none of the hand-off set, and undo whatever the command sets on the way out
    (``delenv`` alone records nothing for an absent name, so it would leave them behind)."""
    for name in _HAND_OFF:
        monkeypatch.setenv(name, "set-by-the-test")
        monkeypatch.delenv(name)


@pytest.mark.parametrize(
    ("argv", "expected"),
    [
        (["webapp"], ["webapp", "--no-new-session"]),
        (
            ["webapp", "--port", "9", "--new-session"],
            ["webapp", "--port", "9", "--no-new-session"],
        ),
        (
            ["webapp", "--no-new-session", "-w", "/srv/ws"],
            ["webapp", "-w", "/srv/ws", "--no-new-session"],
        ),
    ],
    ids=["bare", "new-session", "already-resuming"],
)
def test_a_restart_continues_the_chat_it_was_serving(argv: list[str], expected: list[str]) -> None:
    assert _webapp_restart_args(argv) == expected


@pytest.mark.parametrize(
    "build",
    [
        NewBuild("1111111", "2222222"),
        NewBuild("1111111", "2222222", 'Call Skill(skill="work-loop") now.', "skill"),
    ],
    ids=["idle", "turn-ended-for-the-restart"],
)
def test_a_new_build_execs_the_web_app_into_it(
    monkeypatch: pytest.MonkeyPatch, build: NewBuild
) -> None:
    """Once uvicorn returns, the process is replaced by one on the new build: same arguments
    (the chat kept), and the hand-off the terminal REPL's restart uses (ADR-0034, ADR-0099)."""
    captured: dict[str, Any] = {}
    _patch(monkeypatch, captured)
    _clean_hand_off(monkeypatch)
    execs: list[tuple[str, list[str]]] = []
    monkeypatch.setattr(os, "execv", lambda path, args: execs.append((path, list(args))))
    monkeypatch.setattr(sys, "argv", ["zakcode", "webapp", "--port", "9", "--new-session"])
    captured["installed_while_serving"] = build

    result = runner.invoke(app, ["webapp", "--port", "9", "--new-session"])

    assert result.exit_code == 0, result.stdout
    assert captured["server"].should_exit is True
    assert execs == [
        (
            sys.executable,
            _exec_argv(
                [sys.executable, "-m", "zakcode", "webapp", "--port", "9", "--no-new-session"]
            ),
        )
    ]
    handed_on = {name: os.environ.get(name) for name in _HAND_OFF}
    assert handed_on == {
        "ZAKCODE_RESTARTED_INTO": "2222222",
        "ZAKCODE_RESTART_CONTINUATION": build.continuation,
        "ZAKCODE_RESTART_BOUNDARY": build.boundary,
    }


def test_the_new_build_reads_the_env_files_afresh_and_keeps_the_real_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A value read from a ``.env`` file is exported into the process environment, where it
    outranks the file it came from, so a process exec'd with it ran on what the file said when
    the first one started. It is taken back before the exec, so the new process reads the file
    afresh; a variable the environment itself set is kept (ADR-0333)."""
    import zakcode.server.app as server_app

    home = tmp_path / "home"
    home.mkdir()
    (home / ".env").write_text("ZAKCODE_TEST_FROM_FILE=from-the-file\n", encoding="utf-8")
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.chdir(workspace)
    monkeypatch.setenv("ZAKCODE_HOME", str(home))
    monkeypatch.setenv("ZAKCODE_TEST_REAL", "from-the-environment")
    monkeypatch.setenv("ZAKCODE_TEST_FROM_FILE", "recorded so the test puts it back")
    monkeypatch.delenv("ZAKCODE_TEST_FROM_FILE")
    captured: dict[str, Any] = {}
    _patch(monkeypatch, captured)
    _clean_hand_off(monkeypatch)
    serving = server_app.create_app

    def remembering(**kw: Any) -> object:  # the file's value as the server saw it
        captured["from_file_while_serving"] = os.environ.get("ZAKCODE_TEST_FROM_FILE")
        return serving(**kw)

    monkeypatch.setattr(server_app, "create_app", remembering)
    at_exec: list[dict[str, str]] = []
    monkeypatch.setattr(os, "execv", lambda path, args: at_exec.append(dict(os.environ)))
    monkeypatch.setattr(sys, "argv", ["zakcode", "webapp", "--port", "9"])
    captured["installed_while_serving"] = NewBuild("1111111", "2222222")

    result = runner.invoke(app, ["webapp", "--port", "9"])

    assert result.exit_code == 0, result.stdout
    assert captured["from_file_while_serving"] == "from-the-file"  # the file was read
    [environment] = at_exec
    assert "ZAKCODE_TEST_FROM_FILE" not in environment
    assert environment["ZAKCODE_TEST_REAL"] == "from-the-environment"


def test_serving_with_no_new_build_execs_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}
    _patch(monkeypatch, captured)
    execs: list[Any] = []
    monkeypatch.setattr(os, "execv", lambda path, args: execs.append(args))

    result = runner.invoke(app, ["webapp"])

    assert result.exit_code == 0, result.stdout
    assert execs == []


@pytest.mark.parametrize(
    ("boundary", "carried", "expected_head", "expected_tail"),
    [
        (
            "skill",
            'Call Skill(skill="work-loop") now.',
            "[harness] this session was restarted into build 2222222 (a zakcode update) at a "
            "skill boundary",
            'make that call now:\nCall Skill(skill="work-loop") now.',
        ),
        (
            "stop-hook",
            "Continue: invoke /work-loop",
            "[harness] this session was restarted into build 2222222 (a zakcode update) at a "
            "turn boundary where a Stop hook had asked it to continue",
            "do exactly what it asked:\nContinue: invoke /work-loop",
        ),
    ],
    ids=["skill", "stop-hook"],
)
def test_a_restarted_web_app_takes_the_line_its_predecessor_set_aside(
    monkeypatch: pytest.MonkeyPatch,
    boundary: str,
    carried: str,
    expected_head: str,
    expected_tail: str,
) -> None:
    captured: dict[str, Any] = {}
    _patch(monkeypatch, captured)
    _clean_hand_off(monkeypatch)
    monkeypatch.setenv("ZAKCODE_RESTARTED_INTO", "2222222")
    monkeypatch.setenv("ZAKCODE_RESTART_CONTINUATION", carried)
    monkeypatch.setenv("ZAKCODE_RESTART_BOUNDARY", boundary)

    result = runner.invoke(app, ["webapp"])

    assert result.exit_code == 0, result.stdout
    assert captured["resume_turn"].startswith(expected_head)
    assert captured["resume_turn"].endswith(expected_tail)
    left_behind = [name for name in _HAND_OFF if name in os.environ]
    assert left_behind == [], "a later restart would inherit them"


def test_a_plain_start_has_no_line_to_resume(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}
    _patch(monkeypatch, captured)
    _clean_hand_off(monkeypatch)

    result = runner.invoke(app, ["webapp"])

    assert result.exit_code == 0, result.stdout
    assert captured["resume_turn"] is None
