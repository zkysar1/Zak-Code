"""Tests for the interactive ``zakcode chat`` REPL.

These are fully hermetic: the real :class:`~zakcode.Agent` is replaced with a
fake whose ``astream_turn`` yields a canned sequence of
:class:`~zakcode.events.AgentEvent`s, so no provider, network, or model is ever
touched. The chat command drives that stream through the real
:class:`~zakcode.cli.render.StreamRenderer`, so these exercise the live
token-by-token rendering path end to end. The CLI is exercised through Typer's
``CliRunner`` with scripted stdin.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

import pytest
from typer.testing import CliRunner

import zakcode
from zakcode.cli import ConsolePermissionPrompter, _parse_permission_answer, app
from zakcode.config import load_settings
from zakcode.events import (
    AgentDone,
    AgentEvent,
    AgentTextDelta,
    AgentToolCall,
    AgentToolResult,
)
from zakcode.hooks import HookEvent, HookManager, HookSpec
from zakcode.permissions import (
    PermissionMode,
    PermissionOutcome,
    PermissionPolicy,
    PermissionRequest,
)
from zakcode.providers.base import ProviderError
from zakcode.session.store import Session
from zakcode.usage import Usage

runner = CliRunner()

CANNED_TEXT = "Hello from the fake agent."


class FakeAgent:
    """Drop-in stand-in for :class:`zakcode.Agent` used by the CLI tests.

    Streams a single text delta plus a terminal ``AgentDone`` by default. The
    ``usage`` carried on ``AgentDone`` also gets folded into the session so the
    ``/cost`` command has something to report.
    """

    #: Events yielded for each turn (subclasses override to add tool lines, etc.).
    events: list[AgentEvent] = [
        AgentTextDelta(text=CANNED_TEXT + "\n"),
        AgentDone(
            stop_reason="completed",
            iterations=1,
            usage=Usage(prompt_tokens=3, completion_tokens=5, total_tokens=8),
        ),
    ]

    def __init__(self, **overrides: object) -> None:
        # The CLI now constructs the Agent with prompter=... ; accept and ignore it.
        self.overrides = overrides
        self.settings = load_settings()
        self.session = Session(cwd=".", model=self.settings.default_model)
        self.turns: list[str] = []
        # Minimal stand-ins so /permissions and /hooks have something to render.
        self.permission_policy = PermissionPolicy(self.settings.permission_mode)
        self.hook_manager = HookManager()

    def astream_turn(self, text: str) -> AsyncIterator[AgentEvent]:
        self.turns.append(text)
        # Fold each AgentDone's usage into the session so /cost has data.
        for event in self.events:
            if isinstance(event, AgentDone):
                self.session.add_usage(event.usage)
        return self._gen()

    async def _gen(self) -> AsyncIterator[AgentEvent]:
        for event in self.events:
            yield event


def test_chat_streams_assistant_text_and_exits(monkeypatch) -> None:
    monkeypatch.setattr(zakcode, "Agent", FakeAgent)
    result = runner.invoke(app, ["chat"], input="hello\n/exit\n")
    assert result.exit_code == 0
    assert CANNED_TEXT in result.stdout


def test_chat_eof_exits_cleanly(monkeypatch) -> None:
    monkeypatch.setattr(zakcode, "Agent", FakeAgent)
    # No "/exit" — EOF on the empty stream must still exit 0.
    result = runner.invoke(app, ["chat"], input="")
    assert result.exit_code == 0


def test_chat_slash_help_and_model(monkeypatch) -> None:
    monkeypatch.setattr(zakcode, "Agent", FakeAgent)
    result = runner.invoke(app, ["chat"], input="/help\n/model\n/exit\n")
    assert result.exit_code == 0
    assert "Slash commands" in result.stdout
    assert load_settings().default_model in result.stdout


def test_chat_unknown_slash_is_friendly(monkeypatch) -> None:
    monkeypatch.setattr(zakcode, "Agent", FakeAgent)
    result = runner.invoke(app, ["chat"], input="/bananas\n/exit\n")
    assert result.exit_code == 0
    assert "not yet" in result.stdout


def test_chat_cost_reports_usage(monkeypatch) -> None:
    monkeypatch.setattr(zakcode, "Agent", FakeAgent)
    result = runner.invoke(app, ["chat"], input="hello\n/cost\n/exit\n")
    assert result.exit_code == 0
    assert "total=8" in result.stdout


def test_chat_renders_tool_lines(monkeypatch) -> None:
    class ToolAgent(FakeAgent):
        events = [
            AgentToolCall(id="t1", name="bash", arguments={"command": "ls -la"}),
            AgentToolResult(tool_use_id="t1", output="files", is_error=False),
            AgentTextDelta(text=CANNED_TEXT + "\n"),
            AgentDone(
                stop_reason="completed",
                iterations=2,
                usage=Usage(total_tokens=12),
            ),
        ]

    monkeypatch.setattr(zakcode, "Agent", ToolAgent)
    result = runner.invoke(app, ["chat"], input="run ls\n/exit\n")
    assert result.exit_code == 0
    # Exactly one tool-call summary line is rendered for the single tool use.
    assert result.stdout.count("tool bash(") == 1
    assert "-> ok" in result.stdout
    assert CANNED_TEXT in result.stdout


def test_chat_provider_error_stays_in_repl(monkeypatch) -> None:
    class BoomAgent(FakeAgent):
        def astream_turn(self, text: str) -> AsyncIterator[AgentEvent]:
            raise ProviderError("model unreachable")

    monkeypatch.setattr(zakcode, "Agent", BoomAgent)
    result = runner.invoke(app, ["chat"], input="hello\n/exit\n")
    assert result.exit_code == 0
    assert "Provider error" in result.stdout


def test_chat_permissions_command(monkeypatch) -> None:
    monkeypatch.setattr(zakcode, "Agent", FakeAgent)
    result = runner.invoke(app, ["chat"], input="/permissions\n/exit\n")
    assert result.exit_code == 0
    assert "permission mode" in result.stdout
    assert load_settings().permission_mode in result.stdout


def test_chat_hooks_command_empty(monkeypatch) -> None:
    monkeypatch.setattr(zakcode, "Agent", FakeAgent)
    result = runner.invoke(app, ["chat"], input="/hooks\n/exit\n")
    assert result.exit_code == 0
    assert "no hooks configured" in result.stdout


def test_chat_hooks_command_lists_configured(monkeypatch) -> None:
    class HookedAgent(FakeAgent):
        def __init__(self, **overrides: object) -> None:
            super().__init__(**overrides)
            self.hook_manager = HookManager(
                [HookSpec(event=HookEvent.PRE_TOOL_USE, command=["echo", "hi"], matcher="bash")]
            )

    monkeypatch.setattr(zakcode, "Agent", HookedAgent)
    result = runner.invoke(app, ["chat"], input="/hooks\n/exit\n")
    assert result.exit_code == 0
    assert "PreToolUse" in result.stdout
    assert "echo hi" in result.stdout


# ── the console permission prompter (protocol implementation) ─────────────────


def _request() -> PermissionRequest:
    from zakcode.config import PermissionTier

    return PermissionRequest(
        tool_name="bash",
        tier=PermissionTier.DANGER_FULL_ACCESS,
        arguments={"command": "ls -la"},
        reason="danger tier requires confirmation",
    )


class _FakeConsole:
    """Captures printed lines and returns a scripted answer to input()."""

    def __init__(self, answer: str) -> None:
        self.answer = answer
        self.lines: list[str] = []

    def print(self, *args: object, **_kw: object) -> None:
        self.lines.append(" ".join(str(a) for a in args))

    def input(self, _prompt: str = "") -> str:
        return self.answer


async def test_console_prompter_allow_once() -> None:
    prompter = ConsolePermissionPrompter(_FakeConsole("y"))
    assert await prompter.confirm(_request()) is PermissionOutcome.ALLOW_ONCE


async def test_console_prompter_allow_session() -> None:
    prompter = ConsolePermissionPrompter(_FakeConsole("a"))
    assert await prompter.confirm(_request()) is PermissionOutcome.ALLOW_SESSION


async def test_console_prompter_deny_and_shows_command() -> None:
    console = _FakeConsole("n")
    prompter = ConsolePermissionPrompter(console)
    assert await prompter.confirm(_request()) is PermissionOutcome.DENY_ONCE
    # GUARDRAILS §3: the operator must see the exact command being confirmed.
    assert any("ls -la" in line for line in console.lines)


async def test_console_prompter_unrecognized_answer_denies() -> None:
    prompter = ConsolePermissionPrompter(_FakeConsole("maybe?"))
    assert await prompter.confirm(_request()) is PermissionOutcome.DENY_ONCE


def test_chat_builds_agent_with_prompter(monkeypatch) -> None:
    # The chat command must inject a prompter so 'ask' mode is usable interactively.
    captured: dict[str, object] = {}

    class CapturingAgent(FakeAgent):
        def __init__(self, **overrides: object) -> None:
            captured.update(overrides)
            super().__init__(**{k: v for k, v in overrides.items() if k != "prompter"})

    monkeypatch.setattr(zakcode, "Agent", CapturingAgent)
    result = runner.invoke(app, ["chat"], input="/exit\n")
    assert result.exit_code == 0
    assert isinstance(captured.get("prompter"), ConsolePermissionPrompter)


def _capture_builds(monkeypatch) -> list[dict]:
    """Monkeypatch Agent with a capturer that records every construction's kwargs."""
    builds: list[dict] = []

    class CapturingAgent(FakeAgent):
        def __init__(self, **overrides: object) -> None:
            builds.append(dict(overrides))
            super().__init__(**{k: v for k, v in overrides.items() if k != "prompter"})

    monkeypatch.setattr(zakcode, "Agent", CapturingAgent)
    return builds


def test_chat_clear_preserves_no_memory_and_no_rules(monkeypatch) -> None:
    # The no-drift guarantee the single-builder design rests on: /clear must rebuild
    # with the SAME flag choice, not silently re-enable memory/rules.
    builds = _capture_builds(monkeypatch)
    result = runner.invoke(app, ["chat", "--no-memory", "--no-rules"], input="/clear\n/exit\n")
    assert result.exit_code == 0
    assert len(builds) == 2  # initial build + the /clear rebuild
    for b in builds:
        assert b.get("enable_memory") is False
        assert b.get("enable_rules") is False


def test_chat_clear_default_keeps_memory_and_rules_on(monkeypatch) -> None:
    builds = _capture_builds(monkeypatch)
    result = runner.invoke(app, ["chat"], input="/clear\n/exit\n")
    assert result.exit_code == 0
    assert len(builds) == 2
    assert all(b.get("enable_memory") is True and b.get("enable_rules") is True for b in builds)


# Keep an explicit reference so the unused-import linter is satisfied for the
# scripted-policy helper used indirectly above.
_ = PermissionMode


# ── permission-answer parsing (the [y]/[a]/[n] markup + lenient-words fix) ─────


@pytest.mark.parametrize(
    "answer,expected",
    [
        ("y", PermissionOutcome.ALLOW_ONCE),
        ("yes", PermissionOutcome.ALLOW_ONCE),
        ("allow once", PermissionOutcome.ALLOW_ONCE),
        ("allow", PermissionOutcome.ALLOW_ONCE),
        ("a", PermissionOutcome.ALLOW_SESSION),
        ("always", PermissionOutcome.ALLOW_SESSION),
        ("session", PermissionOutcome.ALLOW_SESSION),
        ("allow for session", PermissionOutcome.ALLOW_SESSION),
        ("ALLOW FOR SESSION", PermissionOutcome.ALLOW_SESSION),
        ("n", PermissionOutcome.DENY_ONCE),
        ("no", PermissionOutcome.DENY_ONCE),
        ("deny", PermissionOutcome.DENY_ONCE),
    ],
)
def test_parse_permission_answer(answer: str, expected: PermissionOutcome) -> None:
    assert _parse_permission_answer(answer) is expected


def test_parse_permission_answer_unrecognized_is_none() -> None:
    assert _parse_permission_answer("maybe?") is None
    assert _parse_permission_answer("") is None


async def test_console_prompter_accepts_spelled_out_session() -> None:
    # The reported bug: typing the words shown ("allow for session") must allow for
    # the session, not silently deny.
    prompter = ConsolePermissionPrompter(_FakeConsole("allow for session"))
    assert await prompter.confirm(_request()) is PermissionOutcome.ALLOW_SESSION


async def test_console_prompter_shows_key_hints_literally() -> None:
    # Regression: the keys must be visible. With rich markup, "[y]" is parsed as a
    # tag and dropped from the rendered line; parens survive.
    console = _FakeConsole("y")
    await ConsolePermissionPrompter(console).confirm(_request())
    blob = "\n".join(console.lines)
    assert "(y)" in blob and "(a)" in blob and "(n)" in blob
    assert "[y]" not in blob


# ── session event loop (one loop per REPL, never one per turn) ─────────────────


def test_run_async_reuses_session_loop_without_closing() -> None:
    # The session-loop fix: _run_async runs on the installed loop and leaves it OPEN
    # for reuse across turns (never a loop-per-call), so library background tasks
    # (e.g. litellm's logging worker) stay bound to a live loop instead of being
    # orphaned when a per-turn loop closes.
    import zakcode.cli as cli

    async def _coro() -> int:
        return 42

    loop = asyncio.new_event_loop()
    prev = cli._SESSION_LOOP
    cli._SESSION_LOOP = loop
    try:
        assert cli._run_async(_coro()) == 42
        assert not loop.is_closed()  # reused, not torn down per call
    finally:
        cli._SESSION_LOOP = prev
        loop.close()


def test_run_async_falls_back_to_fresh_loop_when_no_session() -> None:
    # Outside an interactive session (e.g. a handler called directly in a test),
    # _run_async still works by falling back to asyncio.run.
    import zakcode.cli as cli

    async def _coro() -> int:
        return 7

    prev = cli._SESSION_LOOP
    cli._SESSION_LOOP = None
    try:
        assert cli._run_async(_coro()) == 7
    finally:
        cli._SESSION_LOOP = prev


def test_shutdown_session_loop_closes_and_clears() -> None:
    import zakcode.cli as cli

    loop = asyncio.new_event_loop()
    prev = cli._SESSION_LOOP
    cli._SESSION_LOOP = loop
    try:
        cli._shutdown_session_loop(loop)
        assert loop.is_closed()
        assert cli._SESSION_LOOP is None  # session reference cleared
    finally:
        cli._SESSION_LOOP = prev
