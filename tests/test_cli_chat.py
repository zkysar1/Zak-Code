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
import io
import os
from collections.abc import AsyncIterator
from pathlib import Path

import pytest
from rich.console import Console
from typer.testing import CliRunner

import zakcode
from zakcode.cli import ConsolePermissionPrompter, _parse_permission_answer, app
from zakcode.cli._theme import ZAK_THEME
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
    result = runner.invoke(app, ["cli"], input="hello\n/exit\n")
    assert result.exit_code == 0
    assert CANNED_TEXT in result.stdout


def test_chat_eof_exits_cleanly_with_bookend(monkeypatch) -> None:
    monkeypatch.setattr(zakcode, "Agent", FakeAgent)
    # No "/exit" — EOF on the empty stream must still exit 0, with the close bookend.
    result = runner.invoke(app, ["cli"], input="")
    assert result.exit_code == 0
    assert "session closed" in result.stdout
    assert "goodbye" in result.stdout


def test_chat_exit_prints_session_close_bookend(monkeypatch) -> None:
    monkeypatch.setattr(zakcode, "Agent", FakeAgent)
    result = runner.invoke(app, ["cli"], input="/exit\n")
    assert result.exit_code == 0
    assert "session closed" in result.stdout
    assert "goodbye" in result.stdout


def test_chat_slash_help_and_model(monkeypatch) -> None:
    monkeypatch.setattr(zakcode, "Agent", FakeAgent)
    result = runner.invoke(app, ["cli"], input="/help\n/model\n/exit\n")
    assert result.exit_code == 0
    # The grouped /help layout: three section headings + the full command set.
    for section in ("session", "agent", "integrations"):
        assert section in result.stdout
    for command in ("/permissions", "/plan", "/mcp", "/compact", "/skills"):
        assert command in result.stdout
    assert "ctrl-c interrupts a running reply" in result.stdout
    assert load_settings().default_model in result.stdout


def test_chat_unknown_slash_is_friendly(monkeypatch) -> None:
    monkeypatch.setattr(zakcode, "Agent", FakeAgent)
    result = runner.invoke(app, ["cli"], input="/bananas\n/exit\n")
    assert result.exit_code == 0
    assert "not yet" in result.stdout


def test_chat_cost_reports_usage(monkeypatch) -> None:
    monkeypatch.setattr(zakcode, "Agent", FakeAgent)
    result = runner.invoke(app, ["cli"], input="hello\n/cost\n/exit\n")
    assert result.exit_code == 0
    assert "total=8" in result.stdout


def test_chat_headless_prompt_runs_once_and_exits_zero(monkeypatch) -> None:
    # `-p TASK` runs a single turn (no REPL, no banner) and exits 0 when the turn completes.
    monkeypatch.setattr(zakcode, "Agent", FakeAgent)
    result = runner.invoke(app, ["cli", "-p", "do the thing"])
    assert result.exit_code == 0
    assert CANNED_TEXT in result.stdout  # the turn actually ran
    assert "session closed" not in result.stdout  # it never entered the REPL


def test_chat_headless_prompt_nonzero_when_incomplete(monkeypatch) -> None:
    # A turn that does NOT complete (hit the iteration cap) exits non-zero, so a script knows.
    class CappedAgent(FakeAgent):
        events = [
            AgentTextDelta(text="ran out of road\n"),
            AgentDone(stop_reason="max_iterations", iterations=50, usage=Usage(total_tokens=4)),
        ]

    monkeypatch.setattr(zakcode, "Agent", CappedAgent)
    result = runner.invoke(app, ["cli", "-p", "do the thing"])
    assert result.exit_code == 1


def test_chat_headless_empty_prompt_is_rejected(monkeypatch) -> None:
    # `-p "   "` is a usage error (exit 2), not a silent no-op.
    monkeypatch.setattr(zakcode, "Agent", FakeAgent)
    result = runner.invoke(app, ["cli", "-p", "   "])
    assert result.exit_code == 2
    assert "empty" in result.stdout


class SkillAgent(FakeAgent):
    """FakeAgent + a ``compose_skill_turn`` surface for the one-shot slash-dispatch tests."""

    #: The SkillInvocation the fake compose returns (subclasses override per scenario).
    invocation = zakcode.SkillInvocation(
        invoked=True,
        name="start",
        turn_text=(
            "<command-message>start is running</command-message>\n"
            "<command-name>/start</command-name>\n"
            "<command-args>sera</command-args>\n\nBoot body."
        ),
    )

    async def compose_skill_turn(self, name: str, args: str = "") -> zakcode.SkillInvocation:
        return self.invocation


def test_chat_headless_slash_dispatches_the_skill(monkeypatch) -> None:
    # `-p "/start sera"` (the cron/systemd boot shape) runs the SKILL as the task — the same
    # dispatch + rendering as the REPL — instead of handing the slash line to the model as
    # prose (#148: found on the first live Claude-Mind deployment).
    monkeypatch.setattr(zakcode, "Agent", SkillAgent)
    result = runner.invoke(app, ["cli", "-p", "/start sera"])
    assert result.exit_code == 0
    assert "running skill" in result.stdout and "start" in result.stdout
    assert CANNED_TEXT in result.stdout  # the composed turn actually ran


def test_chat_headless_slash_denied_exits_nonzero(monkeypatch) -> None:
    # A discovered-but-refused skill (user-invocable: false) must FAIL the one-shot loudly —
    # exit 1, no model turn — because a scripted boot that quietly degrades is the #148 class.
    class DeniedAgent(SkillAgent):
        invocation = zakcode.SkillInvocation(
            invoked=True, name="boot", denied_reason="boot is not user-invocable"
        )

    monkeypatch.setattr(zakcode, "Agent", DeniedAgent)
    result = runner.invoke(app, ["cli", "-p", "/boot"])
    assert result.exit_code == 1
    assert "not user-invocable" in result.stdout
    assert CANNED_TEXT not in result.stdout  # no model turn ran


def test_chat_headless_unknown_slash_falls_through_to_model(monkeypatch) -> None:
    # An unknown /token is legitimate one-shot prose (e.g. a slash-PATH) — plain turn, as before.
    class NoSkillAgent(SkillAgent):
        invocation = zakcode.SkillInvocation(invoked=False)

    monkeypatch.setattr(zakcode, "Agent", NoSkillAgent)
    result = runner.invoke(app, ["cli", "-p", "/etc/hosts looks wrong, why?"])
    assert result.exit_code == 0
    assert CANNED_TEXT in result.stdout
    assert "running skill" not in result.stdout


def test_chat_headless_slash_on_thin_agent_falls_through(monkeypatch) -> None:
    # A thin/remote AgentLike with NO compose_skill_turn (the --server client) keeps today's
    # behavior: the prompt goes to the model as plain text, no dispatch attempted.
    monkeypatch.setattr(zakcode, "Agent", FakeAgent)  # FakeAgent has no compose surface
    result = runner.invoke(app, ["cli", "-p", "/start sera"])
    assert result.exit_code == 0
    assert CANNED_TEXT in result.stdout


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
    result = runner.invoke(app, ["cli"], input="run ls\n/exit\n")
    assert result.exit_code == 0
    # Exactly one tool-call line is rendered for the single tool use (its target),
    # followed by the result summary and the assistant text.
    assert result.stdout.count("ls -la") == 1
    assert "files" in result.stdout
    assert CANNED_TEXT in result.stdout


def test_chat_provider_error_stays_in_repl(monkeypatch) -> None:
    class BoomAgent(FakeAgent):
        def astream_turn(self, text: str) -> AsyncIterator[AgentEvent]:
            raise ProviderError("model unreachable")

    monkeypatch.setattr(zakcode, "Agent", BoomAgent)
    result = runner.invoke(app, ["cli"], input="hello\n/exit\n")
    assert result.exit_code == 0
    assert "provider error" in result.stdout


def test_chat_permissions_command(monkeypatch) -> None:
    monkeypatch.setattr(zakcode, "Agent", FakeAgent)
    result = runner.invoke(app, ["cli"], input="/permissions\n/exit\n")
    assert result.exit_code == 0
    assert "permission mode" in result.stdout
    assert load_settings().permission_mode in result.stdout


def test_chat_hooks_command_empty(monkeypatch) -> None:
    monkeypatch.setattr(zakcode, "Agent", FakeAgent)
    result = runner.invoke(app, ["cli"], input="/hooks\n/exit\n")
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
    result = runner.invoke(app, ["cli"], input="/hooks\n/exit\n")
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
    """Renders printed renderables to an in-memory buffer (so assertions inspect the
    real rendered text) and returns a scripted answer to ``input``.

    The prompter prints rich renderables (Panel/Padding/Text), so a naive ``str(arg)``
    capture would only see reprs; this delegates to a real Console writing to a buffer.
    """

    def __init__(self, answer: str) -> None:
        self.answer = answer
        self._buf = io.StringIO()
        self._console = Console(
            file=self._buf, force_terminal=False, no_color=True, width=100, theme=ZAK_THEME
        )

    def print(self, *args: object, **kwargs: object) -> None:
        self._console.print(*args, **kwargs)

    def input(self, prompt: object = "") -> str:
        self._console.print(prompt, end="")
        return self.answer

    @property
    def lines(self) -> list[str]:
        return self._buf.getvalue().splitlines()

    @property
    def encoding(self) -> str:
        return self._console.encoding

    @property
    def file(self) -> io.StringIO:
        return self._buf

    @property
    def width(self) -> int:
        # The prompter's panel() clamps its box width against the console width.
        return self._console.width


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
    result = runner.invoke(app, ["cli"], input="/exit\n")
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


def test_chat_clear_preserves_no_rules(monkeypatch) -> None:
    # The no-drift guarantee the single-builder design rests on: /clear must rebuild
    # with the SAME flag choice, not silently re-enable rules.
    builds = _capture_builds(monkeypatch)
    result = runner.invoke(app, ["cli", "--no-rules"], input="/clear\n/exit\n")
    assert result.exit_code == 0
    assert len(builds) == 2  # initial build + the /clear rebuild
    for b in builds:
        assert b.get("enable_rules") is False


def test_chat_clear_default_keeps_rules_on(monkeypatch) -> None:
    builds = _capture_builds(monkeypatch)
    result = runner.invoke(app, ["cli"], input="/clear\n/exit\n")
    assert result.exit_code == 0
    assert len(builds) == 2
    assert all(b.get("enable_rules") is True for b in builds)


# Keep an explicit reference so the unused-import linter is satisfied for the
# scripted-policy helper used indirectly above.
_ = PermissionMode


# ── permission-answer parsing (the [y]/[a]/[n] markup + lenient-words fix) ─────


@pytest.mark.parametrize(
    "answer,expected",
    [
        ("1", PermissionOutcome.ALLOW_ONCE),
        ("y", PermissionOutcome.ALLOW_ONCE),
        ("yes", PermissionOutcome.ALLOW_ONCE),
        ("allow once", PermissionOutcome.ALLOW_ONCE),
        ("allow", PermissionOutcome.ALLOW_ONCE),
        ("2", PermissionOutcome.ALLOW_SESSION),
        ("a", PermissionOutcome.ALLOW_SESSION),
        ("always", PermissionOutcome.ALLOW_SESSION),
        ("session", PermissionOutcome.ALLOW_SESSION),
        ("allow for session", PermissionOutcome.ALLOW_SESSION),
        ("ALLOW FOR SESSION", PermissionOutcome.ALLOW_SESSION),
        ("3", PermissionOutcome.DENY_ONCE),
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


async def test_console_prompter_shows_numbered_options_and_keys() -> None:
    # The §2 panel: numbered options with their single-key hints, and the prompt
    # advertising both forms. Keys render literally (never parsed as "[y]" markup).
    console = _FakeConsole("y")
    await ConsolePermissionPrompter(console).confirm(_request())
    blob = "\n".join(console.lines)
    assert "allow once" in blob
    assert "allow for this session" in blob
    assert "tell Zak what to do instead" in blob
    for key in ("1", "2", "3", "y", "a", "n"):
        assert key in blob
    assert "permit (1-3 or y/a/n)" in blob
    assert "[y]" not in blob


async def test_console_prompter_accepts_numbered_answer() -> None:
    prompter = ConsolePermissionPrompter(_FakeConsole("2"))
    assert await prompter.confirm(_request()) is PermissionOutcome.ALLOW_SESSION


async def test_console_prompter_humanizes_tier() -> None:
    # The §3 panel: the blast radius reads as words; the raw enum name never prints.
    console = _FakeConsole("n")
    await ConsolePermissionPrompter(console).confirm(_request())
    blob = "\n".join(console.lines)
    assert "full access — touches paths outside the workspace" in blob
    assert "DANGER_FULL_ACCESS" not in blob


async def test_console_prompter_pauses_active_wait_line(monkeypatch) -> None:
    # The prompter pauses the REPL's live wait line before the panel owns the tty
    # and resumes it after the decision (the _ACTIVE_WAIT handle contract).
    import zakcode.cli as cli

    calls: list[str] = []

    class FakeWait:
        def pause(self) -> None:
            calls.append("pause")

        def resume(self) -> None:
            calls.append("resume")

    monkeypatch.setattr(cli, "_ACTIVE_WAIT", FakeWait())
    prompter = ConsolePermissionPrompter(_FakeConsole("y"))
    assert await prompter.confirm(_request()) is PermissionOutcome.ALLOW_ONCE
    assert calls == ["pause", "resume"]


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


# ── audit #4/#5: /plugins + /skills are cp1252-safe and markup-immune ──────────


def _ascii_console() -> tuple[Console, io.StringIO]:
    buf = io.StringIO()
    return Console(file=buf, force_terminal=False, no_color=True, width=100, theme=ZAK_THEME), buf


def test_render_plugins_is_ascii_safe_and_markup_immune(monkeypatch) -> None:
    from types import SimpleNamespace

    from zakcode.cli import _render_plugins

    monkeypatch.setenv("ZAKCODE_ASCII", "1")  # force the cp1252-safe glyph set
    console, buf = _ascii_console()
    report = SimpleNamespace(
        loaded=["good"],
        skipped={"skip [/] me": "needs x"},  # a bare close tag would crash f-string markup
        failed={"bad": "boom [bold]"},
        contributions={"good": {"tools": [1, 2]}},
    )
    agent = SimpleNamespace(plugin_report=report, plugin_discovery_errors={"disc": "nope"})
    _render_plugins(console, agent)  # must not raise MarkupError or UnicodeEncodeError
    out = buf.getvalue()
    assert "✓" not in out and "✗" not in out  # raw unicode glyphs never emitted
    assert "+ good" in out  # ok glyph fell back to the single-char ASCII "+"
    assert "skip [/] me" in out  # the bare close tag rendered literally, not parsed
    assert "boom [bold]" in out  # style tags rendered literally, not consumed


def test_render_skills_is_ascii_safe_and_markup_immune(monkeypatch) -> None:
    from types import SimpleNamespace

    from zakcode.cli import _render_skills

    monkeypatch.setenv("ZAKCODE_ASCII", "1")
    console, buf = _ascii_console()

    class _Registry:
        def __len__(self) -> int:
            return 1

        def catalog(self) -> list[tuple[str, str]]:
            return [("greet [/]", "say hi")]

    agent = SimpleNamespace(skill_registry=_Registry(), skill_errors={"oops": "missing [bold]"})
    _render_skills(console, agent)  # must not raise
    out = buf.getvalue()
    assert "—" not in out  # em-dash fell back to ASCII
    assert "greet [/]" in out  # literal, not parsed as markup
    assert "missing [bold]" in out


# ── the welcome banner (panel + daily tip + value truncation) ──────────────────


def _buffer_console(
    *, force_terminal: bool = False, width: int = 90
) -> tuple[Console, io.StringIO]:
    buf = io.StringIO()
    return (
        Console(
            file=buf,
            force_terminal=force_terminal,
            legacy_windows=False,
            no_color=True,
            width=width,
            theme=ZAK_THEME,
        ),
        buf,
    )


def test_print_banner_welcome_panel_off_tty() -> None:
    from zakcode.cli import _print_banner

    console, buf = _buffer_console()
    _print_banner(console, FakeAgent())
    out = buf.getvalue()
    assert "✦ Zak Code" in out
    for label in ("model", "workspace", "permissions", "session"):
        assert label in out
    assert "/help for commands" in out
    assert "/exit to quit" in out
    assert "tip:" not in out  # off-tty: hermetic output never carries the daily tip


def test_print_banner_tip_on_tty() -> None:
    from zakcode.cli import _TIPS, _print_banner

    console, buf = _buffer_console(force_terminal=True)
    _print_banner(console, FakeAgent())
    out = buf.getvalue()
    assert "tip:" in out
    assert any(tip in out for tip in _TIPS)


def test_print_banner_left_truncates_long_values() -> None:
    from zakcode.cli import _print_banner
    from zakcode.config import Settings

    agent = FakeAgent()
    long_root = "C:\\very\\" + "deep\\" * 30 + "workspace-tail"
    agent.settings = Settings(default_model="scripted/test", workspace_root=long_root)
    console, buf = _buffer_console()
    _print_banner(console, agent)
    out = buf.getvalue()
    assert "workspace-tail" in out  # the filename end survives
    assert long_root not in out  # over-long value was left-truncated
    assert "…" in out  # with a leading ellipsis


# ── the wait line (REPL-owned; auto-disabled off-tty / legacy / by env) ────────


def test_start_wait_disabled_off_tty() -> None:
    from zakcode.cli import _start_wait

    console, _buf = _buffer_console(force_terminal=False)
    assert _start_wait(console) is None


def test_start_wait_honors_no_spinner_env(monkeypatch) -> None:
    from zakcode.cli import _start_wait

    monkeypatch.setenv("ZAKCODE_NO_SPINNER", "1")
    console, _buf = _buffer_console(force_terminal=True)
    assert _start_wait(console) is None


def test_start_wait_disabled_on_legacy_conhost(monkeypatch) -> None:
    from zakcode.cli import _start_wait

    monkeypatch.delenv("ZAKCODE_NO_SPINNER", raising=False)
    buf = io.StringIO()
    console = Console(
        file=buf,
        force_terminal=True,
        legacy_windows=True,
        no_color=True,
        width=90,
        theme=ZAK_THEME,
    )
    assert _start_wait(console) is None


def test_start_wait_sets_and_clears_active_handle(monkeypatch) -> None:
    import zakcode.cli as cli

    monkeypatch.delenv("ZAKCODE_NO_SPINNER", raising=False)
    console, _buf = _buffer_console(force_terminal=True)
    handle = cli._start_wait(console)
    assert handle is not None  # tty + modern terminal: the wait line exists
    try:
        assert cli._ACTIVE_WAIT is handle  # the prompter can find it to pause
    finally:
        handle.stop()
    assert cli._ACTIVE_WAIT is None  # stop() clears the module handle


def test_wait_handle_verb_tracks_outstanding_tools() -> None:
    from zakcode.cli import _WaitHandle

    console, _buf = _buffer_console(force_terminal=True)
    handle = _WaitHandle(console)  # never started: observe() only moves the verb
    handle.observe(AgentToolCall(id="t1", name="bash", arguments={}))
    assert handle.line.verb == "Running"
    handle.observe(AgentToolCall(id="t2", name="bash", arguments={}))
    handle.observe(AgentToolResult(tool_use_id="t1", output="", is_error=False))
    assert handle.line.verb == "Running"  # one call still outstanding
    handle.observe(AgentToolResult(tool_use_id="t2", output="", is_error=False))
    assert handle.line.verb != "Running"  # all results in: back to a gerund


# ── Ctrl-C at the prompt: double-press to exit (issue: one press killed the session) ──


def _scripted_prompt(monkeypatch, script):
    """Script the keyboard: replace the pump's read point (_read_stdin_line) with a
    sequence; an exception INSTANCE raises. The seam moved from read_prompt when
    the input mux became the single stdin owner."""
    seq = iter(script)

    def fake_read_line():
        item = next(seq)
        if isinstance(item, BaseException):
            raise item
        return item

    monkeypatch.setattr("zakcode.cli._read_stdin_line", fake_read_line)


def test_chat_single_ctrl_c_does_not_exit(monkeypatch) -> None:
    monkeypatch.setattr(zakcode, "Agent", FakeAgent)
    _scripted_prompt(monkeypatch, [KeyboardInterrupt(), "/exit"])
    result = runner.invoke(app, ["cli"])
    assert result.exit_code == 0
    assert "press ctrl-c again to exit" in result.stdout
    # The session survived the single interrupt and closed via /exit.
    assert "goodbye" in result.stdout


def test_chat_double_ctrl_c_exits(monkeypatch) -> None:
    monkeypatch.setattr(zakcode, "Agent", FakeAgent)
    _scripted_prompt(monkeypatch, [KeyboardInterrupt(), KeyboardInterrupt()])
    result = runner.invoke(app, ["cli"])
    assert result.exit_code == 0
    assert "press ctrl-c again to exit" in result.stdout
    assert "goodbye" in result.stdout


def test_mid_turn_ctrl_c_arms_the_shared_exit_window(monkeypatch) -> None:
    # Field report 2026-08-26: with a per-prompt-loop window, interrupting a RUNNING
    # turn took interrupt + arm + exit = three presses — the mid-turn press never
    # armed the prompt's double-press window. The interrupt handler now stamps the
    # module-wide window, so the NEXT press exits: two presses total, always.
    import io

    from rich.console import Console

    import zakcode.cli as cli_mod
    from zakcode.cli.render import StreamRenderer

    monkeypatch.setattr(cli_mod, "_LAST_CTRL_C", 0.0)
    loop = asyncio.new_event_loop()
    monkeypatch.setattr(cli_mod, "_SESSION_LOOP", loop)
    buffer = io.StringIO()
    console = Console(file=buffer, width=100, force_terminal=False)

    async def dead_stream() -> AsyncIterator[AgentEvent]:
        # The press lands while the turn runs: the pump re-raises it from the task.
        raise KeyboardInterrupt
        yield  # pragma: no cover — makes this an async generator

    try:
        completed = cli_mod._run_streamed_turn(
            console, dead_stream, StreamRenderer(console=console)
        )
    finally:
        loop.close()
    assert completed is False
    assert cli_mod._LAST_CTRL_C > 0.0  # the window is armed — the next press exits
    assert "ctrl-c again exits" in buffer.getvalue()  # the notice names the affordance


def test_chat_slow_second_ctrl_c_still_does_not_exit(monkeypatch) -> None:
    # Two interrupts OUTSIDE the window are two singles — the session survives both.
    # A negative window makes every interrupt "slow" without patching time.monotonic
    # (which asyncio shares).
    monkeypatch.setattr(zakcode, "Agent", FakeAgent)
    monkeypatch.setattr("zakcode.cli._CTRL_C_EXIT_WINDOW_S", -1.0)
    _scripted_prompt(monkeypatch, [KeyboardInterrupt(), KeyboardInterrupt(), "/exit"])
    result = runner.invoke(app, ["cli"])
    assert result.exit_code == 0
    assert result.stdout.count("press ctrl-c again to exit") == 2


# ── one hammer gesture must never kill the session (field incident 2026-08-26) ──


def test_ctrl_c_disposition_contract(monkeypatch) -> None:
    # The shared window classifier: an idle-prompt rapid double-press is the documented
    # exit gesture, but the rapid tail of a MID-TURN interrupt is the same hammer and is
    # absorbed — however many presses it has — until the presses slow down.
    import time as time_mod

    import zakcode.cli as cli_mod

    # Idle prompt: first press arms, rapid second press exits (the documented gesture).
    monkeypatch.setattr(cli_mod, "_LAST_CTRL_C", 0.0)
    monkeypatch.setattr(cli_mod, "_LAST_CTRL_C_MID_TURN", False)
    assert cli_mod._ctrl_c_disposition() == "arm"
    assert cli_mod._ctrl_c_disposition() == "exit"

    # Mid-turn-armed window: presses inside the gesture refractory absorb, repeatedly.
    monkeypatch.setattr(cli_mod, "_LAST_CTRL_C", time_mod.monotonic())
    monkeypatch.setattr(cli_mod, "_LAST_CTRL_C_MID_TURN", True)
    assert cli_mod._ctrl_c_disposition() == "absorb"
    assert cli_mod._LAST_CTRL_C_MID_TURN is True  # the gesture continues
    assert cli_mod._ctrl_c_disposition() == "absorb"

    # Mid-turn-armed window, DELIBERATE second press (past the refractory, inside the
    # window): exits — two presses total, the requested affordance.
    monkeypatch.setattr(
        cli_mod, "_LAST_CTRL_C", time_mod.monotonic() - (cli_mod._CTRL_C_GESTURE_S + 0.05)
    )
    monkeypatch.setattr(cli_mod, "_LAST_CTRL_C_MID_TURN", True)
    assert cli_mod._ctrl_c_disposition() == "exit"

    # Stale window (mid-turn or not): back to arming.
    monkeypatch.setattr(
        cli_mod, "_LAST_CTRL_C", time_mod.monotonic() - (cli_mod._CTRL_C_EXIT_WINDOW_S + 1.0)
    )
    monkeypatch.setattr(cli_mod, "_LAST_CTRL_C_MID_TURN", True)
    assert cli_mod._ctrl_c_disposition() == "arm"


def test_absorb_interrupts_retries_through_hammered_presses() -> None:
    # The teardown wrapper must complete its step no matter how many presses land in it.
    import zakcode.cli as cli_mod

    calls = {"n": 0}

    def step() -> None:
        calls["n"] += 1
        if calls["n"] <= 2:
            raise KeyboardInterrupt

    cli_mod._absorb_interrupts(step)  # returns instead of letting a press escape
    assert calls["n"] == 3


def test_hammered_press_during_teardown_does_not_escape(monkeypatch) -> None:
    # A press landing INSIDE the interrupt teardown (here: while the notice prints) used
    # to escape as a raw KeyboardInterrupt, unwind the REPL, and — via the cockpit pane
    # chain — kill the whole tmux session. The teardown now absorbs it and completes.
    import io

    from rich.console import Console

    import zakcode.cli as cli_mod
    from zakcode.cli.render import StreamRenderer

    monkeypatch.setattr(cli_mod, "_LAST_CTRL_C", 0.0)
    monkeypatch.setattr(cli_mod, "_LAST_CTRL_C_MID_TURN", False)
    loop = asyncio.new_event_loop()
    monkeypatch.setattr(cli_mod, "_SESSION_LOOP", loop)
    buffer = io.StringIO()
    console = Console(file=buffer, width=100, force_terminal=False)

    real_notice_warn = cli_mod.notice_warn
    hammered = {"n": 0}

    def hammering_notice(*args, **kwargs):
        hammered["n"] += 1
        if hammered["n"] == 1:
            raise KeyboardInterrupt  # the hammer's second press lands mid-teardown
        return real_notice_warn(*args, **kwargs)

    monkeypatch.setattr(cli_mod, "notice_warn", hammering_notice)

    async def dead_stream() -> AsyncIterator[AgentEvent]:
        raise KeyboardInterrupt
        yield  # pragma: no cover — makes this an async generator

    try:
        completed = cli_mod._run_streamed_turn(
            console, dead_stream, StreamRenderer(console=console)
        )
    finally:
        loop.close()
    assert completed is False  # no KeyboardInterrupt escaped
    assert hammered["n"] == 2  # the notice was retried to completion
    assert "ctrl-c again exits" in buffer.getvalue()
    # The window is armed AND flagged mid-turn, so the prompt absorbs the hammer tail.
    assert cli_mod._LAST_CTRL_C > 0.0
    assert cli_mod._LAST_CTRL_C_MID_TURN is True


class _CrashingAgent(FakeAgent):
    """astream_turn blows up synchronously — the shape that used to unwind the REPL."""

    def astream_turn(self, text: str):  # type: ignore[override]
        raise RuntimeError("kaboom mid-turn")


def test_repl_survives_turn_crash(monkeypatch) -> None:
    # A turn-level crash is reported and the prompt survives: in the cockpit, REPL death
    # tears down every tmux pane, so "turn failed" must never become "cockpit gone".
    monkeypatch.setattr(zakcode, "Agent", _CrashingAgent)
    _scripted_prompt(monkeypatch, ["do the thing", "/exit"])
    result = runner.invoke(app, ["cli"])
    assert result.exit_code == 0
    assert "turn failed" in result.stdout
    assert "kaboom mid-turn" in result.stdout
    assert "goodbye" in result.stdout  # reached /exit — the REPL outlived the crash


class _InterruptLeakingAgent(FakeAgent):
    """astream_turn raises KeyboardInterrupt before the turn machinery even starts —
    the escape shape a hammered press can produce outside the teardown's coverage."""

    def astream_turn(self, text: str):  # type: ignore[override]
        raise KeyboardInterrupt


def test_repl_absorbs_interrupt_that_escapes_the_turn(monkeypatch) -> None:
    monkeypatch.setattr(zakcode, "Agent", _InterruptLeakingAgent)
    _scripted_prompt(monkeypatch, ["hello", "/exit"])
    result = runner.invoke(app, ["cli"])
    assert result.exit_code == 0
    assert "goodbye" in result.stdout  # absorbed, prompt survived to /exit
    assert "Traceback" not in result.stdout


# ── /resume: in-REPL twin of the -s/--session flag ──


def _seed_session_store(tmp_path, monkeypatch, *ids):
    """Point the default SessionStore at tmp and persist one session per id."""
    from zakcode.session.store import SessionStore

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    store = SessionStore()
    for sid in ids:
        store.save(Session(id=sid, cwd=".", model="test-model"))
    return store


def test_chat_resume_lists_saved_sessions(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(zakcode, "Agent", FakeAgent)
    _seed_session_store(tmp_path, monkeypatch, "aaaa1111", "bbbb2222")
    result = runner.invoke(app, ["cli"], input="/resume\n/exit\n")
    assert result.exit_code == 0
    assert "aaaa1111" in result.stdout
    assert "bbbb2222" in result.stdout
    assert "a unique prefix works" in result.stdout


def test_chat_resume_empty_store_is_friendly(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(zakcode, "Agent", FakeAgent)
    _seed_session_store(tmp_path, monkeypatch)
    result = runner.invoke(app, ["cli"], input="/resume\n/exit\n")
    assert result.exit_code == 0
    assert "no saved sessions yet" in result.stdout


def test_chat_resume_by_unique_prefix(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(zakcode, "Agent", FakeAgent)
    _seed_session_store(tmp_path, monkeypatch, "aaaa1111", "bbbb2222")
    result = runner.invoke(app, ["cli"], input="/resume aaaa\n/exit\n")
    assert result.exit_code == 0
    assert "resumed session aaaa1111" in result.stdout


def test_chat_resume_unknown_id_keeps_session(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(zakcode, "Agent", FakeAgent)
    _seed_session_store(tmp_path, monkeypatch, "aaaa1111")
    result = runner.invoke(app, ["cli"], input="/resume zzzz\nhello\n/exit\n")
    assert result.exit_code == 0
    assert "no saved session matches" in result.stdout
    # The live session survived the failed resume and still ran a turn.
    assert CANNED_TEXT in result.stdout


def test_chat_resume_ambiguous_prefix_is_refused(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(zakcode, "Agent", FakeAgent)
    _seed_session_store(tmp_path, monkeypatch, "aaaa1111", "aaaa2222")
    result = runner.invoke(app, ["cli"], input="/resume aaaa\n/exit\n")
    assert result.exit_code == 0
    assert "ambiguous session prefix" in result.stdout


# ── /resume replays the transcript (2026-08-24: "resumed" over a blank screen) ──


def test_chat_resume_replays_transcript(tmp_path, monkeypatch) -> None:
    from zakcode.messages import Message

    monkeypatch.setattr(zakcode, "Agent", FakeAgent)
    store = _seed_session_store(tmp_path, monkeypatch)
    sess = Session(id="cafe0001", cwd=".", model="test-model")
    sess.add_message(Message.user("what is our draft strategy?"))
    sess.add_message(Message.assistant_text("Zero-RB: load WR early, backs in rounds 5+."))
    store.save(sess)

    # FakeAgent ignores the loaded session, so patch the builder's product post-hoc:
    # the REPL reads agent.session for the replay — hand it the stored one.
    import zakcode.cli as cli_mod

    real_builder = cli_mod._build_chat_agent

    def builder(*args, **kwargs):
        agent = real_builder(*args, **kwargs)
        if kwargs.get("session_id"):
            agent.session = store.load(kwargs["session_id"])
        return agent

    monkeypatch.setattr(cli_mod, "_build_chat_agent", builder)
    result = runner.invoke(app, ["cli"], input="/resume cafe\n/exit\n")
    assert result.exit_code == 0
    assert "resumed session cafe0001" in result.stdout
    # The old conversation is VISIBLE, both sides, plus the replay bookends.
    assert "what is our draft strategy?" in result.stdout
    assert "Zero-RB: load WR early" in result.stdout
    assert "replaying" in result.stdout
    assert "end of replay" in result.stdout


def test_render_transcript_elides_and_counts_tools() -> None:
    from zakcode.cli import _render_transcript
    from zakcode.messages import Message, TextBlock, ToolUseBlock

    def _rec_console() -> Console:
        return Console(record=True, width=100, force_terminal=True, legacy_windows=False)

    console = _rec_console()
    sess = Session(id="feed0002", cwd=".", model="test-model")
    for i in range(15):
        sess.add_message(Message.user(f"question {i}"))
        sess.add_message(Message.assistant_text(f"answer {i}"))
    tool_msg = Message(
        role="assistant",
        blocks=[
            TextBlock(text="running a check"),
            ToolUseBlock(id="t1", name="run", input={}),
            ToolUseBlock(id="t2", name="read", input={}),
        ],
    )
    sess.add_message(tool_msg)
    _render_transcript(console, sess, limit=5)
    out = console.export_text()
    assert "replaying the last 5 of 31 messages" in out
    assert "question 0" not in out  # elided
    assert "ran 2 tool calls" in out
    long = Session(id="feed0003", cwd=".", model="test-model")
    long.add_message(Message.assistant_text("x" * 3000))
    console2 = _rec_console()
    _render_transcript(console2, long)
    assert "(+1000 chars)" in console2.export_text()


# ── zakcode update: self-update from the recorded install source ──


def test_update_refuses_non_vcs_install(monkeypatch) -> None:
    import zakcode.cli as cli_mod

    monkeypatch.setattr(cli_mod, "build_url", lambda: None)
    monkeypatch.setattr(cli_mod, "build_dir", lambda: None)
    result = runner.invoke(app, ["update"])
    assert result.exit_code == 1
    assert "cannot self-update" in result.stdout


def test_update_runs_pip_with_force_reinstall(monkeypatch) -> None:
    import zakcode.cli as cli_mod

    monkeypatch.setattr(cli_mod, "build_url", lambda: "https://example.com/repo.git")
    calls = []

    class FakeProc:
        returncode = 0
        stdout = "0.0.1 (git abcdef123456)\n"
        stderr = ""

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        return FakeProc()

    monkeypatch.setattr(cli_mod.subprocess, "run", fake_run)
    result = runner.invoke(app, ["update"])
    assert result.exit_code == 0
    pip_cmd = calls[0]
    assert "--force-reinstall" in pip_cmd and "--no-deps" in pip_cmd
    assert "zakcode @ git+https://example.com/repo.git@main" in pip_cmd
    # Second pass: a PLAIN pip install of the same requirement, so a build that
    # ADDS a dependency actually gets it (--no-deps alone skipped new deps —
    # measured live 2026-08-25, prompt_toolkit missing after an update).
    dep_cmd = calls[1]
    assert "pip" in dep_cmd and "zakcode @ git+https://example.com/repo.git@main" in dep_cmd
    assert "--no-deps" not in dep_cmd and "--force-reinstall" not in dep_cmd
    assert "updated" in result.stdout
    assert "abcdef123456" in result.stdout


def test_update_accepts_a_ref(monkeypatch) -> None:
    import zakcode.cli as cli_mod

    monkeypatch.setattr(cli_mod, "build_url", lambda: "https://example.com/repo.git")
    calls = []

    class FakeProc:
        returncode = 0
        stdout = "ok\n"
        stderr = ""

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        return FakeProc()

    monkeypatch.setattr(cli_mod.subprocess, "run", fake_run)
    result = runner.invoke(app, ["update", "v2"])
    assert result.exit_code == 0
    assert any("@v2" in str(part) for part in calls[0])


# ── zakcode update v2: local-checkout installs (uv tool install from a clone) ──


def _make_clone_pair(tmp_path):
    """A bare 'origin' plus a working clone one commit behind it."""
    import subprocess as sp

    def run(cwd, *args):
        sp.run(args, cwd=cwd, check=True, capture_output=True, text=True)

    origin = tmp_path / "origin.git"
    seed = tmp_path / "seed"
    clone = tmp_path / "clone"
    origin.mkdir()
    run(origin, "git", "init", "--bare", "-q", "-b", "main")
    seed.mkdir()
    run(seed, "git", "init", "-q", "-b", "main")
    run(seed, "git", "config", "user.email", "t@t")
    run(seed, "git", "config", "user.name", "t")
    (seed / "pyproject.toml").write_text("v1\n")
    (seed / "uv.lock").write_text("lock1\n")
    run(seed, "git", "add", "-A")
    run(seed, "git", "commit", "-qm", "one")
    run(seed, "git", "remote", "add", "origin", str(origin))
    run(seed, "git", "push", "-q", "origin", "HEAD:main")
    sp.run(["git", "clone", "-q", str(origin), str(clone)], check=True, capture_output=True)
    run(clone, "git", "config", "user.email", "t@t")
    run(clone, "git", "config", "user.name", "t")
    # Advance origin past the clone.
    (seed / "pyproject.toml").write_text("v2\n")
    run(seed, "git", "add", "-A")
    run(seed, "git", "commit", "-qm", "two")
    run(seed, "git", "push", "-q", "origin", "HEAD:main")
    return clone


def test_refresh_checkout_reverts_churn_and_pulls(tmp_path) -> None:
    from zakcode.cli import _refresh_checkout

    clone = _make_clone_pair(tmp_path)
    # The exact operator situation: only the install-churn files are dirty.
    (clone / "pyproject.toml").write_text("churned by uv tool install\n")
    (clone / "uv.lock").write_text("churned\n")
    _refresh_checkout(str(clone))
    assert (clone / "pyproject.toml").read_text() == "v2\n"  # reverted, then pulled


def test_refresh_checkout_refuses_real_local_changes(tmp_path) -> None:
    import typer as typer_mod

    from zakcode.cli import _refresh_checkout

    clone = _make_clone_pair(tmp_path)
    (clone / "my-experiment.py").write_text("precious\n")
    import subprocess as sp

    sp.run(["git", "add", "my-experiment.py"], cwd=clone, check=True, capture_output=True)
    with pytest.raises(typer_mod.Exit):
        _refresh_checkout(str(clone))
    assert (clone / "my-experiment.py").read_text() == "precious\n"  # untouched


def test_update_local_checkout_uses_uv_with_receipt_extras(tmp_path, monkeypatch) -> None:
    import zakcode.cli as cli_mod

    clone = _make_clone_pair(tmp_path)
    monkeypatch.setattr(cli_mod, "build_url", lambda: None)
    monkeypatch.setattr(cli_mod, "build_dir", lambda: str(clone))
    # Simulate a uv tool env: receipt with the [google] extra at the env root.
    fake_prefix = tmp_path / "toolenv"
    fake_prefix.mkdir()
    (fake_prefix / "uv-receipt.toml").write_text(
        '[tool]\nrequirements = [{ name = "zakcode", extras = ["google"] }]\n'
    )
    monkeypatch.setattr(cli_mod.sys, "prefix", str(fake_prefix))
    monkeypatch.setattr(cli_mod.shutil, "which", lambda name: "/usr/bin/uv")
    calls = []
    real_run = cli_mod.subprocess.run

    def fake_run(cmd, **kwargs):
        if cmd and cmd[0] == "git":
            return real_run(cmd, **kwargs)  # the checkout refresh is real
        calls.append(cmd)

        class P:
            returncode = 0
            stdout = "0.0.1 (git feedbead1234)\n"
            stderr = ""

        return P()

    monkeypatch.setattr(cli_mod.subprocess, "run", fake_run)
    result = runner.invoke(app, ["update"])
    assert result.exit_code == 0
    reinstall = calls[0]
    assert reinstall[:5] == ["/usr/bin/uv", "tool", "install", "--force", "--reinstall"]
    assert reinstall[5].startswith("zakcode[google] @ file://")
    assert "updated" in result.stdout


def test_update_local_checkout_without_uv_receipt_uses_pip(tmp_path, monkeypatch) -> None:
    import zakcode.cli as cli_mod

    clone = _make_clone_pair(tmp_path)
    monkeypatch.setattr(cli_mod, "build_url", lambda: None)
    monkeypatch.setattr(cli_mod, "build_dir", lambda: str(clone))
    monkeypatch.setattr(cli_mod.sys, "prefix", str(tmp_path / "no-receipt-here"))
    calls = []
    real_run = cli_mod.subprocess.run

    def fake_run(cmd, **kwargs):
        if cmd and cmd[0] == "git":
            return real_run(cmd, **kwargs)
        calls.append(cmd)

        class P:
            returncode = 0
            stdout = "ok\n"
            stderr = ""

        return P()

    monkeypatch.setattr(cli_mod.subprocess, "run", fake_run)
    result = runner.invoke(app, ["update"])
    assert result.exit_code == 0
    assert "--force-reinstall" in calls[0] and "--no-deps" in calls[0]
    assert any(str(part).startswith("zakcode @ file://") for part in calls[0])
    assert "--no-deps" not in calls[1]  # dependency-resolution pass


# ── terminal hygiene: stale COLUMNS/LINES + bracketed paste (2026-08-25 reports) ──


def test_drop_stale_size_env_removes_mismatched_vars(monkeypatch) -> None:
    import zakcode.cli as cli_mod

    monkeypatch.setattr(cli_mod.os, "get_terminal_size", lambda fd: os.terminal_size((240, 60)))
    monkeypatch.setenv("COLUMNS", "80")  # stale narrow export — the bug
    monkeypatch.setenv("LINES", "60")  # happens to match — deliberate/harmless
    cli_mod._drop_stale_size_env()
    assert "COLUMNS" not in os.environ
    assert os.environ["LINES"] == "60"


def test_drop_stale_size_env_keeps_vars_without_a_tty(monkeypatch) -> None:
    import zakcode.cli as cli_mod

    def boom(fd):
        raise OSError("not a tty")

    monkeypatch.setattr(cli_mod.os, "get_terminal_size", boom)
    monkeypatch.setenv("COLUMNS", "80")
    cli_mod._drop_stale_size_env()
    assert os.environ["COLUMNS"] == "80"  # only width signal there is — untouched


def test_enable_multiline_paste_is_safe_everywhere(monkeypatch) -> None:
    import zakcode.cli as cli_mod

    # Non-tty: early return, no readline import required.
    monkeypatch.setattr(cli_mod.sys.stdin, "isatty", lambda: False)
    cli_mod._enable_multiline_paste()
    # Tty: must not raise whether or not this platform has readline/libedit.
    monkeypatch.setattr(cli_mod.sys.stdin, "isatty", lambda: True)
    cli_mod._enable_multiline_paste()


def test_chat_say_inbox_consumes_messages_as_input(monkeypatch, tmp_path) -> None:
    """The converged input path: a message written to <workspace>/.say (the same
    single-slot file POST /say writes) is consumed exactly like a typed line —
    it runs a turn, is echoed with provenance, and /exit via the inbox ends the
    session. Always on — no flag, no env var. Stdin is blocked the whole time,
    so delivery is proven to be the inbox, not the keyboard."""
    import threading
    import time as _time

    from zakcode.session import say_inbox as si

    monkeypatch.setattr(zakcode, "Agent", FakeAgent)
    monkeypatch.setenv("ZAKCODE_WORKSPACE_ROOT", str(tmp_path))

    block = threading.Event()
    listening = threading.Event()

    def _blocked_read():  # noqa: ANN202
        # The stdin pump parks here; the 30s ceiling bounds the test if the inbox
        # path is broken (EOF then ends the REPL instead of hanging CI). Setting
        # `listening` first tells the writer the session is PAST the pre-session
        # stale-discard — a message written any earlier is (correctly) discarded,
        # which is exactly the race that flaked this test on slower interpreters.
        listening.set()
        block.wait(timeout=30)
        raise EOFError

    monkeypatch.setattr("zakcode.cli._read_stdin_line", _blocked_read)
    inbox = si.say_path(tmp_path)

    def _writer() -> None:
        assert listening.wait(timeout=15), "chat never started listening"
        assert si.write_say(inbox, "hello from the inbox")
        for _ in range(200):  # wait for exactly-once consumption before the next say
            if not si.say_pending(inbox):
                break
            _time.sleep(0.05)
        si.write_say(inbox, "/exit")

    writer = threading.Thread(target=_writer, daemon=True)
    writer.start()
    result = runner.invoke(app, ["cli"], input="")
    block.set()
    assert result.exit_code == 0
    assert "(say) hello from the inbox" in result.output
    assert CANNED_TEXT in result.output
    assert "goodbye" in result.output


def test_chat_delivers_pre_session_say_as_first_input(monkeypatch, tmp_path) -> None:
    """ONE staleness rule, owned by the cockpit: launch_cockpit clears anything
    genuinely stale at session creation, and chat itself never second-guesses the
    inbox — a say already queued when chat starts (typed while the agent booted,
    or queued across a relaunch) is delivered as the first input."""
    from zakcode.session import say_inbox as si

    monkeypatch.setattr(zakcode, "Agent", FakeAgent)
    monkeypatch.setenv("ZAKCODE_WORKSPACE_ROOT", str(tmp_path))
    assert si.write_say(si.say_path(tmp_path), "typed while the agent was booting")
    result = runner.invoke(app, ["cli"], input="/exit\n")
    assert result.exit_code == 0
    assert "discarded" not in result.output
    assert "(say) typed while the agent was booting" in result.output
    assert CANNED_TEXT in result.output  # it ran a real turn
    assert "(say) malicious" not in result.output
    assert not si.say_pending(si.say_path(tmp_path))


def test_interrupt_file_stops_a_running_turn(monkeypatch, tmp_path) -> None:
    """The file-based twin of Ctrl-C: while a turn is streaming, writing
    <workspace>/.interrupt stops it through the exact same path — the
    "interrupted" notice prints and the session survives to take /exit."""
    import threading
    import time as _time

    from zakcode.session import say_inbox as si

    class SlowFakeAgent(FakeAgent):
        async def _gen(self):  # noqa: ANN202
            yield AgentTextDelta(text="starting…\n")
            await asyncio.sleep(30)  # cancelled by the interrupt long before this ends
            yield AgentDone(stop_reason="completed", iterations=1, usage=Usage())

    monkeypatch.setattr(zakcode, "Agent", SlowFakeAgent)
    monkeypatch.setenv("ZAKCODE_WORKSPACE_ROOT", str(tmp_path))

    block = threading.Event()
    listening = threading.Event()

    def _blocked_read():  # noqa: ANN202
        listening.set()
        block.wait(timeout=30)
        raise EOFError

    monkeypatch.setattr("zakcode.cli._read_stdin_line", _blocked_read)
    inbox = si.say_path(tmp_path)
    sig = si.interrupt_path(tmp_path)

    def _driver() -> None:
        assert listening.wait(timeout=15)
        assert si.write_say(inbox, "do something slow")
        for _ in range(200):  # wait until the turn has consumed the say
            if not si.say_pending(inbox):
                break
            _time.sleep(0.05)
        _time.sleep(0.5)  # turn is now streaming (first delta out, then the long sleep)
        si.request_interrupt(sig)
        _time.sleep(1.5)  # give the watcher a poll cycle to stop the turn
        si.write_say(inbox, "/exit")

    threading.Thread(target=_driver, daemon=True).start()
    result = runner.invoke(app, ["cli"], input="")
    block.set()
    assert result.exit_code == 0
    assert "interrupted" in result.output
    assert "goodbye" in result.output
    assert not sig.exists()


# ── the input mux: ONE stdin owner; permission answers ride it ─────────────────────
# 2026-08-25 field incident: the prompter's own console.input raced the stdin pump
# (two blocking readers on one fd) — y/a/n answers were swallowed into the message
# frame and "aa" reached the model as a prompt. These tests pin the fix: the mux is
# the only reader, and a permission prompt can be answered from the keyboard, from
# the say inbox, or denied by the interrupt file.


def _quiet_pump(monkeypatch) -> None:
    """Park the mux's pump thread forever — tests feed the mux by other doors."""
    import threading as _threading

    def _blocked() -> str:
        _threading.Event().wait(timeout=60)
        raise EOFError

    monkeypatch.setattr("zakcode.cli._read_stdin_line", _blocked)


def _make_mux(tmp_path: Path):
    from zakcode.cli import _InputMux
    from zakcode.session import say_inbox as si

    return _InputMux(si.say_path(tmp_path), si.interrupt_path(tmp_path))


def _perm_request() -> PermissionRequest:
    from zakcode.permissions import PermissionTier

    return PermissionRequest(
        tool_name="bash",
        tier=PermissionTier.WORKSPACE_WRITE,
        arguments={"command": "ls"},
        reason="test escalation",
    )


def test_mux_idle_clears_stale_stop_signal_and_delivers_say(monkeypatch, tmp_path) -> None:
    from zakcode.session import say_inbox as si

    _quiet_pump(monkeypatch)
    mux = _make_mux(tmp_path)
    si.request_interrupt(si.interrupt_path(tmp_path))
    assert si.write_say(si.say_path(tmp_path), "hello")
    assert mux.next_input(idle=True) == ("say", "hello")
    # Idle consumers CLEAR a stale stop signal — there is no turn to stop.
    assert not si.interrupt_path(tmp_path).exists()


def test_mux_midturn_reports_stop_without_consuming_the_signal(monkeypatch, tmp_path) -> None:
    from zakcode.session import say_inbox as si

    _quiet_pump(monkeypatch)
    mux = _make_mux(tmp_path)
    si.request_interrupt(si.interrupt_path(tmp_path))
    assert mux.next_input(idle=False) == ("stop", None)
    # Mid-turn the signal is LEFT IN PLACE for the turn's interrupt watcher.
    assert si.interrupt_path(tmp_path).exists()


def test_mux_try_input_takes_waiting_input_without_blocking(monkeypatch, tmp_path) -> None:
    from zakcode.session import say_inbox as si

    _quiet_pump(monkeypatch)
    mux = _make_mux(tmp_path)
    assert mux.try_input() is None
    assert si.write_say(si.say_path(tmp_path), "queued while busy")
    assert mux.try_input() == ("say", "queued while busy")
    assert mux.try_input() is None


def test_mux_answer_line_from_say_echoes_provenance(monkeypatch, tmp_path) -> None:
    from zakcode.session import say_inbox as si

    _quiet_pump(monkeypatch)
    mux = _make_mux(tmp_path)
    console = Console(record=True, width=100, theme=ZAK_THEME)
    assert si.write_say(si.say_path(tmp_path), "y")
    assert mux.answer_line(console) == "y"
    assert "(say) y" in console.export_text()


def test_prompter_denied_by_interrupt_file_leaves_signal_for_turn_watcher(
    monkeypatch, tmp_path
) -> None:
    from zakcode.session import say_inbox as si

    _quiet_pump(monkeypatch)
    mux = _make_mux(tmp_path)
    console = Console(record=True, width=100, theme=ZAK_THEME)
    si.request_interrupt(si.interrupt_path(tmp_path))
    prompter = ConsolePermissionPrompter(
        console, line_source=lambda stop: mux.answer_line(console, stop=stop)
    )
    outcome = asyncio.run(prompter.confirm(_perm_request()))
    assert outcome is PermissionOutcome.DENY_ONCE
    assert "denied" in console.export_text()
    # The signal survives — the turn's interrupt watcher consumes it and cancels.
    assert si.interrupt_path(tmp_path).exists()


class _PermissionAskingAgent(FakeAgent):
    """A fake agent whose turn escalates one tool call through the CLI's prompter."""

    outcomes: list[PermissionOutcome] = []
    confirm_started: object = None  # threading.Event set just before the prompt blocks

    async def _gen(self) -> AsyncIterator[AgentEvent]:
        prompter = self.overrides["prompter"]
        if _PermissionAskingAgent.confirm_started is not None:
            _PermissionAskingAgent.confirm_started.set()
        outcome = await prompter.confirm(_perm_request())
        _PermissionAskingAgent.outcomes.append(outcome)
        yield AgentTextDelta(text=f"outcome={outcome.value}\n")
        yield AgentDone(
            stop_reason="completed",
            iterations=1,
            usage=Usage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
        )


def test_chat_permission_prompt_answered_from_keyboard_through_mux(monkeypatch, tmp_path) -> None:
    """A typed y/a/n reaches the prompter through the mux — the exact path that the
    two-reader race used to swallow. The pump reads ahead ("a" is queued before the
    prompt even opens); the mux still routes it to the prompter, not the model."""
    monkeypatch.setattr(zakcode, "Agent", _PermissionAskingAgent)
    monkeypatch.setenv("ZAKCODE_WORKSPACE_ROOT", str(tmp_path))
    _PermissionAskingAgent.outcomes = []
    _PermissionAskingAgent.confirm_started = None
    _scripted_prompt(monkeypatch, ["please do the thing", "a", EOFError()])
    result = runner.invoke(app, ["cli"], input="")
    assert result.exit_code == 0
    assert _PermissionAskingAgent.outcomes == [PermissionOutcome.ALLOW_SESSION]
    assert "outcome=allow_session" in result.output
    # The answer must NOT have become a model prompt.
    assert "outcome=" in result.output and "\na\n" not in result.output


def test_chat_permission_prompt_answered_from_say_inbox(monkeypatch, tmp_path) -> None:
    """The cockpit say box can approve a tool call: a 'y' written to <workspace>/.say
    while the prompter waits is consumed as the answer, echoed with provenance."""
    import threading
    import time as _time

    from zakcode.session import say_inbox as si

    monkeypatch.setattr(zakcode, "Agent", _PermissionAskingAgent)
    monkeypatch.setenv("ZAKCODE_WORKSPACE_ROOT", str(tmp_path))
    _PermissionAskingAgent.outcomes = []
    confirm_started = threading.Event()
    _PermissionAskingAgent.confirm_started = confirm_started

    block = threading.Event()
    listening = threading.Event()

    def _blocked_read() -> str:
        listening.set()
        block.wait(timeout=30)
        raise EOFError

    monkeypatch.setattr("zakcode.cli._read_stdin_line", _blocked_read)
    inbox = si.say_path(tmp_path)

    def _writer() -> None:
        assert listening.wait(timeout=15), "chat never started listening"
        assert si.write_say(inbox, "run the turn")
        assert confirm_started.wait(timeout=15), "prompter never asked"
        si.write_say(inbox, "y")
        for _ in range(200):  # exactly-once: wait for the answer's consumption
            if not si.say_pending(inbox):
                break
            _time.sleep(0.05)
        si.write_say(inbox, "/exit")

    writer = threading.Thread(target=_writer, daemon=True)
    writer.start()
    result = runner.invoke(app, ["cli"], input="")
    block.set()
    assert result.exit_code == 0
    assert _PermissionAskingAgent.outcomes == [PermissionOutcome.ALLOW_ONCE]
    assert "(say) y" in result.output
    assert "outcome=allow_once" in result.output
    assert "goodbye" in result.output


def test_prompter_holds_non_answer_say_and_delivers_it_after_the_turn(
    monkeypatch, tmp_path
) -> None:
    """A real message sent while a permission prompt is open (2026-08-25 field
    report: 'continue, why did you stop?') must NOT be burned on the prompt's
    re-ask loop — it is held and delivered as ordinary input once idle."""
    import threading
    import time as _time

    from zakcode.cli import _parse_permission_answer
    from zakcode.session import say_inbox as si

    _quiet_pump(monkeypatch)
    mux = _make_mux(tmp_path)
    console = Console(record=True, width=100, theme=ZAK_THEME)
    prompter = ConsolePermissionPrompter(
        console,
        line_source=lambda stop: mux.answer_line(
            console, stop=stop, accept=_parse_permission_answer
        ),
    )
    inbox = si.say_path(tmp_path)
    assert si.write_say(inbox, "also, please add tests for this")

    def _writer() -> None:
        for _ in range(200):  # the non-answer say is consumed (held) first
            if not si.say_pending(inbox):
                break
            _time.sleep(0.05)
        si.write_say(inbox, "y")

    threading.Thread(target=_writer, daemon=True).start()
    outcome = asyncio.run(prompter.confirm(_perm_request()))
    assert outcome is PermissionOutcome.ALLOW_ONCE
    out = console.export_text()
    assert "held" in out and "also, please add tests" in out
    # Once idle, the held message is the next input — with say provenance.
    assert mux.try_input() == ("say", "also, please add tests for this")


def test_answer_line_worker_ends_when_wait_is_abandoned(monkeypatch, tmp_path) -> None:
    """A cancelled permission prompt must END its waiting worker thread — an
    orphaned poller steals the next input and blocks interpreter exit."""
    import threading

    _quiet_pump(monkeypatch)
    mux = _make_mux(tmp_path)
    console = Console(record=True, width=100, theme=ZAK_THEME)
    stop = threading.Event()
    worker = threading.Thread(target=lambda: mux.answer_line(console, stop=stop), daemon=True)
    worker.start()
    worker.join(timeout=0.2)
    assert worker.is_alive()  # waiting, as a live prompt would be
    stop.set()
    worker.join(timeout=5)
    assert not worker.is_alive()


def test_update_local_checkout_reports_real_shas_not_same_commit(tmp_path, monkeypatch) -> None:
    """2026-08-25 field report (serene): a local-path install pulled 2d3134d→c5a1725
    yet printed "same commit — you were already current", because PEP 610 metadata
    for a dir install records no commit and both sides read "0.0.1 (local path)".
    The verdict must come from the checkout's own before/after shas."""
    import subprocess as sp

    import zakcode.cli as cli_mod

    clone = _make_clone_pair(tmp_path)
    old_sha = sp.run(
        ["git", "-C", str(clone), "rev-parse", "--short", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    monkeypatch.setattr(cli_mod, "build_url", lambda: None)
    monkeypatch.setattr(cli_mod, "build_dir", lambda: str(clone))
    monkeypatch.setattr(cli_mod.sys, "prefix", str(tmp_path / "no-receipt-here"))
    real_run = cli_mod.subprocess.run

    def fake_run(cmd, **kwargs):
        if cmd and cmd[0] == "git":
            return real_run(cmd, **kwargs)

        class P:
            returncode = 0
            stdout = "0.0.1 (local path)\n"  # what the post-install probe really prints
            stderr = ""

        return P()

    monkeypatch.setattr(cli_mod.subprocess, "run", fake_run)
    result = runner.invoke(app, ["update"])
    assert result.exit_code == 0
    new_sha = sp.run(
        ["git", "-C", str(clone), "rev-parse", "--short", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert old_sha != new_sha  # the pair really was one commit behind
    assert f"(git {old_sha})" in result.stdout
    assert f"(git {new_sha})" in result.stdout
    assert "already current" not in result.stdout


def test_update_local_checkout_already_current_says_so_with_sha(tmp_path, monkeypatch) -> None:
    import subprocess as sp

    import zakcode.cli as cli_mod

    clone = _make_clone_pair(tmp_path)
    sp.run(["git", "-C", str(clone), "pull", "-q", "--ff-only"], check=True, capture_output=True)
    monkeypatch.setattr(cli_mod, "build_url", lambda: None)
    monkeypatch.setattr(cli_mod, "build_dir", lambda: str(clone))
    monkeypatch.setattr(cli_mod.sys, "prefix", str(tmp_path / "no-receipt-here"))
    real_run = cli_mod.subprocess.run

    def fake_run(cmd, **kwargs):
        if cmd and cmd[0] == "git":
            return real_run(cmd, **kwargs)

        class P:
            returncode = 0
            stdout = "0.0.1 (local path)\n"
            stderr = ""

        return P()

    monkeypatch.setattr(cli_mod.subprocess, "run", fake_run)
    result = runner.invoke(app, ["update"])
    assert result.exit_code == 0
    assert "already current" in result.stdout
