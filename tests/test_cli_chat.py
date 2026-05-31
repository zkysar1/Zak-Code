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

from collections.abc import AsyncIterator

from typer.testing import CliRunner

import zakcode
from zakcode.cli import app
from zakcode.config import load_settings
from zakcode.events import (
    AgentDone,
    AgentEvent,
    AgentTextDelta,
    AgentToolCall,
    AgentToolResult,
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
        self.overrides = overrides
        self.settings = load_settings()
        self.session = Session(cwd=".", model=self.settings.default_model)
        self.turns: list[str] = []

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
