"""Tests for the interactive ``zakcode chat`` REPL.

These are fully hermetic: the real :class:`~zakcode.Agent` is replaced with a
fake whose ``run_turn`` returns a canned :class:`TurnResult`, so no provider,
network, or model is ever touched. The CLI is exercised through Typer's
``CliRunner`` with scripted stdin.
"""

from typer.testing import CliRunner

import zakcode
from zakcode.agent.loop import TurnResult
from zakcode.cli import app
from zakcode.config import load_settings
from zakcode.messages import Message, ToolResultBlock, ToolUseBlock
from zakcode.providers.base import ProviderError
from zakcode.session.store import Session
from zakcode.usage import Usage

runner = CliRunner()

CANNED_TEXT = "Hello from the fake agent."


class FakeAgent:
    """Drop-in stand-in for :class:`zakcode.Agent` used by the CLI tests."""

    def __init__(self, **overrides: object) -> None:
        self.overrides = overrides
        self.settings = load_settings()
        self.session = Session(cwd=".", model=self.settings.default_model)
        self.turns: list[str] = []
        self.canned: TurnResult = TurnResult(
            assistant_messages=[Message.assistant_text(CANNED_TEXT)],
            iterations=1,
            usage=Usage(prompt_tokens=3, completion_tokens=5, total_tokens=8),
            stop_reason="completed",
        )

    def run_turn(self, text: str) -> TurnResult:
        self.turns.append(text)
        self.session.add_usage(self.canned.usage)
        return self.canned


def test_chat_renders_assistant_text_and_exits(monkeypatch) -> None:
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
        def __init__(self, **overrides: object) -> None:
            super().__init__(**overrides)
            use = ToolUseBlock(id="t1", name="bash", input={"command": "ls -la"})
            assistant = Message(role="assistant", blocks=[use])
            self.canned = TurnResult(
                assistant_messages=[assistant, Message.assistant_text(CANNED_TEXT)],
                tool_results=[ToolResultBlock(tool_use_id="t1", output="files")],
                iterations=2,
                usage=Usage(total_tokens=12),
                stop_reason="completed",
            )

    monkeypatch.setattr(zakcode, "Agent", ToolAgent)
    result = runner.invoke(app, ["chat"], input="run ls\n/exit\n")
    assert result.exit_code == 0
    # Exactly one tool-call summary line is rendered for the single tool use.
    assert result.stdout.count("tool bash(") == 1
    assert "-> ok" in result.stdout
    assert CANNED_TEXT in result.stdout


def test_chat_provider_error_stays_in_repl(monkeypatch) -> None:
    class BoomAgent(FakeAgent):
        def run_turn(self, text: str) -> TurnResult:
            raise ProviderError("model unreachable")

    monkeypatch.setattr(zakcode, "Agent", BoomAgent)
    result = runner.invoke(app, ["chat"], input="hello\n/exit\n")
    assert result.exit_code == 0
    assert "Provider error" in result.stdout
