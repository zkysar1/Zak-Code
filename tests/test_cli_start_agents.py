"""The terminal offers a workspace's framework agents on a chat with nothing in it (ADR-0351).

The web page's offer, in the terminal: under the welcome box and after ``/clear``, the two places a
chat starts empty, and never on a resumed chat. Elsewhere nothing new is printed.
"""

from __future__ import annotations

import io
from pathlib import Path

import pytest
from rich.console import Console

import zakcode
from tests.test_cli_chat import FakeAgent, runner
from tests.test_framework_agents import mind
from zakcode.cli import _print_start_agents, app
from zakcode.cli._theme import ZAK_THEME
from zakcode.session.store import Session, SessionStore

_OFFER = "Start an agent before you ask anything"


def _printed(workspace: Path) -> str:
    console = Console(file=io.StringIO(), width=120, theme=ZAK_THEME, color_system=None)
    _print_start_agents(console, workspace)
    out = console.file.getvalue()  # type: ignore[attr-defined]
    assert isinstance(out, str)
    return out


def test_the_offer_names_the_agents_and_the_line_that_starts_one(tmp_path: Path) -> None:
    out = _printed(mind(tmp_path, "scout", "keeper"))
    assert f"{_OFFER}. This workspace has: keeper, scout." in out
    assert "/start keeper --mode assistant   or another agent named above" in out
    assert "answered without the agent" in out


def test_one_agent_is_offered_without_naming_others(tmp_path: Path) -> None:
    out = _printed(mind(tmp_path, "scout"))
    assert "/start scout --mode assistant" in out and "another agent" not in out


def test_a_workspace_without_framework_agents_prints_nothing_new(tmp_path: Path) -> None:
    assert _printed(tmp_path) == ""
    assert _printed(mind(tmp_path)) == ""


@pytest.mark.parametrize(
    ("resume", "typed", "offers"),
    [(False, "/exit\n", 1), (False, "/clear\n/exit\n", 2), (True, "/exit\n", 0)],
    ids=["new-chat", "after-clear", "resumed"],
)
def test_the_chat_offers_them_where_a_chat_starts_empty(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, resume: bool, typed: str, offers: int
) -> None:
    monkeypatch.chdir(mind(tmp_path, "scout"))  # the workspace the chat runs in
    monkeypatch.setattr(zakcode, "Agent", FakeAgent)
    args = []
    if resume:  # a chat saved in the store the conftest roots in tmp_path
        saved = Session(cwd=str(tmp_path), model="scripted/test")
        SessionStore().save(saved)
        args = ["--session", saved.id]
    result = runner.invoke(app, ["cli", *args], input=typed)
    assert result.exit_code == 0, result.output
    assert result.output.count(_OFFER) == offers
