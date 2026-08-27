"""Typed-slash typo tolerance + did-you-mean (ADR-0040).

Field transcript 2026-08-27: ``/enocde-session`` answered "is not yet supported", while the
prose "encode the session" routed to /encode-session through the classifier — the strict
path was dumber than the fuzzy one. A UNIQUE near-identical skill name now runs, visibly
corrected; an ambiguous or weak match is offered back; the REPL's own commands are
suggestion-only (a typo must never run /clear).
"""

from __future__ import annotations

import asyncio
from io import StringIO
from pathlib import Path

from rich.console import Console

from zakcode import Agent
from zakcode.cli import _skill_command_turn, _unknown_command

_SKILL = "---\nname: {name}\ndescription: {name} does a thing\n---\nBody of {name}.\n"


def _write_skill(workspace: Path, name: str) -> None:
    d = workspace / ".zakcode" / "skills" / name
    d.mkdir(parents=True)
    (d / "SKILL.md").write_text(_SKILL.format(name=name), encoding="utf-8")


def _agent(tmp_path: Path) -> Agent:
    return Agent(default_model="zakpick", workspace_root=tmp_path, enable_skills=True)


def _console() -> tuple[Console, StringIO]:
    buf = StringIO()
    return Console(file=buf, force_terminal=False, width=120), buf


def test_unique_close_typo_runs_the_skill_and_names_the_correction(tmp_path: Path) -> None:
    _write_skill(tmp_path, "encode-session")
    result = asyncio.run(_agent(tmp_path).compose_skill_turn("enocde-session"))
    assert result.invoked and result.name == "encode-session"
    assert result.corrected_from == "enocde-session"
    assert result.turn_text is not None
    # The frame carries the skill the operator MEANT — the typo never reaches the model.
    assert result.turn_text.startswith("<command-message>encode-session is running")
    assert "<command-name>/encode-session</command-name>" in result.turn_text
    assert "enocde" not in result.turn_text


def test_exact_name_is_untouched(tmp_path: Path) -> None:
    _write_skill(tmp_path, "encode-session")
    result = asyncio.run(_agent(tmp_path).compose_skill_turn("encode-session"))
    assert result.invoked and result.corrected_from is None and result.suggestions == ()


def test_ambiguous_typo_is_a_did_you_mean_not_a_guess(tmp_path: Path) -> None:
    _write_skill(tmp_path, "encode-session")
    _write_skill(tmp_path, "encode-sessions")
    result = asyncio.run(_agent(tmp_path).compose_skill_turn("enocde-session"))
    assert not result.invoked
    assert set(result.suggestions) == {"encode-session", "encode-sessions"}


def test_weak_match_never_runs_and_is_not_suggested(tmp_path: Path) -> None:
    _write_skill(tmp_path, "encode-session")
    result = asyncio.run(_agent(tmp_path).compose_skill_turn("encore"))
    assert not result.invoked and result.suggestions == ()


def test_fuzzy_off_is_the_exact_only_probe(tmp_path: Path) -> None:
    _write_skill(tmp_path, "encode-session")
    result = asyncio.run(_agent(tmp_path).compose_skill_turn("enocde-session", fuzzy=False))
    assert not result.invoked and result.suggestions == ()


def test_repl_renders_the_correction(tmp_path: Path) -> None:
    _write_skill(tmp_path, "encode-session")
    console, buf = _console()
    outcome = _skill_command_turn(console, _agent(tmp_path), "enocde-session", fuzzy=True)
    assert outcome.handled and outcome.turn_text is not None
    assert "encode-session" in buf.getvalue() and "you typed /enocde-session" in buf.getvalue()


def test_repl_exact_probe_reports_unhandled_without_suggestions(tmp_path: Path) -> None:
    _write_skill(tmp_path, "encode-session")
    console, _ = _console()
    outcome = _skill_command_turn(console, _agent(tmp_path), "enocde-session", fuzzy=False)
    assert not outcome.handled and outcome.suggestions == ()


def test_unknown_command_names_builtins_and_skills_but_runs_nothing() -> None:
    console, buf = _console()
    _unknown_command(console, "/claer", ())
    assert "did you mean /clear" in buf.getvalue()
    console, buf = _console()
    _unknown_command(console, "/zzzz", ("encode-session",))
    assert "did you mean /encode-session" in buf.getvalue()
    console, buf = _console()
    _unknown_command(console, "/qqqqqq", ())
    assert "unknown command /qqqqqq" in buf.getvalue() and "/help" in buf.getvalue()


def test_unknown_command_in_a_skill_less_workspace_names_the_workspace(tmp_path: Path) -> None:
    """Empty catalog + no close builtin: the notice says NO skills are discovered and where
    the chat is rooted (measured 2026-08-27: `/start` in a chat launched outside the
    project read as "the update broke slash commands")."""
    console, buf = _console()
    agent = _agent(tmp_path)  # no project skills written (bundled tier still discovers)
    _unknown_command(console, "/start", (), agent=agent)
    # The console wraps the long notice; glue the wrap back before asserting.
    out = " ".join(buf.getvalue().split())
    assert "has no project skills" in out
    assert str(tmp_path) in "".join(out.split())
    assert "-w <project-root>" in out


def test_unknown_command_with_a_populated_catalog_keeps_the_short_notice(tmp_path: Path) -> None:
    """A discovered catalog means the workspace is fine — the hint would be noise."""
    _write_skill(tmp_path, "encode-session")
    console, buf = _console()
    _unknown_command(console, "/qqqqqq", (), agent=_agent(tmp_path))
    out = " ".join(buf.getvalue().split())
    assert "has no project skills" not in out
    assert "/help lists commands" in out
