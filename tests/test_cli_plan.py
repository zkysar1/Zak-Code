"""Tests for the /agents and /plan CLI commands (M4-5-2).

These exercise the command helpers directly with a captured console, using the
early-return guards (empty task / delegation disabled) so no real model is invoked.
"""

from __future__ import annotations

from io import StringIO
from pathlib import Path

from rich.console import Console

from zakcode import Agent
from zakcode.cli import _render_agents, _run_plan
from zakcode.config import Settings


def _console() -> tuple[Console, StringIO]:
    buf = StringIO()
    return Console(file=buf, width=100), buf


def _agent(tmp_path: Path, *, subagents: bool) -> Agent:
    return Agent(
        settings=Settings(default_model="scripted/test", workspace_root=tmp_path),
        enable_subagents=subagents,
    )


def test_render_agents_lists_types(tmp_path: Path) -> None:
    console, buf = _console()
    _render_agents(console, _agent(tmp_path, subagents=True))
    out = buf.getvalue()
    assert "general-purpose" in out
    assert "plan" in out
    assert "(default)" in out


def test_render_agents_without_delegation(tmp_path: Path) -> None:
    console, buf = _console()
    _render_agents(console, _agent(tmp_path, subagents=False))
    assert "not enabled" in buf.getvalue()


def test_run_plan_empty_task_prints_usage(tmp_path: Path) -> None:
    console, buf = _console()
    _run_plan(console, _agent(tmp_path, subagents=True), "")
    assert "usage: /plan" in buf.getvalue()


def test_run_plan_without_delegation(tmp_path: Path) -> None:
    console, buf = _console()
    _run_plan(console, _agent(tmp_path, subagents=False), "do a thing")
    assert "not enabled" in buf.getvalue()
