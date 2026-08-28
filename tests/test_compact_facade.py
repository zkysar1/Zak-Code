"""M8-2 wiring tests: the Agent facade builds a Compactor only when asked, and the
loop receives it. (Construction-only — no provider calls / network.)"""

from __future__ import annotations

from pathlib import Path

from zakcode import Agent
from zakcode.agent.compact import Compactor
from zakcode.config import Settings


def _agent(tmp_path: Path, **kw: object) -> Agent:
    return Agent(
        settings=Settings(
            default_model="scripted/test", context_window=8192, workspace_root=tmp_path
        ),
        **kw,
    )


def test_compaction_off_by_default(tmp_path: Path) -> None:
    agent = _agent(tmp_path)
    assert agent.compactor is None
    assert agent.loop.compactor is None


def test_enable_compaction_builds_and_wires_compactor(tmp_path: Path) -> None:
    agent = _agent(tmp_path, enable_compaction=True)
    assert isinstance(agent.compactor, Compactor)
    # the same compactor instance is handed to the loop
    assert agent.loop.compactor is agent.compactor


def test_loop_exposes_compact_now(tmp_path: Path) -> None:
    # compact_now is the /compact entry point; it must exist and be a no-op (False)
    # when there is nothing to compact / no compactor.
    agent = _agent(tmp_path)
    assert hasattr(agent.loop, "compact_now")
