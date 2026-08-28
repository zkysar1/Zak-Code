"""Tests for seam A — the quality gate (increment 6b).

The default-path safety (gate OFF → byte-identical) is proven by the rest of the suite staying green
unchanged. These exercise the gate's DECISION via ``AgentLoop._quality_gate`` directly (built with a
one-dimension rubric and a scripted scorer): ship on high, iterate with a brief on low,
and FAIL-OPEN (ship) when the scorer can't judge — it must never trap a finished turn.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from zakcode.agent.loop import AgentLoop, _gather_work
from zakcode.config import Settings
from zakcode.providers.base import Capabilities, LLMResult, Provider
from zakcode.session.store import Session
from zakcode.tools.base import ToolRegistry
from zakcode.usage import Usage


class _ScoreProvider(Provider):
    """Returns a fixed rubric-score JSON for every call (stands in for the gate's scorer)."""

    def __init__(self, text: str) -> None:
        self._text = text

    async def acomplete(self, messages, *, system=None, tools=None, **kwargs: Any) -> LLMResult:  # noqa: ANN001
        return LLMResult(text=self._text, usage=Usage(total_tokens=2, cost_usd=0.001))

    def count_tokens(self, messages, *, system=None) -> int:  # noqa: ANN001
        return 0

    def capabilities(self) -> Capabilities:
        return Capabilities(context_window=8192)

    def model_id(self) -> str:
        return "judge/test"


def _loop(tmp_path: Path, provider: Provider, **settings_kw: Any) -> AgentLoop:
    session = Session(cwd=str(tmp_path), model="scripted/test")
    settings = Settings(quality_gate_dimensions={"q": "is it good"}, **settings_kw)
    return AgentLoop(provider, ToolRegistry(), session, workspace_root=tmp_path, settings=settings)


async def test_quality_gate_ships_on_high_score(tmp_path: Path) -> None:
    loop = _loop(
        tmp_path, _ScoreProvider(json.dumps({"scores": {"q": 1.0}})), quality_gate_threshold=0.8
    )
    ship, weak = await loop._quality_gate("req", "the result", [])
    assert ship is True and weak == ""


async def test_quality_gate_iterates_on_low_score(tmp_path: Path) -> None:
    loop = _loop(
        tmp_path, _ScoreProvider(json.dumps({"scores": {"q": 0.3}})), quality_gate_threshold=0.8
    )
    ship, weak = await loop._quality_gate("req", "the result", [])
    assert ship is False and "q" in weak  # the brief names the failing dimension


async def test_quality_gate_ships_on_scorer_failure(tmp_path: Path) -> None:
    loop = _loop(tmp_path, _ScoreProvider("not json at all"), quality_gate_threshold=0.8)
    ship, weak = await loop._quality_gate("req", "the result", [])
    assert ship is True and weak == ""  # FAIL-OPEN: an unscoreable result must never trap the turn


def test_gather_work_includes_the_written_code(tmp_path: Path) -> None:
    # The gate scores the claim PLUS the actual files the turn wrote — not just the summary.
    f = tmp_path / "store.py"
    f.write_text("def add(a, b):\n    return a + b  # the real code\n", encoding="utf-8")
    artifact = _gather_work("I wrote store.py", [str(f)])
    assert "I wrote store.py" in artifact  # the claim
    assert "the real code" in artifact and "store.py" in artifact  # the file content + name
