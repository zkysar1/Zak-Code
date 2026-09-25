"""``bench/run_task.py`` can run a self-hosted, unpriced model with NO cost ceiling.

A ceiling the meter cannot enforce stops the run: a self-hosted model is absent from
litellm's price map, every call on it is unpriced, and since #574 the loop ends such a
turn with ``budget_unpriced`` instead of running blind under a ceiling that reads $0.00.
Right for a paid lane; a dead instrument for the bench. Measured 2026-09-25 on the ZDS
pod: every run of a five-lane campaign ended after ONE turn at $0.00 — the runner had
handed every run the task's ceiling ($1.00 by default) and nothing could lift it.

``ZBENCH_MAX_COST_USD`` is the lift: empty means no ceiling, a number replaces the
task's, and an unset variable keeps the task's own ceiling byte-for-byte.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

RUN_TASK = Path(__file__).resolve().parent.parent / "bench" / "run_task.py"


@pytest.fixture
def run_task():
    spec = importlib.util.spec_from_file_location("run_task_under_test", RUN_TASK)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_unset_keeps_the_task_s_own_ceiling(run_task, monkeypatch: pytest.MonkeyPatch) -> None:
    """A hosted-provider run never sets the variable and must see exactly what it saw before."""
    monkeypatch.delenv("ZBENCH_MAX_COST_USD", raising=False)
    assert run_task._cost_ceiling({"max_cost_usd": 0.25}) == 0.25
    assert run_task._cost_ceiling({}) == 1.0


def test_empty_means_no_ceiling(run_task, monkeypatch: pytest.MonkeyPatch) -> None:
    """The pod wrapper's setting: a free lane runs uncapped, whatever the task asked for."""
    monkeypatch.setenv("ZBENCH_MAX_COST_USD", "")
    assert run_task._cost_ceiling({"max_cost_usd": 0.25}) is None
    monkeypatch.setenv("ZBENCH_MAX_COST_USD", "   ")
    assert run_task._cost_ceiling({"max_cost_usd": 0.25}) is None


def test_a_number_replaces_the_task_s_ceiling(run_task, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ZBENCH_MAX_COST_USD", "2.5")
    assert run_task._cost_ceiling({"max_cost_usd": 0.25}) == 2.5


def test_a_bad_number_fails_loud(run_task, monkeypatch: pytest.MonkeyPatch) -> None:
    """Never a silent fallback to the task's ceiling: that would re-cap a run meant to be free."""
    monkeypatch.setenv("ZBENCH_MAX_COST_USD", "free")
    with pytest.raises(ValueError):
        run_task._cost_ceiling({"max_cost_usd": 0.25})


def test_the_runner_hands_the_ceiling_to_the_agent_settings() -> None:
    """The helper is wired in: ``_build_agent`` reads ``max_cost_usd`` through it and nowhere
    else. Read from source, like the constructible suite, so this stays outside the runtime
    gates ``bench/`` is excluded from."""
    src = RUN_TASK.read_text(encoding="utf-8")
    assert '"max_cost_usd": _cost_ceiling(spec)' in src
    assert 'spec.get("max_cost_usd"' not in src.split("def _build_agent")[1]
