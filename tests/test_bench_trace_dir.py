"""``bench/determinism_arm.py`` keeps every zakcode run's decision trace on disk.

The engine checkpoints its per-turn decision trace (usage rows, gate and recovery
interventions, the stop) to ``ZAKCODE_TRACE_DIR`` after every tool batch. The arm read that
trace only through the child's report, which a run that outlives the wall cap never prints —
measured 2026-09-25: a 3600 s death on a multi-file cell left a row with no turn count, no
latency and no intervention, so the verdict could say "timeout" and nothing else. The arm now
hands each run its own trace directory, outside the workspace (the digests must never see it),
names it on the row, and for a run with no report tallies the row's interventions from the
checkpointed files with the report's own key. A report's own tally is never second-guessed.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import tempfile
from pathlib import Path

import pytest

ARM = Path(__file__).resolve().parent.parent / "bench" / "determinism_arm.py"
SPEC = {"id": "unit-trace"}


@pytest.fixture
def arm(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    spec = importlib.util.spec_from_file_location("determinism_arm_under_test", ARM)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    monkeypatch.setattr(mod, "BENCH", tmp_path / "bench")
    # The arm creates the per-run trace directory with mkdtemp; keep it under the test's own
    # tmp_path (tempfile.tempdir is the documented runtime override, restored by monkeypatch).
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    return mod


@pytest.fixture
def task(tmp_path: Path) -> Path:
    td = tmp_path / "tasks" / SPEC["id"]
    td.mkdir(parents=True)
    (td / "task.json").write_text(json.dumps(SPEC), encoding="utf-8")
    return td


def _write_turn(env: dict, rows: list[dict]) -> Path:
    """What the engine leaves behind: ``<trace_dir>/<session>/turn_1.jsonl``, one record a line."""
    turn = Path(env["ZAKCODE_TRACE_DIR"]) / "0123456789abcdef0123456789abcdef" / "turn_1.jsonl"
    turn.parent.mkdir(parents=True)
    turn.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return turn


def test_a_capped_run_s_row_carries_the_trace_it_left_behind(arm, task, monkeypatch) -> None:
    """The child outlives the cap: no report, but the checkpointed trace names what fired."""

    def child(cmd, **kw):
        _write_turn(
            kw["env"],
            [
                {"kind": "usage", "detail": "", "data": {"latency_s": 600.0}},
                {"kind": "intervention", "detail": "compacted", "data": {"kind": "compaction"}},
                {"kind": "intervention", "detail": "and again", "data": {"kind": "compaction"}},
                {"kind": "intervention", "detail": "a gate with no kind of its own", "data": {}},
                {"kind": "stop", "detail": "", "data": {}},
            ],
        )
        raise subprocess.TimeoutExpired(cmd, kw["timeout"])

    monkeypatch.setattr(arm.subprocess, "run", child)
    row = arm.one_run_zakcode(task, SPEC, timeout_s=7, pin=True)
    assert row["cli_rc"] == 124 and row["no_report"] is True
    assert row["trace_interventions"] == {"compaction": 2, "a gate with no kind of its own": 1}
    assert row["trace_source"] == "trace-file"
    assert Path(row["trace_dir"]).is_dir()
    assert (Path(row["trace_dir"]) / "0123456789abcdef0123456789abcdef" / "turn_1.jsonl").is_file()


def test_a_reported_run_keeps_the_report_s_own_tally(arm, task, monkeypatch) -> None:
    """A report, even one whose tally is empty (a clean run), is never second-guessed."""

    def child_for(tally: dict):
        def child(cmd, **kw):
            _write_turn(
                kw["env"],
                [{"kind": "intervention", "detail": "", "data": {"kind": "doom_loop"}}],
            )
            rep = {
                "success": True,
                "stop_reason": "completed",
                "iterations": 3,
                "trace_interventions": tally,
                "workspace": "(cleaned)",
            }
            return subprocess.CompletedProcess(cmd, 0, stdout=json.dumps(rep), stderr="")

        return child

    monkeypatch.setattr(arm.subprocess, "run", child_for({"plan_review": 1}))
    row = arm.one_run_zakcode(task, SPEC, pin=True)
    assert row["trace_interventions"] == {"plan_review": 1}
    assert row["trace_source"] == "report" and row["no_report"] is False

    monkeypatch.setattr(arm.subprocess, "run", child_for({}))
    row = arm.one_run_zakcode(task, SPEC, pin=True)
    assert row["trace_interventions"] == {} and row["trace_source"] == "report"


def test_each_run_gets_its_own_trace_directory_outside_the_workspace(
    arm, task, monkeypatch
) -> None:
    seen: list[str] = []

    def child(cmd, **kw):
        seen.append(kw["env"]["ZAKCODE_TRACE_DIR"])
        return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="")

    monkeypatch.setattr(arm.subprocess, "run", child)
    rows = [arm.one_run_zakcode(task, SPEC, pin=True) for _ in range(2)]
    assert len(set(seen)) == 2 and all(Path(p).is_dir() for p in seen)
    assert [r["trace_dir"] for r in rows] == seen
    for r in rows:
        assert not r["trace_dir"].startswith(r["workspace"])
