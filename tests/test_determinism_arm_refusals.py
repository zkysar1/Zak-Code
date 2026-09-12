"""determinism_arm.py must refuse an identity verdict over runs that never measured the agent.

Two vacuous shapes, both measured 2026-09-12 (ADR-0157):

* the workspace was already deleted, so every run digests ZERO files, and an empty set equals
  an empty set (rc=4, shipped first);
* the child crashed at startup (a config refusal on a second machine), so every run digests the
  UNTOUCHED seed, which is identical across runs because nothing ran. The arm printed
  "IDENTICAL across all runs" over three 1.7-second crashes (rc=5, this test).

A run whose report carries ``success`` -- even ``false`` -- is a measurement and stays: three
deterministic wrong answers ARE a determinism result.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
from pathlib import Path

ARM = Path(__file__).resolve().parent.parent / "bench" / "determinism_arm.py"


def _load(monkeypatch, tmp_path):
    spec = importlib.util.spec_from_file_location("determinism_arm_under_test", ARM)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    monkeypatch.setattr(mod, "BENCH", tmp_path / "bench")  # results JSON lands under tmp
    monkeypatch.setattr(mod, "_avail_mb", lambda: 10**6)
    monkeypatch.delenv("ZAKCODE_TEMPERATURE", raising=False)
    monkeypatch.delenv("ZBENCH_PIN_MODE", raising=False)
    task = tmp_path / "tasks" / "unit-task"
    task.mkdir(parents=True)
    (task / "task.json").write_text(json.dumps({"id": "unit-task"}), encoding="utf-8")
    return mod, task


def _run(**over):
    base = {
        "cli_rc": 0,
        "workspace": "/tmp/unit",
        "pin": True,
        "temperature_env": "0",
        "elapsed_s": 1.0,
        "num_turns": 3,
        "total_cost_usd": 0.0,
        "verify_rc": 1,
        "verify_out": "max_iterations",
        "no_report": False,
        "stderr_tail": "",
        "digests": {"a.py": "1" * 64},
        "py_digests": {"a.py": "1" * 64},
        "sources": {},
    }
    base.update(over)
    return base


def test_runs_without_a_report_are_refused(monkeypatch, tmp_path, capsys):
    mod, task = _load(monkeypatch, tmp_path)
    crashed = _run(
        no_report=True,
        num_turns=None,
        stderr_tail="Traceback\nLocalOnlyViolation: local_only is set",
    )
    monkeypatch.setattr(mod, "one_run_zakcode", lambda td, sp, pin=True: dict(crashed))
    rc = mod.main(["--arm", "zakcode", str(task), "3"])
    out = capsys.readouterr().out
    assert rc == 5
    assert "produced NO report" in out
    assert "IDENTICAL across all runs" not in out
    assert "LocalOnlyViolation" in out  # the last stderr line is surfaced, not buried in the JSON
    written = json.loads(
        (
            tmp_path / "bench" / "results" / "determinism-zakcode-pin1-tempdefault-unit-task.json"
        ).read_text()
    )
    assert "produced no report" in written["instrument_failure"]


def test_empty_digests_are_still_refused_with_rc4(monkeypatch, tmp_path, capsys):
    mod, task = _load(monkeypatch, tmp_path)
    monkeypatch.setattr(
        mod, "one_run_zakcode", lambda td, sp, pin=True: _run(digests={}, py_digests={})
    )
    rc = mod.main(["--arm", "zakcode", str(task), "2"])
    assert rc == 4
    assert "captured ZERO files" in capsys.readouterr().out


def test_positive_control_reported_failures_still_render_a_verdict(monkeypatch, tmp_path, capsys):
    """verify_rc=1 on every run WITH a report is a result (deterministic wrong output)."""
    mod, task = _load(monkeypatch, tmp_path)
    monkeypatch.setattr(mod, "one_run_zakcode", lambda td, sp, pin=True: _run())
    rc = mod.main(["--arm", "zakcode", str(task), "3"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "PRIMARY (every file): IDENTICAL across all runs" in out
    assert "0/3 verify_rc==0" in out


def test_one_run_zakcode_marks_a_missing_report(monkeypatch, tmp_path):
    mod, task = _load(monkeypatch, tmp_path)
    spec = {"id": "unit-noreport"}

    def crash(*a, **k):
        return subprocess.CompletedProcess(
            a[0], 1, stdout="", stderr="Traceback\nLocalOnlyViolation: x"
        )

    monkeypatch.setattr(mod.subprocess, "run", crash)
    r = mod.one_run_zakcode(task, spec, pin=True)
    assert r["no_report"] is True and r["verify_rc"] == 1 and r["num_turns"] is None

    def failed_but_reported(*a, **k):
        rep = {
            "success": False,
            "stop_reason": "max_iterations",
            "iterations": 4,
            "workspace": "(cleaned)",
        }
        return subprocess.CompletedProcess(a[0], 0, stdout=json.dumps(rep), stderr="")

    monkeypatch.setattr(mod.subprocess, "run", failed_but_reported)
    r = mod.one_run_zakcode(task, spec, pin=True)
    assert r["no_report"] is False and r["verify_rc"] == 1 and r["num_turns"] == 4
