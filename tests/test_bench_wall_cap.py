"""``bench/determinism_arm.py`` lets a campaign set the per-run wall cap.

The arm kills a child that outlives its cap (rc 124, no report, an INSTRUMENT FAILURE in the
verdict), and until now the cap was a literal in a default argument: 1800 s for the zakcode arm,
900 s for the reference arm. Measured 2026-09-25 on a ten-engine pod with five lanes in flight:
12 of 42 runs hit the 1800 s cap, every one a multi-file task, with the child still working —
the cap was measuring the load on the pod, not the loop. ``ZBENCH_RUN_TIMEOUT_S`` sets the cap
for a campaign; unset or empty keeps each arm's own default; a bad value fails loud, because a
silent fall-back would re-cap a campaign meant to run longer and the timeouts would read as the
agent's fault. The cap travels in the result JSON as ``wall_cap_s`` so a timeout count can be
read against the cap that produced it.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import tempfile
from pathlib import Path

import pytest

ARM = Path(__file__).resolve().parent.parent / "bench" / "determinism_arm.py"


@pytest.fixture
def arm():
    spec = importlib.util.spec_from_file_location("determinism_arm_under_test", ARM)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_unset_keeps_each_arm_s_own_default(arm, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ZBENCH_RUN_TIMEOUT_S", raising=False)
    assert arm._wall_cap(arm.ZAKCODE_WALL_CAP_S) == 1800
    assert arm._wall_cap(arm.CLAUDE_CODE_WALL_CAP_S) == 900


def test_empty_keeps_the_default_too(arm, monkeypatch: pytest.MonkeyPatch) -> None:
    """A wrapper that exports the variable blank has not asked for a cap."""
    monkeypatch.setenv("ZBENCH_RUN_TIMEOUT_S", "")
    assert arm._wall_cap(1800) == 1800
    monkeypatch.setenv("ZBENCH_RUN_TIMEOUT_S", "   ")
    assert arm._wall_cap(1800) == 1800


def test_a_number_replaces_the_default(arm, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ZBENCH_RUN_TIMEOUT_S", "3600")
    assert arm._wall_cap(1800) == 3600


def test_a_bad_value_fails_loud(arm, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ZBENCH_RUN_TIMEOUT_S", "1h")
    with pytest.raises(ValueError):
        arm._wall_cap(1800)


def _task(tmp_path: Path) -> Path:
    td = tmp_path / "task"
    td.mkdir()
    (td / "task.json").write_text(json.dumps({"id": "cap-probe"}), encoding="utf-8")
    return td


def _completed_run() -> dict:
    """The smallest run record the verdict renderer accepts: one file, a report, no timeout."""
    return {
        "verify_rc": 0,
        "num_turns": 1,
        "total_cost_usd": 0.0,
        "elapsed_s": 1.0,
        "digests": {"a.py": "0" * 64},
        "py_digests": {"a.py": "0" * 64},
    }


def _timed_out_run(timeout_s: int, files: dict[str, str] | None = None) -> dict:
    """What one_run_zakcode returns when the child outlives the cap: rc 124 and no report. ``files``
    is what the killed run had written by then -- digested from the workspace the arm made for it
    (ADR-0253); the default models a run that had not written anything."""
    digests = dict(files or {})
    return {
        "cli_rc": 124,
        "verify_rc": None,
        "num_turns": None,
        "total_cost_usd": None,
        "elapsed_s": float(timeout_s),
        "digests": digests,
        "py_digests": {k: v for k, v in digests.items() if k.endswith(".py")},
        "no_report": True,
        "stderr_tail": f"exceeded {timeout_s}s",
    }


def _bench_in(tmp_path: Path, arm, monkeypatch: pytest.MonkeyPatch) -> None:
    """Point the arm's results dir at tmp_path and make the memory guard pass."""
    monkeypatch.setattr(arm, "_avail_mb", lambda: 10**6)
    monkeypatch.setattr(arm, "BENCH", tmp_path)
    monkeypatch.delenv("ZAKCODE_TEMPERATURE", raising=False)  # the cell name reads it


def _result(tmp_path: Path, name: str) -> dict:
    return json.loads((tmp_path / "results" / name).read_text(encoding="utf-8"))


ZAKCODE_RESULT = "determinism-zakcode-pinOFF-tempdefault-cap-probe.json"


def test_main_hands_the_cap_to_the_zakcode_arm_and_records_it(
    arm,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """End to end through ``main``: the env cap reaches the child's timeout and the result JSON."""
    seen: list[dict] = []

    def fake_run(task_dir, spec, timeout_s, pin):
        seen.append({"timeout_s": timeout_s, "pin": pin})
        return _completed_run()

    monkeypatch.setattr(arm, "one_run_zakcode", fake_run)
    _bench_in(tmp_path, arm, monkeypatch)
    monkeypatch.setenv("ZBENCH_RUN_TIMEOUT_S", "42")
    assert arm.main(["--arm", "zakcode", "--no-pin", str(_task(tmp_path)), "1"]) == 0
    assert seen == [{"timeout_s": 42, "pin": False}]
    assert _result(tmp_path, ZAKCODE_RESULT)["wall_cap_s"] == 42


def test_main_hands_the_reference_arm_its_own_default_when_unset(
    arm,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[int] = []

    def fake_run(task_dir, spec, timeout_s):
        seen.append(timeout_s)
        return _completed_run()

    monkeypatch.setattr(arm, "one_run", fake_run)
    _bench_in(tmp_path, arm, monkeypatch)
    monkeypatch.delenv("ZBENCH_RUN_TIMEOUT_S", raising=False)
    assert arm.main([str(_task(tmp_path)), "1"]) == 0
    assert seen == [900]
    assert _result(tmp_path, "determinism-claude-code-asships-cap-probe.json")["wall_cap_s"] == 900


def test_an_instrument_failure_verdict_records_the_cap_as_well(
    arm,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A timed-out child is exactly the run whose cap the reader needs beside the timeout."""
    monkeypatch.setattr(
        arm, "one_run_zakcode", lambda td, sp, timeout_s, pin: _timed_out_run(timeout_s)
    )
    _bench_in(tmp_path, arm, monkeypatch)
    monkeypatch.setenv("ZBENCH_RUN_TIMEOUT_S", "7")
    assert arm.main(["--arm", "zakcode", "--no-pin", str(_task(tmp_path)), "1"]) == 4
    out = _result(tmp_path, ZAKCODE_RESULT)
    assert out["wall_cap_s"] == 7
    assert out["runs"][0]["stderr_tail"] == "exceeded 7s"


# --- ADR-0253: the arm owns each run's workspace, so a run killed at the cap is still digested ---

A_PY = "x = 1\n"
A_PY_SHA = hashlib.sha256(A_PY.encode("utf-8")).hexdigest()

# Stand-ins for bench/run_task.py (the arm runs ``BENCH / "run_task.py"``): each writes one file
# into the workspace the arm handed down, then either outlives the cap or reports like the runner.
CAPPED_CHILD = """\
import os, pathlib, time
ws = pathlib.Path(os.environ["ZBENCH_WORKSPACE"])
(ws / "a.py").write_text("x = 1\\n", encoding="utf-8")
time.sleep(120)
"""
COMPLETED_CHILD = """\
import json, os, pathlib
ws = pathlib.Path(os.environ["ZBENCH_WORKSPACE"])
(ws / "a.py").write_text("x = 1\\n", encoding="utf-8")
report = {"success": True, "iterations": 1, "session_cost_usd": 0.0, "stop_reason": "completed",
          "trace_interventions": {}, "workspace": str(ws)}
print(json.dumps(report))
"""


def _child_in(tmp_path: Path, arm, monkeypatch: pytest.MonkeyPatch, body: str) -> Path:
    """Install ``body`` as the runner the arm will spawn, and keep every temp dir the arm makes
    under tmp_path so the test can see what it left behind."""
    _bench_in(tmp_path, arm, monkeypatch)
    (tmp_path / "run_task.py").write_text(body, encoding="utf-8")
    tmp = tmp_path / "tmp"
    tmp.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(tmp))
    return tmp


def test_a_run_killed_at_the_cap_is_digested_from_the_workspace_the_arm_made(
    arm,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The run the cap kills prints no report, so the arm cannot learn the workspace from it -- it
    must already know, because it made the directory. Before ADR-0253 this row digested as ZERO
    files (the report-less fallback named the full pin's constant path, which an unpinned run
    never has) and the real workspace leaked in the temp dir."""
    tmp = _child_in(tmp_path, arm, monkeypatch, CAPPED_CHILD)
    row = arm.one_run_zakcode(_task(tmp_path), {"id": "cap-probe"}, timeout_s=8, pin=False)
    assert row["cli_rc"] == 124
    assert row["no_report"] is True
    assert row["digests"] == {"a.py": A_PY_SHA}
    assert row["py_digests"] == {"a.py": A_PY_SHA}
    ws = Path(row["workspace"])
    assert ws.parent == tmp and ws.name.startswith("zbench-cap-probe-")  # the per-run suffix stays
    assert not ws.exists()  # digested, then removed: the arm owns the run's directory
    assert [p.name for p in tmp.iterdir() if p.name.startswith("zbench-cap-probe-")] == []


def test_a_completed_run_is_digested_the_same_way(
    arm,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Positive control for the test above: the child that reports still writes where the arm
    said, its report agrees, and the row reads as it always did."""
    tmp = _child_in(tmp_path, arm, monkeypatch, COMPLETED_CHILD)
    row = arm.one_run_zakcode(_task(tmp_path), {"id": "cap-probe"}, timeout_s=60, pin=False)
    assert row["cli_rc"] == 0
    assert row["no_report"] is False
    assert row["verify_rc"] == 0
    assert row["num_turns"] == 1
    assert row["trace_source"] == "report"
    assert row["digests"] == {"a.py": A_PY_SHA}
    assert not Path(row["workspace"]).exists()
    assert [p.name for p in tmp.iterdir() if p.name.startswith("zbench-cap-probe-")] == []


def test_an_instrument_failure_for_a_capped_run_says_it_outlived_the_cap(
    arm,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A capped run's files are in its row now, so the verdict must stop calling them the untouched
    seed of a child that never ran. It is still refused: a half-finished workspace has no verify
    result to compare."""
    row = _timed_out_run(7, files={"a.py": A_PY_SHA})
    monkeypatch.setattr(arm, "one_run_zakcode", lambda td, sp, timeout_s, pin: row)
    _bench_in(tmp_path, arm, monkeypatch)
    monkeypatch.setenv("ZBENCH_RUN_TIMEOUT_S", "7")
    # 5 is the no-report refusal; 4 (zero files captured) is what this row used to trip.
    assert arm.main(["--arm", "zakcode", "--no-pin", str(_task(tmp_path)), "1"]) == 5
    out = capsys.readouterr().out
    assert "outlived the wall cap" in out
    assert "before the agent ran" not in out
    assert _result(tmp_path, ZAKCODE_RESULT)["runs"][0]["digests"] == {"a.py": A_PY_SHA}


def test_an_instrument_failure_for_a_crashed_child_still_says_so(
    arm,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Control: the report-less row of a child that died at startup keeps its own message."""
    row = dict(_completed_run(), cli_rc=1, no_report=True, stderr_tail="ImportError: boom")
    monkeypatch.setattr(arm, "one_run_zakcode", lambda td, sp, timeout_s, pin: row)
    _bench_in(tmp_path, arm, monkeypatch)
    assert arm.main(["--arm", "zakcode", "--no-pin", str(_task(tmp_path)), "1"]) == 5
    out = capsys.readouterr().out
    assert "before the agent ran" in out
    assert "outlived the wall cap" not in out


RUN_TASK = Path(__file__).resolve().parent.parent / "bench" / "run_task.py"


@pytest.fixture
def run_task():
    spec = importlib.util.spec_from_file_location("run_task_under_test", RUN_TASK)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_the_runner_uses_the_directory_the_arm_handed_down(
    run_task,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ws = tmp_path / "handed-down"
    ws.mkdir()
    (ws / "already-here.txt").write_text("keep", encoding="utf-8")
    monkeypatch.setenv("ZBENCH_WORKSPACE", str(ws))
    assert run_task._workspace({"id": "cap-probe"}) == ws
    # The caller made the directory fresh; the runner does not wipe what it does not own.
    assert (ws / "already-here.txt").read_text(encoding="utf-8") == "keep"


def test_a_handed_down_directory_that_is_not_there_fails_loud(
    run_task,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A silent mkdtemp fallback would re-create the leak ADR-0253 closes; a relative path would
    resolve against the runner's cwd, not the caller's."""
    monkeypatch.setenv("ZBENCH_WORKSPACE", str(tmp_path / "missing"))
    with pytest.raises(SystemExit):
        run_task._workspace({"id": "cap-probe"})
    monkeypatch.setenv("ZBENCH_WORKSPACE", "relative/dir")
    with pytest.raises(SystemExit):
        run_task._workspace({"id": "cap-probe"})


def test_without_a_handed_down_directory_the_runner_chooses_as_before(
    run_task,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("ZBENCH_WORKSPACE", raising=False)
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    monkeypatch.delenv("ZBENCH_PIN_IDENTITY", raising=False)
    ws = run_task._workspace({"id": "cap-probe"})
    assert ws.parent == tmp_path and ws.name.startswith("zbench-cap-probe-") and ws.is_dir()
    monkeypatch.setenv("ZBENCH_PIN_IDENTITY", "session")
    ws = run_task._workspace({"id": "cap-probe"})
    assert ws.parent == tmp_path and ws.name.startswith("zbench-cap-probe-") and ws.is_dir()
    monkeypatch.setenv("ZBENCH_PIN_IDENTITY", "1")
    pinned = tmp_path / "zbench-pinned-cap-probe"
    pinned.mkdir()
    (pinned / "stale.txt").write_text("from the previous run", encoding="utf-8")
    assert run_task._workspace({"id": "cap-probe"}) == pinned
    assert not (pinned / "stale.txt").exists()  # a constant NAME with fresh CONTENT
