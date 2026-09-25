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

import importlib.util
import json
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


def _timed_out_run(timeout_s: int) -> dict:
    """What one_run_zakcode returns when the child outlives the cap: no report, no files."""
    return {
        "verify_rc": None,
        "num_turns": None,
        "total_cost_usd": None,
        "elapsed_s": float(timeout_s),
        "digests": {},
        "py_digests": {},
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
