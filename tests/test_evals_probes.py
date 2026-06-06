"""Tests for the M9 behavioral probe suite: every probe passes against the real loop,
and the suite wiring is sound."""

from __future__ import annotations

import asyncio
from pathlib import Path

from zakcode.evals import run_evals
from zakcode.evals.probes import FlakyTool, build_default_suite


def test_default_suite_has_all_probes(tmp_path: Path) -> None:
    suite = build_default_suite(str(tmp_path))
    assert len(suite) == 7
    names = {c.name for c in suite}
    assert names == {
        "completion-detection",
        "safety-rejection",
        "plan-mode-readonly",
        "doom-loop-halt",
        "partial-failure-recovery",
        "stuck-recovery",
        "long-horizon-compaction",
    }


def test_full_suite_all_pass(tmp_path: Path) -> None:
    # The whole point of M9: every behavioral probe passes on a clean tree.
    report = asyncio.run(run_evals(build_default_suite(str(tmp_path))))
    failed = [(r.name, r.error) for r in report.results if not r.passed]
    assert report.ok, f"probes failed: {failed}"
    assert report.passed == 7


# Each probe also runs standalone so a regression names the exact failing behavior.


def _run_one(tmp_path: Path, name: str) -> None:
    suite = build_default_suite(str(tmp_path))
    case = next(c for c in suite if c.name == name)
    result = asyncio.run(run_evals([case]))
    assert result.ok, result.results[0].error


def test_probe_completion(tmp_path: Path) -> None:
    _run_one(tmp_path, "completion-detection")


def test_probe_safety_rejection(tmp_path: Path) -> None:
    _run_one(tmp_path, "safety-rejection")
    # And concretely: the sentinel file must not exist after the denied bash call.
    assert not (tmp_path / "PWNED.txt").exists()


def test_probe_plan_mode_readonly(tmp_path: Path) -> None:
    _run_one(tmp_path, "plan-mode-readonly")


def test_probe_doom_loop_halt(tmp_path: Path) -> None:
    _run_one(tmp_path, "doom-loop-halt")


def test_probe_partial_failure_recovery(tmp_path: Path) -> None:
    _run_one(tmp_path, "partial-failure-recovery")


def test_probe_stuck_recovery(tmp_path: Path) -> None:
    _run_one(tmp_path, "stuck-recovery")


def test_probe_long_horizon_compaction(tmp_path: Path) -> None:
    _run_one(tmp_path, "long-horizon-compaction")


def test_flaky_tool_fails_once_then_succeeds() -> None:
    async def scenario() -> None:
        from zakcode.tools.base import ToolContext

        flaky = FlakyTool()
        ctx = ToolContext(workspace_root=Path("."))
        r1 = await flaky.execute({}, ctx)
        r2 = await flaky.execute({}, ctx)
        assert r1.is_error is True
        assert r2.is_error is False
        assert flaky.calls == 2

    asyncio.run(scenario())
