"""Tests for the `zakcode eval` CLI command (M9-3)."""

from __future__ import annotations

from typer.testing import CliRunner

import zakcode.evals.probes as probes_mod
from zakcode.cli import app
from zakcode.evals.harness import EvalCase

runner = CliRunner()


def test_eval_command_all_pass() -> None:
    result = runner.invoke(app, ["eval"])
    assert result.exit_code == 0, result.stdout
    assert "7/7 passed" in result.stdout
    assert "OK:" in result.stdout
    # all probe names appear in the table
    for name in (
        "completion-detection",
        "safety-rejection",
        "plan-mode-readonly",
        "doom-loop-halt",
        "partial-failure-recovery",
        "stuck-recovery",
        "long-horizon-compaction",
    ):
        assert name in result.stdout


def test_eval_command_verbose_shows_detail() -> None:
    result = runner.invoke(app, ["eval", "--verbose"])
    assert result.exit_code == 0, result.stdout
    # a detail string from a probe (the completion probe returns this phrase)
    assert "completed" in result.stdout.lower()


def test_eval_command_exits_nonzero_on_failure(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    # Inject a failing probe; the command must surface FAIL and exit 1 (CI gate).
    original = probes_mod.build_default_suite

    def with_failure(workspace: str) -> list[EvalCase]:
        suite = original(workspace)

        async def boom() -> str:
            raise AssertionError("forced failure")

        suite.append(EvalCase("forced-fail", "always fails", boom))
        return suite

    monkeypatch.setattr(probes_mod, "build_default_suite", with_failure)
    result = runner.invoke(app, ["eval"])
    assert result.exit_code == 1
    assert "FAIL:" in result.stdout
