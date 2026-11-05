"""The dev channel's publisher opens one lane and refuses every other (g-370-47).

`.github/workflows/ci.yml` builds a candidate into `zak-code-deployments/dev/` on a dispatch.
Production boots `zak-code-deployments/latest/`, so the one error that matters is a lane that falls
through to that name. These tests run the job's own bash block, not a copy of it, against a stub
`aws`, and then break the block three ways to show the refusals are what stop a write.

The workflow also gets a mechanical form of its audit comment: the jobs that write outside the
runner are a fixed set, each gated to the event that may write.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml

WORKFLOW = Path(__file__).resolve().parent.parent / ".github" / "workflows" / "ci.yml"

needs_bash = pytest.mark.skipif(
    sys.platform == "win32" or shutil.which("bash") is None,
    reason="runs the dev-channel job's bash block; that job runs on ubuntu",
)

PROD_GATE = "github.event_name == 'push' && github.ref == 'refs/heads/main'"
DISPATCH_GATE = "github.event_name == 'workflow_dispatch'"


def _jobs() -> dict[str, Any]:
    return yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))["jobs"]


def _publish_script() -> str:
    for step in _jobs()["publish-dev-channel"]["steps"]:
        if step.get("name", "").startswith("Build the candidate"):
            return str(step["run"])
    raise AssertionError("publish-dev-channel has no 'Build the candidate' step")


def _git(repo: Path, env: dict[str, str], *args: str) -> str:
    done = subprocess.run(
        ["git", *args], cwd=repo, env=env, check=True, capture_output=True, text=True
    )
    return done.stdout.strip()


def _run_block(
    script: str,
    tmp_path: Path,
    *,
    lane: str = "dev",
    candidate: str = "alpha/g-1-candidate",
    workflow_ref: str = "refs/heads/main",
    runtime_present: bool = True,
) -> tuple[subprocess.CompletedProcess[str], list[str], str]:
    """Run `script` in a throwaway checkout; return the result, each `aws` call and the sha."""
    repo = tmp_path / "repo"
    server = repo / "src" / "zakcode" / "server"
    server.mkdir(parents=True)
    module = "app.py" if runtime_present else "other.py"
    (server / module).write_text("x = 1\n", encoding="utf-8")
    stubs = tmp_path / "stubs"
    stubs.mkdir()
    stub = stubs / "aws"
    stub.write_text('#!/bin/sh\nprintf \'%s\\n\' "$*" >> "$AWS_LOG"\n', encoding="utf-8")
    stub.chmod(0o755)
    log = tmp_path / "aws.log"
    git_env = {
        **os.environ,
        "GIT_AUTHOR_NAME": "t",
        "GIT_AUTHOR_EMAIL": "t@example.invalid",
        "GIT_COMMITTER_NAME": "t",
        "GIT_COMMITTER_EMAIL": "t@example.invalid",
    }
    for cmd in (["init", "-q"], ["add", "-A"], ["commit", "-q", "-m", "c"]):
        _git(repo, git_env, *cmd)
    sha = _git(repo, git_env, "rev-parse", "HEAD")
    env = {
        **os.environ,
        "PATH": f"{stubs}{os.pathsep}{os.environ['PATH']}",
        "AWS_LOG": str(log),
        "BUCKET": "ayoai-jar-deployments",
        "LANE": lane,
        "CANDIDATE": candidate,
        "WORKFLOW_REF": workflow_ref,
        "REPO": "zkysar1/Zak-Code",
        "RUN": "7",
    }
    script_file = tmp_path / "block.sh"
    script_file.write_text(script, encoding="utf-8")
    done = subprocess.run(
        ["bash", str(script_file)], cwd=repo, env=env, capture_output=True, text=True, timeout=60
    )
    calls = log.read_text(encoding="utf-8").splitlines() if log.exists() else []
    return done, calls, sha


def _refused_without_a_write(done: subprocess.CompletedProcess[str], calls: list[str]) -> bool:
    return done.returncode != 0 and calls == []


def _mutate(script: str, pattern: str, replacement: str) -> str:
    """Break the block; fail loudly when the pattern is absent, so a control is never vacuous."""
    mutated, n = re.subn(pattern, replacement, script, count=1, flags=re.DOTALL)
    assert n == 1, f"the mutation pattern no longer matches the shipped block: {pattern}"
    assert mutated != script
    return mutated


@needs_bash
def test_the_dev_lane_publishes_to_the_dev_prefix_and_nowhere_else(tmp_path: Path) -> None:
    done, calls, sha = _run_block(_publish_script(), tmp_path)
    assert done.returncode == 0, done.stdout + done.stderr
    assert len(calls) == 1, calls
    call = calls[0]
    assert call.startswith(
        "s3 cp zak-code-main.tar.gz "
        "s3://ayoai-jar-deployments/zak-code-deployments/dev/zak-code-main.tar.gz "
    )
    assert f"commit_sha={sha}" in call
    assert "branch=alpha/g-1-candidate" in call
    assert "/latest/" not in call


@needs_bash
@pytest.mark.parametrize(
    "lane", ["", "latest", "prod", "main", "DEV", " dev", "dev ", "dev,latest", "dev;latest", "*"]
)
def test_every_lane_but_dev_is_refused_before_any_write(tmp_path: Path, lane: str) -> None:
    done, calls, _ = _run_block(_publish_script(), tmp_path, lane=lane)
    assert _refused_without_a_write(done, calls), done.stdout + done.stderr
    assert "Unrecognised lane" in done.stdout + done.stderr


@needs_bash
@pytest.mark.parametrize(
    "candidate", ["", "-x", "a..b", "a b", "a,b", "a;b", "$(id)", "`id`", "a\nb", "a:b", "a=b"]
)
def test_a_candidate_that_could_inject_metadata_or_shell_is_refused(
    tmp_path: Path, candidate: str
) -> None:
    done, calls, _ = _run_block(_publish_script(), tmp_path, candidate=candidate)
    assert _refused_without_a_write(done, calls), done.stdout + done.stderr
    assert "ref must be" in done.stdout + done.stderr


@needs_bash
@pytest.mark.parametrize("candidate", ["main", "v1.2.3", "alpha/g-370-47-x", "ea5572a" * 5 + "ab"])
def test_a_plain_ref_is_accepted(tmp_path: Path, candidate: str) -> None:
    done, calls, _ = _run_block(_publish_script(), tmp_path, candidate=candidate)
    assert done.returncode == 0, done.stdout + done.stderr
    assert len(calls) == 1


@needs_bash
def test_a_dispatch_from_a_candidate_branch_is_refused(tmp_path: Path) -> None:
    done, calls, _ = _run_block(
        _publish_script(), tmp_path, workflow_ref="refs/heads/alpha/g-1-candidate"
    )
    assert _refused_without_a_write(done, calls), done.stdout + done.stderr
    assert "dispatch on main" in done.stdout + done.stderr


@needs_bash
def test_a_tarball_without_the_runtime_is_not_published(tmp_path: Path) -> None:
    done, calls, _ = _run_block(_publish_script(), tmp_path, runtime_present=False)
    assert _refused_without_a_write(done, calls), done.stdout + done.stderr
    assert "app.py absent" in done.stdout + done.stderr


@needs_bash
def test_a_lane_that_falls_through_to_the_production_name_would_be_caught(tmp_path: Path) -> None:
    mutant = _mutate(
        _publish_script(),
        r'(case "\$LANE" in\n\s*dev\) CHANNEL="dev" ;;\n)\s*\*\).*?;;\n(\s*esac)',
        r'\1  *) CHANNEL="latest" ;;\n\2',
    )
    done, calls, _ = _run_block(mutant, tmp_path, lane="latest")
    assert not _refused_without_a_write(done, calls)
    assert any("/zak-code-deployments/latest/zak-code-main.tar.gz" in c for c in calls), calls


@needs_bash
def test_dropping_the_main_only_check_would_be_caught(tmp_path: Path) -> None:
    mutant = _mutate(
        _publish_script(),
        r'\[ "\$WORKFLOW_REF" = "refs/heads/main" \] \\\n[^\n]*\n',
        "",
    )
    done, calls, _ = _run_block(mutant, tmp_path, workflow_ref="refs/heads/alpha/g-1-candidate")
    assert not _refused_without_a_write(done, calls)


@needs_bash
def test_dropping_the_candidate_check_would_be_caught(tmp_path: Path) -> None:
    mutant = _mutate(_publish_script(), r'case "\$CANDIDATE" in.*?\nesac\n', "")
    done, calls, _ = _run_block(mutant, tmp_path, candidate="a,b")
    assert not _refused_without_a_write(done, calls)
    assert "branch=a,b" in calls[0]


def test_no_expression_is_interpolated_into_the_dev_job_script() -> None:
    for step in _jobs()["publish-dev-channel"]["steps"]:
        assert "${{" not in step.get("run", ""), step.get("name")


def test_a_dispatch_can_open_the_dev_lane_and_no_other() -> None:
    raw = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    on = raw.get("on", raw.get(True))
    lane = on["workflow_dispatch"]["inputs"]["lane"]
    assert lane["type"] == "choice"
    assert lane["options"] == ["dev"]


def test_a_dispatch_runs_the_dev_job_and_none_of_the_test_jobs() -> None:
    jobs = _jobs()
    assert jobs["publish-dev-channel"]["if"] == DISPATCH_GATE
    for name in ("verify", "update-e2e", "cockpit-e2e"):
        assert jobs[name]["if"] == "github.event_name != 'workflow_dispatch'", name


def test_the_jobs_that_write_outside_the_runner_are_a_fixed_set_each_gated() -> None:
    """The guard-5410 audit in ci.yml, held mechanically. A new writer must be added to it."""
    writes = re.compile(r"aws s3 (cp|sync|mv|rm)|aws s3api put|aws sesv2 send|upload-artifact")
    found: set[str] = set()
    for name, job in _jobs().items():
        text = "\n".join(f"{s.get('run', '')}\n{s.get('uses', '')}" for s in job["steps"])
        if writes.search(text):
            found.add(name)
    assert found == {"verify", "publish-sidecar", "publish-dev-channel", "alert-on-failure"}, (
        "a job now writes outside the runner that the guard-5410 audit comment in ci.yml does not "
        "list: say what tells the consumer which lane produced it, and gate the step"
    )
    jobs = _jobs()
    assert jobs["publish-sidecar"]["if"] == PROD_GATE
    assert "github.ref == 'refs/heads/main'" in jobs["alert-on-failure"]["if"]
    assert jobs["publish-dev-channel"]["if"] == DISPATCH_GATE


def test_the_production_pointer_is_written_by_the_production_job_alone() -> None:
    for name, job in _jobs().items():
        for step in job["steps"]:
            if "zak-code-deployments/latest" in step.get("run", ""):
                assert name == "publish-sidecar", f"{name} can write the production channel"


def test_the_production_publish_names_its_lane_in_the_metadata() -> None:
    steps = _jobs()["publish-sidecar"]["steps"]
    publish = [s for s in steps if s.get("name", "").startswith("Publish to S3")]
    assert len(publish) == 1
    assert "branch=main" in publish[0]["run"]
