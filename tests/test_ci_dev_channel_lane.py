"""The dev channel's publisher opens one lane and refuses every other.

`.github/workflows/dev-channel.yml` builds a candidate into `zak-code-deployments/dev/` on a
dispatch. Production boots `zak-code-deployments/latest/`, so the one error that matters is a lane
that falls through to that name. These tests run the job's own steps, in order, against a stub
`aws`, and then break the guard three ways to show the refusals are what stop a write.

The file also holds the audit of every workflow step that writes outside the runner. They are a
fixed set: a new one is listed here, with its reason, before it is added to a workflow.
"""

from __future__ import annotations

import hashlib
import os
import re
import shutil
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest
import yaml

WORKFLOWS = Path(__file__).resolve().parent.parent / ".github" / "workflows"
CI = WORKFLOWS / "ci.yml"
DEV = WORKFLOWS / "dev-channel.yml"
JOB = "publish-dev-channel"
DEV_KEY = "zak-code-deployments/dev/zak-code-main.tar.gz"
LATEST = "zak-code-deployments/latest"
PROD_GATE = "github.event_name == 'push' && github.ref == 'refs/heads/main'"
CANDIDATE = "feature/some-change"

needs_bash = pytest.mark.skipif(
    sys.platform == "win32" or shutil.which("bash") is None,
    reason="runs the dev-channel job's bash steps; that job runs on ubuntu",
)


def _load(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _triggers(path: Path) -> dict[str, Any]:
    doc = _load(path)
    # PyYAML reads the bare key `on` as the boolean True.
    return doc.get("on", doc.get(True))


def _steps(path: Path, job: str = JOB) -> list[dict[str, Any]]:
    return list(_load(path)["jobs"][job]["steps"])


def _label(step: dict[str, Any]) -> str:
    return str(step.get("name") or step["uses"])


def _bucket() -> str:
    """The bucket the job names, read from the workflow so this file never spells it."""
    for step in _steps(DEV):
        if "BUCKET" in (step.get("env") or {}):
            return str(step["env"]["BUCKET"])
    raise AssertionError("no step of the dev job names the bucket")


# --- running the job ---------------------------------------------------------------------------

_EXPRESSION = re.compile(r"\$\{\{\s*([\w.]+)\s*\}\}")


def _resolve(text: str, context: dict[str, str]) -> str:
    """Fill the `${{ }}` expressions as the runner does; refuse one the model cannot fill."""
    out = _EXPRESSION.sub(lambda m: context[m.group(1)], text)
    assert "${{" not in out, f"an expression this model cannot resolve: {text}"
    return out


@dataclass
class JobRun:
    reached: list[str] = field(default_factory=list)  # each step the job started, in order
    failed: str | None = None  # the step that exited non-zero, which ends the job
    output: str = ""
    calls: list[str] = field(default_factory=list)  # each `aws` call the stub saw
    sha: str = ""

    @property
    def refused_without_a_write(self) -> bool:
        return self.failed is not None and self.calls == []


def _git(repo: Path, *args: str) -> str:
    env = {
        **os.environ,
        "GIT_AUTHOR_NAME": "t",
        "GIT_AUTHOR_EMAIL": "t@example.invalid",
        "GIT_COMMITTER_NAME": "t",
        "GIT_COMMITTER_EMAIL": "t@example.invalid",
    }
    done = subprocess.run(
        ["git", *args], cwd=repo, env=env, check=True, capture_output=True, text=True
    )
    return done.stdout.strip()


def _checked_out(tmp_path: Path, *, runtime_present: bool) -> tuple[Path, str]:
    """A repository of one commit: what `actions/checkout` leaves in the workspace."""
    repo = tmp_path / "repo"
    server = repo / "src" / "zakcode" / "server"
    server.mkdir(parents=True)
    (server / ("app.py" if runtime_present else "other.py")).write_text("x = 1\n", encoding="utf-8")
    for cmd in (["init", "-q"], ["add", "-A"], ["commit", "-q", "-m", "c"]):
        _git(repo, *cmd)
    return repo, _git(repo, "rev-parse", "HEAD")


def _stub_aws(tmp_path: Path) -> tuple[Path, Path]:
    stubs = tmp_path / "stubs"
    stubs.mkdir()
    stub = stubs / "aws"
    stub.write_text('#!/bin/sh\nprintf \'%s\\n\' "$*" >> "$AWS_LOG"\n', encoding="utf-8")
    stub.chmod(0o755)
    return stubs, tmp_path / "aws.log"


def _unchanged(script: str) -> str:
    return script


def _run_job(
    tmp_path: Path,
    *,
    lane: str = "dev",
    ref: str = CANDIDATE,
    workflow_ref: str = "refs/heads/main",
    runtime_present: bool = True,
    mutate_guard: Callable[[str], str] = _unchanged,
    skip: tuple[int, ...] = (),
) -> JobRun:
    """Run the dev job's steps in order, as the runner does, until one fails.

    `actions/checkout` becomes "the working directory is now the candidate's repository" and the
    credentials action does nothing. Any other action is refused, so a step added to the job
    cannot go unmodelled. The first step runs in an empty workspace, which is how it runs with no
    checkout before it. `mutate_guard` rewrites that step's script, and `skip` leaves steps out.
    """
    work = tmp_path / "work"
    work.mkdir()
    repo, sha = _checked_out(tmp_path, runtime_present=runtime_present)
    stubs, log = _stub_aws(tmp_path)
    github_env = tmp_path / "github-env"
    github_env.write_text("", encoding="utf-8")
    context = {
        "inputs.lane": lane,
        "inputs.ref": ref,
        "github.ref": workflow_ref,
        "github.repository": "owner/repo",
        "github.run_number": "7",
    }
    env = {
        **os.environ,
        "PATH": f"{stubs}{os.pathsep}{os.environ['PATH']}",
        "AWS_LOG": str(log),
        "GITHUB_ENV": str(github_env),
    }
    cwd = work
    run = JobRun(sha=sha)
    for index, step in enumerate(_steps(DEV)):
        if index in skip:
            continue
        run.reached.append(_label(step))
        uses = step.get("uses", "")
        if uses.startswith("actions/checkout@"):
            cwd = repo
        elif uses.startswith("aws-actions/configure-aws-credentials@"):
            continue
        elif uses:
            raise AssertionError(f"the model cannot run {uses}: teach it before adding the step")
        else:
            script = mutate_guard(step["run"]) if index == 0 else step["run"]
            script_file = tmp_path / f"step-{index}.sh"
            script_file.write_text(script, encoding="utf-8")
            step_env = {
                **env,
                **{k: _resolve(str(v), context) for k, v in (step.get("env") or {}).items()},
            }
            done = subprocess.run(
                ["bash", "-e", str(script_file)],
                cwd=cwd,
                env=step_env,
                capture_output=True,
                text=True,
                timeout=60,
            )
            run.output += done.stdout + done.stderr
            # A later step sees what this one wrote to GITHUB_ENV, and this one never does.
            for line in github_env.read_text(encoding="utf-8").splitlines():
                name, sep, value = line.partition("=")
                if sep:
                    env[name] = value
            if done.returncode != 0:
                run.failed = _label(step)
                break
    run.calls = log.read_text(encoding="utf-8").splitlines() if log.exists() else []
    return run


def _mutate(script: str, pattern: str, replacement: str) -> str:
    """Break the block; fail loudly when the pattern is absent, so a control is never vacuous."""
    mutated, n = re.subn(pattern, replacement, script, count=1, flags=re.DOTALL)
    assert n == 1, f"the mutation pattern no longer matches the shipped block: {pattern}"
    assert mutated != script
    return mutated


def _refused_at_the_guard(run: JobRun) -> bool:
    """Refused at the first step, so no checkout and no credentials came before the refusal."""
    return run.refused_without_a_write and run.reached == [_label(_steps(DEV)[0])]


# --- the lane, the dispatch and the ref ----------------------------------------------------------


@needs_bash
def test_the_dev_lane_publishes_to_the_dev_prefix_and_nowhere_else(tmp_path: Path) -> None:
    run = _run_job(tmp_path)
    assert run.failed is None, run.output
    assert run.reached == [_label(s) for s in _steps(DEV)]
    assert run.calls == [
        f"s3 cp zak-code-main.tar.gz s3://{_bucket()}/{DEV_KEY} --metadata "
        f"commit_sha={run.sha},repo=owner/repo,build_number=7,branch={CANDIDATE},lane=dev"
    ]


@needs_bash
@pytest.mark.parametrize(
    "lane",
    ["", "latest", "prod", "main", "DEV", " dev", "dev ", "dev\n", "dev,latest", "dev;latest", "*"],
)
def test_every_lane_but_dev_is_refused_at_the_first_step(tmp_path: Path, lane: str) -> None:
    run = _run_job(tmp_path, lane=lane)
    assert _refused_at_the_guard(run), run.output
    assert "Unrecognised lane" in run.output


@needs_bash
@pytest.mark.parametrize(
    "ref",
    ["", "-x", "a..b", "a b", "a,b", "a;b", "$(id)", "`id`", "a\nb", "a:b", "a=b", "café"],
)
def test_a_ref_that_could_inject_metadata_or_shell_is_refused_at_the_first_step(
    tmp_path: Path, ref: str
) -> None:
    run = _run_job(tmp_path, ref=ref)
    assert _refused_at_the_guard(run), run.output
    assert "ref must be" in run.output


@needs_bash
@pytest.mark.parametrize(
    "ref", ["main", "v1.2.3", CANDIDATE, hashlib.sha1(b"candidate").hexdigest()]
)
def test_a_plain_ref_is_accepted(tmp_path: Path, ref: str) -> None:
    run = _run_job(tmp_path, ref=ref)
    assert run.failed is None, run.output
    assert len(run.calls) == 1
    assert run.calls[0].endswith(f"branch={ref},lane=dev")


@needs_bash
@pytest.mark.parametrize(
    "workflow_ref", ["refs/heads/" + CANDIDATE, "refs/tags/v1", "refs/pull/1/merge", "main", ""]
)
def test_a_dispatch_from_anywhere_but_main_is_refused_at_the_first_step(
    tmp_path: Path, workflow_ref: str
) -> None:
    run = _run_job(tmp_path, workflow_ref=workflow_ref)
    assert _refused_at_the_guard(run), run.output
    assert "dispatch on main" in run.output


@needs_bash
def test_a_tarball_without_the_runtime_is_not_published(tmp_path: Path) -> None:
    run = _run_job(tmp_path, runtime_present=False)
    assert run.refused_without_a_write, run.output
    assert run.reached == [_label(s) for s in _steps(DEV)]
    assert "app.py absent" in run.output


@needs_bash
def test_the_publish_step_cannot_write_without_the_channel_the_guard_resolves(
    tmp_path: Path,
) -> None:
    run = _run_job(tmp_path, skip=(0,))
    assert run.refused_without_a_write, run.output
    assert "CHANNEL" in run.output


# --- the refusals are what stop a write: each one broken, and the write follows ------------------


@needs_bash
def test_a_lane_that_falls_through_to_the_production_name_would_be_caught(tmp_path: Path) -> None:
    def fall_through(script: str) -> str:
        return _mutate(
            script,
            r'(case "\$LANE" in\n\s*dev\) CHANNEL="dev" ;;\n)\s*\*\).*?;;\n(\s*esac)',
            r'\1  *) CHANNEL="latest" ;;\n\2',
        )

    run = _run_job(tmp_path, lane="latest", mutate_guard=fall_through)
    assert run.failed is None, run.output
    assert any(f"{LATEST}/zak-code-main.tar.gz" in call for call in run.calls), run.calls


@needs_bash
def test_dropping_the_main_only_check_would_be_caught(tmp_path: Path) -> None:
    def drop(script: str) -> str:
        return _mutate(script, r'\[ "\$WORKFLOW_REF" = "refs/heads/main" \] \\\n[^\n]*\n', "")

    run = _run_job(tmp_path, workflow_ref="refs/heads/" + CANDIDATE, mutate_guard=drop)
    assert run.failed is None, run.output
    assert len(run.calls) == 1


@needs_bash
def test_dropping_the_ref_check_would_be_caught(tmp_path: Path) -> None:
    def drop(script: str) -> str:
        return _mutate(script, r'case "\$CANDIDATE" in.*?\nesac\n', "")

    run = _run_job(tmp_path, ref="a,b", mutate_guard=drop)
    assert run.failed is None, run.output
    assert "branch=a,b," in run.calls[0]


# --- the shape of the workflow -------------------------------------------------------------------


def test_a_dispatch_can_open_the_dev_lane_and_no_other() -> None:
    inputs = _triggers(DEV)["workflow_dispatch"]["inputs"]
    assert inputs["lane"]["type"] == "choice"
    assert inputs["lane"]["options"] == ["dev"]
    assert inputs["lane"]["required"] is True
    assert inputs["ref"]["required"] is True


def test_the_dev_workflow_is_dispatch_only_and_ci_has_no_dispatch() -> None:
    assert set(_triggers(DEV)) == {"workflow_dispatch"}
    assert "workflow_dispatch" not in _triggers(CI)


def test_the_dev_workflow_shares_no_job_name_with_ci() -> None:
    """A skipped job posts a check run under its name on the ref a dispatch targets."""

    def names(path: Path) -> set[str]:
        jobs = _load(path)["jobs"]
        return {*jobs, *(job["name"] for job in jobs.values() if "name" in job)}

    assert names(DEV) & names(CI) == set()


def test_every_step_of_the_dev_job_runs_on_every_dispatch() -> None:
    job = _load(DEV)["jobs"][JOB]
    assert "if" not in job
    assert all("if" not in step for step in job["steps"])


def test_the_checkout_takes_the_ref_the_dispatch_names() -> None:
    checkouts = [s for s in _steps(DEV) if s.get("uses", "").startswith("actions/checkout@")]
    assert [s["with"]["ref"] for s in checkouts] == ["${{ inputs.ref }}"]


def test_no_expression_is_interpolated_into_a_dev_job_script() -> None:
    for step in _steps(DEV):
        assert "${{" not in step.get("run", ""), _label(step)


def test_the_production_pointer_is_written_by_the_production_job_alone() -> None:
    for path in sorted(WORKFLOWS.glob("*.yml")):
        for job_id, job in _load(path)["jobs"].items():
            for step in job["steps"]:
                if LATEST in step.get("run", ""):
                    assert (path.name, job_id) == ("ci.yml", "publish-sidecar"), (path.name, job_id)


def test_the_production_publish_is_gated_and_names_its_lane_in_the_metadata() -> None:
    jobs = _load(CI)["jobs"]
    assert jobs["publish-sidecar"]["if"] == PROD_GATE
    publish = [s for s in _steps(CI, "publish-sidecar") if s.get("name", "").startswith("Publish")]
    assert len(publish) == 1
    assert "branch=main,lane=prod" in publish[0]["run"]


# --- every step that writes outside the runner ---------------------------------------------------

# What a step can do to the world beyond its own runner. A command is matched on its script with
# the line continuations joined, an action on its `uses:` and `with:` text.
COMMAND_WRITES = (
    r"aws\s+s3\s+(?:cp|sync|mv|rm)\b",
    r"aws\s+s3api\s+(?:put|delete|copy)",
    r"aws\s+(?:sesv2|ses)\s+send",
    r"aws\s+sns\s+publish",
    r"aws\s+dynamodb\s+(?:put|update|delete|batch-write)",
    r"git\s+push\b",
    r"gh\s+(?:release|api|issue|workflow\s+run|pr\s+(?:create|merge|comment|edit|close))\b",
    r"curl\b[^\n]*\s(?:-X|--request|-d|--data\S*|-F|--form|-T|--upload-file)\b",
    r"docker\s+(?:push|login)\b",
    r"twine\s+upload\b",
    r"uv\s+publish\b",
    r"npm\s+publish\b",
)
ACTION_WRITES = (
    r"actions/upload-artifact",
    r"actions/cache",
    r"enable-cache: true",
    r"gh-release|gh-pages|pypi-publish|build-push-action|login-action",
)

CACHE = "the uv cache, in GitHub's cache store, scoped to the ref that saved it; not a channel"
ARTIFACT = "GitHub's artifact store, for this run only; not a channel"
MAIL = "a notification, not an artifact"
MAIL_ON_PUSH = MAIL + "; a failed push to main only"
MAIL_ON_SCHEDULE = MAIL + "; a failed scheduled run"

# (workflow file, job, step) -> why it may write outside the runner.
KNOWN_WRITERS = {
    ("bench.yml", "bench", "Install uv"): CACHE,
    ("bench.yml", "bench", "Upload results JSON"): ARTIFACT,
    ("ci.yml", "verify", "Install uv"): CACHE,
    ("ci.yml", "verify", "Upload coverage report"): ARTIFACT,
    ("ci.yml", "update-e2e", "Install uv"): CACHE,
    ("ci.yml", "cockpit-e2e", "Install uv"): CACHE,
    (
        "ci.yml",
        "publish-sidecar",
        "Publish to S3 (immutable <sha> + latest pointer)",
    ): "the production channel and a sha-keyed copy; push to main only, lane=prod in the metadata",
    ("ci.yml", "alert-on-failure", "Send SES alert"): MAIL_ON_PUSH,
    (
        "dev-channel.yml",
        JOB,
        "Build the candidate and publish it to the dev channel",
    ): "the dev channel, lane=dev in the metadata; a dispatch on main only, the lane refused first",
    ("say-watch-smoke.yml", "alert-on-failure", "Send SES alert"): MAIL_ON_SCHEDULE,
}


def _writes(step: dict[str, Any]) -> bool:
    script = re.sub(r"\\\n\s*", " ", str(step.get("run", "")))
    with_text = "\n".join(f"{k}: {str(v).lower()}" for k, v in (step.get("with") or {}).items())
    action = f"{step.get('uses', '')}\n{with_text}"
    return any(re.search(p, script) for p in COMMAND_WRITES) or any(
        re.search(p, action) for p in ACTION_WRITES
    )


def _writers() -> set[tuple[str, str, str]]:
    return {
        (path.name, job_id, _label(step))
        for path in sorted(WORKFLOWS.glob("*.yml"))
        for job_id, job in _load(path)["jobs"].items()
        for step in job["steps"]
        if _writes(step)
    }


def test_the_steps_that_write_outside_the_runner_are_a_fixed_set() -> None:
    found = _writers()
    assert found == set(KNOWN_WRITERS), (
        f"a step writes outside the runner that this list does not hold: "
        f"{sorted(found - set(KNOWN_WRITERS))}; listed here but gone: "
        f"{sorted(set(KNOWN_WRITERS) - found)}. List a new writer here with its reason, and say "
        "what tells a consumer which lane produced what it wrote, before it is added to a workflow."
    )


def test_the_alert_job_runs_for_a_push_to_main_only() -> None:
    gate = _load(CI)["jobs"]["alert-on-failure"]["if"]
    assert "github.event_name == 'push'" in gate
    assert "github.ref == 'refs/heads/main'" in gate


@pytest.mark.parametrize(
    "script",
    [
        "aws s3 cp a s3://b/k",
        "aws s3 sync . s3://b/",
        "aws s3 rm s3://b/k",
        "aws s3api put-object --bucket b --key k",
        "aws sesv2 send-email --cli-input-json file://m.json",
        "aws sns publish --topic-arn t --message m",
        "aws dynamodb put-item --table-name t --item '{}'",
        "git push origin HEAD:main",
        "gh release upload v1 dist/x.tgz",
        "gh api -X POST repos/o/r/issues",
        "gh pr comment 1 --body hi",
        "curl -X POST https://example.invalid/hook",
        "curl --data @body.json https://example.invalid/hook",
        "curl -sS \\\n  -T build.tgz https://example.invalid/up",
        "docker push registry.example.invalid/img",
        "twine upload dist/*",
        "uv publish",
        "npm publish",
    ],
)
def test_the_audit_scan_sees_each_write_shape(script: str) -> None:
    assert _writes({"run": script})


@pytest.mark.parametrize(
    "step",
    [
        {"run": "aws sts get-caller-identity"},
        {"run": "git archive --format=tar.gz -o x.tgz HEAD"},
        {"run": "curl -sSL https://example.invalid/x -o x"},
        {"run": "tar tzf x.tgz > list.txt"},
        {"uses": "actions/checkout@v6"},
        {"uses": "astral-sh/setup-uv@v7", "with": {"enable-cache": False}},
    ],
)
def test_the_audit_scan_passes_a_step_that_writes_nothing_outside_the_runner(
    step: dict[str, Any],
) -> None:
    assert not _writes(step)


@pytest.mark.parametrize(
    "step",
    [
        {"uses": "actions/upload-artifact@v4"},
        {"uses": "actions/cache@v4"},
        {"uses": "astral-sh/setup-uv@v7", "with": {"enable-cache": True}},
        {"uses": "softprops/action-gh-release@v2"},
    ],
)
def test_the_audit_scan_sees_an_action_that_writes(step: dict[str, Any]) -> None:
    assert _writes(step)
