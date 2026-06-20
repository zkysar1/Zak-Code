"""Seam B: on a stalled turn, run N isolated attempts and adopt the first that VERIFIES.

The risky step, done safely. When a turn stalls AND ``best_of_attempts > 1`` AND a verifier is
configured, fan out fresh attempts in ISOLATED copies of the workspace SOURCE, run the verifier in
each, and adopt the FIRST that passes by applying ONLY ITS DIFF back — never a blind overwrite, so a
file the attempt didn't touch is left exactly as it was. Source-only + size-capped (a workspace too
big to copy is skipped, not thrashed), and OFF unless opted in. The small-model bet on exactly the
hard/stalled tasks the suite measurement said it pays off on.

The file-manipulation helpers (:func:`copy_source`, :func:`changed_paths`, :func:`apply_diff`) are
pure and unit-tested; the orchestration that spawns isolated siblings lives in :func:`run_best_of_
attempts` and is gated OFF by default.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from zakcode.quality import best_attempt

#: Directories/files never copied into an isolated attempt (vcs, deps, caches, build output): they
#: bloat the copy and are never "the work". Source-only isolation keeps the copy small and the size
#: cap meaningful.
_SKIP = frozenset(
    {
        ".git",
        ".hg",
        ".svn",
        ".venv",
        "venv",
        "env",
        "node_modules",
        "__pycache__",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".tox",
        "dist",
        "build",
        ".idea",
        ".vscode",
        ".next",
        "target",
        ".gradle",
        ".DS_Store",
    }
)

#: Default ceiling on the source bytes copied per attempt — above this the workspace is too big to
#: isolate by copy, so the retry is SKIPPED rather than thrash the disk.
_MAX_SOURCE_BYTES = 64 * 1024 * 1024  # 64 MiB


def _skipped(rel: Path) -> bool:
    return any(part in _SKIP or part.endswith(".pyc") for part in rel.parts)


def _source_files(root: Path) -> list[Path]:
    """Relative paths of the source files under ``root`` (skipping vcs/deps/caches/build)."""
    return [
        p.relative_to(root)
        for p in root.rglob("*")
        if p.is_file() and not _skipped(p.relative_to(root))
    ]


def copy_source(src: Path, dest: Path, *, max_bytes: int = _MAX_SOURCE_BYTES) -> bool:
    """Copy ``src``'s SOURCE files (skipping vcs/deps/caches) into ``dest``. Returns ``False`` and
    copies nothing if the source exceeds ``max_bytes`` — too big to isolate by copy, so skip."""
    files = _source_files(src)
    if sum((src / rel).stat().st_size for rel in files) > max_bytes:
        return False
    for rel in files:
        target = dest / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src / rel, target)
    return True


def changed_paths(attempt: Path, snapshot: Path) -> tuple[list[Path], list[Path]]:
    """``(changed_or_added, deleted)`` source paths in ``attempt`` vs ``snapshot`` — the attempt's
    DIFF. Compared by CONTENT, so an untouched file is never in the change set."""
    after = set(_source_files(attempt))
    before = set(_source_files(snapshot))
    changed = [
        rel
        for rel in sorted(after)
        if rel not in before or (attempt / rel).read_bytes() != (snapshot / rel).read_bytes()
    ]
    deleted = [rel for rel in sorted(before) if rel not in after]
    return changed, deleted


def apply_diff(changed: list[Path], deleted: list[Path], attempt: Path, workspace: Path) -> None:
    """Apply ONLY the diff to ``workspace``: copy each ``changed`` file from ``attempt``, remove
    each ``deleted`` file. Untouched files are left as they were — never a blind overwrite."""
    for rel in changed:
        target = workspace / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(attempt / rel, target)
    for rel in deleted:
        (workspace / rel).unlink(missing_ok=True)


def _verify(command: str, cwd: Path, *, timeout: int = 120) -> bool:
    """Run the verifier ``command`` in ``cwd``; ``True`` iff it exits 0. Never raises."""
    try:
        proc = subprocess.run(  # noqa: S602 — the verifier is operator-configured (verify_command)
            command, shell=True, cwd=str(cwd), capture_output=True, timeout=timeout
        )
        return proc.returncode == 0
    except (subprocess.SubprocessError, OSError):
        return False


async def run_best_of_attempts(agent: Any, user_text: str, original: Any) -> Any:
    """Seam B: on a stalled turn, run up to ``best_of_attempts`` isolated attempts and adopt (diff-
    apply) the first that VERIFIES; else return ``original``. Off unless ``best_of_attempts > 1``
    AND ``verify_command`` is set. Isolated siblings are built via ``type(agent)`` and run headless;
    only the winner's DIFF is applied back. Temp copies are always cleaned up.
    """
    settings = agent.settings
    n = settings.best_of_attempts
    command = settings.verify_command
    if n <= 1 or not command:
        return original

    workspace = Path(settings.workspace_root)
    snapshot = Path(tempfile.mkdtemp(prefix="zb-snap-"))
    attempts: list[Path] = []
    try:
        if not copy_source(workspace, snapshot):
            return original  # workspace too big to isolate safely — skip the retry

        async def run(i: int) -> tuple[Path, Any]:
            attempt = Path(tempfile.mkdtemp(prefix=f"zb-att{i}-"))
            attempts.append(attempt)
            copy_source(snapshot, attempt)  # every attempt starts from the same snapshot
            sibling = type(agent)(
                settings=settings.model_copy(
                    update={
                        "workspace_root": str(attempt),
                        "best_of_attempts": 1,  # no recursion
                        "permission_mode": "autonomous",  # headless attempt; isolated copy
                    }
                )
            )
            result = await sibling.arun_turn(user_text)
            return attempt, result

        async def verify(handle: tuple[Path, Any]) -> bool:
            return _verify(command, handle[0])

        winner = await best_attempt(n, run=run, verify=verify)
        if winner is None:
            return original
        attempt, result = winner
        changed, deleted = changed_paths(attempt, snapshot)
        apply_diff(changed, deleted, attempt, workspace)
        return result  # the verified attempt's (completed) result
    finally:
        shutil.rmtree(snapshot, ignore_errors=True)
        for attempt in attempts:
            shutil.rmtree(attempt, ignore_errors=True)


__all__ = ["apply_diff", "changed_paths", "copy_source", "run_best_of_attempts"]
