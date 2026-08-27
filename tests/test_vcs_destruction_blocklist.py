"""VCS-internals destruction joins the never-waivable blocklist (ADR-0038).

Field incident 2026-08-27: an unattended model ran ``rm -f .git/objects/pack/*``,
``rm -rf .git/refs/mind`` and re-cloned the repository chasing a phantom stale ref. The
recursive-root ``rm`` check exempts relative paths and the destructive-git entry knew only
force-push and hard-reset, so all of it was auto-allowed.
"""

from __future__ import annotations

import pytest

from zakcode.permissions import scan_command_danger


@pytest.mark.parametrize(
    "cmd",
    [
        "rm -f .git/objects/pack/*",
        "rm -rf .git/refs/mind",
        "rm -rf .git",
        "rm -rf /opt/coach-mind/.git/refs/mind && echo done",
        'rm -rf "$repo/.git"',
        "rm -rf ./.git",
        "mv .git .git.bak",
        "shred -u .git/packed-refs",
        "echo x > .git/HEAD",
        "cat pack >> .git/packed-refs",
        "git gc --prune=now",
        "git prune",
        "git reflog expire --expire=now --all",
        "git reflog expire --expire-unreachable=now --all && git gc --prune=now",
    ],
)
def test_vcs_destruction_is_caught(cmd: str) -> None:
    assert scan_command_danger(cmd) is not None


@pytest.mark.parametrize(
    "cmd",
    [
        "rm .gitignore",
        "rm -rf .github/workflows",
        "rm -rf build && git status",
        "rm -rf ./build/.gitkeep",
        "git rm --cached foo.py",
        "git prune-packed",
        "git gc",
        "git gc --aggressive",
        "git reflog expire --expire=30.days.ago --all",
        "git update-ref -d refs/mind/claim/x/y",
        "git fetch origin '+refs/mind/claim/*:refs/mind/claim/*'",
        "cat .git/config",
        "ls .git/refs",
        "echo x > .gitignore",
        "git for-each-ref refs/mind/",
    ],
)
def test_ordinary_git_and_rm_pass(cmd: str) -> None:
    assert scan_command_danger(cmd) is None
