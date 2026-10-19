"""Tests for ``write_text_atomic`` — publish a text file whole, or not at all (ADR-0318).

The concurrency half lives in ``test_server_staged_write_race.py``, where it is driven through
the two routes that depend on it. What is pinned here is the failure contract the routes
cannot reach: a write that dies midway must not leave its temp file behind or touch the file
it was meant to replace.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from zakcode.session import atomic_write
from zakcode.session.atomic_write import write_text_atomic


@pytest.fixture(autouse=True)
def _no_pause_between_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    """The retry tests count calls; none of them should wait out the pauses between retries."""
    monkeypatch.setattr(atomic_write, "_RENAME_PAUSE_S", 0)


class _RefusingReplace:
    """Stands in for ``os.replace``: refuses the first ``refusals`` calls, then really renames."""

    def __init__(self, refusals: int) -> None:
        self.refusals = refusals
        self.calls = 0
        self._real = os.replace

    def __call__(self, src: str | os.PathLike[str], dst: str | os.PathLike[str]) -> None:
        self.calls += 1
        if self.calls <= self.refusals:
            raise PermissionError(13, "Access is denied")
        self._real(src, dst)


def test_write_text_atomic_publishes_and_replaces(tmp_path: Path) -> None:
    target = tmp_path / ".nudge"
    write_text_atomic(target, "first\n")
    assert target.read_text(encoding="utf-8") == "first\n"

    write_text_atomic(target, "second\n")
    assert target.read_text(encoding="utf-8") == "second\n"
    # Only the published file is left: no temp survives a successful write.
    assert sorted(entry.name for entry in tmp_path.iterdir()) == [".nudge"]


def test_write_text_atomic_failed_rename_cleans_up_and_keeps_the_old_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / ".observation"
    target.write_text("old\n", encoding="utf-8")

    def refuse(src: object, dst: object) -> None:  # noqa: ARG001
        raise PermissionError("rename refused")

    monkeypatch.setattr(os, "replace", refuse)
    with pytest.raises(PermissionError, match="rename refused"):
        write_text_atomic(target, "new\n")

    assert target.read_text(encoding="utf-8") == "old\n"
    assert sorted(entry.name for entry in tmp_path.iterdir()) == [".observation"]


def test_write_text_atomic_retries_a_refused_rename_where_the_refusal_is_transient(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Windows refuses a rename onto a file another rename holds open, then lets it through.
    target = tmp_path / ".observation"
    replace = _RefusingReplace(refusals=3)
    monkeypatch.setattr(atomic_write, "_RETRY_RENAME_REFUSAL", True)
    monkeypatch.setattr(os, "replace", replace)

    write_text_atomic(target, "new\n")

    assert replace.calls == 4
    assert target.read_text(encoding="utf-8") == "new\n"
    assert sorted(entry.name for entry in tmp_path.iterdir()) == [".observation"]


@pytest.mark.parametrize(
    ("retry", "expected_calls"), [(False, 1), (True, atomic_write._RENAME_ATTEMPTS)]
)
def test_write_text_atomic_raises_a_refusal_that_does_not_clear(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, retry: bool, expected_calls: int
) -> None:
    # Where the refusal is final (POSIX) it is raised on the first call; where it may be
    # transient it is raised once the bound is spent. Either way nothing is left behind.
    target = tmp_path / ".observation"
    target.write_text("old\n", encoding="utf-8")
    replace = _RefusingReplace(refusals=10**6)
    monkeypatch.setattr(atomic_write, "_RETRY_RENAME_REFUSAL", retry)
    monkeypatch.setattr(os, "replace", replace)

    with pytest.raises(PermissionError):
        write_text_atomic(target, "new\n")

    assert replace.calls == expected_calls
    assert target.read_text(encoding="utf-8") == "old\n"
    assert sorted(entry.name for entry in tmp_path.iterdir()) == [".observation"]
