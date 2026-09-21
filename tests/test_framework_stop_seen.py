"""g-373-120 — the cap watcher must tell "never saw the stop" from "saw it, still working".

THE DEFECT. The conductor's grace loop had exactly ONE observable,
``framework_stop_complete``, which is a COMPLETION signal. So a raise the mind never
received and a raise the mind is actively working on were indistinguishable, and BOTH
burned the entire consolidation reserve in silence. Measured on a clean dev vessel
(g-373-94, env fileworld-smoke): 8 framework_stop raises, 5 endings on the overrun
interrupt, 1 graceful-stop invocation, **0** ``framework_stop_complete`` sign-offs — so
on that vessel the poll never had anything to find, and serve.log could not say why.

``framework_stop_seen`` supplies the missing SEEN half by reading the stamp the
framework's PreToolUse hook already writes (``bash-agent-inject.py``
``_maybe_surface_stop``). The write side was already built and tested in the Mind repo;
what did not exist was a reader, and a written-down contract for what it keys on.

THE THREE-VALUED RESULT IS THE POINT. ``None`` (unknowable) must never collapse into
``False`` (no stamp), because ``False`` is itself FOUR-WAY AMBIGUOUS — no Bash tool call
since the raise, a worker Body, a reader session, or the hook failing open. That is why
every caller reports rather than acts, and why ``test_absence_is_reported_not_acted_on``
pins the report-only posture at the call site.
"""

from __future__ import annotations

import os
from pathlib import Path

from zakcode.session.framework_stop import (
    SIDECAR_RAISE_MARKER,
    SIGNAL_SET_SCRIPT,
    STOP_SURFACE_DIRNAME,
    STOP_SURFACE_INTERVAL_S,
    framework_session_dir,
    framework_stop_seen,
    sidecar_raise_time,
    stop_surface_seen_at,
)

AGENT = "bravo"
RAISED_AT = "2026-09-17T20:29:29Z"


def _signed_raise(root: Path, raised_at: str = RAISED_AT) -> float:
    """A signed stop pair on disk; returns the raise time in epoch seconds."""
    session_dir = framework_session_dir(root, AGENT)
    session_dir.mkdir(parents=True, exist_ok=True)
    (session_dir / "stop-target-mode").write_text("assistant", encoding="utf-8")
    (session_dir / "stop-requested").write_text(
        f"{SIDECAR_RAISE_MARKER}\nraised_at: {raised_at}\n", encoding="utf-8"
    )
    raised = sidecar_raise_time(root, AGENT)
    assert raised is not None, "fixture must produce a parseable signed raise"
    return raised


def _stamp(root: Path, sid: str, mtime: float) -> Path:
    """One zero-byte stop-surface stamp at an exact mtime, as the hook writes it."""
    stamp_dir = root / STOP_SURFACE_DIRNAME
    stamp_dir.mkdir(parents=True, exist_ok=True)
    stamp = stamp_dir / sid
    stamp.touch()
    os.utime(stamp, (mtime, mtime))
    return stamp


# ───────────────────────────── the stamp reader ─────────────────────────────


def test_no_stamp_directory_reads_as_none_not_as_zero(tmp_path: Path) -> None:
    """Fail-open: an absent log dir is UNKNOWN, never a timestamp of 0.

    A 0.0 here would compare as "older than every raise" and manufacture a confident
    never-seen report on a vessel whose hook simply has not run yet.
    """
    assert stop_surface_seen_at(tmp_path) is None


def test_empty_stamp_directory_reads_as_none(tmp_path: Path) -> None:
    (tmp_path / STOP_SURFACE_DIRNAME).mkdir(parents=True)
    assert stop_surface_seen_at(tmp_path) is None


def test_reader_takes_the_newest_stamp_across_sessions(tmp_path: Path) -> None:
    """The conductor never learns the harness session id, so it reads the newest.

    Pinned because a per-session read is impossible here and a FIRST-match read would
    silently report the oldest session's stamp on any vessel with more than one.
    """
    _stamp(tmp_path, "sid-old", 1_000.0)
    _stamp(tmp_path, "sid-new", 5_000.0)
    _stamp(tmp_path, "nosid", 3_000.0)
    assert stop_surface_seen_at(tmp_path) == 5_000.0


def test_reader_ignores_subdirectories(tmp_path: Path) -> None:
    """Only files are stamps; a stray dir must not raise or count."""
    stamp_dir = tmp_path / STOP_SURFACE_DIRNAME
    stamp_dir.mkdir(parents=True)
    (stamp_dir / "a-directory").mkdir()
    assert stop_surface_seen_at(tmp_path) is None


# ───────────────────────────── the SEEN predicate ────────────────────────────


def test_unknowable_when_no_signed_raise(tmp_path: Path) -> None:
    """None, NOT False — nothing was asked, so nothing can have gone unseen.

    The load-bearing distinction: collapsing this into False would report "the mind never
    saw the stop" on every run that never raised one.
    """
    _stamp(tmp_path, "sid-a", 10_000.0)
    assert framework_stop_seen(tmp_path, AGENT) is None


def test_unknowable_without_an_agent(tmp_path: Path) -> None:
    _signed_raise(tmp_path)
    assert framework_stop_seen(tmp_path, "") is None


def test_raised_and_never_stamped_is_false(tmp_path: Path) -> None:
    """The escalation case this goal exists to make visible."""
    _signed_raise(tmp_path)
    assert framework_stop_seen(tmp_path, AGENT) is False


def test_stamp_after_the_raise_is_seen(tmp_path: Path) -> None:
    raised = _signed_raise(tmp_path)
    _stamp(tmp_path, "sid-a", raised + 5.0)
    assert framework_stop_seen(tmp_path, AGENT) is True


def test_a_stamp_stale_within_the_throttle_still_reads_as_seen(tmp_path: Path) -> None:
    """THE THROTTLE IS PART OF THE COMPARISON — the subtlest rule in the contract.

    The hook returns early WITHOUT re-stamping inside ``STOP_SURFACE_INTERVAL_S``, so the
    mtime is a LAST-SURFACED time with that much granularity. A mind that surfaced the
    stop 1s before the raise landed and then hit the throttle is SEEN, and a naive
    ``stamp >= raised`` would report it as never-seen — a false alarm on exactly the
    healthy path.
    """
    raised = _signed_raise(tmp_path)
    _stamp(tmp_path, "sid-a", raised - (STOP_SURFACE_INTERVAL_S / 2))
    assert framework_stop_seen(tmp_path, AGENT) is True


def test_a_stamp_older_than_the_throttle_is_not_seen(tmp_path: Path) -> None:
    """The other side of the same boundary: a PREVIOUS run's stamp is not this raise.

    Without this, any vessel whose log dir carried an old stamp would report every
    subsequent raise as seen — the failure that would make the new signal useless while
    looking healthy.
    """
    raised = _signed_raise(tmp_path)
    _stamp(tmp_path, "sid-a", raised - (STOP_SURFACE_INTERVAL_S * 3))
    assert framework_stop_seen(tmp_path, AGENT) is False


def test_the_newest_stamp_decides_even_when_an_older_one_is_stale(tmp_path: Path) -> None:
    raised = _signed_raise(tmp_path)
    _stamp(tmp_path, "sid-stale", raised - 600.0)
    _stamp(tmp_path, "sid-live", raised + 1.0)
    assert framework_stop_seen(tmp_path, AGENT) is True


# ──────────────────── the root is the one that already works ─────────────────


def test_the_stamp_root_is_the_root_the_raise_already_resolves_against(tmp_path: Path) -> None:
    """REACHABILITY, by code-trace rather than assumption (guard-3476).

    The stamp is NOT under the env's logs dir — it is under the mind repo root. This pins
    that the root used here is the SAME ``workspace_root`` that ``SIGNAL_SET_SCRIPT`` is
    resolved against, whose ``is_file()`` check must already pass for any raise to happen
    at all. So wherever this sidecar can raise a stop, it can read this stamp; if someone
    later re-roots one of the two, this fails instead of silently reading nothing forever.
    """
    signal_script = tmp_path / SIGNAL_SET_SCRIPT
    signal_script.parent.mkdir(parents=True, exist_ok=True)
    signal_script.write_text("#!/usr/bin/env bash\n", encoding="utf-8")

    stamp = _stamp(tmp_path, "sid-a", 42.0)

    # Both resolve against the same root, and the stamp sits beside core/scripts/.
    assert stamp.is_relative_to(tmp_path)
    assert signal_script.is_relative_to(tmp_path)
    assert STOP_SURFACE_DIRNAME.parts[0] == SIGNAL_SET_SCRIPT.parts[0] == "core"
    assert stop_surface_seen_at(tmp_path) == 42.0


def test_absence_is_reported_not_acted_on() -> None:
    """REPORT-ONLY posture, pinned at the call site (outcome 2).

    ``_retire_unconsumed_framework_stop`` must LOG the seen-vs-done distinction and then
    retire the pair exactly as before. Two things must never appear: an early ending keyed
    on a four-way-ambiguous absence, and a new fixed reserve number as the remedy (a bet
    against an unbounded tail, and it moves the raise earlier into /start's own clear).
    """
    from zakcode.server import app

    source = Path(app.__file__).read_text(encoding="utf-8")
    body = source.split("def _retire_unconsumed_framework_stop", 1)[1]
    body = body.split("\n    async def ", 1)[0]

    assert "framework_stop_seen(" in body, "the SEEN half must be read here"
    assert "abandon_framework_stop(" in body, "retirement behaviour must be unchanged"
    assert body.count("logger.warning") == 3, "all three seen-states must be reported"
    # No new ending, and no new reserve arithmetic, introduced by the report.
    for forbidden in ("request_interrupt(", "run_stopping.set(", "effective_reserve ="):
        assert forbidden not in body, f"report-only: {forbidden} must not appear here"
