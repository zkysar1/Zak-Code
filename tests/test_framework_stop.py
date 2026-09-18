"""Unit tests for the framework graceful-stop contract (``zakcode.session.framework_stop``).

The invariant these pin is an ORDER, and an order is exactly the kind of thing a
comment cannot enforce: ``stop-target-mode`` must be on disk BEFORE ``stop-requested``,
because the framework's stop handler reads the target mode with no fallback. So the
fake signal-setter below asserts the mode file from INSIDE the script — the assertion
runs at the only moment that can distinguish the two orders.
"""

from __future__ import annotations

import os
import stat
from pathlib import Path

import pytest

from zakcode.session import framework_stop
from zakcode.session.framework_stop import (
    AGENT_MODE_FILENAME,
    DEFAULT_STOP_TARGET_MODE,
    SIDECAR_RAISE_MARKER,
    SIGNAL_SET_SCRIPT,
    STOP_CHECKPOINT_FILENAME,
    STOP_TARGET_MODE_FILENAME,
    abandon_framework_stop,
    framework_session_dir,
    framework_stop_complete,
    request_framework_stop,
    retire_expired_sidecar_stop,
    sidecar_raise_time,
)

AGENT = "alpha"


def _plant_signal_setter(root: Path, body: str) -> Path:
    """Plant a stand-in for the framework's ``session-signal-set.sh`` at the real path."""
    script = root / SIGNAL_SET_SCRIPT
    script.parent.mkdir(parents=True, exist_ok=True)
    script.write_text(body, encoding="utf-8")
    script.chmod(script.stat().st_mode | stat.S_IXUSR)
    return script


def _real_setter_body() -> str:
    """A faithful stand-in: touches the marker the real script touches, nothing else."""
    return (
        "#!/usr/bin/env bash\n"
        "set -eu\n"
        'dir="agents/${AYOAI_AGENT}/session"\n'
        # The order assertion, made where it is decidable: the real handler would read
        # this file with no fallback, so its absence here is the bug we are pinning.
        '[ -f "$dir/stop-target-mode" ] '
        '|| { echo "target mode absent at signal time" >&2; exit 9; }\n'
        'mkdir -p "$dir"\n'
        'touch "$dir/$1"\n'
    )


def test_raises_the_stop_and_writes_target_mode_first(tmp_path: Path) -> None:
    _plant_signal_setter(tmp_path, _real_setter_body())

    assert request_framework_stop(tmp_path, AGENT) is True

    session = framework_session_dir(tmp_path, AGENT)
    assert (session / "stop-requested").exists()
    # No trailing newline — byte-identical to the framework's own callers.
    assert (session / "stop-target-mode").read_text(encoding="utf-8") == DEFAULT_STOP_TARGET_MODE


def test_signal_lands_agent_level_not_per_session(tmp_path: Path) -> None:
    """``sessions/<SID>/stop-requested`` is a turn-end permit; it does NOT stop the loop."""
    _plant_signal_setter(tmp_path, _real_setter_body())

    assert request_framework_stop(tmp_path, AGENT) is True

    assert (tmp_path / "agents" / AGENT / "session" / "stop-requested").exists()
    assert not (tmp_path / "agents" / AGENT / "sessions").exists()


def test_is_idempotent(tmp_path: Path) -> None:
    """A second /run/stop, or a cap landing after an explicit stop, is still one stop."""
    _plant_signal_setter(tmp_path, _real_setter_body())
    assert request_framework_stop(tmp_path, AGENT) is True

    # A setter that would fail if it ran again proves the second call short-circuits.
    _plant_signal_setter(tmp_path, "#!/usr/bin/env bash\nexit 1\n")
    assert request_framework_stop(tmp_path, AGENT) is True


def test_reverts_target_mode_when_the_setter_fails(tmp_path: Path) -> None:
    """A dangling target mode tells the next reader a stop is happening when none is."""
    _plant_signal_setter(tmp_path, "#!/usr/bin/env bash\necho boom >&2\nexit 3\n")

    assert request_framework_stop(tmp_path, AGENT) is False

    session = framework_session_dir(tmp_path, AGENT)
    assert not (session / "stop-target-mode").exists()
    assert not (session / "stop-requested").exists()


def test_zero_exit_over_an_absent_marker_is_a_failure(tmp_path: Path) -> None:
    """Verify by READING: a zero exit that armed nothing must not buy a grace period."""
    _plant_signal_setter(tmp_path, "#!/usr/bin/env bash\nexit 0\n")

    assert request_framework_stop(tmp_path, AGENT) is False

    session = framework_session_dir(tmp_path, AGENT)
    assert not (session / "stop-target-mode").exists()


def test_absent_script_is_not_an_error_and_writes_nothing(tmp_path: Path) -> None:
    """A non-framework workspace simply keeps the ending the conductor already had."""
    assert request_framework_stop(tmp_path, AGENT) is False
    assert not (tmp_path / "agents").exists()


def test_binds_the_agent_under_both_names(tmp_path: Path) -> None:
    """The seed's scripts read MIND_AGENT; its daemon-backed steps read AYOAI_AGENT."""
    _plant_signal_setter(
        tmp_path,
        "#!/usr/bin/env bash\n"
        "set -eu\n"
        'dir="agents/${MIND_AGENT}/session"\n'
        '[ "${AYOAI_AGENT}" = "${MIND_AGENT}" ] || exit 8\n'
        'mkdir -p "$dir"\n'
        'touch "$dir/$1"\n',
    )

    assert request_framework_stop(tmp_path, AGENT) is True
    assert (framework_session_dir(tmp_path, AGENT) / "stop-requested").exists()


def test_target_mode_is_overridable(tmp_path: Path) -> None:
    _plant_signal_setter(tmp_path, _real_setter_body())

    assert request_framework_stop(tmp_path, AGENT, target_mode="reader") is True

    assert (framework_session_dir(tmp_path, AGENT) / "stop-target-mode").read_text(
        encoding="utf-8"
    ) == "reader"


def test_order_assertion_actually_fires(tmp_path: Path) -> None:
    """Positive control: the order check in _real_setter_body() can FAIL.

    Without this, a setter that never looked at the mode file would pass every test
    above and the order invariant would be pinned by nothing.
    """
    script = _plant_signal_setter(tmp_path, _real_setter_body())
    session = framework_session_dir(tmp_path, AGENT)
    session.mkdir(parents=True, exist_ok=True)

    import subprocess

    from zakcode._subprocess import find_bash

    bash = find_bash()
    assert bash, "no bash interpreter on this box"
    env = dict(os.environ, AYOAI_AGENT=AGENT, MIND_AGENT=AGENT)
    completed = subprocess.run(
        [bash, str(script), "stop-requested"],
        cwd=str(tmp_path),
        env=env,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 9, completed.stderr.decode("utf-8", "replace")


# ── g-373-92: the orphaned pair must not outlive its run ─────────────────────


def _raise(tmp_path: Path):
    _plant_signal_setter(tmp_path, _real_setter_body())
    assert request_framework_stop(tmp_path, AGENT) is True
    return framework_session_dir(tmp_path, AGENT)


def test_abandon_retires_a_pair_no_mind_consumed(tmp_path: Path) -> None:
    """The measured defect: the pair survives the run and poisons the next boot."""
    d = _raise(tmp_path)
    assert (d / "stop-requested").exists() and (d / "stop-target-mode").exists()
    assert abandon_framework_stop(tmp_path, AGENT) is True
    assert not (d / "stop-requested").exists(), "the next vessel boot reads this as live"
    assert not (d / "stop-target-mode").exists(), "a dangling target mode lies to the next reader"


def test_abandon_refuses_when_a_stop_actually_started(tmp_path: Path) -> None:
    """stop-checkpoint.json means a mind DID consume the ask -- not an orphan.

    Clearing it would destroy the framework's own interrupted-stop resume path, which is
    the ONLY thing that can finish a stop that got underway.
    """
    d = _raise(tmp_path)
    (d / STOP_CHECKPOINT_FILENAME).write_text("{}", encoding="utf-8")
    assert abandon_framework_stop(tmp_path, AGENT) is False
    assert (d / "stop-requested").exists(), "a stop in progress must keep its ask"
    assert (d / STOP_CHECKPOINT_FILENAME).exists(), "never touch the framework's own record"


def test_abandon_is_idempotent_and_quiet_on_a_clean_dir(tmp_path: Path) -> None:
    d = _raise(tmp_path)
    assert abandon_framework_stop(tmp_path, AGENT) is True
    assert abandon_framework_stop(tmp_path, AGENT) is True
    assert not (d / "stop-requested").exists()


def test_abandon_clears_a_dangling_target_mode_alone(tmp_path: Path) -> None:
    """The mid-abandon interrupt leaves exactly this, so the next call must finish it."""
    d = framework_session_dir(tmp_path, AGENT)
    d.mkdir(parents=True, exist_ok=True)
    (d / "stop-target-mode").write_text(DEFAULT_STOP_TARGET_MODE, encoding="utf-8")
    assert abandon_framework_stop(tmp_path, AGENT) is True
    assert not (d / "stop-target-mode").exists()


def test_abandon_refuses_without_an_agent(tmp_path: Path) -> None:
    assert abandon_framework_stop(tmp_path, "") is False


def test_abandon_never_touches_a_per_session_stop_requested(tmp_path: Path) -> None:
    """The sibling under sessions/<SID>/ is a turn-end permit, not a loop stop."""
    d = _raise(tmp_path)
    sid = d.parent / "sessions" / "sid-1"
    sid.mkdir(parents=True, exist_ok=True)
    (sid / "stop-requested").write_text("", encoding="utf-8")
    assert abandon_framework_stop(tmp_path, AGENT) is True
    assert (sid / "stop-requested").exists(), "different file, different meaning"


def test_both_overrun_branches_actually_call_the_retire(tmp_path: Path) -> None:
    """The wire-in is REAL, not merely present (guard-1943 shape).

    Both endings mean 'the grace is spent and nothing consumed the stop'; a fix wired
    into only one of them leaves the measured path open.
    """
    src = (Path(__file__).resolve().parents[1] / "src" / "zakcode" / "server" / "app.py").read_text(
        encoding="utf-8"
    )
    assert "abandon_framework_stop," in src, "imported"
    assert src.count("_retire_unconsumed_framework_stop()") == 4, (
        "one definition + both overrun branches + the window closing on a loop at rest"
    )
    # The third orphan path (ADR-0189): no turn to interrupt, the window just closes.
    idle_close = src.index(
        "if framework_stop_until is not None and time.monotonic() >= framework_stop_until:"
    )
    assert "_retire_unconsumed_framework_stop()" in src[idle_close : idle_close + 200]
    mid = src.index("overran its %.0fs window")
    cap = src.index("run cap: framework stop overran its window")
    for start in (mid, cap):
        window = src[start : start + 400]
        assert "_retire_unconsumed_framework_stop()" in window, "retire missing from an overrun"
        assert window.index("_retire_unconsumed_framework_stop()") < window.index(
            "request_interrupt("
        ), "retire the pair BEFORE the interrupt -- the interrupt can end this process"


# ── the sidecar's signature (ADR-0188) ───────────────────────────────────────────
#
# Measured 2026-09-17 (prod vessel debc47de, run B): the raise landed at 20:29:29,
# the framework's /start wrote its binding — the stop-clear guard's "session start" —
# at 20:31:43, and the guard read mtime < started_at as "stale" and deleted the run's
# only ending. The framework now keeps a SIGNED signal without consulting time, so
# the signature has to be there, and its lifetime has to be owned here.


def test_signal_is_signed_after_the_setter_creates_it(tmp_path: Path) -> None:
    _plant_signal_setter(tmp_path, _real_setter_body())
    assert request_framework_stop(tmp_path, AGENT) is True
    text = (framework_session_dir(tmp_path, AGENT) / "stop-requested").read_text(encoding="utf-8")
    first, second = text.splitlines()[:2]
    assert first == SIDECAR_RAISE_MARKER
    assert second.startswith("raised_at: ") and second.endswith("Z")
    assert sidecar_raise_time(tmp_path, AGENT) is not None


def test_signing_never_recreates_an_ask_the_mind_already_consumed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The ask is live from the setter's touch, so a quick mind can consume it before the
    signature is written. The signature must not put it back.

    Measured 2026-09-18 on windows-latest: the R4 end-to-end test timed out at 10 s against
    a 30 s grace. ``Path.write_text`` had re-created the ``stop-requested`` the mind had just
    removed, so ``framework_stop_complete`` read "not yet" until the grace ran out. The
    window is opened here by hand, with no timing: the mind's whole stop lands between the
    setter's verification and the signing.
    """
    _plant_signal_setter(tmp_path, _real_setter_body())
    session_dir = framework_session_dir(tmp_path, AGENT)
    session_dir.mkdir(parents=True)
    (session_dir / AGENT_MODE_FILENAME).write_text("autonomous\n", encoding="utf-8")
    real_setter = framework_stop.invoke_signal_setter

    def setter_then_the_mind_stops(*args: object) -> bool:
        raised = real_setter(*args)  # type: ignore[arg-type]
        assert raised and (session_dir / "stop-requested").exists()
        (session_dir / "stop-requested").unlink()  # D3: the mind consumes the ask
        (session_dir / AGENT_MODE_FILENAME).write_text("assistant\n", encoding="utf-8")  # D7
        (session_dir / STOP_TARGET_MODE_FILENAME).unlink()  # D7: the target mode is consumed
        return raised

    monkeypatch.setattr(framework_stop, "invoke_signal_setter", setter_then_the_mind_stops)

    assert request_framework_stop(tmp_path, AGENT) is True, "the ask WAS raised, and read"
    assert not (session_dir / "stop-requested").exists(), "the signature re-created the ask"
    assert framework_stop_complete(tmp_path, AGENT) is True, "a finished stop reads finished"


def test_an_unsigned_or_absent_signal_has_no_raise_time(tmp_path: Path) -> None:
    assert sidecar_raise_time(tmp_path, AGENT) is None
    session_dir = framework_session_dir(tmp_path, AGENT)
    session_dir.mkdir(parents=True)
    (session_dir / "stop-requested").touch()  # the framework's own writers: empty
    assert sidecar_raise_time(tmp_path, AGENT) is None
    (session_dir / "stop-requested").write_text("note\n" + SIDECAR_RAISE_MARKER + "\n")
    assert sidecar_raise_time(tmp_path, AGENT) is None, "first line only"
    assert sidecar_raise_time(tmp_path, "") is None


def _signed_pair(root: Path, raised_at: str) -> Path:
    session_dir = framework_session_dir(root, AGENT)
    session_dir.mkdir(parents=True, exist_ok=True)
    (session_dir / "stop-target-mode").write_text("assistant", encoding="utf-8")
    (session_dir / "stop-requested").write_text(
        f"{SIDECAR_RAISE_MARKER}\nraised_at: {raised_at}\n", encoding="utf-8"
    )
    return session_dir


def test_startup_retires_a_signed_stop_only_past_the_grace(tmp_path: Path) -> None:
    session_dir = _signed_pair(tmp_path, "2026-09-17T20:29:29Z")
    raised = sidecar_raise_time(tmp_path, AGENT)
    assert raised is not None
    # inside the window: a restart mid-grace leaves the ask for the mind
    assert retire_expired_sidecar_stop(tmp_path, AGENT, grace_s=350, now=raised + 100) is False
    assert (session_dir / "stop-requested").exists()
    assert (session_dir / "stop-target-mode").exists()
    # past it: the pair is an orphan of a previous process
    assert retire_expired_sidecar_stop(tmp_path, AGENT, grace_s=350, now=raised + 351) is True
    assert not (session_dir / "stop-requested").exists()
    assert not (session_dir / "stop-target-mode").exists()


def test_startup_leaves_an_unsigned_stop_and_a_started_stop_alone(tmp_path: Path) -> None:
    session_dir = framework_session_dir(tmp_path, AGENT)
    session_dir.mkdir(parents=True)
    (session_dir / "stop-requested").touch()  # unsigned: the framework's mtime rule owns it
    assert retire_expired_sidecar_stop(tmp_path, AGENT, grace_s=0, now=10**12) is False
    assert (session_dir / "stop-requested").exists()
    _signed_pair(tmp_path, "2026-09-17T20:29:29Z")
    (session_dir / STOP_CHECKPOINT_FILENAME).write_text("{}", encoding="utf-8")  # a stop began
    assert retire_expired_sidecar_stop(tmp_path, AGENT, grace_s=0, now=10**12) is False
    assert (session_dir / "stop-requested").exists()
