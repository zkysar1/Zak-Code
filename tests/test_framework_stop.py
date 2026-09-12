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

from zakcode.session.framework_stop import (
    DEFAULT_STOP_TARGET_MODE,
    SIGNAL_SET_SCRIPT,
    framework_session_dir,
    request_framework_stop,
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
