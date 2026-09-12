"""Raise the FRAMEWORK's own graceful stop for a served Mind vessel.

A vessel runs a Mind: a perpetual agent loop whose ending is a real protocol —
consolidate, write the handoff, drop to `assistant`, go IDLE. The conductor in this
process owns WHEN a run ends (credit, the human, the clock); it must not own WHAT
the mind does at the end. Before this module the two were fused: a cap-hit
interrupted the in-flight turn and then spent the reserve on an INJECTED PROMPT
("write a short first-person recap..."), so the mind's own ending never ran and the
framework's consolidation + handoff were simply skipped. The run was severed and
the receipt was a prompt we wrote for it.

So the ending is handed back: this module raises the framework's stop the way the
framework's own sanctioned external callers raise it, and the mind ends itself.

THE SHAPE IS NOT OURS TO INVENT — it is the framework's, and both halves are
load-bearing:

1. ``stop-target-mode`` is written FIRST, then ``stop-requested``. The framework's
   graceful-stop handler reads the target mode with NO fallback, so the reverse
   order crashes it by design. Every sanctioned caller does it in this order
   (``productivity-stop-gate.sh``, ``reducer-self-fence.sh``,
   ``loop-exhaustion-fence.sh``); this sidecar is the fourth, and the framework's
   ``.claude/rules/stop-hook-compliance.md`` names it as such.
2. If the signal write fails, the target-mode file is REVERTED. That file must exist
   ONLY while a stop is actually in progress — a dangling one tells the next reader a
   stop is happening when none is.

AGENT-LEVEL, NEVER PER-SESSION. The signal goes to ``agents/<agent>/session/``. The
sibling ``agents/<agent>/sessions/<SID>/stop-requested`` is a different file with a
different meaning — a turn-end permit that does NOT stop the loop — so writing there
would end one turn and leave the run perpetual, which is exactly the severed ending
this module exists to remove.

We shell out to the framework's own ``session-signal-set.sh`` rather than touching
the marker ourselves: it is the single writer of that file, it validates the signal
name, and a future change to what "set a signal" means reaches us for free. The
seed's scripts read ``MIND_AGENT`` and never ``AYOAI_*``, while its daemon-backed
steps read ``AYOAI_AGENT`` — an agent bound under only one of the two names is
UNBOUND for the other half — so both are exported.

Fail-open throughout: a vessel whose framework stop cannot be raised still ends by
the interrupt backstop the conductor already owns. Losing the graceful ending is
bad; hanging a paid run is worse.
"""

from __future__ import annotations

import contextlib
import logging
import os
import subprocess
from pathlib import Path

from zakcode._subprocess import find_bash

logger = logging.getLogger(__name__)

#: Written under the agent's session dir, read by the framework's stop handler.
STOP_TARGET_MODE_FILENAME = "stop-target-mode"

#: The signal name the framework's ``session-signal-set.sh`` accepts.
STOP_REQUESTED_SIGNAL = "stop-requested"

#: Where a stopped run lands: user-directed, reconciliation-ready, loop off.
DEFAULT_STOP_TARGET_MODE = "assistant"

#: The framework's single writer of session signal markers, relative to the seed root.
SIGNAL_SET_SCRIPT = Path("core") / "scripts" / "session-signal-set.sh"

#: A local marker write that cannot finish in this long is a wedged filesystem, and the
#: conductor's grace is ticking — fall back to the interrupt rather than block on it.
SIGNAL_SET_TIMEOUT_S = 30.0


def framework_session_dir(workspace_root: str | os.PathLike[str], agent: str) -> Path:
    """The AGENT-LEVEL session dir whose signals stop the perpetual loop.

    Deliberately not ``sessions/<SID>/`` — see the module docstring.
    """
    return Path(workspace_root) / "agents" / str(agent) / "session"


def request_framework_stop(
    workspace_root: str | os.PathLike[str],
    agent: str,
    *,
    target_mode: str = DEFAULT_STOP_TARGET_MODE,
) -> bool:
    """Ask the workspace's Mind to run its OWN graceful stop. Idempotent.

    Returns True when ``stop-requested`` is on disk for ``agent`` — either because
    this call set it or because a previous one already had. False on every failure
    path, which the caller reads as "fall back to the interrupt".
    """
    root = Path(workspace_root)
    session_dir = framework_session_dir(root, agent)
    signal_file = session_dir / STOP_REQUESTED_SIGNAL
    mode_file = session_dir / STOP_TARGET_MODE_FILENAME

    # Idempotent: a second /run/stop, or a cap landing after an explicit stop, is still
    # one stop. Checked BEFORE the mode write so a retry cannot clobber the target mode
    # of a stop that is already in flight.
    if signal_file.exists():
        return True

    script = root / SIGNAL_SET_SCRIPT
    if not script.is_file():
        # Not a framework seed, or a seed without the scripts. Not an error here — the
        # conductor simply keeps the ending it already had.
        logger.info(
            "framework stop: no %s under %s; leaving the ending to the interrupt",
            SIGNAL_SET_SCRIPT,
            root,
        )
        return False

    bash_bin = find_bash()
    if not bash_bin:
        # Resolved, never bare: a bare "bash" argv0 on Windows is hijacked by the WSL
        # app-execution-alias stub and can block forever on a dead LxssManager.
        logger.warning(
            "framework stop: no bash interpreter found; leaving the ending to the interrupt"
        )
        return False

    try:
        session_dir.mkdir(parents=True, exist_ok=True)
        # ORDER CRITICAL (see docstring): target mode first, and with no trailing
        # newline — byte-identical to what the framework's own callers write.
        mode_file.write_text(target_mode, encoding="utf-8")
    except OSError as exc:
        logger.warning("framework stop: could not write %s (%s)", mode_file, exc)
        return False

    env = dict(os.environ)
    env["AYOAI_AGENT"] = agent
    env["MIND_AGENT"] = agent
    try:
        completed = subprocess.run(
            [bash_bin, str(script), STOP_REQUESTED_SIGNAL],
            cwd=str(root),
            env=env,
            capture_output=True,
            timeout=SIGNAL_SET_TIMEOUT_S,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        _revert_target_mode(mode_file)
        logger.warning("framework stop: %s did not run (%s)", SIGNAL_SET_SCRIPT, exc)
        return False

    if completed.returncode != 0:
        _revert_target_mode(mode_file)
        logger.warning(
            "framework stop: %s exited %s (%s)",
            SIGNAL_SET_SCRIPT,
            completed.returncode,
            completed.stderr.decode("utf-8", "replace").strip()[:200],
        )
        return False

    # Verify by READING, never by the exit code: the marker is what the framework's stop
    # handler actually reads, and a zero exit over an absent file would hand the
    # conductor a grace period to wait out for a stop that was never armed.
    if not signal_file.exists():
        _revert_target_mode(mode_file)
        logger.warning(
            "framework stop: %s exited 0 but %s is absent", SIGNAL_SET_SCRIPT, signal_file
        )
        return False

    logger.info("framework stop requested for agent %s (target mode %s)", agent, target_mode)
    return True


def _revert_target_mode(mode_file: Path) -> None:
    """Leave no dangling target mode: it must exist only while a stop is real."""
    with contextlib.suppress(OSError):
        mode_file.unlink()
