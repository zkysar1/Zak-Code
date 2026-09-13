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

HOW a signal is raised at all — shelling out to the framework's own
``session-signal-set.sh``, resolving bash, exporting the agent under both names,
verifying by READING the marker — is :mod:`zakcode.session.framework_signal`, shared
with the observe lane. It is deliberately NOT restated here: the two lanes differ only
in their POLICY, and a second copy of the mechanism is a second thing to drift.

Fail-open throughout: a vessel whose framework stop cannot be raised still ends by
the interrupt backstop the conductor already owns. Losing the graceful ending is
bad; hanging a paid run is worse.
"""

from __future__ import annotations

import contextlib
import logging
import os
from pathlib import Path

from zakcode.session.framework_signal import (
    SIGNAL_SET_SCRIPT,
    SIGNAL_SET_TIMEOUT_S,
    framework_session_dir,
    invoke_signal_setter,
    signal_setter_command,
)

logger = logging.getLogger(__name__)

#: Written under the agent's session dir, read by the framework's stop handler.
STOP_TARGET_MODE_FILENAME = "stop-target-mode"

#: The signal name the framework's ``session-signal-set.sh`` accepts.
STOP_REQUESTED_SIGNAL = "stop-requested"

#: The framework's own EXIT PERMIT, written under the agent's session dir only after
#: the graceful stop's obligations complete (consolidate, hand off, set target mode).
STOP_LOOP_SIGNAL = "stop-loop"

#: Where a stopped run lands: user-directed, reconciliation-ready, loop off.
DEFAULT_STOP_TARGET_MODE = "assistant"

#: Re-exported: the signal-raising mechanism is shared with the observe lane
#: (:mod:`zakcode.session.framework_signal`), and callers already import these names from
#: here. Kept as re-exports rather than moved so this module's public surface is unchanged.
__all__ = [
    "DEFAULT_STOP_TARGET_MODE",
    "SIGNAL_SET_SCRIPT",
    "SIGNAL_SET_TIMEOUT_S",
    "STOP_LOOP_SIGNAL",
    "STOP_REQUESTED_SIGNAL",
    "STOP_TARGET_MODE_FILENAME",
    "framework_session_dir",
    "framework_stop_complete",
    "request_framework_stop",
]


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

    # PREFLIGHT BEFORE THE MODE WRITE, and that order is load-bearing in its own right:
    # a non-seed workspace (or a box with no bash) must leave NOTHING behind, and the
    # revert below only covers failures that happen after the file exists.
    command = signal_setter_command(root)
    if command is None:
        # Not a framework seed, or a seed without the scripts. Not an error here — the
        # conductor simply keeps the ending it already had.
        logger.info(
            "framework stop: cannot reach %s under %s; leaving the ending to the interrupt",
            SIGNAL_SET_SCRIPT,
            root,
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

    # Every failure from here on REVERTS: the shared helper verifies by reading the
    # marker and logs which way it failed, and a dangling target mode would tell the next
    # reader a stop is in progress when none is.
    if not invoke_signal_setter(command, root, agent, STOP_REQUESTED_SIGNAL, signal_file):
        _revert_target_mode(mode_file)
        return False

    logger.info("framework stop requested for agent %s (target mode %s)", agent, target_mode)
    return True


def framework_stop_complete(workspace_root: str | os.PathLike[str], agent: str) -> bool:
    """True once the Mind has signed its OWN stop off.

    ``stop-loop`` is the framework's exit permit: a SINGLE writer sets it, and only
    after the graceful stop's obligations have run. So it cannot fire early, which is
    exactly the property a completion waiter must key on (guard-5809) -- unlike
    ``stop-requested``, which this process wrote itself and which is therefore true
    the instant the ask is made rather than when the work is done.

    Absent means "not yet", NEVER "failed": absence is indistinguishable from a mind
    that is still consolidating, so the caller bounds the wait with its own grace
    instead of reading anything into a False here. Fail-open on OSError for the same
    reason -- an unreadable session dir must not hold a paid vessel open.
    """
    if not agent:
        return False
    try:
        return (framework_session_dir(workspace_root, agent) / STOP_LOOP_SIGNAL).exists()
    except OSError:
        return False


def _revert_target_mode(mode_file: Path) -> None:
    """Leave no dangling target mode: it must exist only while a stop is real."""
    with contextlib.suppress(OSError):
        mode_file.unlink()
