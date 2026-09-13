"""Raise one of the FRAMEWORK's own session signal markers from this sidecar.

A seed workspace runs a Mind whose loop reads markers under ``agents/<agent>/session/``:
``stop-requested`` ends the perpetual loop, ``perception-received`` breaks a sleeping one
early. TWO lanes in this process raise such a signal — the conductor's run-end
(:mod:`zakcode.session.framework_stop`) and the vessel's perception intake
(``POST /observe``) — and the SHAPE of raising one is the framework's, not ours:

* Shell out to the framework's own ``session-signal-set.sh`` rather than touching the
  marker directly. It is the single writer of that file, it VALIDATES the signal name
  against the framework's allow-list, it creates the session dir, and a future change to
  what "set a signal" means reaches us for free.
* Resolve ``bash``, never a bare argv0: on Windows a bare "bash" is hijacked by the WSL
  app-execution-alias stub and can block forever on a dead LxssManager.
* Export the agent under BOTH names. The seed's scripts read ``MIND_AGENT`` and never
  ``AYOAI_*``, while its daemon-backed steps read ``AYOAI_AGENT`` — an agent bound under
  only one of the two is UNBOUND for the other half.
* Verify by READING the marker, never by the exit code. A zero exit over an absent file
  hands the caller a signal it never actually armed.
* Fail open. A signal that cannot be raised must not take the process with it: the stop
  lane falls back to the interrupt it already owned, and the observe lane still stages
  its frame.

This module is the MECHANISM only. Which signal, under what condition, and what else must
already be on disk are the callers' policy — ``framework_stop`` carries an ordering
obligation (target mode first, reverted on failure) that nothing here knows about, which
is why the two halves below are exposed separately as well as composed.
"""

from __future__ import annotations

import logging
import os
import subprocess
from pathlib import Path

from zakcode._subprocess import find_bash

logger = logging.getLogger(__name__)

#: The framework's single writer of session signal markers, relative to the seed root.
SIGNAL_SET_SCRIPT = Path("core") / "scripts" / "session-signal-set.sh"

#: A local marker write that cannot finish in this long is a wedged filesystem, and a
#: caller's grace may be ticking — fall back rather than block on it.
SIGNAL_SET_TIMEOUT_S = 30.0


def framework_session_dir(workspace_root: str | os.PathLike[str], agent: str) -> Path:
    """The AGENT-LEVEL session dir whose markers the Mind's loop reads.

    Deliberately not ``sessions/<SID>/``: that per-session dir holds same-named files with
    DIFFERENT meanings (a ``stop-requested`` there is a turn-end permit, not a loop stop),
    so a signal written there reaches a different reader than the one intended.
    """
    return Path(workspace_root) / "agents" / str(agent) / "session"


def signal_setter_command(workspace_root: str | os.PathLike[str]) -> list[str] | None:
    """``[bash, script]`` for the framework's signal writer, or None when unavailable.

    None is the ordinary answer in a NON-SEED workspace, not an error: there is no
    framework here to signal. A caller with an ordering obligation checks this BEFORE
    writing anything of its own, so a missing seed leaves no partial state behind.
    """
    root = Path(workspace_root)
    script = root / SIGNAL_SET_SCRIPT
    if not script.is_file():
        logger.debug("framework signal: no %s under %s", SIGNAL_SET_SCRIPT, root)
        return None
    bash_bin = find_bash()
    if not bash_bin:
        logger.warning("framework signal: no bash interpreter found")
        return None
    return [bash_bin, str(script)]


def invoke_signal_setter(
    command: list[str],
    workspace_root: str | os.PathLike[str],
    agent: str,
    signal: str,
    marker: str | os.PathLike[str],
) -> bool:
    """Run the framework's setter for ``signal`` and confirm ``marker`` reached disk."""
    env = dict(os.environ)
    env["AYOAI_AGENT"] = str(agent)
    env["MIND_AGENT"] = str(agent)
    try:
        completed = subprocess.run(
            [*command, signal],
            cwd=str(Path(workspace_root)),
            env=env,
            capture_output=True,
            timeout=SIGNAL_SET_TIMEOUT_S,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        logger.warning(
            "framework signal: %s did not run for %s (%s)", SIGNAL_SET_SCRIPT, signal, exc
        )
        return False

    if completed.returncode != 0:
        logger.warning(
            "framework signal: %s exited %s for %s (%s)",
            SIGNAL_SET_SCRIPT,
            completed.returncode,
            signal,
            completed.stderr.decode("utf-8", "replace").strip()[:200],
        )
        return False

    # Verify by READING (see module docstring): the marker is what the loop reads.
    if not Path(marker).exists():
        logger.warning("framework signal: %s exited 0 but %s is absent", SIGNAL_SET_SCRIPT, marker)
        return False
    return True


def set_framework_signal(workspace_root: str | os.PathLike[str], agent: str, signal: str) -> bool:
    """Raise one framework session signal for ``agent``. Idempotent, fail-open.

    The whole of the SIMPLE case: nothing has to exist first and nothing needs reverting
    on failure. A caller carrying either obligation composes :func:`signal_setter_command`
    and :func:`invoke_signal_setter` itself rather than reaching for this.

    Returns True when the marker is on disk — because this call set it, or because it was
    already there. An already-present marker is success, not a conflict: these signals are
    levels the loop consumes, so two raises before one read are still one wake.
    """
    marker = framework_session_dir(workspace_root, agent) / signal
    if marker.exists():
        return True
    command = signal_setter_command(workspace_root)
    if command is None:
        return False
    return invoke_signal_setter(command, workspace_root, agent, signal, marker)
