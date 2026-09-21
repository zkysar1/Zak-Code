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
import datetime
import logging
import os
import time
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

#: The graceful stop's own in-progress record, under the agent's session dir: written at
#: its entry (GS-0) and cleared as its LAST act (D7.1), after the target mode is applied.
STOP_CHECKPOINT_FILENAME = "stop-checkpoint.json"

#: The agent's runtime mode, under the agent's session dir. The graceful stop applies its
#: target mode here at D7, after consolidation and cleanup.
AGENT_MODE_FILENAME = "agent-mode"

#: Where a stopped run lands: user-directed, reconciliation-ready, loop off.
DEFAULT_STOP_TARGET_MODE = "assistant"

#: Where the framework's PreToolUse hook stamps "this session has been TOLD about a
#: pending stop", relative to the MIND REPO ROOT (i.e. ``workspace_root`` -- the same root
#: ``SIGNAL_SET_SCRIPT`` resolves against). Zero-byte files; the mtime is the whole datum.
#: Exactly ONE writer exists in the framework -- ``bash-agent-inject.py``
#: ``_maybe_surface_stop`` -- which is what makes the mtime attributable at all
#: (guard-1504: enumerate every writer before reading an mtime as evidence).
STOP_SURFACE_DIRNAME = Path("core") / "logs" / "stop-surface-hook"

#: The hook's own re-stamp throttle (``STOP_SURFACE_INTERVAL_S`` in that hook). Inside
#: this window it returns early WITHOUT touching the stamp, so the mtime is a
#: last-surfaced time with this much granularity -- never a per-tool-call counter.
STOP_SURFACE_INTERVAL_S = 20.0

#: First line this module writes INTO ``stop-requested`` after the framework's setter
#: has created it. The framework's own writers leave the marker EMPTY, so the line is
#: this sidecar's signature, and the framework's /start Step 2.5 guard
#: (``session.py::live_stop_decision``) keeps a signed signal WITHOUT consulting time.
#: Why time was not enough (measured 2026-09-17, prod vessel debc47de, run B): the
#: raise landed at 20:29:29, /start wrote its binding — the guard's notion of "session
#: start" — at 20:31:43 after two minutes of onboarding on a slow model, and the guard
#: read mtime < started_at as "stale" and deleted the run's only ending.
SIDECAR_RAISE_MARKER = "raised_by: vessel-sidecar"

#: The modes a graceful stop can land in. A mind whose loop still runs reads ``autonomous``.
_STOPPED_MODES = frozenset({"assistant", "reader"})

#: Re-exported: the signal-raising mechanism is shared with the observe lane
#: (:mod:`zakcode.session.framework_signal`), and callers already import these names from
#: here. Kept as re-exports rather than moved so this module's public surface is unchanged.
__all__ = [
    "AGENT_MODE_FILENAME",
    "DEFAULT_STOP_TARGET_MODE",
    "SIGNAL_SET_SCRIPT",
    "SIGNAL_SET_TIMEOUT_S",
    "SIDECAR_RAISE_MARKER",
    "STOP_CHECKPOINT_FILENAME",
    "STOP_REQUESTED_SIGNAL",
    "STOP_SURFACE_DIRNAME",
    "STOP_SURFACE_INTERVAL_S",
    "STOP_TARGET_MODE_FILENAME",
    "framework_session_dir",
    "abandon_framework_stop",
    "framework_stop_complete",
    "framework_stop_seen",
    "request_framework_stop",
    "retire_expired_sidecar_stop",
    "sidecar_raise_time",
    "stop_surface_seen_at",
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

    _sign_signal(signal_file)
    logger.info("framework stop requested for agent %s (target mode %s)", agent, target_mode)
    return True


def _sign_signal(signal_file: Path) -> None:
    """Write the sidecar's signature into the marker the setter just created.

    Best effort: the signal is already raised, and an unsigned one still works
    through the framework's mtime rule, so a write failure is logged, never raised.

    NEVER CREATES the marker. The ask is live from the moment the setter touches it, and a
    mind that reads it at once can consume it (D3 removes the file) before this line runs.
    ``Path.write_text`` creates, so it put a consumed ``stop-requested`` BACK: the raise
    reported success, ``framework_stop_complete`` read "not yet" for the whole grace, and a
    run whose mind had finished its stop beat on until the window closed. Measured
    2026-09-18 on windows-latest as a bare TimeoutError in the R4 end-to-end test (30 s
    grace against a 10 s wait), then reproduced with no timing at all by consuming the ask
    between the setter's verification and this write. Opened without ``O_CREAT``, a
    consumed ask stays consumed.
    """
    stamp = datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    try:
        descriptor = os.open(signal_file, os.O_WRONLY | os.O_TRUNC)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(f"{SIDECAR_RAISE_MARKER}\nraised_at: {stamp}\n")
    except FileNotFoundError:
        logger.info(
            "framework stop: %s was consumed before it could be signed; leaving it consumed",
            signal_file,
        )
    except OSError as exc:
        logger.warning("framework stop: raised but could not sign %s (%s)", signal_file, exc)


def sidecar_raise_time(workspace_root: str | os.PathLike[str], agent: str) -> float | None:
    """Epoch seconds of the signed raise on disk for ``agent``, or None when the signal is
    absent, unsigned, or carries no parseable ``raised_at``."""
    if not agent:
        return None
    try:
        text = (framework_session_dir(workspace_root, agent) / STOP_REQUESTED_SIGNAL).read_text(
            encoding="utf-8", errors="replace"
        )
    except OSError:
        return None
    lines = text.splitlines()
    if not lines or lines[0].strip() != SIDECAR_RAISE_MARKER:
        return None
    for line in lines[1:]:
        if line.startswith("raised_at:"):
            raw = line.split(":", 1)[1].strip()
            try:
                return datetime.datetime.fromisoformat(raw.replace("Z", "+00:00")).timestamp()
            except ValueError:
                return None
    return None


def stop_surface_seen_at(workspace_root: str | os.PathLike[str]) -> float | None:
    """Epoch seconds of the NEWEST stop-surface stamp, or None when there is none.

    THE STAMP IS THE MIND'S "I HAVE BEEN TOLD". The framework's PreToolUse hook
    (``core/scripts/bash-agent-inject.py`` ``_maybe_surface_stop``) touches a zero-byte
    file per session when it notices a pending stop, so the MTIME IS THE ONLY DATUM --
    there is no content to parse, by design.

    WHY THIS ROOT IS REACHABLE. The stamp lives under the MIND REPO ROOT, not under the
    env's logs dir -- and that root is exactly ``workspace_root``, the same one
    ``SIGNAL_SET_SCRIPT`` is resolved against. That resolution already does an
    ``is_file()`` check which MUST pass for any raise to happen at all, so wherever this
    sidecar can raise a stop it can also read this stamp. Verified by code-trace of the
    working raise path rather than assumed (guard-3476: a healthy write is never evidence
    the artifact is readable).

    NEWEST ACROSS THE DIRECTORY, not keyed by session. The stamp is keyed on the
    HARNESS's session id, which this process never learns (it is not the binding's sid),
    and a session whose id is unusable shares one literal ``nosid`` file. A served vessel
    runs one mind, so the newest stamp is the only reading available here. Both error
    directions are cheap because every caller is REPORT-ONLY: reading one mind's stamp as
    another's can only cost a missed report, never an early ending.

    Fail-open on OSError: an unreadable log dir must never affect a run.
    """
    stamp_dir = Path(workspace_root) / STOP_SURFACE_DIRNAME
    try:
        mtimes = [entry.stat().st_mtime for entry in stamp_dir.iterdir() if entry.is_file()]
    except OSError:
        return None
    return max(mtimes) if mtimes else None


def framework_stop_seen(workspace_root: str | os.PathLike[str], agent: str) -> bool | None:
    """Has the mind SEEN this raise? ``True`` / ``False`` / ``None`` when unknowable.

    The companion to :func:`framework_stop_complete`, which answers DONE. Together they
    are two booleans where the watcher used to have one: before this, "the mind never saw
    the raise" and "the mind saw it and is still working" were indistinguishable, and both
    burned the whole reserve in silence.

    ``None`` means the question could not be asked -- no agent, or no parseable signed
    raise time -- and is deliberately distinct from ``False``.

    THE THROTTLE IS PART OF THE COMPARISON. The hook returns early without re-stamping
    inside ``STOP_SURFACE_INTERVAL_S``, so the mtime is a LAST-SURFACED time with that
    much granularity; a stamp up to one interval OLDER than the raise still means seen.

    ``False`` IS NOT "THE MIND IGNORED THE STOP" -- absence is four-way ambiguous: the
    mind has made no Bash tool call since the raise (the common case, and exactly the gap
    the hook exists to close), a worker Body (whose stop is its own per-session file), a
    reader/assistant session, or the hook failing open. Every caller must report it as an
    observation, never act on it as a verdict.
    """
    raised = sidecar_raise_time(workspace_root, agent)
    if raised is None:
        return None
    seen = stop_surface_seen_at(workspace_root)
    if seen is None:
        return False
    return seen + STOP_SURFACE_INTERVAL_S >= raised


def retire_expired_sidecar_stop(
    workspace_root: str | os.PathLike[str],
    agent: str,
    *,
    grace_s: float,
    now: float | None = None,
) -> bool:
    """At server start: retire a SIGNED stop whose grace ran out before this process began.

    A signed signal never reads as stale to the framework (that is the point of the
    signature), so its lifetime has to be owned here. ``abandon_framework_stop`` already
    retires the pair when THIS process's grace expires; this is the same decision for a
    pair a previous process left behind — a sidecar that died between its raise and its
    retirement. Only a raise older than ``grace_s`` qualifies: the docstring rule
    "call it only where the grace is already spent" holds, and a fresher one (a restart
    inside the window) is left for the mind. Returns True when a pair was retired.
    """
    raised = sidecar_raise_time(workspace_root, agent)
    if raised is None:
        return False
    current = time.time() if now is None else now
    if current - raised < grace_s:
        return False
    logger.warning(
        "framework stop for agent %s was raised %.0fs ago by a previous sidecar and never "
        "consumed -- retiring it at startup rather than letting this run open on it",
        agent,
        current - raised,
    )
    return abandon_framework_stop(workspace_root, agent)


def framework_stop_complete(workspace_root: str | os.PathLike[str], agent: str) -> bool:
    """True once the Mind has signed its OWN stop off.

    Keyed on what the framework's graceful stop leaves behind once it has FINISHED, never
    on a marker it only passes through (guard-5809). Its last writes are D7 (apply the
    target mode, then remove ``stop-target-mode``) and D7.1 (clear
    ``stop-checkpoint.json``, which the framework names as the one signal that the stop ran
    to completion). So complete means ALL of:

    - ``agent-mode`` reads a mode a stop lands in. This is the positive half: before any
      stop is raised the three files below are absent too, and absence must never read
      as a finished stop.
    - ``stop-requested`` is gone: the mind consumed the ask (D3).
    - ``stop-target-mode`` is gone: D7 ran.
    - ``stop-checkpoint.json`` is gone: D7.1 ran.

    The mode alone is not enough. This process never saw what it read before the ask, and
    a mode an earlier stop left at ``assistant`` would pass. Each of the three files closes
    a different early reading.

    NOT ``stop-loop``: D2 sets it BEFORE consolidation and D6 removes it, so it marks a
    stop in progress, not a finished one. Keyed on it, this waiter could fire during
    consolidation, or never. A stop that runs inside one turn is always past D6 by the time
    the loop looks, so the loop beat on until the grace ran out (g-373-16 R4).

    False means "not yet", NEVER "failed", so the caller bounds the wait with its own grace
    instead of reading anything into it. Fail-open on OSError for the same reason -- an
    unreadable session dir must not hold a paid vessel open.
    """
    if not agent:
        return False
    try:
        session_dir = framework_session_dir(workspace_root, agent)
        mode = (session_dir / AGENT_MODE_FILENAME).read_text(encoding="utf-8", errors="replace")
        if mode.strip() not in _STOPPED_MODES:
            return False
        in_progress = (STOP_REQUESTED_SIGNAL, STOP_TARGET_MODE_FILENAME, STOP_CHECKPOINT_FILENAME)
        return not any((session_dir / name).exists() for name in in_progress)
    except OSError:
        return False


def abandon_framework_stop(workspace_root: str | os.PathLike[str], agent: str) -> bool:
    """Retire a stop THIS process raised that no mind ever consumed. Idempotent.

    THE DEFECT THIS EXISTS FOR (g-373-92, measured twice in two days on
    fileworld-smoke). ``request_framework_stop`` writes a pair whose only consumer is
    the framework's Phase -1.4, which runs inside a LIVE loop. When the grace expires
    with the pair still on disk, no loop ever read it -- the vessel is idle, or the run
    was torn down through the API -- and nothing else clears it. The pair lives on EFS,
    so the NEXT vessel boot reads a 15-hour-old ask as a live stop: one measured run
    ($0.145362, instance i-04d10d2ca472cde6a) opened its whole 350s grace AT BOOT for a
    stop nobody requested and then ended severed, the outcome the reserve exists to
    prevent. Two hand cleanups preceded this (omni 2026-09-14, echo 2026-09-15); a third
    occurrence was already paid for while the signal had no lifetime.

    Call it ONLY where the grace is already spent -- the overrun branches in
    :mod:`zakcode.server.app`, which have just decided to interrupt. Never call it while
    a stop may still land: a mind mid-consumption would lose the ask it is acting on.

    REMOVAL ORDER IS THE WRITE ORDER REVERSED, and that direction is load-bearing. The
    signal goes FIRST, so an interrupt mid-abandon can only ever leave a dangling
    ``stop-target-mode`` -- inert (every consumer keys on ``stop-requested``) and
    overwritten by the next raise. The other order would leave a signal with no target
    mode, which the framework's stop handler reads with no fallback and crashes on BY
    DESIGN -- the same asymmetry ``request_framework_stop`` encodes on the way in.

    ``stop-checkpoint.json`` is deliberately NOT touched. That file is the FRAMEWORK's
    own record that a graceful stop got underway, and its interrupted-stop resume path
    (``stop-checkpoint.sh resume-needed``) is the only thing that can finish one. Its
    presence means a mind DID consume the ask, so this is not an orphan at all.

    Returns True when the pair is gone afterwards. Fail-open on OSError like its
    siblings: an unreadable session dir must never stall the interrupt that follows.
    """
    if not agent:
        return False
    try:
        session_dir = framework_session_dir(workspace_root, agent)
        if (session_dir / STOP_CHECKPOINT_FILENAME).exists():
            logger.info(
                "framework stop: %s present for agent %s -- a stop DID start; leaving the "
                "pair for the framework's own resume",
                STOP_CHECKPOINT_FILENAME,
                agent,
            )
            return False
        signal_file = session_dir / STOP_REQUESTED_SIGNAL
        mode_file = session_dir / STOP_TARGET_MODE_FILENAME
        had = signal_file.exists() or mode_file.exists()
        with contextlib.suppress(FileNotFoundError):
            signal_file.unlink()
        _revert_target_mode(mode_file)
        if had:
            logger.warning(
                "framework stop for agent %s went unconsumed within its window -- "
                "retiring the orphaned signal pair so the next vessel boot does not "
                "read it as a live stop (g-373-92)",
                agent,
            )
        return not (signal_file.exists() or mode_file.exists())
    except OSError as exc:
        logger.warning("framework stop: could not retire the orphaned pair (%s)", exc)
        return False


def _revert_target_mode(mode_file: Path) -> None:
    """Leave no dangling target mode: it must exist only while a stop is real."""
    with contextlib.suppress(OSError):
        mode_file.unlink()
