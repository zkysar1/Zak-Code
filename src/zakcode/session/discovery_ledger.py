"""The discovery ledger — what this mind has unlocked by exploring, kept because the vessel forgets.

The vessel's ``discoveryPerception`` slice (env-server ``SpatialPerceptionVerticle``, g-368-15)
is a **projection, not a store**: the perception tick CLEARS the folder and rebuilds it every
round from the entities currently inside the character's 27-stud bubble. That is the right
design for the vessel — nothing accumulates, a departing entity prunes itself, and the slice
cannot drift from the ``touchCount`` it mirrors. But it means an object the character explored
and then walked away from is simply ABSENT from the next envelope.

So "which objects has exploring unlocked?" cannot be answered from any single envelope.
Unlocking is monotone and the evidence for it is not: entities LEAVE the reported population as
part of normal operation (walking away), which is exactly the shape where reading the current
frame under-reports the accumulated truth. Accumulating across frames is the MIND's job, and
this module is where it happens.

What the ledger holds, and why it holds nothing else: exactly one entry per unlocked ayoKey,
each carrying only facts the next envelope cannot re-supply.

- **Membership IS the unlock, and it is sticky.** A key enters the map once and never leaves.
  A later envelope that omits it, or reports it with ``discovered: false``, does not re-lock
  it — absence is how the vessel says "not within reach right now", never "never found". There
  is no per-entry ``discovered`` flag because presence already is one, and a second copy of a
  fact is a second thing to keep true.
- **Entities merely SEEN are not recorded.** "In the bubble but never touched" is re-derivable
  from the very next envelope, so storing it would grow the file with a copy of something the
  vessel already says. The unlock is the only fact that outlives the frame.
- **``touchCount`` and ``lastObservedAt`` are latest-wins, and they travel together.** The
  count is meaningless without the time it was read at, which is why the stamp is here at all.
  ``touchCount: 0`` beside a present key is the visible signature of a vessel-side counter
  reset; nothing hides it and nothing corrects it — the ledger reports what it was told.
- **Fail-open, like every other step on this path.** An unreadable or malformed ledger yields
  an empty one and the fold proceeds; a failed WRITE is swallowed. A perception must never be
  lost because bookkeeping about it failed — ``_deliver_observation`` is fail-open by
  inheritance and this module must not be the thing that breaks that.
- **Whose clock.** ``discoveredAt`` / ``lastObservedAt`` are the envelope's ``observedAt``,
  which is the VESSEL's clock, recorded verbatim and never compared against this machine's.
  They are provenance, not a basis for local time-window arithmetic.

The keys are the vessel's ``ayoKey`` values, kept verbatim: ayoKey is the mind-side identity
for an entity across the border, so rewriting it here would break the only join that exists.
The inner field names are likewise the envelope's camelCase rather than this package's
snake_case — they are a foreign schema being recorded, and renaming them would quietly decouple
this file from the producer it mirrors.
"""

from __future__ import annotations

import contextlib
import json
import os
from pathlib import Path
from typing import Any

#: The ledger filename under the workspace root, a sibling of ``.observation`` and ``.say``.
DISCOVERY_FILENAME = ".discovery"

#: The envelope slice this ledger folds. Named by the vessel; the "Perception" suffix is what
#: gets it across the bridge at all (``PerceptionBridgeVerticle.PERCEPTION_SUFFIX``).
DISCOVERY_SLICE = "discoveryPerception"

#: The ledger format this reader/writer speaks. A ledger stamped with anything else is treated
#: as empty rather than guessed at — the same posture the observation envelope takes, for the
#: same reason: misreading accumulated state is worse than rebuilding it.
DISCOVERY_LEDGER_VERSION = 1


def discovery_path(workspace_root: str | os.PathLike[str]) -> Path:
    """The discovery ledger file for a workspace."""
    return Path(workspace_root) / DISCOVERY_FILENAME


def read_ledger(path: Path) -> dict[str, Any]:
    """Load the unlocked map, or an empty one.

    Returns ``{}`` when the file is absent, unreadable, does not parse as a JSON object, or is
    stamped with a version this code does not speak. An absent ledger is the NORMAL first-run
    state, not a contract violation — which is why a default is synthesised here and would not
    be for a file that is authoritative and expected to exist.
    """
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError:
        return {}
    try:
        ledger = json.loads(raw)
    except (ValueError, TypeError):
        return {}
    if not isinstance(ledger, dict):
        return {}
    if ledger.get("ledgerVersion") != DISCOVERY_LEDGER_VERSION:
        return {}
    discovered = ledger.get("discovered")
    return discovered if isinstance(discovered, dict) else {}


def write_ledger(path: Path, discovered: dict[str, Any]) -> bool:
    """Persist the unlocked map atomically. Returns False if it could not be written.

    Temp-write + replace, matching how ``POST /observe`` stages its envelope: a torn ledger
    read by the next turn is indistinguishable from a corrupt one, and a corrupt one reads as
    empty — which would silently re-lock everything already found.
    """
    payload = {"ledgerVersion": DISCOVERY_LEDGER_VERSION, "discovered": discovered}
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        body = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        tmp.write_text(body + "\n", encoding="utf-8")
        os.replace(tmp, path)
    except OSError:
        with contextlib.suppress(OSError):
            tmp.unlink()
        return False
    return True


def _slice_of(envelope: dict[str, Any] | None) -> dict[str, Any]:
    """The ``discoveryPerception`` map from an envelope, or empty when it carries none.

    An envelope with no discovery slice is the EXPECTED case on any vessel that predates the
    producer, so it is silently empty rather than an error: this consumer ships before the
    producer is merged and armed, and must be inert until then rather than noisy.
    """
    if not isinstance(envelope, dict):
        return {}
    observation = envelope.get("observation")
    if not isinstance(observation, dict):
        return {}
    found = observation.get(DISCOVERY_SLICE)
    return found if isinstance(found, dict) else {}


def _reports_discovered(reported: dict[str, Any], touch_count: int) -> bool:
    """Whether the vessel says this entity is unlocked.

    The explicit flag wins where it exists, and the counter it is derived from is the fallback.
    Both are read because the producer computes ``discovered`` as ``touchCount > 0`` today: if
    that rule ever changes vessel-side, the flag is the half that should decide, and this
    consumer must not be silently re-deriving a rule it does not own.
    """
    flag = reported.get("discovered")
    if isinstance(flag, bool):
        return flag
    return touch_count > 0


def fold_observation(path: Path, envelope: dict[str, Any] | None) -> list[str]:
    """Fold one envelope's discovery slice into the ledger; return the keys newly unlocked.

    The return value is the UNLOCK EVENT — keys that crossed into the ledger on THIS envelope —
    and deliberately not "everything currently unlocked", which the ledger already holds. An
    event is reportable once; a standing set re-announced every round is noise from the second
    round on.

    Returns ``[]`` and writes nothing when the envelope carries no discovery slice, and writes
    nothing when the slice changed neither the membership nor any recorded value.
    """
    incoming = _slice_of(envelope)
    if not incoming:
        return []

    observed_at = ""
    if isinstance(envelope, dict):
        raw_at = envelope.get("observedAt")
        if isinstance(raw_at, str):
            observed_at = raw_at

    unlocked = read_ledger(path)
    newly: list[str] = []
    changed = False

    for ayo_key, reported in incoming.items():
        # A malformed row is skipped, not guessed at, and does not abort the rest of the fold:
        # one bad entry must not cost every good one in the same envelope.
        if not isinstance(ayo_key, str) or not isinstance(reported, dict):
            continue

        touch_count = reported.get("touchCount")
        if not isinstance(touch_count, int) or isinstance(touch_count, bool):
            touch_count = 0

        entry = unlocked.get(ayo_key)
        if not isinstance(entry, dict):
            # Not in the ledger. Record it ONLY on an unlock — a merely-seen entity is in the
            # next envelope anyway, and the frontier is not what this ledger is for.
            if not _reports_discovered(reported, touch_count):
                continue
            entry = {"discoveredAt": observed_at}
            unlocked[ayo_key] = entry
            newly.append(ayo_key)
            changed = True

        # Already unlocked (or unlocked just now): refresh the two travelling values. Stickiness
        # lives in the fact that nothing below can remove the key — a `discovered: false` here
        # is the vessel reporting a reset counter, not a retraction of the unlock.
        if entry.get("touchCount") != touch_count or entry.get("lastObservedAt") != observed_at:
            changed = True
        entry["touchCount"] = touch_count
        entry["lastObservedAt"] = observed_at

    if changed:
        write_ledger(path, unlocked)
    return sorted(newly)
