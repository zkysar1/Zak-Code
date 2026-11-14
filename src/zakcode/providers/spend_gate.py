"""The per-Mind spend gate: refuse a model call once this Mind has spent its cap.

Before this, the per-Mind cap was an accounting line. The vessel metered a Mind's spend from
outside and, at the cap, asked the run to end (``POST /run/stop``, reason ``mind_budget_cap``),
fire-and-forget. Nothing refused a call, so a Mind past its cap kept calling the model until
its run ended. The meter also under-read the Mind's own record by 27-32% (measured
2026-11-02). This gate is the refusal, in the process that makes the calls, counted from the
calls themselves.

INERT UNTIL ARMED. Zak-Code ``main`` reaches every sidecar at its next boot, so a gate that
armed itself would change every running Mind at once. It arms only when the vessel pushes the
Mind's budget (``POST /budget``): the cap, the reserve, and what the vessel's row says this
Mind has already spent (the floor). The row stays the one source of truth for the cap.

Once armed:

* :meth:`SpendGate.admit` runs before every model call and raises :class:`SpendCapReached`
  once ``spent >= cap``. It is a :class:`QuotaExhausted`, so the loop ends the turn on the
  first refusal and never retries it. Tools, files, ``/sidecar/health`` and ``/run/stop``
  are never refused, and the process never exits: a tripped Mind stays up and answers each
  turn with the refusal.
* :meth:`SpendGate.charge` adds each answered call's cost. When ``spent`` first reaches
  ``cap - reserve`` it calls the soft-trip hook ONCE, and the app starts the run's graceful
  ending with reason ``mind_budget_cap``. The reserve is what that ending may spend.

``spent = max(base + counted, floor)``. ``base`` is the floor when the gate armed, ``counted``
is what this process charged since, and ``floor`` is the newest value the vessel pushed. The
row lags and under-reads, so a push can raise ``spent`` and never lowers it. The one
exception is a floor BELOW ``base``: the row started a new period, so the gate re-bases.

THE SOFT TRIP is a fact about the cap the gate holds now. The ledger keeps it set across a
restart, so a trip an earlier cap left would keep a raised cap from ever starting its ending.
A push that leaves ``spent`` short of the new ``cap - reserve`` clears it, and the hook fires
once more at the new line. A push that leaves ``spent`` at or past the line leaves it set, so a
repeated push never fires it twice.

THE LEDGER (``ZAKCODE_SPEND_LEDGER``, a JSON file outside the mind workspace) carries the
state across a process restart inside one run. A ledger that exists but cannot be read fails
CLOSED: the gate is armed with nothing it can trust and refuses until a push re-arms it. A
ledger that can be read restores the cap it last held, and the row may have been raised since,
so until the vessel's first push lands in this process that cap may be stale: a refusal decided
against it waits for that push (THE FIRST PUSH, below).

THE FIRST PUSH lands about 70 s after the sidecar starts serving, and the first turn starts about
8 s after it (measured 2026-11-14, when four calls were refused in the 45 s before the push,
against a cap the row had already raised). So the provider's async dispatch asks
:meth:`SpendGate.admit_async`, not :meth:`SpendGate.admit`. A call that would be refused against a
cap restored ONLY from the ledger, with no push yet in this process, is held for the first push.
The bound, ``FIRST_PUSH_WAIT_S`` (the first-contact bound), is ONE window per process: the first
call held opens it and later calls share what is left of it, so a vessel that never pushes costs
one window, not one per call, and a call that would be refused after the window has closed is
refused at once. When the push lands, or the bound is reached, the call is decided against the cap
the gate holds: admitted if the pushed cap allows it, refused exactly as :meth:`SpendGate.admit`
refuses if it does not or if no push came. The wait polls a ``threading.Event`` in
``asyncio.sleep`` slices instead of parking in an executor thread, because a thread cannot be
cancelled: it would outlive a cancelled call and hold a process exit open for the rest of the
bound. A poll also yields to the event loop, which is the loop that serves ``POST /budget``, so
the push is never starved. Only a refusal waits: a call under the restored cap, an unarmed gate, a
gate this process has taken a push on and an unreadable ledger all decide at once, and
:meth:`SpendGate.admit` itself never waits.

AN UNPRICED CALL (``Usage.cost_unpriced``, or a stream that reported no usage) is counted, not
refused, the same rule as :class:`~zakcode.agent.budget.IterationBudget`: the price is never
guessed. The health snapshot reports ``unpriced_calls`` so a reader can tell a cap it cannot
enforce from one that is not spent.

No estimate is made before a call. A call admitted just under the cap may still run, so the
overshoot is at most the calls already in flight when the cap is reached.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import threading
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from zakcode.providers.base import QuotaExhausted

logger = logging.getLogger(__name__)

#: Where the gate keeps its state across a restart. Set by the vessel's provisioning.
LEDGER_ENV = "ZAKCODE_SPEND_LEDGER"

#: The run stop reason the soft trip ends the run with (matches the vessel's own stop).
STOP_REASON = "mind_budget_cap"

#: The first-contact bound, in seconds: ONE window per process, opened by the first refusal that
#: waits and shared by the later ones, in which a refusal decided against a cap the gate holds
#: only from its ledger waits for the vessel's first push. That push lands about 70 s after the
#: sidecar starts serving (measured 2026-11-14), so 150 s covers it with margin.
FIRST_PUSH_WAIT_S = 150.0

#: How often a waiting call looks for the push. A look is a yield to the event loop, so the wait
#: holds neither the loop nor a thread; 0.1 s is nothing against a push that takes about 70 s.
_FIRST_PUSH_POLL_S = 0.1

_LEDGER_VERSION = 1


class SpendCapReached(QuotaExhausted):
    """This Mind has spent its cap, so the model call was refused before it was sent."""


def _money(value: Any) -> float | None:
    """A finite, non-negative dollar amount, or None. ``bool`` is not money."""
    if isinstance(value, bool):
        return None
    try:
        amount = float(value)
    except (TypeError, ValueError):
        return None
    return amount if math.isfinite(amount) and amount >= 0 else None


class SpendGate:
    """One process's spend gate. Use :func:`get_gate` for the process-wide instance."""

    def __init__(self, ledger_path: str | os.PathLike[str] | None = None) -> None:
        self._lock = threading.Lock()
        self._ledger = Path(ledger_path) if ledger_path else None
        self._armed = False
        self._cap = 0.0
        self._reserve = 0.0
        self._base = 0.0
        self._floor = 0.0
        self._counted = 0.0
        self._refused = 0
        self._unpriced = 0
        self._soft_fired = False
        self._load_failed = False
        self._write_failed = False
        self._first_push = threading.Event()  # set by the first push this process takes
        self._first_contact_ends: float | None = None  # monotonic end of the first-contact window
        self._on_soft_trip: Callable[[], None] | None = None
        self._load()

    # ── state ────────────────────────────────────────────────────────────────────

    def _spent(self) -> float:
        return max(self._base + self._counted, self._floor)

    def set_soft_trip_hook(self, hook: Callable[[], None] | None) -> None:
        """What to call, once, when spend first reaches ``cap - reserve``."""
        self._on_soft_trip = hook

    def set_budget(self, cap_usd: Any, reserve_usd: Any, floor_usd: Any) -> dict[str, Any]:
        """Arm, or update, the gate from the vessel's push. Raises ValueError on bad values.

        A push that leaves ``spent`` short of the new ``cap - reserve`` clears a soft trip an
        earlier cap left, so the ending starts at the new line. The first push this process takes
        also releases the calls :meth:`admit_async` is holding for it.
        """
        cap, reserve, floor = _money(cap_usd), _money(reserve_usd), _money(floor_usd)
        if cap is None or reserve is None or floor is None:
            raise ValueError("cap_usd, reserve_usd and floor_usd must be finite and >= 0")
        if reserve > cap:
            raise ValueError("reserve_usd must not exceed cap_usd")
        with self._lock:
            if not self._armed or self._load_failed or floor < self._base:
                # First push, a re-arm after an unreadable ledger, or a new period on the row.
                self._base = floor
                self._counted = 0.0
                self._soft_fired = False
                self._load_failed = False
            self._armed = True
            self._cap = cap
            self._reserve = reserve
            self._floor = floor
            self._first_push.set()  # the vessel has spoken: no call waits for it any more
            if self._soft_fired and not self._at_soft_line():
                self._soft_fired = False
            self._save()
            fire = self._soft_due()
            snapshot = self._snapshot()
        if fire:
            self._fire_soft_trip()
        return snapshot

    def admit(self) -> None:
        """Refuse the call about to be sent when this Mind has spent its cap."""
        with self._lock:
            if not self._armed:
                return
            if self._load_failed:
                self._refused += 1
                raise SpendCapReached(
                    "This Mind's spend ledger could not be read, so its model calls are "
                    "refused until the vessel pushes its budget again."
                )
            spent = self._spent()
            if spent >= self._cap:
                self._refused += 1
                raise SpendCapReached(
                    f"This Mind has spent ${spent:.2f} of its ${self._cap:.2f} cap, so the "
                    "model call was refused before it was sent."
                )

    async def admit_async(self) -> None:
        """:meth:`admit` for the async dispatch: a refusal against a ledger-only cap waits first.

        A call this gate would refuse, against a cap restored only from its ledger with no push
        yet in this process, waits for the vessel's first push (THE FIRST PUSH, in the module
        docstring) and is then decided by :meth:`admit` against the cap the gate holds. Every
        other call is decided at once, exactly as :meth:`admit` decides it.
        """
        bound_s = self._first_push_bound()
        if bound_s > 0.0:
            await self._wait_for_first_push(bound_s)
        self.admit()

    def _first_push_bound(self) -> float:
        """Seconds a refusal may still wait for the vessel's first push; 0.0 decides now.

        The first refusal it sees opens the first-contact window; later ones get what is left.
        """
        with self._lock:
            if not self._armed or self._load_failed or self._first_push.is_set():
                return 0.0  # nothing armed, nothing to trust, or the vessel has already spoken
            if self._spent() < self._cap:
                return 0.0  # admit() admits it: only a refusal waits
            now = time.monotonic()
            if self._first_contact_ends is None:
                self._first_contact_ends = now + FIRST_PUSH_WAIT_S
            return max(0.0, self._first_contact_ends - now)

    async def _wait_for_first_push(self, bound_s: float) -> None:
        """Wait up to ``bound_s`` for the first push, yielding to the event loop between looks."""
        started = time.monotonic()
        held = self.snapshot()
        logger.info(
            "spend gate: holding a model call up to %.0f s for the vessel's first push: this Mind "
            "has spent $%.2f of the $%.2f cap its ledger holds, and no push has reached this "
            "process yet",
            bound_s,
            held["spent_usd"],
            held["cap_usd"],
        )
        deadline = started + bound_s
        while not self._first_push.is_set():
            left = deadline - time.monotonic()
            if left <= 0:
                break
            await asyncio.sleep(min(_FIRST_PUSH_POLL_S, left))
        waited = time.monotonic() - started
        if self._first_push.is_set():
            logger.info(
                "spend gate: the vessel's first push arrived after %.1f s; deciding the call "
                "against the cap the gate holds now",
                waited,
            )
        else:
            logger.warning(
                "spend gate: no first push from the vessel in %.1f s (bound %.0f s); deciding the "
                "call against the cap its ledger holds",
                waited,
                FIRST_PUSH_WAIT_S,
            )

    def charge(self, usage: Any) -> None:
        """Count one answered call. ``usage`` is a :class:`~zakcode.usage.Usage` or None."""
        with self._lock:
            if not self._armed:
                return
            cost = _money(getattr(usage, "cost_usd", None)) if usage is not None else None
            if usage is None or cost is None or getattr(usage, "cost_unpriced", False):
                self._unpriced += 1
            self._counted += cost or 0.0
            self._save()
            fire = self._soft_due()
        if fire:
            self._fire_soft_trip()

    def _at_soft_line(self) -> bool:
        """True when spend has reached ``cap - reserve``. Lock held."""
        return self._spent() >= self._cap - self._reserve

    def _soft_due(self) -> bool:
        """True once each time spend reaches ``cap - reserve``; a set trip stays set. Lock held."""
        if self._soft_fired or not self._at_soft_line():
            return False
        self._soft_fired = True
        self._save()
        return True

    def _fire_soft_trip(self) -> None:
        hook = self._on_soft_trip
        logger.warning(
            "spend gate: this Mind reached its cap less the reserve; starting the run's "
            "graceful ending (%s)",
            STOP_REASON,
        )
        if hook is None:
            return
        try:
            hook()
        except Exception:  # noqa: BLE001 - the ending is best effort; the hard cap still holds
            logger.exception("spend gate: the soft-trip hook failed")

    def snapshot(self) -> dict[str, Any]:
        """The health view: ``spend_gate`` in ``/sidecar/health``."""
        with self._lock:
            return self._snapshot()

    def _snapshot(self) -> dict[str, Any]:
        spent = self._spent()
        return {
            "armed": self._armed,
            "cap_usd": round(self._cap, 6),
            "reserve_usd": round(self._reserve, 6),
            "spent_usd": round(spent, 6),
            "soft_tripped": self._soft_fired,
            "tripped": self._armed and (self._load_failed or spent >= self._cap),
            "refused_calls": self._refused,
            "unpriced_calls": self._unpriced,
            "ledger_ok": not (self._load_failed or self._write_failed),
        }

    # ── the ledger ───────────────────────────────────────────────────────────────

    def _load(self) -> None:
        if self._ledger is None or not self._ledger.exists():
            return
        try:
            data = json.loads(self._ledger.read_text(encoding="utf-8"))
            if data.get("version") != _LEDGER_VERSION:
                raise ValueError(f"unknown ledger version {data.get('version')!r}")
            names = ("cap", "reserve", "base", "floor", "counted")
            values = {k: v for k in names if (v := _money(data.get(k))) is not None}
            if len(values) != len(names) or not isinstance(data.get("armed"), bool):
                raise ValueError("ledger fields are missing or not money")
            refused = int(data.get("refused", 0) or 0)
            unpriced = int(data.get("unpriced", 0) or 0)
        except (OSError, ValueError, TypeError, AttributeError) as exc:
            logger.error(
                "spend gate: the ledger %s could not be read (%s); failing closed",
                self._ledger,
                exc,
            )
            self._armed = True
            self._load_failed = True
            return
        self._armed = data["armed"]
        self._cap = values["cap"]
        self._reserve = values["reserve"]
        self._base = values["base"]
        self._floor = values["floor"]
        self._counted = values["counted"]
        self._soft_fired = bool(data.get("soft_fired", False))
        self._refused = refused
        self._unpriced = unpriced

    def _save(self) -> None:
        """Write the ledger atomically. Lock held. A failed write is reported, not fatal."""
        if self._ledger is None:
            return
        data = {
            "version": _LEDGER_VERSION,
            "armed": self._armed,
            "cap": self._cap,
            "reserve": self._reserve,
            "base": self._base,
            "floor": self._floor,
            "counted": self._counted,
            "soft_fired": self._soft_fired,
            "refused": self._refused,
            "unpriced": self._unpriced,
        }
        tmp = self._ledger.with_name(self._ledger.name + ".tmp")
        try:
            self._ledger.parent.mkdir(parents=True, exist_ok=True)
            tmp.write_text(json.dumps(data), encoding="utf-8")
            os.replace(tmp, self._ledger)
            self._write_failed = False
        except OSError as exc:
            if not self._write_failed:
                logger.error(
                    "spend gate: the ledger %s could not be written (%s)", self._ledger, exc
                )
            self._write_failed = True


_GATE: SpendGate | None = None
_GATE_LOCK = threading.Lock()


def get_gate(env: Mapping[str, str] | None = None) -> SpendGate:
    """The process-wide gate, made on first use with the ledger ``ZAKCODE_SPEND_LEDGER`` names."""
    global _GATE
    with _GATE_LOCK:
        if _GATE is None:
            source = os.environ if env is None else env
            _GATE = SpendGate(source.get(LEDGER_ENV) or None)
        return _GATE


def reset_gate_for_tests(gate: SpendGate | None = None) -> None:
    """Replace the process-wide gate (tests only)."""
    global _GATE
    with _GATE_LOCK:
        _GATE = gate
