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

THE LEDGER (``ZAKCODE_SPEND_LEDGER``, a JSON file outside the mind workspace) carries the
state across a process restart inside one run. A ledger that exists but cannot be read fails
CLOSED: the gate is armed with nothing it can trust and refuses until a push re-arms it.

AN UNPRICED CALL (``Usage.cost_unpriced``, or a stream that reported no usage) is counted, not
refused, the same rule as :class:`~zakcode.agent.budget.IterationBudget`: the price is never
guessed. The health snapshot reports ``unpriced_calls`` so a reader can tell a cap it cannot
enforce from one that is not spent.

No estimate is made before a call. A call admitted just under the cap may still run, so the
overshoot is at most the calls already in flight when the cap is reached.
"""

from __future__ import annotations

import json
import logging
import math
import os
import threading
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from zakcode.providers.base import QuotaExhausted

logger = logging.getLogger(__name__)

#: Where the gate keeps its state across a restart. Set by the vessel's provisioning.
LEDGER_ENV = "ZAKCODE_SPEND_LEDGER"

#: The run stop reason the soft trip ends the run with (matches the vessel's own stop).
STOP_REASON = "mind_budget_cap"

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
        self._on_soft_trip: Callable[[], None] | None = None
        self._load()

    # ── state ────────────────────────────────────────────────────────────────────

    def _spent(self) -> float:
        return max(self._base + self._counted, self._floor)

    def set_soft_trip_hook(self, hook: Callable[[], None] | None) -> None:
        """What to call, once, when spend first reaches ``cap - reserve``."""
        self._on_soft_trip = hook

    def set_budget(self, cap_usd: Any, reserve_usd: Any, floor_usd: Any) -> dict[str, Any]:
        """Arm, or update, the gate from the vessel's push. Raises ValueError on bad values."""
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

    def _soft_due(self) -> bool:
        """True exactly once: the first time spend reaches ``cap - reserve``. Lock held."""
        if self._soft_fired or self._spent() < self._cap - self._reserve:
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
