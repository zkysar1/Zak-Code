"""The Anthropic credit pool: ordered accounts, one per session, moved only on a hard refusal.

A fleet, a local run or CI can hold several Anthropic accounts whose credit resets on the first of
each month and is lost then. Sparing an account saves nothing, so the pool drains them in a fixed
rank order. A session keeps the account it started on, because the prompt cache and the thinking
blocks belong to the account, and it moves only when that account says it is spent or refuses the
key.

This module is the Python half of ONE policy. The policy text, the registry and the test vectors
are authored with the other runtime's implementation, and the vectors are the contract:
``tests/test_anthropic_pool_vectors.py`` runs every one of them against :class:`Pool`, so a rule is
never changed in one place alone. ``anthropic_credit_registry.json`` beside this module is a
verbatim copy of that registry, and ``tests/fixtures/anthropic_pool/SOURCE.json`` pins the git blob
ids of both copied files.

The policy, in the order a call meets it:

* **assign**: a new session takes the lowest-ranked account that is eligible for its ``use``. An
  account is eligible when it is active, its key is in the environment, it serves the ``use``, it is
  not invalid, exhausted or cooling down, a card account has a monthly cap it is under, and a promo
  account is under the steer-off fraction of its credit (new sessions only).
* **sticky**: a live session never moves because of the steer-off line.
* **credit refusal** (HTTP 402 ``billing_error``, or a 400 saying the balance is too low): the
  account is spent until the next month, the session moves to the next eligible account and the
  request is sent once more there. A refused call costs nothing, so reactive failover is cheap.
* **429**: the same account, after ``retry-after``. Three 429s inside a minute cool the account
  down for NEW sessions only.
* **401 / 403**: the account is invalid until restart, an alert is logged, the session moves.
* **529 / 5xx**: backoff on the same account. Another org does not cure a provider-wide fault.
* **ledger**: one row per call, which is what steers a new session off an account near its credit.

Three layers: :class:`Pool` is the decision engine (state and rules, no I/O and no clock of its
own); :class:`PoolLedger` writes the rows; :class:`PoolRuntime` and :class:`AnthropicPoolRing` are
the wiring. One runtime per process, so an account one session finds spent is skipped by the next;
one ring per provider, holding that provider's session. The provider asks the ring for credentials
before each request and tells it what a refusal was.

A key is never logged, returned or put in a message: everything here names an account by its id and
slot (rule 10 of the policy, and the repository's own rule 4).
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
import uuid
import weakref
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from zakcode.usage import Usage

logger = logging.getLogger(__name__)

#: The registry this module ships: a verbatim copy, see the module docstring.
DEFAULT_REGISTRY_PATH = Path(__file__).with_name("anthropic_credit_registry.json")

#: What a decision names when no account may serve.
NO_MEMBER = "NONE"

#: ``use`` values a session can be seated for when none is configured.
DEFAULT_USE = "dev"

_STATUS_CREDIT_JSON = 402
_AUTH_STATUSES = (401, 403)
_OVERLOAD_STATUS = 529

#: The error type Anthropic nests in a JSON body, as it appears inside an exception's text. The
#: backslashes are optional because litellm quotes the body once more when it wraps it.
_ERROR_TYPE = re.compile(r'\\?"error\\?"\s*:\s*\{\s*\\?"type\\?"\s*:\s*\\?"([A-Za-z_]+)\\?"')

#: The status Anthropic documents for an error type, for the types litellm's exception mapper has
#: no branch for: a 402 or a 403 comes out as its generic connection error, status 500, with the
#: body kept only as text. The type is one-to-one with the status, so the status is read back.
_STATUS_OF_UNMAPPED_TYPE = {"billing_error": _STATUS_CREDIT_JSON, "permission_error": 403}


# ---------------------------------------------------------------------------------------------
# Time
# ---------------------------------------------------------------------------------------------


def _utcnow() -> datetime:
    return datetime.now(UTC)


def parse_iso(value: str | None) -> datetime | None:
    """An ISO 8601 UTC time, written with ``Z`` or ``+00:00``."""
    if value is None:
        return None
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def format_iso(moment: datetime | None) -> str | None:
    """The form the vectors compare: whole seconds, trailing ``Z``."""
    if moment is None:
        return None
    return moment.strftime("%Y-%m-%dT%H:%M:%SZ")


def _month_start(moment: datetime) -> datetime:
    return datetime(moment.year, moment.month, 1, tzinfo=UTC)


def _next_month_start(moment: datetime) -> datetime:
    if moment.month == 12:
        return datetime(moment.year + 1, 1, 1, tzinfo=UTC)
    return datetime(moment.year, moment.month + 1, 1, tzinfo=UTC)


# ---------------------------------------------------------------------------------------------
# The decision engine
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Decision:
    """What the policy says to do with one assignment or one response."""

    member: str
    retry: bool = False
    moved: bool = False
    alert: bool = False
    retry_after: float | None = None

    def as_dict(self) -> dict[str, Any]:
        """The shape the vectors assert on. ``retry_after`` appears only when there is one."""
        out: dict[str, Any] = {
            "member": self.member,
            "retry": self.retry,
            "moved": self.moved,
            "alert": self.alert,
        }
        if self.retry_after is not None:
            out["retry_after"] = self.retry_after
        return out


class MemberState:
    """One account: its registry definition and what this process has learned about it."""

    def __init__(
        self, definition: Mapping[str, Any], runtime: Mapping[str, Any] | None = None
    ) -> None:
        runtime = runtime or {}
        self.id: str = definition["id"]
        self.slot: str = definition["slot"]
        self.rank: int = definition["rank"]
        self.billing: str = definition["billing"]
        self.credit_usd: float | None = definition.get("credit_usd")
        self.monthly_cap_usd: float | None = definition.get("monthly_cap_usd")
        self.use: list[str] = list(definition["use"])
        self.status: str = definition["status"]
        self.exhausted_until: datetime | None = parse_iso(runtime.get("exhausted_until"))
        self.cool_until: datetime | None = parse_iso(runtime.get("cool_until"))
        self.invalid: bool = bool(runtime.get("invalid", False))
        self.spend_mtd: float = float(runtime.get("spend_mtd", 0.0))
        self.recent_429s: list[datetime] = [
            t for t in (parse_iso(s) for s in runtime.get("recent_429s", [])) if t is not None
        ]

    def snapshot(self) -> dict[str, Any]:
        """The mutable state, as the vectors read it."""
        return {
            "exhausted_until": format_iso(self.exhausted_until),
            "cool_until": format_iso(self.cool_until),
            "invalid": self.invalid,
            "spend_mtd": self.spend_mtd,
        }


class Pool:
    """The selection policy over a registry. No I/O: the caller passes ``now`` to every call."""

    def __init__(
        self,
        registry: Mapping[str, Any],
        env_present: list[str] | set[str],
        state: Mapping[str, Any] | None = None,
        now: datetime | None = None,
    ) -> None:
        state = state or {}
        runtime = state.get("members", {})
        self.steer_off_fraction: float = registry["steer_off_fraction"]
        self._rate_limit: Mapping[str, Any] = registry["rate_limit"]
        self._overload: Mapping[str, Any] = registry["overload"]
        self._reset_grace: Mapping[str, Any] = registry["reset_grace"]
        #: Slot numbers whose key variable is non-empty in the environment.
        self.env_present: set[str] = set(env_present)
        self.members: dict[str, MemberState] = {
            m["id"]: MemberState(m, runtime.get(m["id"])) for m in registry["members"]
        }
        #: session id -> {"member": account id, "use": the session's use}
        self.sessions: dict[str, dict[str, str]] = {}
        for sid, value in state.get("sessions", {}).items():
            self.sessions[sid] = (
                dict(value)
                if isinstance(value, dict)
                else {
                    "member": value,
                    "use": DEFAULT_USE,
                }
            )
        # Consecutive failures of one request chain, reset on success or on a move.
        self._run_429: dict[str, int] = {}
        self._run_5xx: dict[str, int] = {}
        self._month: tuple[int, int] | None = (now.year, now.month) if now else None
        #: Held across every call below; the runtime holds it too, across a whole decision.
        self.lock = threading.RLock()

    # -- rules ------------------------------------------------------------------------------

    def _advance_time(self, now: datetime) -> None:
        """A calendar-month rollover resets every account's month-to-date spend."""
        month = (now.year, now.month)
        if self._month is not None and month != self._month:
            for member in self.members.values():
                member.spend_mtd = 0.0
        self._month = month

    def _ranked(self) -> list[MemberState]:
        return sorted(self.members.values(), key=lambda m: m.rank)

    def why_not(
        self, member: MemberState, use: str, now: datetime, *, new_session: bool
    ) -> str | None:
        """Why ``member`` may not serve ``use`` now (rule 1), or None when it may.

        The one place the eligibility conditions are written: :meth:`assign` and the message a
        refused call carries both read it, so they cannot disagree.
        """
        if member.status != "active":
            return f"registry status is {member.status!r}"
        if member.slot not in self.env_present:
            return f"no key in the environment (slot {member.slot})"
        if use not in member.use:
            return f"not for use {use!r}"
        if member.invalid:
            return "its key was refused (invalid until restart)"
        if member.exhausted_until is not None and member.exhausted_until > now:
            return f"out of credit until {format_iso(member.exhausted_until)}"
        if member.cool_until is not None and member.cool_until > now:
            return f"cooling down until {format_iso(member.cool_until)}"
        if member.billing == "card-refill":
            # Fail closed: real card money is never spent without a cap.
            if member.monthly_cap_usd is None:
                return "card billing with no monthly cap set"
            if member.spend_mtd >= member.monthly_cap_usd:
                return "card billing at its monthly cap"
        if (
            member.billing == "promo"
            and new_session
            and member.credit_usd is not None
            and member.spend_mtd >= self.steer_off_fraction * member.credit_usd
        ):
            return "promo credit at the steer-off line for new sessions"
        return None

    def _reassign(self, session_id: str, use: str, now: datetime, exclude: str) -> str | None:
        """The next eligible account for a LIVE session. Steer-off does not apply to it."""
        for member in self._ranked():
            if member.id != exclude and self.why_not(member, use, now, new_session=False) is None:
                self.sessions[session_id] = {"member": member.id, "use": use}
                return member.id
        return None

    @staticmethod
    def is_credit_refusal(status: int, error_type: str | None, message: str | None) -> bool:
        """Rule 4: a 402 ``billing_error``, or a 400 whose text says the balance is too low."""
        if status == _STATUS_CREDIT_JSON and error_type == "billing_error":
            return True
        return (
            status == 400 and bool(message) and "credit balance is too low" in str(message).lower()
        )

    # -- the API ----------------------------------------------------------------------------

    def assign(self, session_id: str, use: str, now: datetime) -> Decision:
        """Rule 2: seat a new session on the lowest-ranked eligible account, or on NONE."""
        with self.lock:
            self._advance_time(now)
            for member in self._ranked():
                if self.why_not(member, use, now, new_session=True) is None:
                    self.sessions[session_id] = {"member": member.id, "use": use}
                    return Decision(member=member.id)
            return Decision(member=NO_MEMBER)

    def handle_response(
        self,
        session_id: str,
        status: int,
        now: datetime,
        *,
        error_type: str | None = None,
        message: str | None = None,
        retry_after: float | None = None,
        stop_reason: str | None = None,
    ) -> Decision:
        """Rules 3 to 8: what to do with one HTTP response to a session's request."""
        with self.lock:
            self._advance_time(now)
            session = self.sessions.get(session_id)
            if not session:
                return Decision(member=NO_MEMBER)
            member_id, use = session["member"], session["use"]
            member = self.members[member_id]

            if self.is_credit_refusal(status, error_type, message):
                grace_end = _month_start(now) + timedelta(hours=self._reset_grace["grace_h"])
                if now < grace_end:
                    # Credit may not have reset yet when month-start re-enabled the account.
                    member.exhausted_until = now + timedelta(hours=self._reset_grace["recheck_h"])
                else:
                    member.exhausted_until = _next_month_start(now)
                return self._moved(session_id, use, now, member_id, alert=False)

            if status == 429:
                run = self._run_429.get(session_id, 0) + 1
                self._run_429[session_id] = run
                window_s = self._rate_limit["window_s"]
                member.recent_429s = [
                    t for t in member.recent_429s if (now - t).total_seconds() <= window_s
                ]
                member.recent_429s.append(now)
                if len(member.recent_429s) >= self._rate_limit["cooldown_after"]:
                    member.cool_until = now + timedelta(seconds=self._rate_limit["cooldown_s"])
                return Decision(
                    member=member_id,
                    retry=run <= self._rate_limit["max_retries"],
                    retry_after=retry_after,
                )

            if status in _AUTH_STATUSES:
                member.invalid = True
                return self._moved(session_id, use, now, member_id, alert=True)

            if status == _OVERLOAD_STATUS or 500 <= status < 600:
                run = self._run_5xx.get(session_id, 0) + 1
                self._run_5xx[session_id] = run
                return Decision(member=member_id, retry=run <= self._overload["max_retries"])

            if status == 200 and stop_reason == "refusal":
                return Decision(member=member_id)
            if 400 <= status < 500:
                return Decision(member=member_id)

            self._run_429.pop(session_id, None)
            self._run_5xx.pop(session_id, None)
            return Decision(member=member_id)

    def _moved(
        self, session_id: str, use: str, now: datetime, failed: str, *, alert: bool
    ) -> Decision:
        """The session's account refused it for good: seat it on the next one and retry there."""
        target = self._reassign(session_id, use, now, exclude=failed)
        self._run_429.pop(session_id, None)
        self._run_5xx.pop(session_id, None)
        if target is None:
            return Decision(member=NO_MEMBER, moved=True, alert=alert)
        return Decision(member=target, retry=True, moved=True, alert=alert)

    def add_spend(self, member_id: str, usd: float, now: datetime) -> None:
        """Count a call's cost against its account's month-to-date spend."""
        with self.lock:
            self._advance_time(now)
            self.members[member_id].spend_mtd += usd

    def raise_spend_to(self, totals: Mapping[str, float], now: datetime) -> None:
        """Take the larger of what this process counted and what the ledger holds per account."""
        with self.lock:
            self._advance_time(now)
            for member in self.members.values():
                member.spend_mtd = max(member.spend_mtd, totals.get(member.id, 0.0))

    def forget(self, session_id: str) -> None:
        """Drop a finished session's seat and retry counters."""
        with self.lock:
            self.sessions.pop(session_id, None)
            self._run_429.pop(session_id, None)
            self._run_5xx.pop(session_id, None)

    def get_member_state(self, member_id: str) -> dict[str, Any]:
        return self.members[member_id].snapshot()

    def unavailable(self, use: str, now: datetime, *, new_session: bool) -> str:
        """Every account that may not serve ``use``, with why: the text of a refused call.

        ``new_session`` is the kind of assignment that found none, because the steer-off line
        applies to a new session only and the reasons must be the ones that applied.
        """
        with self.lock:
            reasons = [
                f"{m.id} (slot {m.slot}): {why}"
                for m in self._ranked()
                if (why := self.why_not(m, use, now, new_session=new_session)) is not None
            ]
        return "; ".join(reasons) or "none"


def apply_registry_overrides(
    registry: Mapping[str, Any], overrides: Mapping[str, Any] | None
) -> dict[str, Any]:
    """A copy of the registry with per-account fields replaced (the vectors use this)."""
    out: dict[str, Any] = json.loads(json.dumps(registry))
    for member in out["members"]:
        member.update((overrides or {}).get("members", {}).get(member["id"], {}))
    return out


def load_registry(path: Path | str | None = None) -> dict[str, Any]:
    """The registry at ``path``, or the one this module ships."""
    with open(path or DEFAULT_REGISTRY_PATH, encoding="utf-8") as handle:
        loaded: dict[str, Any] = json.load(handle)
    return loaded


# ---------------------------------------------------------------------------------------------
# The ledger
# ---------------------------------------------------------------------------------------------


class PoolLedger:
    """One JSON row per call, in ``anthropic-pool-ledger-YYYY-MM.jsonl`` under ``directory``.

    The file is named for the UTC month, so a rollover starts an empty one and no row is ever
    deleted. Several processes on a machine append to the same file; the totals a new session is
    seated by are read back from it, which is how an account several processes are draining is seen
    to be near its credit by all of them.
    """

    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self._warned = False

    def path_for(self, moment: datetime) -> Path:
        return self.directory / f"anthropic-pool-ledger-{moment:%Y-%m}.jsonl"

    def append(self, row: Mapping[str, Any], moment: datetime) -> None:
        """Add one row. A ledger that cannot be written is said so ONCE, and the call goes on:
        the call already happened and was billed, and refusing its result would not unbill it."""
        line = (json.dumps(row, separators=(",", ":")) + "\n").encode("utf-8")
        try:
            self.directory.mkdir(parents=True, exist_ok=True)
            fd = os.open(self.path_for(moment), os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
            try:
                os.write(fd, line)
            finally:
                os.close(fd)
        except OSError as exc:
            if not self._warned:
                self._warned = True
                logger.warning(
                    "anthropic pool ledger %s was not written (%s); month-to-date spend "
                    "will read low",
                    self.directory,
                    exc,
                )

    def month_totals(self, moment: datetime) -> dict[str, float]:
        """Estimated dollars per account over the month ``moment`` falls in. A row that is not
        readable counts for nothing."""
        totals: dict[str, float] = {}
        try:
            text = self.path_for(moment).read_text(encoding="utf-8")
        except OSError:
            return totals
        for line in text.splitlines():
            try:
                row = json.loads(line)
                member = row["member"]
                totals[member] = totals.get(member, 0.0) + float(row["est_usd"])
            except (ValueError, KeyError, TypeError):
                continue
        return totals


# ---------------------------------------------------------------------------------------------
# Reading a refusal
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class RefusalFacts:
    """What a failed call says about itself: the HTTP status, Anthropic's error type, the text."""

    status: int | None
    error_type: str | None
    message: str


def refusal_facts(exc: BaseException) -> RefusalFacts:
    """Read the status, the error type and the text off an exception litellm raised."""
    for _ in range(4):  # a wrapper is read as the failure it wraps
        inner = getattr(exc, "original_exception", None)
        if not isinstance(inner, BaseException) or inner is exc:
            break
        exc = inner
    raw_status = getattr(exc, "status_code", None)
    status = (
        raw_status if isinstance(raw_status, int) and not isinstance(raw_status, bool) else None
    )
    text = str(exc)
    error_type: str | None = None
    body = getattr(exc, "body", None)
    if isinstance(body, Mapping):
        nested = body.get("error")
        candidate = nested.get("type") if isinstance(nested, Mapping) else body.get("type")
        if isinstance(candidate, str) and candidate != "error":
            error_type = candidate
    if error_type is None and (found := _ERROR_TYPE.search(text)):
        error_type = found.group(1)
    if error_type in _STATUS_OF_UNMAPPED_TYPE and (status is None or status >= 500):
        # No real 5xx carries these types, so a 5xx beside one is the mapper's stand-in status.
        status = _STATUS_OF_UNMAPPED_TYPE[error_type]
    return RefusalFacts(status=status, error_type=error_type, message=text)


# ---------------------------------------------------------------------------------------------
# The wiring
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class PoolCredentials:
    """What one request carries for its account. Nothing here is printed."""

    member: str
    slot: str
    api_key: str = field(repr=False)
    workspace_id: str = field(default="", repr=False)


class PoolEmpty(Exception):
    """No account may serve this session. The text names accounts and reasons, never a key."""


class PoolRuntime:
    """The pool of one process: the engine every session shares, plus the ledger."""

    def __init__(
        self,
        registry: Mapping[str, Any],
        *,
        env: Mapping[str, str],
        ledger: PoolLedger | None = None,
        clock: Callable[[], datetime] = _utcnow,
    ) -> None:
        self.registry = registry
        self.env = env
        self.ledger = ledger
        self._clock = clock
        self.pool = Pool(registry, self._slots_with_keys(), {}, clock())
        #: The most account moves one call may make: each move marks an account spent or invalid,
        #: so a chain ends by itself, and this only bounds a chain nothing else would end.
        self.hop_limit = len(registry["members"]) + 1

    @property
    def models(self) -> list[str]:
        return list(self.registry.get("models", []))

    def _variable(self, kind: str, slot: str) -> str:
        return str(self.registry["secrets"][kind]).replace("{slot}", slot)

    def _slots_with_keys(self) -> set[str]:
        return {
            m["slot"]
            for m in self.registry["members"]
            if self.env.get(self._variable("api_key", m["slot"]), "").strip()
        }

    def now(self) -> datetime:
        return self._clock()

    def credentials_for(self, ring: AnthropicPoolRing) -> PoolCredentials:
        """The account ``ring``'s session is on, seating it first when it has no seat."""
        with self.pool.lock:
            now = self._clock()
            seat = ring.member
            if seat is None:
                self.pool.env_present = self._slots_with_keys()
                if self.ledger is not None:
                    self.pool.raise_spend_to(self.ledger.month_totals(now), now)
                seat = self.pool.assign(ring.session_id, ring.use, now).member
                if seat == NO_MEMBER:
                    raise PoolEmpty(
                        f"no Anthropic pool account can serve use {ring.use!r}: "
                        f"{self.pool.unavailable(ring.use, now, new_session=True)}"
                    )
            member = self.pool.members[seat]
            return PoolCredentials(
                member=member.id,
                slot=member.slot,
                api_key=self.env.get(self._variable("api_key", member.slot), "").strip(),
                workspace_id=self.env.get(self._variable("workspace_id", member.slot), "").strip(),
            )

    def answer(
        self,
        ring: AnthropicPoolRing,
        used: str,
        facts: RefusalFacts,
        retry_after: float | None,
    ) -> Decision | None:
        """What to do after a call on account ``used`` failed. None: the pool has no opinion."""
        if facts.status is None:
            return None
        with self.pool.lock:
            now = self._clock()
            seated = ring.member
            credit = Pool.is_credit_refusal(facts.status, facts.error_type, facts.message)
            if seated != used:
                # A concurrent call on this session already moved it: this refusal belongs to the
                # account it left, which is already dealt with. Say so and re-issue on the new one.
                if (credit or facts.status in _AUTH_STATUSES) and seated is not None:
                    return Decision(member=seated, retry=True, moved=True)
                return None
            decision = self.pool.handle_response(
                ring.session_id,
                facts.status,
                now,
                error_type=facts.error_type,
                message=facts.message,
                retry_after=retry_after,
            )
            if credit:
                logger.warning(
                    "anthropic pool: %s is out of credit until %s; session moves to %s",
                    used,
                    format_iso(self.pool.members[used].exhausted_until),
                    decision.member,
                )
            if decision.alert:
                logger.error(
                    "anthropic pool ALERT: %s (slot %s) refused its key (HTTP %s) and is "
                    "invalid until restart; session moves to %s",
                    used,
                    self.pool.members[used].slot,
                    facts.status,
                    decision.member,
                )
            return decision

    def record(self, ring: AnthropicPoolRing, used: str, usage: Usage, model: str) -> None:
        """A call succeeded on ``used``: clear the retry run, count the spend, write the row."""
        with self.pool.lock:
            now = self._clock()
            if ring.member == used:
                self.pool.handle_response(ring.session_id, 200, now)
            cost = 0.0 if usage.cost_unpriced else usage.cost_usd
            self.pool.add_spend(used, cost, now)
        if self.ledger is not None:
            written = usage.cache_creation_tokens
            row: dict[str, Any] = {
                "ts": format_iso(now),
                "member": used,
                "model": model.removeprefix("anthropic/"),
                "input_uncached": max(usage.prompt_tokens - usage.cache_read_tokens - written, 0),
                "cache_read": usage.cache_read_tokens,
                "cache_write": written,
                "output": usage.completion_tokens,
                "est_usd": cost,
                "session": ring.session_id,
            }
            if usage.cost_unpriced:
                row["unpriced"] = True
            self.ledger.append(row, now)

    def release(self, session_id: str) -> None:
        self.pool.forget(session_id)


class AnthropicPoolRing:
    """One provider's seat in the pool.

    The provider calls :meth:`apply` for every request it builds and :meth:`on_error` for every
    failure; the seat is the account the pool gave the session, and it changes only when a refusal
    moves it. The seat lives in the pool, not here, so there is one copy of it.
    """

    def __init__(self, runtime: PoolRuntime, use: str = DEFAULT_USE) -> None:
        self._runtime = runtime
        self.use = use
        self.session_id = uuid.uuid4().hex
        weakref.finalize(self, runtime.release, self.session_id)

    @property
    def member(self) -> str | None:
        """The account id this session is on, or None before its first request."""
        return self._runtime.pool.sessions.get(self.session_id, {}).get("member")

    @property
    def hop_limit(self) -> int:
        return self._runtime.hop_limit

    def apply(self, call_kwargs: dict[str, Any]) -> str:
        """Put this session's account on a request, and return the account id it names.

        Raises :class:`PoolEmpty` when no account can serve the session.
        """
        creds = self._runtime.credentials_for(self)
        registry = self._runtime.registry
        call_kwargs["api_key"] = creds.api_key
        if registry.get("base_url"):
            call_kwargs["api_base"] = registry["base_url"]
        headers = dict(call_kwargs.get("extra_headers") or {})
        headers.update(registry.get("headers", {}))
        if creds.workspace_id:
            headers[registry["workspace_header"]] = creds.workspace_id
        call_kwargs["extra_headers"] = headers
        return creds.member

    def on_error(
        self, exc: BaseException, *, used: str, retry_after: float | None = None
    ) -> Decision | None:
        """Tell the pool what a failed call was. A decision with ``moved`` set means the session
        is on another account now (or on none, when ``member`` is NONE)."""
        return self._runtime.answer(self, used, refusal_facts(exc), retry_after)

    def on_success(self, usage: Usage, model: str, *, used: str) -> None:
        self._runtime.record(self, used, usage, model)

    def unavailable(self) -> str:
        """Why no account is left for this session, for the error a refused call raises."""
        text = self._runtime.pool.unavailable(self.use, self._runtime.now(), new_session=False)
        return f"every Anthropic pool account is spent or unavailable for use {self.use!r}: {text}"


# ---------------------------------------------------------------------------------------------
# One runtime per process
# ---------------------------------------------------------------------------------------------

_RUNTIMES: dict[tuple[str, str], PoolRuntime] = {}
_RUNTIMES_LOCK = threading.Lock()


def reset_runtimes() -> None:
    """Forget every process-wide runtime (tests)."""
    with _RUNTIMES_LOCK:
        _RUNTIMES.clear()


def runtime_for(settings: Any, env: Mapping[str, str] | None = None) -> PoolRuntime:
    """The runtime for the registry and ledger ``settings`` name.

    Built once per process for the process environment, so every session shares what any of them
    learns. A caller that passes its own ``env`` (a test, an embedding) gets a runtime of its own.
    """
    from zakcode.config import zakcode_home  # lazy: config imports this package

    registry_path = getattr(settings, "anthropic_pool_registry", None) or DEFAULT_REGISTRY_PATH
    ledger_dir = getattr(settings, "anthropic_pool_ledger_dir", None) or (
        zakcode_home() / "anthropic-pool"
    )

    def build(environment: Mapping[str, str]) -> PoolRuntime:
        return PoolRuntime(
            load_registry(registry_path), env=environment, ledger=PoolLedger(Path(ledger_dir))
        )

    if env is not None:
        return build(env)
    key = (str(registry_path), str(ledger_dir))
    with _RUNTIMES_LOCK:
        if key not in _RUNTIMES:
            _RUNTIMES[key] = build(os.environ)
        return _RUNTIMES[key]


def ring_for(
    settings: Any, *, model: str, via_gateway: bool, env: Mapping[str, str] | None = None
) -> AnthropicPoolRing | None:
    """The seat a provider should take, or None when the pool is off or does not serve the route.

    Off unless ``anthropic_pool`` is set, and never inferred. Never on for a home under
    ``hipaa_mode``: patient data does not go through accounts that several runs share. Never behind
    a gateway, which holds its own credentials. Only for the models the registry names.
    """
    if settings is None or not getattr(settings, "anthropic_pool", False):
        return None
    if getattr(settings, "hipaa_mode", False) or via_gateway:
        return None
    runtime = runtime_for(settings, env)
    bare = model.removeprefix("anthropic/")
    if bare not in runtime.models:
        return None
    return AnthropicPoolRing(
        runtime, use=getattr(settings, "anthropic_pool_use", None) or DEFAULT_USE
    )
