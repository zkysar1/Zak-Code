"""``zakcode throughput`` — where a fleet's turn time goes (ADR-0104).

The question "are we bottlenecked, and on what?" was answered on 2026-08-29 by hand: eight
sessions' documents read with an ad-hoc script, paired with the router's ``/v1/models``
listing. The answer (four engine slots, seven to eight requests in flight, a p50 turn of
~55 s of which ~10 s was decode) drove a hardware decision. Nothing in the CLI could show
it, so the next person would reconstruct it — or guess. This command reads the same two
sources every persisted session already carries and prints the table.

Per session, over the window: assistant turns, p50/p90 turn latency, median output and
prompt tokens, the prefix-cache hit share, and effective output tokens per second. A
turn's latency is the event-time gap from the message BEFORE the assistant message (the
tool result or prompt the model was answering) to the assistant message itself — model
queue + prefill + decode, which is what an operator waits for. A gap over
``IDLE_GAP_SECONDS`` is a session sitting idle, not a slow turn, and is left out. One
known distortion: a document persisted BEFORE event-time stamps (ADR-0049) loads every
message with a load-time stamp, so a resumed old transcript can pair a backfilled stamp
with a real one and read as a turn that never took that long — zero gaps are dropped, a
gap over the idle cap is dropped, and anything between is a lead to check against the
transcript, not a measurement (the Mind's guard-3265: histogram a stamp before pairing
on it). Documents written since ADR-0049 carry real event times.

The router line (``slots``, ``in flight``, ``capacity available``) comes from the
configured endpoint's ``/models`` listing when it carries the ``zds`` block; any other
server, or an unreachable one, degrades to a one-line note. The report is read-only and
never prints a key or a credentialed URL.
"""

from __future__ import annotations

import json
import statistics
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Annotated, Any

import typer
from rich.console import Console
from rich.table import Table

from zakcode.cli._layout import kv_table, notice_info, notice_warn
from zakcode.cli._theme import ZAK_THEME
from zakcode.session.store import Session, SessionError, SessionStore
from zakcode.usage import Usage

__all__ = [
    "IDLE_GAP_SECONDS",
    "Capacity",
    "Report",
    "SessionRow",
    "Turn",
    "build_report",
    "fetch_capacity",
    "register_throughput_command",
    "render",
    "session_turns",
    "throughput",
]

#: A gap longer than this between a message and the assistant reply that follows it is a
#: session waiting (for a wake-up, an operator, a parked reducer) — not a turn the model
#: took that long over. Half an hour: the longest measured genuine turn was ~4 min.
IDLE_GAP_SECONDS = 1800.0

#: The ``/models`` listing is one GET off the request path; a slow router costs at most this.
CAPACITY_TIMEOUT = 3.0


@dataclass(frozen=True)
class Turn:
    """One assistant turn with the usage record that produced it."""

    session: str
    at: datetime
    latency_s: float
    prompt_tokens: int
    cache_read_tokens: int
    completion_tokens: int

    @property
    def tokens_per_second(self) -> float:
        return self.completion_tokens / self.latency_s if self.latency_s > 0 else 0.0


@dataclass(frozen=True)
class SessionRow:
    """A session's turns over the window, summarised."""

    session: str
    turns: int
    p50_latency_s: float
    p90_latency_s: float
    median_output_tokens: int
    median_prompt_tokens: int
    cache_hit_share: float
    median_tokens_per_second: float
    last_turn_at: str


@dataclass(frozen=True)
class Capacity:
    """The router's slot picture, or why it could not be read."""

    ok: bool
    detail: str = ""
    model: str = ""
    replicas: int | None = None
    slots_total: int | None = None
    inflight_now: int | None = None
    capacity_available: int | None = None
    max_inflight_total: int | None = None

    @property
    def queue_factor(self) -> float | None:
        """Requests in flight per slot — above 1.0 every request is waiting on another."""
        if self.slots_total and self.inflight_now is not None:
            return self.inflight_now / self.slots_total
        return None


@dataclass
class Report:
    window_hours: float
    generated_at: str
    rows: list[SessionRow] = field(default_factory=list)
    turns_total: int = 0
    skipped: list[str] = field(default_factory=list)
    capacity: Capacity | None = None

    def to_json(self) -> str:
        payload: dict[str, Any] = asdict(self)
        if self.capacity is not None:
            payload["capacity"]["queue_factor"] = self.capacity.queue_factor
        return json.dumps(payload, indent=2, sort_keys=True)


def _parse_iso(stamp: str) -> datetime | None:
    """A message ``created_at`` as a naive UTC datetime, or ``None`` when unparseable."""
    try:
        parsed = datetime.fromisoformat(stamp.replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    if parsed.tzinfo is not None:
        parsed = parsed.astimezone(UTC).replace(tzinfo=None)
    return parsed


def _paired(session: Session) -> list[tuple[int, Usage]]:
    """Assistant message indexes zipped with their usage records, aligned from the TAIL.

    A usage is appended per model call and an assistant message per reply, so the two
    lists run in step — except where a call ended without a reply (an interrupted turn,
    a provider error) or a compaction wrote a message without a call. Aligning the last
    ``n`` of each keeps the most recent turns exact, which is where a throughput question
    lives; the oldest ``|len(a) - len(b)|`` records are dropped rather than mis-paired.
    """
    assistant = [i for i, m in enumerate(session.messages) if m.role == "assistant"]
    usages = session.usages
    n = min(len(assistant), len(usages))
    if n == 0:
        return []
    return list(zip(assistant[-n:], usages[-n:], strict=True))


def session_turns(session: Session, *, since: datetime) -> list[Turn]:
    """The session's assistant turns at or after ``since`` (naive UTC), oldest first."""
    turns: list[Turn] = []
    for index, usage in _paired(session):
        if index == 0:
            continue
        ended = _parse_iso(session.messages[index].created_at)
        began = _parse_iso(session.messages[index - 1].created_at)
        if ended is None or began is None or ended < since:
            continue
        latency = (ended - began).total_seconds()
        if latency <= 0 or latency > IDLE_GAP_SECONDS:
            continue
        turns.append(
            Turn(
                session=session.id,
                at=ended,
                latency_s=latency,
                prompt_tokens=usage.prompt_tokens,
                cache_read_tokens=usage.cache_read_tokens,
                completion_tokens=usage.completion_tokens,
            )
        )
    return turns


def _percentile(values: list[float], share: float) -> float:
    ordered = sorted(values)
    position = max(0, min(len(ordered) - 1, round(share * len(ordered)) - 1))
    return ordered[position]


def _row(session_id: str, turns: list[Turn]) -> SessionRow:
    latencies = [t.latency_s for t in turns]
    prompts = [t.prompt_tokens for t in turns]
    cached = sum(t.cache_read_tokens for t in turns)
    prompt_total = sum(prompts)
    return SessionRow(
        session=session_id,
        turns=len(turns),
        p50_latency_s=round(statistics.median(latencies), 1),
        p90_latency_s=round(_percentile(latencies, 0.9), 1),
        median_output_tokens=int(statistics.median(t.completion_tokens for t in turns)),
        median_prompt_tokens=int(statistics.median(prompts)),
        cache_hit_share=round(cached / prompt_total, 3) if prompt_total else 0.0,
        median_tokens_per_second=round(statistics.median(t.tokens_per_second for t in turns), 1),
        last_turn_at=max(t.at for t in turns).isoformat(timespec="seconds"),
    )


def build_report(
    store: SessionStore,
    *,
    hours: float,
    now: datetime | None = None,
    capacity: Capacity | None = None,
) -> Report:
    """Summarise every session with a turn inside the last ``hours``.

    A session whose file is older than the window is not opened at all (its mtime is the
    cheap pre-filter); one that cannot be loaded is named under ``skipped`` and the
    report goes on — a corrupt transcript must not hide the seven healthy ones.
    """
    moment = now or datetime.now(UTC).replace(tzinfo=None)
    since = moment - timedelta(hours=hours)
    report = Report(
        window_hours=hours, generated_at=moment.isoformat(timespec="seconds"), capacity=capacity
    )
    for session_id, mtime in store.list_recent():
        if datetime.fromtimestamp(mtime, UTC).replace(tzinfo=None) < since:
            continue
        try:
            session = store.load(session_id)
        except SessionError as exc:
            report.skipped.append(f"{session_id}: {type(exc).__name__}")
            continue
        turns = session_turns(session, since=since)
        if not turns:
            continue
        report.rows.append(_row(session_id, turns))
        report.turns_total += len(turns)
    report.rows.sort(key=lambda r: r.turns, reverse=True)
    return report


def fetch_capacity(
    api_base: str | None,
    api_key: str | None,
    model: str,
    *,
    timeout: float = CAPACITY_TIMEOUT,
) -> Capacity:
    """The ``zds`` slot block for ``model`` from the endpoint's ``/models`` listing.

    Never raises and never echoes the key or the base URL: an unreachable or foreign
    server is a one-line ``detail`` naming the failure class, not a traceback.
    """
    if not api_base:
        return Capacity(ok=False, detail="no api_base configured — nothing to ask")
    from zakcode.providers.litellm_provider import _fetch_models, _strip_provider_prefix

    try:
        listing = _fetch_models(api_base, api_key, timeout)
    except Exception as exc:  # noqa: BLE001 — a probe; the report must still print
        return Capacity(ok=False, detail=f"/models unreachable ({type(exc).__name__})")
    entries = listing.get("data") if isinstance(listing, dict) else listing
    if not isinstance(entries, list):
        return Capacity(ok=False, detail="/models listing has no data array")
    wanted = {model.strip().lower(), _strip_provider_prefix(model.strip()).lower()}
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        zds = entry.get("zds")
        if not isinstance(zds, dict):
            continue
        if (
            str(entry.get("id", "")).lower() not in wanted
            and str(zds.get("canonical", "")).lower() not in wanted
        ):
            continue
        return Capacity(
            ok=True,
            model=str(zds.get("canonical") or entry.get("id") or model),
            replicas=_zds_int(zds, "replicas"),
            slots_total=_zds_int(zds, "slots_total"),
            inflight_now=_zds_int(zds, "inflight_now"),
            capacity_available=_zds_int(zds, "capacity_available"),
            max_inflight_total=_zds_int(zds, "max_inflight_total"),
        )
    return Capacity(ok=False, detail=f"/models lists no zds block for {model!r}")


def _zds_int(zds: dict[str, Any], key: str) -> int | None:
    """A positive-or-zero integer field of the ``zds`` block, or ``None`` (bool is not int)."""
    value = zds.get(key)
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def render(report: Report, console: Console) -> None:
    """Print the per-session table, the router line, and what they mean together."""
    if not report.rows:
        notice_info(console, f"no assistant turns in the last {report.window_hours:g}h")
    else:
        table = Table(
            title=f"turns in the last {report.window_hours:g}h  ({report.turns_total} total)",
            box=None,
            padding=(0, 1),
            pad_edge=False,
            header_style="banner.label",
        )
        table.add_column("session", no_wrap=True)
        for column in (
            "turns",
            "p50 lat",
            "p90 lat",
            "med out",
            "med prompt",
            "cache hit",
            "out tok/s",
        ):
            table.add_column(column, justify="right", no_wrap=True)
        table.add_column("last turn", no_wrap=True)
        for row in report.rows:
            table.add_row(
                row.session[:8],
                str(row.turns),
                f"{row.p50_latency_s:.0f}s",
                f"{row.p90_latency_s:.0f}s",
                str(row.median_output_tokens),
                f"{row.median_prompt_tokens:,}",
                f"{row.cache_hit_share:.0%}",
                f"{row.median_tokens_per_second:.1f}",
                row.last_turn_at[11:19],
            )
        console.print(table)
    for skipped in report.skipped:
        notice_warn(console, f"skipped {skipped}")

    capacity = report.capacity
    if capacity is None:
        return
    if not capacity.ok:
        notice_info(console, f"router: {capacity.detail}")
        return
    rows = [
        ("model", capacity.model),
        ("engines", str(capacity.replicas) if capacity.replicas is not None else "?"),
        ("slots", str(capacity.slots_total) if capacity.slots_total is not None else "?"),
        ("in flight", str(capacity.inflight_now) if capacity.inflight_now is not None else "?"),
        (
            "available",
            str(capacity.capacity_available) if capacity.capacity_available is not None else "?",
        ),
        (
            "max in flight",
            str(capacity.max_inflight_total) if capacity.max_inflight_total is not None else "?",
        ),
    ]
    console.print()
    console.print(kv_table(rows))
    factor = capacity.queue_factor
    if factor is None:
        return
    if factor > 1.0:
        notice_warn(
            console,
            f"{factor:.1f} requests per slot — each turn waits on ~{factor - 1:.1f} others; "
            "more engines shorten the queue, more sessions lengthen it",
        )
    else:
        notice_info(console, f"{factor:.1f} requests per slot — no queueing at the router")


def throughput(
    hours: Annotated[
        float, typer.Option("--hours", "-H", help="How far back to read turns.", min=0.05)
    ] = 1.0,
    store_dir: Annotated[
        Path | None,
        typer.Option(
            "--store", help="Session store to read (default: this user's ~/.zakcode/sessions)."
        ),
    ] = None,
    as_json: Annotated[bool, typer.Option("--json", help="Machine-readable report.")] = False,
    no_router: Annotated[
        bool, typer.Option("--no-router", help="Skip the endpoint's /models slot probe.")
    ] = False,
) -> None:
    """Where the turn time goes: per-session latency, tokens/s, cache hits, router slots.

    Reads the persisted session documents (read-only) and, unless ``--no-router``, the
    configured endpoint's ``/models`` listing for its ``zds`` slot block. Nothing is
    sent to a model; no key or credentialed URL is printed.
    """
    from zakcode.config import load_settings

    capacity: Capacity | None = None
    if not no_router:
        settings = load_settings()
        capacity = fetch_capacity(settings.api_base, settings.api_key, settings.default_model)
    store = SessionStore(store_dir) if store_dir is not None else SessionStore()
    report = build_report(store, hours=hours, capacity=capacity)
    if as_json:
        typer.echo(report.to_json())
        return
    render(report, Console(theme=ZAK_THEME, highlight=False))


def register_throughput_command(app: typer.Typer) -> None:
    """Attach ``zakcode throughput`` to the root Typer app."""
    app.command()(throughput)
