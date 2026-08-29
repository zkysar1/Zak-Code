"""Tests for ``zakcode throughput`` (ADR-0104) — session documents in, the turn-time table out."""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path

import pytest
from rich.console import Console

from zakcode.cli import throughput as tp
from zakcode.cli._theme import ZAK_THEME
from zakcode.messages import Message
from zakcode.session.store import Session, SessionStore
from zakcode.usage import Usage

NOW = datetime(2026, 8, 29, 22, 0, 0)


def _stamp(offset_s: float) -> str:
    return (NOW + timedelta(seconds=offset_s)).isoformat() + "Z"


def _session(
    session_id: str,
    turns: list[tuple[float, float, int, int, int]],
    *,
    extra_assistant: int = 0,
) -> Session:
    """A session whose i-th turn began ``began_s`` and ended ``ended_s`` after NOW.

    Each tuple is ``(began_s, ended_s, prompt, cached, completion)``. ``extra_assistant``
    prepends assistant messages that have NO usage record (an older interrupted turn).
    """
    session = Session(id=session_id, cwd="/tmp/x", model="openai/zds-qwen3.6-35b")
    for k in range(extra_assistant):
        session.add_message(
            Message(role="user", blocks=[], created_at=_stamp(-100000 - 10 * k - 5))
        )
        session.add_message(
            Message(role="assistant", blocks=[], created_at=_stamp(-100000 - 10 * k))
        )
    for began, ended, prompt, cached, completion in turns:
        session.add_message(Message(role="user", blocks=[], created_at=_stamp(began)))
        session.add_message(Message(role="assistant", blocks=[], created_at=_stamp(ended)))
        session.add_usage(
            Usage(
                prompt_tokens=prompt,
                completion_tokens=completion,
                total_tokens=prompt + completion,
                cache_read_tokens=cached,
            )
        )
    return session


def _rec_console() -> Console:
    return Console(theme=ZAK_THEME, highlight=False, record=True, width=120, force_terminal=False)


class TestSessionTurns:
    def test_latency_runs_from_the_message_before_to_the_assistant_message(self) -> None:
        s = _session("aaaa", [(-120, -60, 60000, 58000, 300), (-30, -10, 61000, 60500, 100)])
        turns = tp.session_turns(s, since=NOW - timedelta(hours=1))
        assert [t.latency_s for t in turns] == [60.0, 20.0]
        assert [t.completion_tokens for t in turns] == [300, 100]
        assert turns[0].tokens_per_second == pytest.approx(5.0)

    def test_an_idle_gap_is_not_a_turn(self) -> None:
        s = _session("bbbb", [(-4000, -100, 60000, 58000, 300), (-30, -10, 61000, 60500, 100)])
        turns = tp.session_turns(s, since=NOW - timedelta(hours=2))
        assert [t.latency_s for t in turns] == [20.0]

    def test_usages_align_to_the_newest_assistant_messages(self) -> None:
        # Two older assistant messages carry no usage: the two records that exist belong
        # to the two NEWEST replies, not the two oldest.
        s = _session(
            "cccc",
            [(-120, -60, 60000, 58000, 300), (-30, -10, 61000, 60500, 100)],
            extra_assistant=2,
        )
        turns = tp.session_turns(s, since=NOW - timedelta(days=3))
        assert [t.latency_s for t in turns] == [60.0, 20.0]
        assert [t.completion_tokens for t in turns] == [300, 100]

    def test_turns_before_the_window_are_left_out(self) -> None:
        s = _session("dddd", [(-7200, -7150, 1, 0, 1), (-30, -10, 61000, 60500, 100)])
        turns = tp.session_turns(s, since=NOW - timedelta(hours=1))
        assert [t.completion_tokens for t in turns] == [100]

    def test_no_usage_records_means_no_turns(self) -> None:
        s = Session(id="eeee", cwd="/tmp/x", model="m")
        s.add_message(Message.user("hi"))
        s.add_message(Message.assistant_text("hello"))
        assert tp.session_turns(s, since=NOW - timedelta(days=1)) == []


class TestBuildReport:
    def test_rows_summarise_each_session_and_skip_a_corrupt_one(self, tmp_path: Path) -> None:
        store = SessionStore(tmp_path)
        store.save(
            _session("aaaa", [(-120, -60, 60000, 58000, 300), (-30, -10, 61000, 60500, 100)])
        )
        store.save(_session("bbbb", [(-300, -200, 40000, 0, 1000)]))
        (tmp_path / "cccc.json").write_text("{not json", encoding="utf-8")
        report = tp.build_report(store, hours=1, now=NOW)
        assert [r.session for r in report.rows] == ["aaaa", "bbbb"]
        a = report.rows[0]
        assert a.turns == 2
        assert a.p50_latency_s == 40.0
        assert a.p90_latency_s == 60.0
        assert a.median_output_tokens == 200
        assert a.cache_hit_share == pytest.approx((58000 + 60500) / (60000 + 61000), abs=1e-3)
        assert a.median_tokens_per_second == 5.0
        assert report.rows[1].cache_hit_share == 0.0
        assert report.turns_total == 3
        assert report.skipped == ["cccc: SessionCorruptError"]

    def test_a_session_file_older_than_the_window_is_not_opened(self, tmp_path: Path) -> None:
        store = SessionStore(tmp_path)
        path = store.save(_session("aaaa", [(-120, -60, 60000, 58000, 300)]))
        import os

        old = (NOW - timedelta(hours=5)).timestamp()
        os.utime(path, (old, old))
        report = tp.build_report(store, hours=1, now=NOW)
        assert report.rows == [] and report.turns_total == 0

    def test_json_carries_the_rows_and_the_queue_factor(self, tmp_path: Path) -> None:
        store = SessionStore(tmp_path)
        store.save(_session("aaaa", [(-120, -60, 60000, 58000, 300)]))
        cap = tp.Capacity(ok=True, model="m", replicas=4, slots_total=4, inflight_now=8)
        payload = json.loads(tp.build_report(store, hours=1, now=NOW, capacity=cap).to_json())
        assert payload["rows"][0]["session"] == "aaaa"
        assert payload["capacity"]["queue_factor"] == 2.0
        assert payload["turns_total"] == 1


class TestCapacity:
    def test_reads_the_zds_block_of_the_configured_model(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        listing = {
            "data": [
                {
                    "id": "zds-qwen3.5-35b",
                    "zds": {
                        "canonical": "zds-qwen3.6-35b",
                        "replicas": 4,
                        "slots_total": 4,
                        "inflight_now": 7,
                        "capacity_available": 5,
                        "max_inflight_total": 12,
                    },
                }
            ]
        }
        seen: dict[str, object] = {}

        def fake_fetch(api_base: str, api_key: str | None, timeout: float) -> object:
            seen.update(api_base=api_base, api_key=api_key, timeout=timeout)
            return listing

        monkeypatch.setattr("zakcode.providers.litellm_provider._fetch_models", fake_fetch)
        cap = tp.fetch_capacity("http://router:9090/v1", "sk-secret", "openai/zds-qwen3.6-35b")
        assert cap.ok and cap.slots_total == 4 and cap.inflight_now == 7
        assert cap.queue_factor == pytest.approx(1.75)
        assert seen["api_key"] == "sk-secret" and seen["timeout"] == tp.CAPACITY_TIMEOUT

    def test_an_unreachable_router_is_a_note_that_names_no_secret(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def boom(api_base: str, api_key: str | None, timeout: float) -> object:
            raise OSError("connection refused to http://user:sk-secret@router")

        monkeypatch.setattr("zakcode.providers.litellm_provider._fetch_models", boom)
        cap = tp.fetch_capacity("http://user:sk-secret@router:9090/v1", "sk-secret", "m")
        assert not cap.ok
        assert "OSError" in cap.detail
        assert "sk-secret" not in cap.detail and "router" not in cap.detail

    def test_no_api_base_asks_nothing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def boom(*args: object) -> object:
            raise AssertionError("must not be called")

        monkeypatch.setattr("zakcode.providers.litellm_provider._fetch_models", boom)
        cap = tp.fetch_capacity(None, None, "m")
        assert not cap.ok and "no api_base" in cap.detail

    def test_a_listing_without_the_model_is_a_note(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "zakcode.providers.litellm_provider._fetch_models",
            lambda *a: {"data": [{"id": "other", "zds": {"slots_total": 1}}]},
        )
        cap = tp.fetch_capacity("http://r/v1", None, "mine")
        assert not cap.ok and "no zds block" in cap.detail


class TestRender:
    def test_table_and_queue_warning(self, tmp_path: Path) -> None:
        store = SessionStore(tmp_path)
        store.save(_session("abcdef12", [(-120, -60, 60000, 58000, 300)]))
        cap = tp.Capacity(ok=True, model="zds", replicas=4, slots_total=4, inflight_now=8)
        console = _rec_console()
        tp.render(tp.build_report(store, hours=1, now=NOW, capacity=cap), console)
        text = console.export_text()
        assert "abcdef12" in text and "60s" in text and "97%" in text
        assert "2.0 requests per slot" in text and "more engines" in text

    def test_unreachable_router_prints_the_note_not_a_traceback(self, tmp_path: Path) -> None:
        console = _rec_console()
        report = tp.build_report(SessionStore(tmp_path), hours=1, now=NOW)
        report.capacity = tp.Capacity(ok=False, detail="/models unreachable (OSError)")
        tp.render(report, console)
        text = console.export_text()
        assert "no assistant turns" in text and "unreachable (OSError)" in text


class TestCommand:
    def test_is_registered_on_the_root_app(self) -> None:
        from zakcode.cli import app

        names = {
            c.name or (c.callback.__name__ if c.callback else "") for c in app.registered_commands
        }
        assert "throughput" in names

    def test_json_flag_prints_the_report(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        store = SessionStore(tmp_path)
        store.save(_session("aaaa", [(-60, -30, 100, 90, 10)]))
        monkeypatch.setattr(tp, "datetime", _FrozenDatetime)
        tp.throughput(hours=24 * 400, store_dir=tmp_path, as_json=True, no_router=True)
        payload = json.loads(capsys.readouterr().out)
        assert payload["rows"][0]["session"] == "aaaa" and payload["capacity"] is None


class _FrozenDatetime(datetime):
    """``datetime.now`` pinned to NOW so the JSON test is not calendar-dependent."""

    @classmethod
    def now(cls, tz: object = None) -> datetime:  # type: ignore[override]
        from datetime import UTC

        return NOW.replace(tzinfo=UTC) if tz is not None else NOW
