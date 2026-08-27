"""Resume safety (ADR-0033): build + stop-reason stamps and the compaction notice.

Field incident 2026-08-26 (serene): `zakcode update` printed "running chat sessions keep
the old build until restarted"; the process was not restarted, and the next `/resume`
replayed a transcript that had already collapsed once. Two facts a resume needs were not
recorded anywhere — which build wrote the document, and how its last turn ended. These
tests pin both stamps (both turn paths), the pure notice, and the append-only load path.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest

from zakcode.agent.loop import AgentLoop
from zakcode.build_info import build_commit
from zakcode.messages import Message
from zakcode.providers.base import (
    Capabilities,
    LLMResult,
    Provider,
    ProviderStreamEvent,
    StreamDone,
    StreamTextDelta,
)
from zakcode.session.store import RESUME_COMPACT_STOP_REASONS, Session, SessionStore
from zakcode.tools.base import ToolRegistry

# ── the pure notice ──────────────────────────────────────────────────────────


def test_same_build_and_clean_stop_is_quiet() -> None:
    session = Session(cwd=".", model="m", build="0c28c8b", last_stop_reason="completed")
    assert session.resume_notice(running_build="0c28c8b") is None


def test_build_mismatch_names_both_builds() -> None:
    session = Session(cwd=".", model="m", build="c4edaa4", last_stop_reason="completed")
    notice = session.resume_notice(running_build="0c28c8b")
    assert notice is not None
    assert "c4edaa4" in notice and "0c28c8b" in notice and "compacting" in notice


def test_unstamped_transcript_on_a_stamped_build_is_flagged() -> None:
    # Every pre-ADR-0033 document is unstamped — the serene transcript's exact shape.
    session = Session(cwd=".", model="m", last_stop_reason="completed")
    notice = session.resume_notice(running_build="0c28c8b")
    assert notice is not None and "older build" in notice


def test_unstamped_dev_resume_is_quiet() -> None:
    # An editable/dev install has no VCS build id: unstamped on both sides is a match.
    session = Session(cwd=".", model="m")
    assert session.resume_notice(running_build=None) is None


@pytest.mark.parametrize("reason", sorted(RESUME_COMPACT_STOP_REASONS))
def test_collapsed_last_turn_is_flagged(reason: str) -> None:
    session = Session(cwd=".", model="m", build="abc", last_stop_reason=reason)
    notice = session.resume_notice(running_build="abc")
    assert notice is not None and reason in notice


@pytest.mark.parametrize("reason", ["completed", "max_iterations", "provider_error", ""])
def test_ordinary_stop_reasons_are_quiet(reason: str) -> None:
    session = Session(cwd=".", model="m", build="abc", last_stop_reason=reason)
    assert session.resume_notice(running_build="abc") is None


def test_older_documents_load_with_empty_stamps() -> None:
    # Append-only schema v1: a document written before the stamps existed loads with the
    # empty defaults (and therefore reads as "an older build" on resume).
    session = Session.model_validate(
        {"version": 1, "id": "abc", "cwd": ".", "model": "m", "messages": []}
    )
    assert session.build == ""
    assert session.last_stop_reason == ""


# ── the stamps, written by the loop ──────────────────────────────────────────


class _Once(Provider):
    async def acomplete(
        self, messages: list[Message], *, system: str | None = None, tools: Any = None, **kw: Any
    ) -> LLMResult:
        return LLMResult(text="done")

    async def astream(
        self, messages: list[Message], *, system: str | None = None, tools: Any = None, **kw: Any
    ) -> AsyncIterator[ProviderStreamEvent]:
        yield StreamTextDelta(text="done")
        yield StreamDone(finish_reason="stop")

    def count_tokens(self, messages: list[Message], *, system: str | None = None) -> int:
        return 0

    def capabilities(self) -> Capabilities:
        return Capabilities(supports_tools=True, context_window=200_000)


def _loop(tmp_path: Path, store: SessionStore) -> AgentLoop:
    return AgentLoop(
        _Once(),
        ToolRegistry(),
        Session(cwd=str(tmp_path), model="test"),
        workspace_root=tmp_path,
        max_iterations=4,
        store=store,
    )


def test_buffered_turn_stamps_build_and_stop_reason(tmp_path: Path) -> None:
    store = SessionStore(tmp_path)
    loop = _loop(tmp_path, store)
    result = asyncio.run(loop.arun_turn("hi"))
    assert result.stop_reason == "completed"
    loaded = store.load(loop.session.id)
    assert loaded.last_stop_reason == "completed"
    assert loaded.build == (build_commit() or "")


def test_streaming_turn_stamps_build_and_stop_reason(tmp_path: Path) -> None:
    store = SessionStore(tmp_path)
    loop = _loop(tmp_path, store)

    async def _drain() -> None:
        async for _ev in loop.astream_turn("hi"):
            pass

    asyncio.run(_drain())
    loaded = store.load(loop.session.id)
    assert loaded.last_stop_reason == "completed"
    assert loaded.build == (build_commit() or "")
