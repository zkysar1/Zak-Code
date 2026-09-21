"""The transcript on disk is an append-only record, as Claude Code's is (ADR-0206).

Until ADR-0206 the file handed to hooks as ``transcript_path`` was re-rendered from the LIVE
history on every fire, so each compaction erased everything before it from disk. Measured on
a served run of 575 calls and 19 compactions: the file held 64 lines at the end, and nothing
on the box could say what the loop had been doing in its last 300 calls. Claude Code's own
transcript never loses a line: a compaction ADDS a ``compact_boundary`` row and a row flagged
``isCompactSummary`` (shape read off a real Claude Code transcript, keys only), and
consumers written against it count those rows and walk forward from them.

Every test reads the FILE, the way a consumer would. The two readers at the foot are written
here from the read pattern, not copied from any framework.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

from zakcode.agent.compact import (
    CONTINUATION_NOTE,
    ELISION_MARKER,
    SUMMARY_MARKER,
    CompactionConfig,
    Compactor,
)
from zakcode.agent.loop import AgentLoop
from zakcode.hooks.transcript import render_compaction_rows
from zakcode.messages import Message, ToolResultBlock, ToolUseBlock
from zakcode.providers.base import Capabilities, LLMResult, Provider
from zakcode.session.store import Session, SessionStore
from zakcode.tools.base import ToolRegistry


class _Summarizer(Provider):
    """Answers every completion with one canned summary; a fixed token count."""

    def __init__(self, summary: str = "what happened so far") -> None:
        self._summary = summary

    async def acomplete(
        self, messages: list[Message], *, system: str | None = None, tools: Any = None, **kw: Any
    ) -> LLMResult:
        return LLMResult(text=self._summary)

    def count_tokens(self, messages: list[Message], *, system: str | None = None) -> int:
        return 100

    def capabilities(self) -> Capabilities:
        return Capabilities(supports_tools=True, context_window=8192)


def _loop(
    tmp_path: Path,
    *,
    store: SessionStore | None = None,
    session: Session | None = None,
    keep: int = 2,
) -> AgentLoop:
    return AgentLoop(
        _Summarizer(),
        ToolRegistry(),
        session or Session(cwd=str(tmp_path), model="test"),
        workspace_root=tmp_path,
        store=store,
        compactor=Compactor(CompactionConfig(preserve_recent=keep)),
    )


def _say(loop: AgentLoop, *texts: str) -> None:
    """Alternate user / assistant messages: ``q0, a0, q1, a1, ...``."""
    for index, text in enumerate(texts):
        make = Message.user if index % 2 == 0 else Message.assistant_text
        loop.session.add_message(make(text))


def _rows(path: str | Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()]


def _said(row: dict[str, Any]) -> str:
    """The prose of one row, whichever of the two content shapes it uses."""
    content = row.get("message", {}).get("content", row.get("content", ""))
    if isinstance(content, str):
        return content
    return "".join(str(block.get("text") or block.get("content") or "") for block in content)


# ── one line per message, however often the file is asked for ────────────────────────


def test_a_message_is_written_once_however_often_the_file_is_asked_for(tmp_path: Path) -> None:
    loop = _loop(tmp_path)
    _say(loop, "q0", "a0")
    first = loop._cc_transcript_path()
    assert [_said(r) for r in _rows(first)] == ["q0", "a0"]

    assert loop._cc_transcript_path() == first  # a second fire with nothing new
    _say(loop, "q1")
    loop._cc_transcript_path()
    assert [_said(r) for r in _rows(first)] == ["q0", "a0", "q1"]
    assert loop.session.transcript_cursor == 3


def test_an_empty_session_still_hands_a_hook_a_real_file(tmp_path: Path) -> None:
    path = _loop(tmp_path)._cc_transcript_path()
    assert path and Path(path).exists() and Path(path).read_text(encoding="utf-8") == ""


# ── a compaction removes nothing from the file ───────────────────────────────────────


def test_a_compaction_adds_two_rows_and_removes_none(tmp_path: Path) -> None:
    loop = _loop(tmp_path, keep=2)
    _say(loop, "q0", "a0", "q1", "a1", "q2", "a2")
    path = loop._cc_transcript_path()

    assert asyncio.run(loop.compact_now(trigger="manual")) is True
    assert len(loop.session.messages) == 3  # the summary and the kept tail of two
    _say(loop, "q3")
    loop._cc_transcript_path()

    rows = _rows(path)
    assert [_said(r) for r in rows[:6]] == ["q0", "a0", "q1", "a1", "q2", "a2"]
    boundary, summary = rows[6], rows[7]
    assert (boundary["type"], boundary["subtype"]) == ("system", "compact_boundary")
    assert boundary["compactMetadata"] == {
        "trigger": "manual",
        "preMessages": 6,
        "postMessages": 3,
    }
    assert summary["type"] == "user" and summary["isCompactSummary"] is True
    assert [_said(r) for r in rows[8:]] == ["q3"]
    assert len(rows) == 9  # the kept tail (q2, a2) was NOT written a second time


def test_the_summary_row_carries_what_the_model_now_reads(tmp_path: Path) -> None:
    loop = _loop(tmp_path, keep=2)
    _say(loop, "q0", "a0", "q1", "a1")
    asyncio.run(loop.compact_now())
    (summary,) = [r for r in _rows(loop._cc_transcript_path()) if r.get("isCompactSummary")]
    text = summary["message"]["content"]
    assert text == loop.session.messages[0].text  # the head of the live list, verbatim
    assert text.startswith(SUMMARY_MARKER) and "what happened so far" in text
    assert text.endswith(CONTINUATION_NOTE)


def test_what_a_compaction_drops_is_written_first_even_if_nothing_asked(tmp_path: Path) -> None:
    # No hook fired, no store: nothing had flushed these six messages when the compaction ran.
    loop = _loop(tmp_path, keep=2)
    _say(loop, "q0", "a0", "q1", "a1", "q2", "a2")
    assert loop.session.transcript_cursor == 0
    asyncio.run(loop.compact_now())
    rows = _rows(loop._cc_transcript_path())
    assert [_said(r) for r in rows[:6]] == ["q0", "a0", "q1", "a1", "q2", "a2"]
    assert rows[6]["subtype"] == "compact_boundary"


def test_the_boundary_row_says_the_measured_prompt_size_only_when_there_is_one(
    tmp_path: Path,
) -> None:
    loop = _loop(tmp_path, keep=2)
    _say(loop, "q0", "a0", "q1", "a1")
    loop.session.prompt_anchor_tokens = 123_456  # what the provider last REPORTED (ADR-0077)
    loop.session.prompt_anchor_index = 4
    asyncio.run(loop.compact_now(trigger="auto"))
    _say(loop, "q2", "a2", "q3", "a3")
    asyncio.run(loop.compact_now(trigger="auto"))  # the anchor was forgotten by the first one
    first, second = [r for r in _rows(loop._cc_transcript_path()) if r.get("subtype")]
    assert first["compactMetadata"]["preTokens"] == 123_456
    assert "preTokens" not in second["compactMetadata"]
    assert second["compactMetadata"]["trigger"] == "auto"


# ── a model-free elision summarizes nothing, so it marks nothing ─────────────────────


def test_an_elision_adds_no_rows_and_the_file_keeps_the_output_whole(tmp_path: Path) -> None:
    loop = _loop(tmp_path)
    long_output = "row of data\n" * 400  # well past the elision floor
    loop.session.add_message(Message.user("run it"))
    loop.session.add_message(
        Message(role="assistant", blocks=[ToolUseBlock(id="t1", name="Bash", input={})])
    )
    loop.session.add_message(
        Message(role="tool", blocks=[ToolResultBlock(tool_use_id="t1", output=long_output)])
    )
    path = loop._cc_transcript_path()
    before = Path(path).read_text(encoding="utf-8")

    assert asyncio.run(loop.elide_now()) is True
    live = loop.session.messages[2].blocks[0]
    assert isinstance(live, ToolResultBlock) and live.output.startswith(ELISION_MARKER)

    assert Path(loop._cc_transcript_path()).read_text(encoding="utf-8") == before
    assert ELISION_MARKER not in before
    (result,) = _rows(path)[2]["message"]["content"]
    assert result["type"] == "tool_result" and result["content"] == long_output


def test_shrinking_an_old_message_in_place_leaves_the_file_alone(tmp_path: Path) -> None:
    # The one other way the live list changes: a same-index replacement (ADR-0045 elides a
    # skill body once its turn ended). The file says what was said when it was said.
    loop = _loop(tmp_path)
    _say(loop, "a long skill body", "ok")
    path = loop._cc_transcript_path()
    loop.session.messages[0] = Message.user("[elided]")
    _say(loop, "next")
    loop._cc_transcript_path()
    assert [_said(r) for r in _rows(path)] == ["a long skill body", "ok", "next"]


# ── the file is as durable, and as current, as the session document ──────────────────


def test_a_stored_session_is_written_at_every_persist_with_no_hook_at_all(tmp_path: Path) -> None:
    store = SessionStore(tmp_path / "home" / "sessions")
    loop = _loop(tmp_path, store=store)
    _say(loop, "q0", "a0")
    loop._persist()
    path = tmp_path / "home" / "transcripts" / f"{loop.session.id}.jsonl"
    assert [_said(r) for r in _rows(path)] == ["q0", "a0"]
    assert store.load(loop.session.id).transcript_cursor == 2  # the document saved the cursor


def test_a_storeless_persist_writes_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / "zk"
    monkeypatch.setenv("ZAKCODE_HOME", str(home))
    loop = _loop(tmp_path)
    _say(loop, "q0", "a0")
    loop._persist()
    assert not (home / "transcripts").exists()


def test_a_resumed_session_appends_where_the_last_process_stopped(tmp_path: Path) -> None:
    store = SessionStore(tmp_path / "home" / "sessions")
    first = _loop(tmp_path, store=store)
    _say(first, "q0", "a0")
    first._persist()

    resumed = _loop(tmp_path, store=store, session=store.load(first.session.id))
    _say(resumed, "q1")
    resumed._persist()
    path = tmp_path / "home" / "transcripts" / f"{first.session.id}.jsonl"
    assert [_said(r) for r in _rows(path)] == ["q0", "a0", "q1"]


def test_nothing_in_the_file_is_ever_overwritten(tmp_path: Path) -> None:
    # A session document from before ADR-0206 has no cursor (it loads as 0) and may sit beside
    # a file an older build rendered. That file's bytes stay the prefix of what follows.
    loop = _loop(tmp_path)
    _say(loop, "q0", "a0")
    path = Path(loop._cc_transcript_path())
    older = '{"type": "user", "message": {"role": "user", "content": "from an older build"}}\n'
    path.write_text(older, encoding="utf-8")
    loop.session.transcript_cursor = 0

    loop._cc_transcript_path()
    text = path.read_text(encoding="utf-8")
    assert text.startswith(older)
    assert [_said(r) for r in _rows(path)] == ["from an older build", "q0", "a0"]


def test_a_cursor_past_the_end_of_the_list_takes_nothing_back(tmp_path: Path) -> None:
    loop = _loop(tmp_path)
    _say(loop, "q0", "a0")
    path = loop._cc_transcript_path()
    loop.session.transcript_cursor = 99  # a road this loop does not know shortened the list
    _say(loop, "q1")
    loop._cc_transcript_path()
    assert [_said(r) for r in _rows(path)] == ["q0", "a0"]  # nothing rewritten, nothing doubled
    assert loop.session.transcript_cursor == 3
    _say(loop, "a1")
    loop._cc_transcript_path()
    assert [_said(r) for r in _rows(path)] == ["q0", "a0", "a1"]


def test_a_file_that_cannot_be_written_breaks_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = SessionStore(tmp_path / "home" / "sessions")
    loop = _loop(tmp_path, store=store, keep=2)
    _say(loop, "q0", "a0", "q1", "a1")

    def _refuse(path: Path, text: str) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(loop, "_append_transcript", _refuse)
    assert loop._cc_transcript_path() == ""  # a hook is handed no path rather than a crash
    loop._persist()  # the session document still saves
    assert store.load(loop.session.id).messages
    assert asyncio.run(loop.compact_now()) is True  # and a compaction still compacts
    assert loop.session.transcript_cursor == len(loop.session.messages)


# ── the two rows, rendered ───────────────────────────────────────────────────────────


def test_compaction_rows_are_two_newline_terminated_json_lines() -> None:
    text = render_compaction_rows(
        "S", trigger="resume", messages_before=9, messages_after=3, session_id="sid", cwd="/w"
    )
    assert text.endswith("\n") and text.count("\n") == 2
    boundary, summary = (json.loads(line) for line in text.splitlines())
    assert boundary["content"] == "Conversation compacted" and boundary["level"] == "info"
    assert boundary["sessionId"] == summary["sessionId"] == "sid"
    assert boundary["timestamp"] == summary["timestamp"] and boundary["timestamp"].endswith("Z")
    assert summary["message"] == {"role": "user", "content": "S"}


# ── read the way its consumers read it ───────────────────────────────────────────────


def _compactions(path: str | Path) -> int:
    """How a report counts a session's compactions: rows flagged ``isCompactSummary``."""
    return sum(1 for row in _rows(path) if row.get("isCompactSummary"))


def _first_call_after_last_boundary(path: str | Path) -> str | None:
    """How an audit asks what the model did first after a compaction: find the last
    ``compact_boundary`` row, then the first assistant row after it that holds a tool_use."""
    rows = _rows(path)
    cuts = [i for i, row in enumerate(rows) if row.get("subtype") == "compact_boundary"]
    if not cuts:
        return None
    for row in rows[cuts[-1] + 1 :]:
        content = row.get("message", {}).get("content")
        if row.get("type") == "assistant" and isinstance(content, list):
            for block in content:
                if block.get("type") == "tool_use":
                    return str(block.get("name"))
    return None


def test_a_consumer_can_count_compactions_and_find_the_first_call_after_one(
    tmp_path: Path,
) -> None:
    loop = _loop(tmp_path, keep=2)
    _say(loop, "q0", "a0", "q1", "a1")
    path = loop._cc_transcript_path()
    assert _compactions(path) == 0 and _first_call_after_last_boundary(path) is None

    asyncio.run(loop.compact_now())
    _say(loop, "q2", "a2", "q3", "a3")
    asyncio.run(loop.compact_now())
    loop.session.add_message(
        Message(role="assistant", blocks=[ToolUseBlock(id="w", name="ScheduleWakeup", input={})])
    )
    loop.session.add_message(
        Message(role="assistant", blocks=[ToolUseBlock(id="b", name="Bash", input={})])
    )
    loop._cc_transcript_path()
    assert _compactions(path) == 2
    assert _first_call_after_last_boundary(path) == "ScheduleWakeup"
