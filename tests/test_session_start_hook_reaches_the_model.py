"""What a SessionStart hook SAYS reaches the model, the way Claude Code hands it over (ADR-0211).

Why this file exists. Claude Code adds a SessionStart hook's stdout to the model's context, and
a framework written for it leans on that hardest right after a compaction: its hook prints what
was in flight (the goal, the loop state, what to call first), because the summary that replaced
the conversation cannot be trusted to hold it. Zak Code fired the event and threw the words
away. Measured 2026-09-21 on a served gpt-5.6-luna run: 19 compactions in 35 minutes, the
framework's checkpoint written at every one of them, and read by nobody.

So these tests do not inspect a list the harness built. A REAL loop compacts a REAL transcript
in the middle of a turn, a hook SCRIPT decides from its own stdin whether to speak, and the
assertion is on the REQUEST the provider is handed next: the last thing the model reads before
it acts. Hermetic: tmp workspaces, a scripted provider, no network. The hook bodies are
synthesized; no framework's text is quoted.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

from zakcode.agent.compact import CompactionConfig, Compactor
from zakcode.agent.loop import AgentLoop
from zakcode.hooks import (
    _MAX_SESSION_START_CHARS,
    HookEvent,
    HookManager,
    HookSpec,
    LifecyclePayload,
)
from zakcode.messages import Message
from zakcode.providers.base import Capabilities, LLMResult, Provider, ToolCall, Usage
from zakcode.session.store import Session
from zakcode.tools.builtins.default_registry import default_registry

MODEL = "scripted/test"
FILES = 12  # one small file per tool call, so no two tool results are alike

#: The consumer this change is for, in its own shape: it WRITES DOWN that it ran (a side effect,
#: whatever happens next) and SPEAKS only for the sources named on its command line.
_SPEAKER = """
import json, sys
doc = json.load(sys.stdin)
row = {{"source": doc.get("source"), "event": doc.get("hook_event_name")}}
with open({log!r}, "a", encoding="utf-8") as out:
    out.write(json.dumps(row) + "\\n")
if doc.get("source") in sys.argv[1:]:
    print("RESTORED after %s: goal g-7 was in flight; first call the wake-up tool" % doc["source"])
"""

#: The JSON shape Claude Code documents for the same thing.
_JSON_SPEAKER = """
import json, sys
doc = json.load(sys.stdin)
print(json.dumps({"hookSpecificOutput": {"hookEventName": "SessionStart",
      "additionalContext": "JSON-RESTORED after %s" % doc.get("source")}}))
"""

#: Ran, left its mark, FAILED, and printed on the way out.
_FAILING = """
import json, sys
doc = json.load(sys.stdin)
with open({log!r}, "a", encoding="utf-8") as out:
    out.write(json.dumps({{"source": doc.get("source")}}) + "\\n")
print("words from a hook that failed")
sys.exit(3)
"""

_FLOOD = 'print("x" * 100000)'
_SILENT = "import sys; sys.stdin.read()"


class _Recording(Provider):
    """Walks a script (the last entry repeats) and keeps every MAIN request it was handed. A
    summarization request gets a summary and is kept apart. Tokens grow with the transcript,
    so a modest window makes the loop compact by itself in the middle of a turn."""

    def __init__(self, script: list[LLMResult], *, window: int = 2000, per_message: int = 100):
        self._script, self._window, self._per = script, window, per_message
        self.main: list[list[Message]] = []
        self.summaries = 0

    async def acomplete(self, messages: Any, *, system: Any = None, tools: Any = None, **kw: Any):  # type: ignore[override]
        if system and "compacting" in str(system).lower():
            self.summaries += 1
            return LLMResult(text="summary: earlier files were read", usage=Usage(total_tokens=1))
        self.main.append(list(messages))
        return self._script[min(len(self.main), len(self._script)) - 1]

    def model_id(self) -> str:
        return MODEL

    def count_tokens(self, messages: Any, *, system: Any = None) -> int:
        return self._per * len(messages)

    def capabilities(self) -> Capabilities:
        return Capabilities(supports_tools=True, context_window=self._window)


def _read(index: int) -> LLMResult:
    call = ToolCall(id=f"r{index}", name="Read", arguments={"path": f"f{index}.txt"})
    return LLMResult(text="", tool_calls=[call], usage=Usage(total_tokens=1))


def _say(text: str) -> LLMResult:
    return LLMResult(text=text, tool_calls=[], usage=Usage(total_tokens=1))


def _reads_then_done() -> list[LLMResult]:
    return [*[_read(i) for i in range(FILES)], _say("done")]


def _hook(tmp_path: Path, name: str, body: str, event: HookEvent, *argv: str) -> HookSpec:
    path = tmp_path / name
    path.write_text(body, encoding="utf-8")
    return HookSpec(event=event, command=[sys.executable, str(path), *argv])


def _loop(tmp_path: Path, provider: Provider, hooks: list[HookSpec]) -> AgentLoop:
    for index in range(FILES):
        (tmp_path / f"f{index}.txt").write_text(f"file number {index}\n", encoding="utf-8")
    return AgentLoop(
        provider,
        default_registry(),
        Session(cwd=str(tmp_path), model=MODEL),
        workspace_root=tmp_path,
        max_iterations=40,
        hook_manager=HookManager(hooks),
        compactor=Compactor(CompactionConfig()),
    )


async def _run(loop: AgentLoop, text: str, *, streamed: bool) -> str:
    if not streamed:
        return (await loop.arun_turn(text)).stop_reason
    async for _ in loop.astream_turn(text):
        pass
    return str(loop.session.last_stop_reason)


def _text(message: Message) -> str:
    return message.text or ""


def _notes(loop: AgentLoop, kind: str) -> list[dict[str, Any]]:
    return [e.data for e in loop._trace.of_kind("intervention") if e.data.get("kind") == kind]


def _log(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _said(loop: AgentLoop) -> list[str]:
    """Every persisted message that carries a SessionStart hook's words, in order."""
    return [_text(m) for m in loop.session.messages if "SessionStart hook ran" in _text(m)]


# ── after a compaction: the case the change is for ────────────────────────────────────────


@pytest.mark.asyncio
@pytest.mark.parametrize("streamed", [False, True], ids=["buffered", "streaming"])
async def test_what_the_hook_says_after_a_compaction_is_the_last_thing_the_next_request_reads(
    tmp_path: Path, streamed: bool
) -> None:
    log = tmp_path / "ran.jsonl"
    speaker = _hook(
        tmp_path, "speak.py", _SPEAKER.format(log=str(log)), HookEvent.SESSION_START, "compact"
    )
    provider = _Recording(_reads_then_done())
    loop = _loop(tmp_path, provider, [speaker])
    assert await _run(loop, "read the files", streamed=streamed) == "completed"

    compactions = [n for n in _notes(loop, "compaction") if n.get("compacted")]
    assert compactions and provider.summaries == len(compactions)  # it really compacted, mid-turn
    said = _notes(loop, "session_start_said")
    assert [n["source"] for n in said] == ["compact"] * len(compactions)
    # The wire to the MODEL: some request ends with the hook's words, tagged and explained.
    endings = [_text(request[-1]) for request in provider.main]
    restored = [e for e in endings if "RESTORED after compact" in e]
    assert len(restored) == len(compactions), endings
    assert all(e.startswith("[hook] ") and "just compacted" in e for e in restored), restored
    # The hook ran once per event, and spoke for the one source it was asked to speak for.
    assert [row["source"] for row in _log(log)] == ["startup", *["compact"] * len(compactions)]
    assert all(row["event"] == "SessionStart" for row in _log(log))


@pytest.mark.asyncio
async def test_the_words_are_said_once_and_then_age_like_any_other_message(tmp_path: Path) -> None:
    """Persisted where the hook fired, not re-sent on every request of the turn."""
    log = tmp_path / "ran.jsonl"
    speaker = _hook(
        tmp_path, "speak.py", _SPEAKER.format(log=str(log)), HookEvent.SESSION_START, "compact"
    )
    provider = _Recording(_reads_then_done())
    loop = _loop(tmp_path, provider, [speaker])
    await loop.arun_turn("read the files")
    carrying = [i for i, req in enumerate(provider.main) if "RESTORED" in _text(req[-1])]
    assert carrying, "no request ended with the hook's words"
    first = carrying[0]
    later = provider.main[first + 1]
    assert "RESTORED" not in _text(later[-1])  # the next request ends with a tool result again
    assert sum("RESTORED" in _text(m) for m in later) == 1  # and still holds the words, ONCE


@pytest.mark.asyncio
async def test_the_words_are_the_hooks_own_and_not_fenced_as_untrusted_data(tmp_path: Path) -> None:
    """The untrusted-context fence tells the model NOT to follow what is inside. A restore is
    an instruction from the workspace's own configuration, so it must not wear that fence."""
    speaker = _hook(
        tmp_path,
        "speak.py",
        _SPEAKER.format(log=str(tmp_path / "ran.jsonl")),
        HookEvent.SESSION_START,
        "compact",
    )
    loop = _loop(tmp_path, _Recording(_reads_then_done()), [speaker])
    await loop.arun_turn("read the files")
    said = _said(loop)
    assert said and all("<injected_context>" not in text for text in said)
    assert all("untrusted" not in text.lower() for text in said)
    assert all(m.role == "user" for m in loop.session.messages if "RESTORED" in _text(m))


# ── the controls: what must NOT change ────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_a_hook_that_says_nothing_leaves_the_conversation_as_it_was(tmp_path: Path) -> None:
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    quiet = _loop(tmp_path / "a", _Recording(_reads_then_done()), [])
    silent = _hook(tmp_path / "b", "silent.py", _SILENT, HookEvent.SESSION_START)
    hooked = _loop(tmp_path / "b", _Recording(_reads_then_done()), [silent])
    await quiet.arun_turn("read the files")
    await hooked.arun_turn("read the files")
    assert [(_text(m), m.role) for m in hooked.session.messages] == [
        (_text(m), m.role) for m in quiet.session.messages
    ]
    assert _notes(hooked, "session_start_said") == []
    assert [n for n in _notes(hooked, "compaction") if n.get("compacted")]  # and it did compact


@pytest.mark.asyncio
async def test_a_hook_that_fails_still_ran_once_and_says_nothing(tmp_path: Path) -> None:
    log = tmp_path / "ran.jsonl"
    failing = _hook(tmp_path, "fail.py", _FAILING.format(log=str(log)), HookEvent.SESSION_START)
    provider = _Recording(_reads_then_done())
    loop = _loop(tmp_path, provider, [failing])
    await loop.arun_turn("read the files")
    compactions = [n for n in _notes(loop, "compaction") if n.get("compacted")]
    assert compactions
    # Its side effect stands, ONCE per event: the seam still runs the hook exactly one time.
    assert [row["source"] for row in _log(log)] == ["startup", *["compact"] * len(compactions)]
    assert _said(loop) == [] and _notes(loop, "session_start_said") == []
    assert not any("words from a hook that failed" in _text(m) for r in provider.main for m in r)


@pytest.mark.asyncio
async def test_every_other_lifecycle_event_stays_observe_only(tmp_path: Path) -> None:
    """PreCompact and SessionEnd hooks may print what they like: Claude Code adds neither to the
    model's context, and neither do we."""
    talkative = "print('words from a hook nobody should hear')"
    hooks = [
        _hook(tmp_path, "pre.py", talkative, HookEvent.PRE_COMPACT),
        _hook(tmp_path, "end.py", talkative, HookEvent.SESSION_END),
    ]
    provider = _Recording(_reads_then_done())
    loop = _loop(tmp_path, provider, hooks)
    await loop.arun_turn("read the files")
    assert [n for n in _notes(loop, "compaction") if n.get("compacted")]
    assert not any("nobody should hear" in _text(m) for r in provider.main for m in r)
    for event in (HookEvent.PRE_COMPACT, HookEvent.SESSION_END):
        payload = LifecyclePayload(event=event, session_id="s", cwd=str(tmp_path))
        assert await loop.hook_manager.fire(payload) == []


# ── a new session and a resumed one ───────────────────────────────────────────────────────


@pytest.mark.asyncio
@pytest.mark.parametrize("streamed", [False, True], ids=["buffered", "streaming"])
async def test_at_startup_the_ask_stays_first_and_the_words_follow_it_once(
    tmp_path: Path, streamed: bool
) -> None:
    speaker = _hook(
        tmp_path,
        "speak.py",
        _SPEAKER.format(log=str(tmp_path / "ran.jsonl")),
        HookEvent.SESSION_START,
        "startup",
        "resume",
    )
    provider = _Recording([_say("hello")])
    loop = _loop(tmp_path, provider, [speaker])
    await _run(loop, "the ask", streamed=streamed)
    first = provider.main[0]
    assert [_text(m) for m in first[:1]] == ["the ask"]  # the fold of the first ask keys on it
    assert _text(first[1]).startswith("[hook] ") and "RESTORED after startup" in _text(first[1])
    assert "a new session is starting" in _text(first[1])
    await _run(loop, "a second ask", streamed=streamed)
    assert len(_said(loop)) == 1  # SessionStart fires once per session, and so do its words


@pytest.mark.asyncio
async def test_a_resumed_session_is_told_so(tmp_path: Path) -> None:
    speaker = _hook(
        tmp_path,
        "speak.py",
        _SPEAKER.format(log=str(tmp_path / "ran.jsonl")),
        HookEvent.SESSION_START,
        "resume",
    )
    loop = _loop(tmp_path, _Recording([_say("hello")]), [speaker])
    loop.session.messages.extend([Message.user("earlier ask"), Message.assistant_text("earlier")])
    await loop.arun_turn("the ask")
    (said,) = _said(loop)
    assert "RESTORED after resume" in said and "a saved session is being resumed" in said
    assert _text(loop.session.messages[2]) == "the ask" and _text(loop.session.messages[3]) == said


# ── the file a consumer reads ─────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_in_the_transcript_the_words_follow_the_compaction_rows(tmp_path: Path) -> None:
    """An audit of what a model did first after a compaction walks forward from the boundary
    row. What the hook said has to be there before any call the model makes next, as it is in
    a Claude Code transcript, and the rows before the boundary stay as they were."""
    speaker = _hook(
        tmp_path,
        "speak.py",
        _SPEAKER.format(log=str(tmp_path / "ran.jsonl")),
        HookEvent.SESSION_START,
        "compact",
    )
    loop = _loop(tmp_path, _Recording([_say("hello")]), [speaker])
    for index in range(4):
        loop.session.add_message(Message.user(f"q{index}"))
        loop.session.add_message(Message.assistant_text(f"a{index}"))
    path = Path(loop._cc_transcript_path())
    before = path.read_bytes()
    assert await loop.compact_now(trigger="auto") is True
    loop._cc_transcript_path()
    assert path.read_bytes().startswith(before)  # append-only: nothing already written moved
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    cut = next(i for i, row in enumerate(rows) if row.get("subtype") == "compact_boundary")
    assert rows[cut + 1].get("isCompactSummary") is True
    after = rows[cut + 2]
    assert after["type"] == "user" and "RESTORED after compact" in json.dumps(after)
    assert len(rows) == cut + 3  # the boundary, the summary, the words: nothing else was added


# ── the two stdout shapes, and the bound ──────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_the_json_shape_claude_code_documents_is_read(tmp_path: Path) -> None:
    # Not "json.py": a script shadows the module of its own name, and the hook would die on
    # its first import, which reads exactly like a hook that said nothing.
    speaker = _hook(tmp_path, "speak_json.py", _JSON_SPEAKER, HookEvent.SESSION_START)
    loop = _loop(tmp_path, _Recording([_say("hello")]), [speaker])
    await loop.arun_turn("the ask")
    (said,) = _said(loop)
    assert "JSON-RESTORED after startup" in said and "hookSpecificOutput" not in said


@pytest.mark.asyncio
async def test_a_flood_is_bounded(tmp_path: Path) -> None:
    flood = _hook(tmp_path, "flood.py", _FLOOD, HookEvent.SESSION_START)
    loop = _loop(tmp_path, _Recording([_say("hello")]), [flood])
    await loop.arun_turn("the ask")
    (said,) = _said(loop)
    assert said.count("x") == _MAX_SESSION_START_CHARS and said.endswith("[hook output truncated]")


@pytest.mark.asyncio
async def test_two_hooks_speak_in_the_order_they_are_registered(tmp_path: Path) -> None:
    one = _hook(tmp_path, "one.py", "print('first voice')", HookEvent.SESSION_START)
    two = _hook(tmp_path, "two.py", "print('second voice')", HookEvent.SESSION_START)
    loop = _loop(tmp_path, _Recording([_say("hello")]), [one, two])
    await loop.arun_turn("the ask")
    (said,) = _said(loop)
    assert said.index("first voice") < said.index("second voice")
