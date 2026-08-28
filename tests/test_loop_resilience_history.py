"""Empty-completion history repair + per-child trace labels (2026-08-22 field defects).

Both defects were measured driving a live agent on a local pod: an empty (thinking-only)
assistant completion stored with no blocks poisons the session — OpenAI-compat providers
reject the whole HISTORY on every later call — and sub-agent loops sharing the parent's
trace_dir clobbered the root's turn_N.jsonl files.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Sequence
from pathlib import Path

from zakcode.agent.loop import _EMPTY_COMPLETION_PLACEHOLDER, _MAX_EMPTY_RETRIES, AgentLoop
from zakcode.config import load_settings
from zakcode.providers.base import Capabilities, LLMResult, Provider
from zakcode.session.store import Session
from zakcode.tools.base import ToolRegistry
from zakcode.usage import Usage


class ScriptedProvider(Provider):
    """Returns canned results in order (same shape as test_loop's helper)."""

    def __init__(self, results: Sequence[LLMResult]) -> None:
        self._results = list(results)
        self.calls = 0

    def capabilities(self) -> Capabilities:
        return Capabilities(supports_tools=True, context_window=8192)

    async def acomplete(self, messages, tools=None, **kwargs):  # type: ignore[override]
        self.calls += 1
        return self._results.pop(0)

    def count_tokens(self, messages, tools=None) -> int:  # type: ignore[override]
        return 0


def _usage() -> Usage:
    return Usage(prompt_tokens=1, completion_tokens=1, total_tokens=2)


def _loop(provider: Provider, *, trace_dir: str | None = None, **kw) -> AgentLoop:
    settings = load_settings(workspace_root=Path.cwd())
    if trace_dir is not None:
        settings = settings.model_copy(update={"trace_dir": trace_dir})
    session = Session(cwd="/tmp/work", model="ollama_chat/llama3.1")
    return AgentLoop(provider, ToolRegistry(), session, settings=settings, **kw)


def test_empty_completion_stores_placeholder_not_empty_message() -> None:
    # The empty completion draws a "say something" nudge (2026-08-26 give-up gate);
    # the model answers on the second call. The empty one must still land as a
    # placeholder message, never as an empty-blocks assistant message.
    provider = ScriptedProvider(
        [
            LLMResult(text="", finish_reason="stop", usage=_usage()),
            LLMResult(text="ok", finish_reason="stop", usage=_usage()),
        ]
    )
    loop = _loop(provider)

    asyncio.run(loop.arun_turn("hello"))

    assistant = [m for m in loop.session.messages if m.role == "assistant"]
    assert len(assistant) == 2
    # The stored message must carry content — an empty-blocks assistant message is
    # rejected by OpenAI-compat providers on EVERY subsequent call (transcript poison).
    assert assistant[0].blocks, "empty-blocks assistant message poisons the transcript"
    assert assistant[0].text == _EMPTY_COMPLETION_PLACEHOLDER
    assert assistant[1].text == "ok"


def test_session_survives_empty_completion_and_continues() -> None:
    # Turn 1 stays silent through every "say something" nudge and ends gave_up;
    # every one of its empty completions must land as a placeholder so turn 2
    # still parses at the provider.
    empties = 1 + _MAX_EMPTY_RETRIES
    provider = ScriptedProvider(
        [LLMResult(text="", finish_reason="stop", usage=_usage())] * empties
        + [LLMResult(text="second turn works", finish_reason="stop", usage=_usage())]
    )
    loop = _loop(provider)

    first = asyncio.run(loop.arun_turn("first"))
    result = asyncio.run(loop.arun_turn("second"))

    assert first.stop_reason == "gave_up"
    # Before the placeholder fix the second call died at the provider with
    # "Assistant message must contain either 'content' or 'tool_calls'".
    assert result.stop_reason == "completed"
    assert result.assistant_messages[-1].text == "second turn works"
    for msg in loop.session.messages:
        if msg.role == "assistant":
            assert msg.blocks


def test_trace_label_separates_child_dumps(tmp_path: Path) -> None:
    root = ScriptedProvider([LLMResult(text="root", finish_reason="stop", usage=_usage())])
    child = ScriptedProvider([LLMResult(text="child", finish_reason="stop", usage=_usage())])

    root_loop = _loop(root, trace_dir=str(tmp_path))
    child_loop = _loop(child, trace_dir=str(tmp_path), trace_label="sub1-general-purpose")

    asyncio.run(root_loop.arun_turn("go"))
    asyncio.run(child_loop.arun_turn("go"))

    names = sorted(p.name for p in tmp_path.glob("*.jsonl"))
    assert "turn_1.jsonl" in names  # root keeps the bare name (zakcode-usage-stats compat)
    assert "sub1-general-purpose_turn_1.jsonl" in names  # child no longer clobbers it
    # Both are real trace files, not empty artifacts.
    for name in names:
        assert (tmp_path / name).read_text(encoding="utf-8").strip()
        json.loads((tmp_path / name).read_text(encoding="utf-8").splitlines()[0])
