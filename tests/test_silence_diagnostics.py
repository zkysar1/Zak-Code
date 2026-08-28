"""An empty completion names what the backend actually sent.

Measured 2026-08-28 (coach, zc-03): six silences in one boot — "622 tokens generated, none
delivered" — with no text, no reasoning, no tool call, and nothing in the trace to say which
channel the tokens took or how the response ended. The empty-completion note now carries the
backend's finish reason, the buffered message object, and (streaming) the provider's sample of
raw deltas, so the next silence is diagnosed from the trace instead of guessed at.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from zakcode.agent.loop import AgentLoop, _raw_message_excerpt, _silent_detail
from zakcode.config import load_settings
from zakcode.evals.harness import ScriptedProvider, reply
from zakcode.providers.base import LLMResult
from zakcode.session.store import Session
from zakcode.tools import ToolRegistry
from zakcode.usage import Usage


def _loop(tmp_path: Path, provider: ScriptedProvider) -> AgentLoop:
    settings = load_settings(workspace_root=tmp_path)
    session = Session(cwd=str(tmp_path), model="scripted/test")
    return AgentLoop(provider, ToolRegistry(), session, settings=settings, workspace_root=tmp_path)


def _empty_notes(loop: AgentLoop) -> list[dict[str, Any]]:
    return [e.data for e in loop._trace.events if e.data.get("kind") == "empty_completion"]


def test_silent_detail_names_the_finish_reason_when_known() -> None:
    assert _silent_detail(0) == ""
    assert _silent_detail(622) == " (622 tokens generated, none delivered)"
    assert _silent_detail(622, "stop") == " (622 tokens generated, none delivered; finish=stop)"


def test_raw_message_excerpt_is_the_message_object_bounded() -> None:
    raw = {"choices": [{"message": {"role": "assistant", "content": "", "tool_calls": None}}]}
    excerpt = _raw_message_excerpt(raw)
    assert excerpt is not None and '"role": "assistant"' in excerpt
    assert _raw_message_excerpt(None) is None
    long = {"choices": [{"message": {"content": "x" * 5000}}]}
    assert len(_raw_message_excerpt(long) or "") <= 601


def test_buffered_empty_completion_note_carries_finish_reason_and_raw(tmp_path: Path) -> None:
    silent = LLMResult(
        finish_reason="stop",
        usage=Usage(prompt_tokens=10, completion_tokens=622, total_tokens=632),
        raw={"choices": [{"message": {"role": "assistant", "content": "", "reasoning": "hm"}}]},
    )
    loop = _loop(tmp_path, ScriptedProvider([silent, reply("here is the answer")]))
    result = asyncio.run(loop.arun_turn("hi"))
    assert result.stop_reason == "completed"
    (note,) = _empty_notes(loop)
    assert note["completion_tokens"] == 622 and note["finish_reason"] == "stop"
    assert '"reasoning": "hm"' in note["raw"]
    (event,) = [e for e in loop._trace.events if e.data.get("kind") == "empty_completion"]
    assert "622 tokens generated, none delivered; finish=stop" in event.detail


class _SamplingProvider(ScriptedProvider):
    """A scripted provider that, like LiteLLMProvider, keeps a sample of its last stream."""

    last_stream_sample: dict[str, Any] | None = {
        "chunks": 4,
        "finish_reason": "stop",
        "deltas": ['{"role": "assistant"}', '{"content": ""}'],
    }


def test_streaming_empty_completion_note_carries_the_stream_sample(tmp_path: Path) -> None:
    silent = LLMResult(
        finish_reason="stop",
        usage=Usage(prompt_tokens=10, completion_tokens=334, total_tokens=344),
    )
    loop = _loop(tmp_path, _SamplingProvider([silent, reply("here is the answer")]))

    async def run() -> list[Any]:
        return [ev async for ev in loop.astream_turn("hi")]

    events = asyncio.run(run())
    assert events
    (note,) = _empty_notes(loop)
    assert note["completion_tokens"] == 334
    assert note["stream"] == _SamplingProvider.last_stream_sample
    statuses = [getattr(ev, "message", "") for ev in events]
    assert any("334 tokens generated, none delivered" in s for s in statuses)
