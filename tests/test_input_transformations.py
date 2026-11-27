"""The API's report of what it did to a request's thinking blocks reaches the log.
https://platform.claude.com/docs/en/build-with-claude/preserved-thinking/fix-bound-to-a-different-conversation

With the thinking-binding-controls beta header, a streamed response names each thinking block the
API dropped or let through in the ``message_start`` event's ``message.input_transformations``.
litellm's parser reads only that event's usage, so the provider reads the list before the parser.

Hermetic: ``httpx.AsyncClient.send`` is replaced by a canned event stream, so the events pass
through litellm's own Anthropic client and parser. No network, no credentials.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import httpx
import pytest

import zakcode.providers.litellm_provider as lp
from zakcode.config import Settings
from zakcode.messages import Message

ENTRY = {
    "type": "thinking_dropped",
    "path": "messages.3.content.0",
    "reason": "prefix_binding_mismatch",
}


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("ZAKCODE_HOME", str(tmp_path / "confighome"))
    for var in ("ZAKCODE_DEFAULT_MODEL", "ZAKCODE_API_BASE", "ZAKCODE_ANTHROPIC_POOL"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(lp, "_MIN_REQUEST_INTERVAL_S", 0.0)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")


def _stream(transformations: list[dict[str, str]]) -> bytes:
    usage = {"input_tokens": 3, "output_tokens": 1}
    events: list[dict[str, Any]] = [
        {
            "type": "message_start",
            "message": {
                "id": "msg_1",
                "type": "message",
                "role": "assistant",
                "model": "claude-sonnet-5-5",
                "content": [],
                "stop_reason": None,
                "stop_sequence": None,
                "usage": usage,
                "input_transformations": transformations,
            },
        },
        {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "ok"}},
        {"type": "content_block_stop", "index": 0},
        {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": usage},
        {"type": "message_stop"},
    ]
    return "".join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in events).encode()


async def _streamed_text(monkeypatch: pytest.MonkeyPatch, body: bytes) -> str:
    async def send(self: httpx.AsyncClient, request: httpx.Request, **kw: Any) -> httpx.Response:
        headers = {"content-type": "text/event-stream"}
        return httpx.Response(200, content=body, headers=headers, request=request)

    monkeypatch.setattr(httpx.AsyncClient, "send", send)
    provider = lp.LiteLLMProvider(
        Settings(
            default_model="anthropic/claude-sonnet-5-5",
            api_key="sk-test",
            context_window=200_000,
            _env_file=None,
        )
    )
    events = [event async for event in provider.astream([Message.user("hello")])]
    return "".join(getattr(event, "text", "") or "" for event in events)


def _reports(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [r.getMessage() for r in caplog.records if "input_transformations" in r.getMessage()]


async def test_a_streamed_report_of_a_dropped_block_is_logged(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO, logger="zakcode.providers")

    assert await _streamed_text(monkeypatch, _stream([ENTRY])) == "ok"

    assert _reports(caplog) == [
        "input_transformations: the API reported 1 thinking block change(s) on this request: "
        "thinking_dropped prefix_binding_mismatch messages.3.content.0"
    ]


async def test_an_empty_report_logs_nothing(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO, logger="zakcode.providers")

    assert await _streamed_text(monkeypatch, _stream([])) == "ok"

    assert _reports(caplog) == []


def test_a_long_report_names_the_first_entries_and_counts_them_all(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO, logger="zakcode.providers")
    entries = [{**ENTRY, "path": f"messages.{i}.content.0"} for i in range(20)]

    lp._log_input_transformations(
        {"type": "message_start", "message": {"input_transformations": entries}}
    )

    (line,) = _reports(caplog)
    assert "reported 20 thinking block change(s)" in line
    assert "messages.7.content.0" in line
    assert "messages.8.content.0" not in line
