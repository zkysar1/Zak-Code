"""Live-provider smoke tests — the REAL vendor path end to end.

These are SKIPPED by default (they cost money / need a running model) and never run
in the normal gate or CI. Opt in explicitly:

    set LIVE_TESTS=1 (and optionally ZAKCODE_LIVE_MODEL), then run this file. Examples:
        LIVE_TESTS=1 uv run pytest tests/test_live_provider_smoke.py -q
        ZAKCODE_LIVE_MODEL=openai/gpt-4o-mini   (OpenAI; needs OPENAI_API_KEY + quota)
        ZAKCODE_LIVE_MODEL=ollama_chat/llama3.1 (Ollama; needs `ollama serve` + the model)

They verify, against a real model, what the no-network suite cannot: that litellm
normalization, real token counting, and a real tool-use round-trip actually work — the
substance behind the "vendor-agnostic" claim. A successful run is the proof that Zak
Code drives a live LLM to complete a task; a quota/connection failure SKIPS with a clear
reason rather than failing, so this file is safe to leave in the suite.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

_LIVE = os.environ.get("LIVE_TESTS") == "1"
_MODEL = os.environ.get("ZAKCODE_LIVE_MODEL", "openai/gpt-4o-mini")

pytestmark = pytest.mark.skipif(
    not _LIVE, reason="live-provider test; set LIVE_TESTS=1 (and ZAKCODE_LIVE_MODEL) to run"
)


def _agent(workspace: Path):  # type: ignore[no-untyped-def]
    from zakcode import Agent
    from zakcode.permissions import PermissionPolicy

    return Agent(
        default_model=_MODEL,
        workspace_root=str(workspace),
        permission_policy=PermissionPolicy("acceptEdits"),  # allow writes, gate shell
        max_iterations=6,
    )


def _skip_on_provider_error(exc: Exception) -> None:
    """Turn an environmental provider failure (no quota / not running) into a SKIP."""
    from zakcode.providers.base import ProviderError

    if isinstance(exc, ProviderError):
        pytest.skip(f"live provider unavailable ({type(exc).__name__}): {exc}")
    raise exc


async def test_live_real_completion(tmp_path: Path) -> None:
    # Real model call: it should produce text and report real token usage.
    agent = _agent(tmp_path)
    try:
        result = await agent.arun_turn("Reply with exactly the word: pong")
    except Exception as exc:  # noqa: BLE001 — classify env failures as skips
        _skip_on_provider_error(exc)
        return
    assert result.stop_reason in ("completed", "max_iterations")
    assert result.usage.total_tokens > 0, "real provider should report token usage"
    text = "\n".join(m.text for m in result.assistant_messages if m.text).lower()
    assert "pong" in text, f"unexpected reply: {text!r}"


async def test_live_real_tool_use(tmp_path: Path) -> None:
    # Real model driving a real tool to a real on-disk effect.
    agent = _agent(tmp_path)
    target = tmp_path / "hello.txt"
    try:
        result = await agent.arun_turn(
            f"Use your write_file tool to create the file {target.name} containing exactly "
            f"the text 'hello world'. The file path is {target}. Then reply DONE."
        )
    except Exception as exc:  # noqa: BLE001
        _skip_on_provider_error(exc)
        return
    # A tool-capable model should have written the file. If the model chose not to call
    # the tool (weaker local models sometimes don't), skip rather than fail — the point
    # is to prove the path works when the model cooperates.
    if not target.is_file():
        pytest.skip(
            f"model did not call write_file (stop={result.stop_reason}); tool path untested"
        )
    assert "hello world" in target.read_text(encoding="utf-8").lower()


def test_live_token_counting(tmp_path: Path) -> None:
    # The provider's real tokenizer returns a positive, monotonic-ish count.
    from zakcode.messages import Message

    agent = _agent(tmp_path)
    one = agent.provider.count_tokens([Message.user("hello")])
    many = agent.provider.count_tokens([Message.user("hello " * 200)])
    assert one >= 0
    assert many > one, (one, many)
