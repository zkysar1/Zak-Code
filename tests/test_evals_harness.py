"""Tests for the M9 eval-harness core: ScriptedProvider, provider injection, runner."""

from __future__ import annotations

from pathlib import Path

from zakcode import Agent
from zakcode.evals import (
    EvalCase,
    EvalReport,
    ScriptedProvider,
    call_tool,
    hermetic_env,
    make_agent,
    reply,
    run_evals,
)
from zakcode.providers.base import LLMResult

# ── ScriptedProvider ──────────────────────────────────────────────────────────


def test_scripted_provider_requires_script_or_responder() -> None:
    try:
        ScriptedProvider()
    except ValueError:
        pass
    else:  # pragma: no cover - guard
        raise AssertionError("expected ValueError")


def test_scripted_provider_repeats_last() -> None:
    async def scenario() -> None:
        p = ScriptedProvider([reply("a"), reply("b")])
        r0 = await p.acomplete([])
        r1 = await p.acomplete([])
        r2 = await p.acomplete([])  # past the end → repeats last
        assert (r0.text, r1.text, r2.text) == ("a", "b", "b")
        assert p.calls == 3

    import asyncio

    asyncio.run(scenario())


def test_scripted_provider_responder_takes_precedence() -> None:
    async def scenario() -> None:
        def responder(messages, system, i):  # type: ignore[no-untyped-def]
            return reply(f"dynamic-{i}")

        p = ScriptedProvider([reply("static")], responder=responder)
        r = await p.acomplete([])
        assert r.text == "dynamic-0"

    import asyncio

    asyncio.run(scenario())


def test_count_tokens_grows_with_history() -> None:
    from zakcode.messages import Message

    p = ScriptedProvider([reply("x")], tokens_per_message=10)
    assert p.count_tokens([]) == 0
    assert p.count_tokens([Message.user("a"), Message.user("b")]) == 20
    assert p.count_tokens([Message.user("a")], system="sys") == 20  # +1 message unit


def test_summarize_calls_counted() -> None:
    async def scenario() -> None:
        p = ScriptedProvider([reply("s")])
        await p.acomplete([], system="You are compacting a long conversation...")
        await p.acomplete([], system="ordinary system prompt")
        assert p.summarize_calls == 1

    import asyncio

    asyncio.run(scenario())


# ── provider injection into the real Agent (no network) ─────────────────────────


def test_agent_accepts_injected_provider(tmp_path: Path) -> None:
    async def scenario() -> None:
        provider = ScriptedProvider([reply("hello from script")])
        agent = make_agent(provider, workspace_root=str(tmp_path))
        assert agent.provider is provider
        result = await agent.arun_turn("hi")
        assert result.stop_reason == "completed"
        assert result.assistant_messages[-1].text == "hello from script"

    import asyncio

    asyncio.run(scenario())


def test_injected_provider_can_drive_a_tool(tmp_path: Path) -> None:
    async def scenario() -> None:
        # First a write_file call, then a closing reply. permission_mode="allow" so the
        # write is not gated. Proves the harness can exercise a full tool round-trip.
        target = tmp_path / "out.txt"
        provider = ScriptedProvider(
            [
                call_tool("write_file", {"path": str(target), "content": "hi"}),
                reply("done"),
            ]
        )
        agent = make_agent(provider, workspace_root=str(tmp_path))
        result = await agent.arun_turn("write the file")
        assert result.stop_reason == "completed"
        assert target.read_text(encoding="utf-8") == "hi"

    import asyncio

    asyncio.run(scenario())


def test_default_agent_still_builds_litellm(tmp_path: Path) -> None:
    # No provider injected → the facade builds its normal provider (no network at init).
    agent = Agent(default_model="ollama/llama3", workspace_root=str(tmp_path))
    assert type(agent.provider).__name__ == "LiteLLMProvider"


# ── runner ───────────────────────────────────────────────────────────────────────


def test_run_evals_aggregates_pass_and_fail() -> None:
    async def scenario() -> None:
        async def good() -> str:
            return "all good"

        async def bad() -> str:
            raise AssertionError("nope")

        report = await run_evals(
            [
                EvalCase("good", "passes", good),
                EvalCase("bad", "fails", bad),
            ]
        )
        assert isinstance(report, EvalReport)
        assert report.total == 2
        assert report.passed == 1
        assert report.failed == 1
        assert report.ok is False
        bad_result = next(r for r in report.results if r.name == "bad")
        assert bad_result.error is not None and "nope" in bad_result.error

    import asyncio

    asyncio.run(scenario())


def test_report_ok_when_all_pass() -> None:
    async def scenario() -> None:
        async def good() -> None:
            return None

        report = await run_evals([EvalCase("g", "d", good)])
        assert report.ok is True

    import asyncio

    asyncio.run(scenario())


def test_hermetic_env_restores(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    import os

    monkeypatch.setenv("OPENAI_API_KEY", "sentinel")
    with hermetic_env():
        assert os.environ.get("OPENAI_API_KEY") is None
        assert os.environ.get("TZ") == "UTC"
    assert os.environ.get("OPENAI_API_KEY") == "sentinel"


def test_llmresult_builders() -> None:
    assert reply("hi").finish_reason == "stop"
    assert reply("hi").has_tool_calls is False
    c: LLMResult = call_tool("grep", {"pattern": "x"})
    assert c.has_tool_calls is True
    assert c.tool_calls[0].name == "grep"
    assert c.tool_calls[0].arguments == {"pattern": "x"}
