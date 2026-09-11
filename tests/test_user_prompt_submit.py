"""Tests for the USER_PROMPT_SUBMIT context-injection seam (ADR-0134).

Mirrors the PRE_LLM_CALL context-hook tests in ``test_hooks.py`` but pins the two
contract differences that make UserPromptSubmit its own seam rather than a mapping
entry: the stdin field is ``prompt`` (not ``user_text``), and the stdout it injects
is Claude Code's ``hookSpecificOutput.additionalContext`` (not the ``{"context": ...}``
key PRE_LLM_CALL uses). Shell hooks are tiny throwaway Python scripts run via
``sys.executable`` so the tests are hermetic and cross-platform.
"""

from __future__ import annotations

import sys
from pathlib import Path

from zakcode.evals.harness import ScriptedProvider, call_tool, make_agent, reply
from zakcode.hooks import HookEvent, HookManager, HookSpec, UserPromptSubmitPayload
from zakcode.messages import Message
from zakcode.providers.base import LLMResult


def _script(tmp_path: Path, name: str, body: str) -> list[str]:
    path = tmp_path / name
    path.write_text(body, encoding="utf-8")
    return [sys.executable, str(path)]


def _payload(prompt: str = "fix the bug", cwd: str = "") -> UserPromptSubmitPayload:
    return UserPromptSubmitPayload(prompt=prompt, cwd=cwd)


def test_has_user_prompt_hooks_precheck() -> None:
    # No hooks -> False (the loop skips the whole seam on a single boolean).
    assert HookManager().has_user_prompt_hooks() is False
    ups = HookSpec(event=HookEvent.USER_PROMPT_SUBMIT, command=["x"])
    assert HookManager([ups]).has_user_prompt_hooks() is True
    # A different-event shell hook does NOT count.
    other = HookSpec(event=HookEvent.PRE_LLM_CALL, command=["x"])
    assert HookManager([other]).has_user_prompt_hooks() is False


async def test_reads_prompt_from_stdin_and_injects_additional_context(tmp_path: Path) -> None:
    # Pins BOTH gotchas at once: the hook reads data['prompt'] (NOT user_text) from stdin,
    # and returns the Claude Code hookSpecificOutput.additionalContext shape.
    body = (
        "import sys, json\n"
        "data = json.load(sys.stdin)\n"
        "print(json.dumps({'hookSpecificOutput': {'hookEventName': 'UserPromptSubmit', "
        "'additionalContext': 'recalled for ' + data['prompt']}}))\n"
    )
    cmd = _script(tmp_path, "inject.py", body)
    mgr = HookManager([HookSpec(event=HookEvent.USER_PROMPT_SUBMIT, command=cmd)])
    texts = await mgr.gather_user_prompt_context(_payload(prompt="ship it"))
    assert texts == ["recalled for ship it"]


async def test_plain_text_stdout_is_injected(tmp_path: Path) -> None:
    body = "import sys, json\ndata = json.load(sys.stdin)\nprint('plain ' + data['prompt'])\n"
    cmd = _script(tmp_path, "plain.py", body)
    mgr = HookManager([HookSpec(event=HookEvent.USER_PROMPT_SUBMIT, command=cmd)])
    assert await mgr.gather_user_prompt_context(_payload(prompt="go")) == ["plain go"]


async def test_nonzero_exit_injects_nothing(tmp_path: Path) -> None:
    # Injection-only scope: a non-zero exit (incl. Claude Code's exit-2 prompt block)
    # contributes NO context here. Prompt-blocking is a documented follow-up, not this seam.
    body = "import sys\nsys.stderr.write('would block')\nsys.exit(2)\n"
    cmd = _script(tmp_path, "block.py", body)
    mgr = HookManager([HookSpec(event=HookEvent.USER_PROMPT_SUBMIT, command=cmd)])
    assert await mgr.gather_user_prompt_context(_payload()) == []


async def test_failure_is_isolated(tmp_path: Path) -> None:
    # A hook that raises/does-not-exist contributes nothing; a healthy sibling still runs.
    boom = HookSpec(event=HookEvent.USER_PROMPT_SUBMIT, command=[sys.executable, "/no/such/x.py"])
    good_body = "import sys, json\njson.load(sys.stdin)\nprint('survived')\n"
    good = HookSpec(
        event=HookEvent.USER_PROMPT_SUBMIT, command=_script(tmp_path, "ok.py", good_body)
    )
    mgr = HookManager([boom, good])
    assert await mgr.gather_user_prompt_context(_payload()) == ["survived"]


async def test_pre_llm_and_user_prompt_are_distinct_seams(tmp_path: Path) -> None:
    # A PRE_LLM_CALL shell hook must NOT be run by gather_user_prompt_context, and vice versa.
    body = "import sys, json\ndata = json.load(sys.stdin)\nprint('ctx')\n"
    pre = HookSpec(event=HookEvent.PRE_LLM_CALL, command=_script(tmp_path, "pre.py", body))
    mgr = HookManager([pre])
    assert await mgr.gather_user_prompt_context(_payload()) == []
    assert mgr.has_user_prompt_hooks() is False


def test_payload_serializes_prompt_key() -> None:
    # The wire shape a shell hook reads on stdin carries `prompt` (the CC field name).
    import json

    doc = json.loads(_payload(prompt="hello", cwd="/tmp").model_dump_json(by_alias=True))
    assert doc["prompt"] == "hello"
    assert doc["hook_event_name"] == "UserPromptSubmit"


# ── loop-level integration: the injected context reaches the model, once per call, unpersisted ──

_UPS_BODY = (
    "import sys, json\n"
    "data = json.load(sys.stdin)\n"
    "print(json.dumps({'hookSpecificOutput': {'hookEventName': 'UserPromptSubmit', "
    "'additionalContext': 'UPS-CTX for ' + data['prompt']}}))\n"
)


def _capturing_provider() -> tuple[ScriptedProvider, list[list[Message]]]:
    seen: list[list[Message]] = []

    def responder(messages: list[Message], system: str | None, idx: int) -> LLMResult:
        seen.append(list(messages))
        return reply("done")

    return ScriptedProvider(responder=responder), seen


def _agent_with_ups_hook(tmp_path: Path, provider: ScriptedProvider):
    agent = make_agent(provider, workspace_root=str(tmp_path), permission_mode="allow")
    cmd = _script(tmp_path, "ups_inject.py", _UPS_BODY)
    # Mutate the SHARED hook_manager in place (Agent.hook_manager IS agent.loop.hook_manager;
    # reassigning the facade attr would orphan the loop's reference). Mirrors the loader's
    # ``shell_hooks.extend`` path (__init__.py) rather than settings.json round-trip.
    agent.hook_manager.shell_hooks.append(HookSpec(event=HookEvent.USER_PROMPT_SUBMIT, command=cmd))
    return agent


async def test_user_prompt_context_injected_buffered(tmp_path: Path) -> None:
    provider, seen = _capturing_provider()
    agent = _agent_with_ups_hook(tmp_path, provider)

    await agent.arun_turn("hello world")

    last = seen[0][-1]
    assert last.role == "user"
    assert "UPS-CTX for hello world" in last.text
    assert "<injected_context>" in last.text and "</injected_context>" in last.text
    # Never persisted — the cached system+history prefix and on-disk session stay clean.
    assert all("<injected_context>" not in m.text for m in agent.session.messages)
    assert [m.role for m in agent.session.messages] == ["user", "assistant"]


async def test_user_prompt_context_injected_streaming(tmp_path: Path) -> None:
    provider, seen = _capturing_provider()
    agent = _agent_with_ups_hook(tmp_path, provider)

    _events = [ev async for ev in agent.astream_turn("go now")]

    last = seen[0][-1]
    assert last.role == "user"
    assert "UPS-CTX for go now" in last.text
    assert all("<injected_context>" not in m.text for m in agent.session.messages)


async def test_no_user_prompt_hook_leaves_messages_untouched(tmp_path: Path) -> None:
    # The has_user_prompt_hooks() gate: with nothing wired the provider sees exactly the
    # session history (just the user turn on the first call) — no injected tail.
    provider, seen = _capturing_provider()
    agent = make_agent(provider, workspace_root=str(tmp_path), permission_mode="allow")

    await agent.arun_turn("hello")

    assert len(seen[0]) == 1
    assert seen[0][0].role == "user"
    assert "<injected_context>" not in seen[0][0].text


async def test_user_prompt_context_does_not_accumulate_over_iterations(tmp_path: Path) -> None:
    # A tool round-trip drives two provider calls; the once-computed UserPromptSubmit context
    # rides EXACTLY ONE tail message per call (never stacking) and never persists.
    seen: list[list[Message]] = []

    def responder(messages: list[Message], system: str | None, idx: int) -> LLMResult:
        seen.append(list(messages))
        return call_tool("list_dir", {}) if idx == 0 else reply("done")

    provider = ScriptedProvider(responder=responder)
    agent = _agent_with_ups_hook(tmp_path, provider)

    await agent.arun_turn("enumerate")

    assert len(seen) == 2
    for call_msgs in seen:
        injected = [m for m in call_msgs if "UPS-CTX for enumerate" in m.text]
        assert len(injected) == 1
    assert all("<injected_context>" not in m.text for m in agent.session.messages)
