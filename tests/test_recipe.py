"""Tests for the Recipe Cursor (Slice 2): a verify-before-finish gate.

Pure-state tests for :class:`RecipeCursor`, plus loop-integration tests proving that
with ``recipe_mode`` on, a create-and-run turn cannot end until the written .py has been
run successfully — and ends gracefully as ``recipe_stalled`` if it never is.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

from zakcode.agent.loop import AgentLoop
from zakcode.agent.recipe import RecipeCursor, extract_acceptance, resolve_python_run
from zakcode.messages import Message, ToolResultBlock
from zakcode.permissions import PermissionMode, PermissionPolicy
from zakcode.providers.base import Capabilities, LLMResult, Provider, ToolCall
from zakcode.session.store import Session
from zakcode.tools import default_registry


def _c(call_id: str, name: str, **args: object) -> ToolCall:
    return ToolCall(id=call_id, name=name, arguments=dict(args))


def _r(
    tool_use_id: str, *, path: str | None = None, is_error: bool = False, output: str = "ok"
) -> ToolResultBlock:
    data = {"path": path} if path else None
    return ToolResultBlock(tool_use_id=tool_use_id, output=output, is_error=is_error, data=data)


# ── pure cursor logic ─────────────────────────────────────────────────────────


def test_cursor_disabled_never_gates() -> None:
    c = RecipeCursor(enabled=False)
    c.observe([_c("w", "write_file", path="a.py")], [_r("w", path="a.py")])
    assert c.needs_verification() is False


def test_cursor_gates_after_python_write() -> None:
    c = RecipeCursor(enabled=True)
    c.observe([_c("w", "write_file", path="a.py")], [_r("w", path="a.py")])
    assert c.needs_verification() is True


def test_cursor_ignores_non_python_write() -> None:
    c = RecipeCursor(enabled=True)
    c.observe([_c("w", "write_file", path="notes.txt")], [_r("w", path="notes.txt")])
    assert c.needs_verification() is False


def test_cursor_verified_by_run_referencing_file() -> None:
    c = RecipeCursor(enabled=True)
    c.observe([_c("w", "write_file", path="fizz.py")], [_r("w", path="fizz.py")])
    c.observe([_c("r", "bash", command="py fizz.py")], [_r("r")])
    assert c.needs_verification() is False


def test_cursor_run_not_referencing_file_does_not_verify() -> None:
    c = RecipeCursor(enabled=True)
    c.observe([_c("w", "write_file", path="fizz.py")], [_r("w", path="fizz.py")])
    c.observe([_c("r", "bash", command="echo hello")], [_r("r")])
    assert c.needs_verification() is True


def test_cursor_failed_run_does_not_verify() -> None:
    c = RecipeCursor(enabled=True)
    c.observe([_c("w", "write_file", path="fizz.py")], [_r("w", path="fizz.py")])
    c.observe([_c("r", "bash", command="py fizz.py")], [_r("r", is_error=True)])
    assert c.needs_verification() is True


def test_cursor_nudge_cap() -> None:
    c = RecipeCursor(enabled=True, attempt_cap=2)
    assert c.can_nudge()
    c.nudge()
    assert c.can_nudge()
    c.nudge()
    assert c.can_nudge() is False


# ── loop integration ──────────────────────────────────────────────────────────


class _ScriptedProvider(Provider):
    """Returns canned LLMResults in order; the last repeats once exhausted."""

    def __init__(self, script: Sequence[LLMResult]) -> None:
        self._script = list(script)
        self.calls = 0

    async def acomplete(
        self, messages: list[Message], *, system: str | None = None, tools: Any = None, **kw: Any
    ) -> LLMResult:
        idx = min(self.calls, len(self._script) - 1)
        self.calls += 1
        return self._script[idx]

    def count_tokens(self, messages: list[Message], *, system: str | None = None) -> int:
        return 0

    def capabilities(self) -> Capabilities:
        return Capabilities(supports_tools=True, context_window=8192)


def _loop(provider: _ScriptedProvider, tmp_path: Path, **kw: Any) -> AgentLoop:
    return AgentLoop(
        provider,
        default_registry(),
        Session(cwd=str(tmp_path), model="test"),
        workspace_root=tmp_path,
        max_iterations=10,
        **kw,
    )


def test_recipe_gate_forces_a_run_before_completing(tmp_path: Path) -> None:
    write = LLMResult(tool_calls=[_c("w1", "write_file", path="prog.py", content="print('hi')\n")])
    done = LLMResult(text="All done!")
    run = LLMResult(tool_calls=[_c("r1", "bash", command="echo prog.py")])
    provider = _ScriptedProvider([write, done, run, done])
    result = asyncio.run(_loop(provider, tmp_path, recipe_mode=True).arun_turn("make prog.py"))
    assert result.stop_reason == "completed"
    # write -> "done"(blocked+nudged) -> run -> "done"(now verified) = 4 provider calls.
    assert provider.calls == 4


def test_recipe_gate_stalls_gracefully_after_cap(tmp_path: Path) -> None:
    write = LLMResult(tool_calls=[_c("w1", "write_file", path="p.py", content="print('x')\n")])
    done = LLMResult(text="done")  # never runs the file
    provider = _ScriptedProvider([write, done])
    loop = _loop(provider, tmp_path, recipe_mode=True, recipe_attempt_cap=2)
    result = asyncio.run(loop.arun_turn("make p.py"))
    assert result.stop_reason == "recipe_stalled"


def test_recipe_disabled_completes_without_a_run(tmp_path: Path) -> None:
    write = LLMResult(tool_calls=[_c("w1", "write_file", path="p.py", content="print('x')\n")])
    done = LLMResult(text="done")
    provider = _ScriptedProvider([write, done])
    result = asyncio.run(_loop(provider, tmp_path, recipe_mode=False).arun_turn("make p.py"))
    assert result.stop_reason == "completed"
    assert provider.calls == 2  # no nudge: write, done


# ── Slice 2b-C: deterministic acceptance COMPARE ──────────────────────────────


def test_extract_acceptance_positive() -> None:
    assert extract_acceptance("write a script that prints `Hello, World!`") == "Hello, World!"
    assert extract_acceptance('it should output "pong"') == "pong"
    assert extract_acceptance("the program prints 'ok'") == "ok"


def test_extract_acceptance_none_cases() -> None:
    assert extract_acceptance("write a fibonacci function") is None  # no verb+literal
    assert extract_acceptance("print `a` and also print `b`") is None  # >1 distinct
    assert extract_acceptance("it prints `src/main.py`") is None  # path-like
    assert extract_acceptance("it prints `output.txt`") is None  # code/file extension
    assert extract_acceptance("print ``") is None  # empty literal


def test_extract_acceptance_is_verbatim_and_case_sensitive() -> None:
    assert extract_acceptance("prints `Hello`") == "Hello"  # not lowercased


def test_cursor_acceptance_requires_output_match() -> None:
    c = RecipeCursor(enabled=True, acceptance="Hello, World!")
    c.observe([_c("w", "write_file", path="hi.py")], [_r("w", path="hi.py")])
    # ran the file but printed the wrong thing -> not verified
    c.observe([_c("r", "bash", command="py hi.py")], [_r("r", output="Goodbye\n[exit code: 0]")])
    assert c.needs_verification() is True
    # ran the file and printed the expected string -> verified
    c.observe(
        [_c("r2", "bash", command="py hi.py")],
        [_r("r2", output="Hello, World!\n[exit code: 0]")],
    )
    assert c.needs_verification() is False


def test_cursor_no_acceptance_exit0_suffices() -> None:
    c = RecipeCursor(enabled=True, acceptance=None)
    c.observe([_c("w", "write_file", path="hi.py")], [_r("w", path="hi.py")])
    c.observe([_c("r", "bash", command="py hi.py")], [_r("r", output="anything at all")])
    assert c.needs_verification() is False


def test_nudge_cites_acceptance() -> None:
    c = RecipeCursor(enabled=True, acceptance="pong")
    assert "pong" in c.nudge()


def test_recipe_acceptance_stalls_on_wrong_output(tmp_path: Path) -> None:
    write = LLMResult(tool_calls=[_c("w1", "write_file", path="p.py", content="print('nope')\n")])
    done = LLMResult(text="done")
    run = LLMResult(tool_calls=[_c("r1", "bash", command="echo ran p.py and printed nothing")])
    provider = _ScriptedProvider([write, done, run, done])
    loop = _loop(
        provider, tmp_path, recipe_mode=True, recipe_acceptance_compare=True, recipe_attempt_cap=1
    )
    result = asyncio.run(loop.arun_turn("create p.py that prints `pong`"))
    assert result.stop_reason == "recipe_stalled"  # ran, but output lacked "pong"


def test_recipe_acceptance_completes_on_right_output(tmp_path: Path) -> None:
    write = LLMResult(tool_calls=[_c("w1", "write_file", path="p.py", content="print('pong')\n")])
    done = LLMResult(text="done")
    run = LLMResult(tool_calls=[_c("r1", "bash", command="echo ran p.py and printed pong")])
    provider = _ScriptedProvider([write, done, run, done])
    loop = _loop(provider, tmp_path, recipe_mode=True, recipe_acceptance_compare=True)
    result = asyncio.run(loop.arun_turn("create p.py that prints `pong`"))
    assert result.stop_reason == "completed"  # output contained "pong"


# ── Slice 2b-A: harness-issued verification run ───────────────────────────────


def _bash_spec() -> Any:
    return default_registry().get("bash").spec


def test_auto_allows_only_without_a_prompt() -> None:
    assert PermissionPolicy(PermissionMode.ALLOW).auto_allows(_bash_spec(), {"command": "py x.py"})
    assert not PermissionPolicy(PermissionMode.ASK).auto_allows(
        _bash_spec(), {"command": "py x.py"}
    )
    granted = PermissionPolicy(PermissionMode.ASK)
    granted._session_allow.add("bash")  # a prior "allow for session" grant
    assert granted.auto_allows(_bash_spec(), {"command": "py x.py"})


def test_auto_allows_never_for_dangerous_even_granted() -> None:
    p = PermissionPolicy(PermissionMode.ALLOW)
    p._session_allow.add("bash")
    assert p.auto_allows(_bash_spec(), {"command": "rm -rf /"}) is False


def test_resolve_python_run() -> None:
    cmd = resolve_python_run("/tmp/x.py")
    assert cmd is not None  # a python interpreter exists in the test env
    assert "/tmp/x.py" in cmd


def test_cursor_pending_target_and_consume() -> None:
    c = RecipeCursor(enabled=True)
    assert c.pending_target() is None
    c.observe([_c("w", "write_file", path="a.py")], [_r("w", path="a.py")])
    assert c.pending_target() == "a.py"
    c.consume_attempt()
    assert c.nudges == 1
    c.observe([_c("r", "bash", command="py a.py")], [_r("r")])
    assert c.pending_target() is None  # verified -> nothing pending


def test_harness_run_verifies_without_the_model(tmp_path: Path) -> None:
    if resolve_python_run("x.py") is None:
        pytest.skip("no python interpreter available")
    write = LLMResult(tool_calls=[_c("w1", "write_file", path="p.py", content="print('pong')\n")])
    done = LLMResult(text="done")  # the model NEVER runs it
    provider = _ScriptedProvider([write, done])
    loop = _loop(provider, tmp_path, recipe_mode=True, recipe_harness_run=True)
    result = asyncio.run(loop.arun_turn("make p.py"))
    assert result.stop_reason == "completed"  # the HARNESS ran it
    transcript = "\n".join(m.text or "" for m in loop.session.messages)
    assert "[harness]" in transcript


def test_harness_run_off_falls_back_to_nudge(tmp_path: Path) -> None:
    write = LLMResult(tool_calls=[_c("w1", "write_file", path="p.py", content="print('x')\n")])
    done = LLMResult(text="done")
    provider = _ScriptedProvider([write, done])
    loop = _loop(
        provider, tmp_path, recipe_mode=True, recipe_harness_run=False, recipe_attempt_cap=1
    )
    result = asyncio.run(loop.arun_turn("make p.py"))
    assert result.stop_reason == "recipe_stalled"
    transcript = "\n".join(m.text or "" for m in loop.session.messages)
    assert "[harness]" not in transcript  # gated off -> never auto-ran


def test_harness_run_is_bounded_on_a_broken_file(tmp_path: Path) -> None:
    if resolve_python_run("x.py") is None:
        pytest.skip("no python interpreter available")
    # 1/0 compiles (passes the firewall) but errors at runtime -> the harness run never
    # verifies; it must still stall gracefully after the cap, not loop forever.
    write = LLMResult(tool_calls=[_c("w1", "write_file", path="p.py", content="1 / 0\n")])
    done = LLMResult(text="done")
    provider = _ScriptedProvider([write, done])
    loop = _loop(
        provider, tmp_path, recipe_mode=True, recipe_harness_run=True, recipe_attempt_cap=2
    )
    result = asyncio.run(loop.arun_turn("make p.py"))
    assert result.stop_reason == "recipe_stalled"
