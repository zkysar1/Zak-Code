"""Tests for the Recipe Cursor (Slice 2): a verify-before-finish gate.

Pure-state tests for :class:`RecipeCursor`, plus loop-integration tests proving that the
always-on gate will not let a create-and-run turn end until the written runnable file has
been run successfully — and that it ends gracefully as ``recipe_stalled`` when it cannot.
The gate self-arms from the observed write and the harness runs the file itself whenever
that run would not prompt; there is no feature flag (one way of doing things). When the
harness run WOULD prompt, the gate falls back to nudging the model instead.
"""

from __future__ import annotations

import asyncio
import importlib.util
import shutil
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

from zakcode.agent.budget import IterationBudget
from zakcode.agent.loop import AgentLoop
from zakcode.agent.recipe import RecipeCursor, extract_acceptance, resolve_run_command
from zakcode.events import AgentDone, AgentStatus, AgentToolCall, AgentToolResult
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


def test_cursor_gates_after_js_and_shell_write() -> None:
    # Generalized beyond Python: a .js / .sh write arms the gate the same way (Bet 1).
    for fname in ("app.js", "build.sh", "task.rb", "deploy.ps1"):
        c = RecipeCursor(enabled=True)
        c.observe([_c("w", "write_file", path=fname)], [_r("w", path=fname)])
        assert c.needs_verification() is True, fname


def test_cursor_verified_by_run_referencing_file() -> None:
    c = RecipeCursor(enabled=True)
    c.observe([_c("w", "write_file", path="fizz.py")], [_r("w", path="fizz.py")])
    c.observe([_c("r", "bash", command="py fizz.py")], [_r("r")])
    assert c.needs_verification() is False


def test_cursor_verified_by_node_run() -> None:
    # A .js file is satisfied by `node app.js`, mirroring the Python path.
    c = RecipeCursor(enabled=True)
    c.observe([_c("w", "write_file", path="app.js")], [_r("w", path="app.js")])
    c.observe([_c("r", "bash", command="node app.js")], [_r("r")])
    assert c.needs_verification() is False


def test_cursor_verified_by_typescript_runners() -> None:
    # A .ts file run via tsx / ts-node (the runners resolve_run_command may emit when deno/bun
    # are absent) must be credited as executed — else a clean run falsely stalls.
    for cmd in ("tsx app.ts", "ts-node app.ts", "deno run app.ts", "bun app.ts"):
        c = RecipeCursor(enabled=True)
        c.observe([_c("w", "write_file", path="app.ts")], [_r("w", path="app.ts")])
        c.observe([_c("r", "bash", command=cmd)], [_r("r")])
        assert c.needs_verification() is False, cmd


def test_interpreter_set_covers_every_resolvable_runner() -> None:
    # Invariant guard: every head-position executable resolve_run_command can emit must be
    # recognized by _executed_targets (i.e. live in _INTERPRETERS). 'deno' is the documented
    # exception — it runs via the `deno run` subcommand, so 'deno' is the head and the file is
    # a body token. Catches a future extension whose runner is added to only one of the sets.
    from zakcode.agent import recipe as _recipe

    for ext, runners in _recipe._INTERPRETER_BY_EXT.items():
        for exe in runners:
            if exe == "deno":
                continue
            assert exe in _recipe._INTERPRETERS, f"{exe} (for {ext}) missing from _INTERPRETERS"


def test_cursor_verified_by_windows_exe_interpreter() -> None:
    # `C:\...\python.exe x.py` — the sys.executable form weak models emit on Windows — must
    # count as a real run exactly like bare `python x.py` (the .exe suffix is normalized off).
    for head in (r"C:\Py\python.exe", "python.exe", "node.cmd", "py.exe"):
        target = "a.js" if "node" in head else "a.py"
        c = RecipeCursor(enabled=True)
        c.observe([_c("w", "write_file", path=target)], [_r("w", path=target)])
        c.observe([_c("r", "bash", command=f'{head} "{target}"')], [_r("r")])
        assert c.needs_verification() is False, head


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


def test_cursor_non_executing_commands_do_not_verify() -> None:
    # A command that merely NAMES the file (never runs it) must not satisfy the gate.
    for cmd in ("echo prog.py", "cat prog.py", "ls -l prog.py", "rm prog.py", "git add prog.py"):
        c = RecipeCursor(enabled=True)
        c.observe([_c("w", "write_file", path="prog.py")], [_r("w", path="prog.py")])
        c.observe([_c("r", "bash", command=cmd)], [_r("r")])
        assert c.needs_verification() is True, cmd


def test_cursor_substring_filename_does_not_false_verify() -> None:
    # Running a DIFFERENT, longer file must not satisfy a shorter target's basename.
    c = RecipeCursor(enabled=True)
    c.observe([_c("w", "write_file", path="a.py")], [_r("w", path="a.py")])
    c.observe([_c("r", "bash", command="py aa.py")], [_r("r")])
    assert c.needs_verification() is True


def test_cursor_interpreter_with_flags_and_quoted_path_verifies() -> None:
    c = RecipeCursor(enabled=True)
    c.observe([_c("w", "write_file", path="src/app.py")], [_r("w", path="src/app.py")])
    c.observe([_c("r", "bash", command='python -u "src/app.py"')], [_r("r")])
    assert c.needs_verification() is False


def test_cursor_run_in_later_segment_verifies() -> None:
    # `cd build && py app.py` — the interpreter is in the second segment.
    c = RecipeCursor(enabled=True)
    c.observe([_c("w", "write_file", path="app.py")], [_r("w", path="app.py")])
    c.observe([_c("r", "bash", command="cd build && py app.py")], [_r("r")])
    assert c.needs_verification() is False


def test_cursor_named_in_one_segment_run_in_another_is_not_confused() -> None:
    # `python a.py && echo b.py` runs a.py but only NAMES b.py; b.py must stay unverified.
    c = RecipeCursor(enabled=True)
    c.observe([_c("w1", "write_file", path="a.py")], [_r("w1", path="a.py")])
    c.observe([_c("w2", "write_file", path="b.py")], [_r("w2", path="b.py")])
    c.observe([_c("r", "bash", command="python a.py && echo b.py")], [_r("r")])
    assert c.needs_verification() is True  # b.py was only echoed
    assert c.pending_target() == "b.py"


def test_cursor_multi_file_requires_running_each() -> None:
    c = RecipeCursor(enabled=True)
    c.observe([_c("w1", "write_file", path="a.py")], [_r("w1", path="a.py")])
    c.observe([_c("w2", "write_file", path="b.py")], [_r("w2", path="b.py")])
    c.observe([_c("r1", "bash", command="py a.py")], [_r("r1")])
    assert c.needs_verification() is True  # b.py not run yet
    assert c.pending_target() == "b.py"
    c.observe([_c("r2", "bash", command="py b.py")], [_r("r2")])
    assert c.needs_verification() is False


def test_cursor_rewrite_unverifies_a_file() -> None:
    c = RecipeCursor(enabled=True)
    c.observe([_c("w", "write_file", path="a.py")], [_r("w", path="a.py")])
    c.observe([_c("r", "bash", command="py a.py")], [_r("r")])
    assert c.needs_verification() is False
    # Editing the file again means it must be re-verified.
    c.observe([_c("w2", "edit_file", path="a.py")], [_r("w2", path="a.py")])
    assert c.needs_verification() is True


# ── green test-runner run satisfies the gate (item 3: recipe_stalled over-fire) ──


def test_cursor_verified_by_pytest_module_run() -> None:
    # The canonical over-fire case: a module + its pytest file are written, and the model
    # verifies with `python -m pytest test_x.py`. pytest IMPORTS the module (never names it as
    # a run token), so the per-target check alone would falsely stall — the green suite satisfies
    # the gate. (This is the bench 01-wordfreq / 03-lru shape.)
    c = RecipeCursor(enabled=True)
    c.observe([_c("w1", "write_file", path="wordfreq.py")], [_r("w1", path="wordfreq.py")])
    c.observe(
        [_c("w2", "write_file", path="test_wordfreq.py")], [_r("w2", path="test_wordfreq.py")]
    )
    assert c.needs_verification() is True
    c.observe([_c("r", "bash", command="python -m pytest test_wordfreq.py")], [_r("r")])
    assert c.needs_verification() is False


def test_cursor_verified_by_bare_pytest() -> None:
    # Bare `pytest -q` (no file arg) discovers and runs the suite; a green run satisfies the gate.
    c = RecipeCursor(enabled=True)
    c.observe([_c("w1", "write_file", path="lru.py")], [_r("w1", path="lru.py")])
    c.observe([_c("w2", "write_file", path="test_lru.py")], [_r("w2", path="test_lru.py")])
    c.observe([_c("r", "bash", command="pytest -q")], [_r("r")])
    assert c.needs_verification() is False


def test_cursor_multifile_library_verified_by_pytest() -> None:
    # 04-todo-cli shape: two library modules imported by the test file. Neither store.py nor
    # cli.py is ever executed directly, yet a green `pytest test_todo.py` verifies them both —
    # the per-target gate could never be satisfied for an imported-only library otherwise.
    c = RecipeCursor(enabled=True)
    for f in ("store.py", "cli.py", "test_todo.py"):
        c.observe([_c(f, "write_file", path=f)], [_r(f, path=f)])
    assert c.needs_verification() is True
    c.observe([_c("r", "bash", command="python -m pytest test_todo.py")], [_r("r")])
    assert c.needs_verification() is False


def test_cursor_failed_pytest_does_not_verify() -> None:
    # A failing suite (non-zero exit -> is_error) verifies nothing; the gate still holds.
    c = RecipeCursor(enabled=True)
    c.observe([_c("w", "write_file", path="a.py")], [_r("w", path="a.py")])
    c.observe([_c("w2", "write_file", path="test_a.py")], [_r("w2", path="test_a.py")])
    c.observe([_c("r", "bash", command="pytest")], [_r("r", is_error=True)])
    assert c.needs_verification() is True


def test_cursor_pytest_does_not_bypass_acceptance() -> None:
    # With a stated expected-output literal, a green suite is NOT enough (it does not demonstrate
    # the program prints the literal) — a direct run that prints it is still required.
    c = RecipeCursor(enabled=True, acceptance="pong")
    c.observe([_c("w", "write_file", path="p.py")], [_r("w", path="p.py")])
    c.observe([_c("r", "bash", command="pytest")], [_r("r", output="1 passed")])
    assert c.needs_verification() is True
    c.observe([_c("r2", "bash", command="py p.py")], [_r("r2", output="pong\n[exit code: 0]")])
    assert c.needs_verification() is False


def test_cursor_rewrite_after_pytest_re_requires_verification() -> None:
    # A green suite satisfies the gate; editing a module afterward invalidates it (the new code
    # was not exercised) — the turn must run its tests again before finishing.
    c = RecipeCursor(enabled=True)
    c.observe([_c("w1", "write_file", path="a.py")], [_r("w1", path="a.py")])
    c.observe([_c("w2", "write_file", path="test_a.py")], [_r("w2", path="test_a.py")])
    c.observe([_c("r", "bash", command="pytest")], [_r("r")])
    assert c.needs_verification() is False
    c.observe([_c("w3", "edit_file", path="a.py")], [_r("w3", path="a.py")])
    assert c.needs_verification() is True


def test_runs_test_suite_recognizes_runners() -> None:
    from zakcode.agent import recipe as _recipe

    runs = [
        "pytest",
        "pytest -q test_x.py",
        "py.test",
        "python -m pytest",
        "py -m unittest",
        "python -m unittest discover",
        "cd sub && pytest",
        "jest",
        "vitest run",
        "npm test",
        "pnpm test",
        "yarn test",
        "deno test",
        "bun test",
        "go test ./...",
        "cargo test",
        "node --test",
        "rspec",
    ]
    for cmd in runs:
        assert _recipe._runs_test_suite(cmd) is True, cmd
    not_runs = [
        "py app.py",
        "node app.js",
        "echo test",
        "cat test_x.py",
        "python build.py",
        "git add test_x.py",
        "python -m http.server",
        "go build ./...",
    ]
    for cmd in not_runs:
        assert _recipe._runs_test_suite(cmd) is False, cmd


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
    # No permission_policy by default → the harness verification run is never suppressed
    # (the gate is always on; see the module docstring). Tests that need the harness
    # SUPPRESSED pass an acceptEdits/ask policy so a shell run would prompt.
    return AgentLoop(
        provider,
        default_registry(),
        Session(cwd=str(tmp_path), model="test"),
        workspace_root=tmp_path,
        max_iterations=10,
        **kw,
    )


def test_recipe_gate_completes_when_model_runs_the_file(tmp_path: Path) -> None:
    # The model runs the file ITSELF before finishing, so the gate is satisfied by the
    # model's own run and no harness run is needed.
    run_cmd = resolve_run_command("prog.py")
    if run_cmd is None:
        pytest.skip("no python interpreter available")
    write = LLMResult(tool_calls=[_c("w1", "write_file", path="prog.py", content="print('hi')\n")])
    run = LLMResult(tool_calls=[_c("r1", "bash", command=run_cmd)])
    done = LLMResult(text="All done!")
    provider = _ScriptedProvider([write, run, done])
    result = asyncio.run(_loop(provider, tmp_path).arun_turn("make prog.py"))
    assert result.stop_reason == "completed"
    assert provider.calls == 3  # write, run, done — gate satisfied by the model's own run


def test_recipe_gate_completes_when_model_runs_pytest(tmp_path: Path) -> None:
    # item 3 regression: a create-with-tests turn whose verification is a GREEN pytest run must
    # finish `completed`, not stall as `recipe_stalled` — even though the module under test is
    # imported by the suite and never executed directly. (Reproduces the bench over-fire where
    # 01-wordfreq / 04-todo-cli passed the held-out oracle but reported recipe_stalled.)
    if importlib.util.find_spec("pytest") is None:
        pytest.skip("pytest not importable")
    pyrun = f'"{sys.executable}" -m pytest -q'
    write_mod = LLMResult(
        tool_calls=[
            _c("w1", "write_file", path="mymod.py", content="def add(a, b):\n    return a + b\n")
        ]
    )
    write_test = LLMResult(
        tool_calls=[
            _c(
                "w2",
                "write_file",
                path="test_mymod.py",
                content="from mymod import add\n\n\ndef test_add():\n    assert add(1, 2) == 3\n",
            )
        ]
    )
    run = LLMResult(tool_calls=[_c("r1", "bash", command=pyrun)])
    done = LLMResult(text="All done!")
    provider = _ScriptedProvider([write_mod, write_test, run, done])
    result = asyncio.run(_loop(provider, tmp_path).arun_turn("build mymod.py and its tests"))
    assert result.stop_reason == "completed"  # the green suite satisfied the gate
    assert result.degraded is False  # ...and the turn is not flagged a struggle


# ── completion-review gate: a code-changing turn re-verifies before finishing ──


_REVIEW_PHRASE = "An independent reviewer flagged"  # the independent-critic send-back nudge


def test_completion_review_critic_disapproval_sends_back_once(tmp_path: Path) -> None:
    # With completion_review_attempts=1, a code-changing turn is reviewed by the INDEPENDENT
    # critic; when it WITHHOLDS approval the turn is sent back ONCE, carrying the critic's specific
    # flagged gaps (bounded so it converges). The critic shares the scripted provider here (no
    # zakpick), so its verdict is the scripted result right after the completion attempt.
    run_cmd = resolve_run_command("prog.py")
    if run_cmd is None:
        pytest.skip("no python interpreter available")
    write = LLMResult(tool_calls=[_c("w1", "write_file", path="prog.py", content="print('hi')\n")])
    run = LLMResult(tool_calls=[_c("r1", "bash", command=run_cmd)])
    done = LLMResult(text="All done!")
    verdict = LLMResult(text='{"approved": false, "issues": "the --count flag was not added"}')
    provider = _ScriptedProvider([write, run, done, verdict, done])
    loop = _loop(provider, tmp_path, completion_review_attempts=1)
    result = asyncio.run(loop.arun_turn("make prog.py with a --count flag"))
    assert result.stop_reason == "completed"
    transcript = "\n".join(m.text or "" for m in loop.session.messages)
    assert transcript.count(_REVIEW_PHRASE) == 1  # the critic sent it back exactly once (bounded)
    assert "--count flag was not added" in transcript  # carrying the critic's SPECIFIC issue


def test_completion_review_critic_approval_finishes_immediately(tmp_path: Path) -> None:
    # When the critic APPROVES, the turn finishes immediately — an already-correct turn pays one
    # cheap side-call, not a wasted self-review re-entry.
    run_cmd = resolve_run_command("prog.py")
    if run_cmd is None:
        pytest.skip("no python interpreter available")
    write = LLMResult(tool_calls=[_c("w1", "write_file", path="prog.py", content="print('hi')\n")])
    run = LLMResult(tool_calls=[_c("r1", "bash", command=run_cmd)])
    done = LLMResult(text="All done!")
    verdict = LLMResult(text='{"approved": true, "issues": ""}')
    provider = _ScriptedProvider([write, run, done, verdict])
    loop = _loop(provider, tmp_path, completion_review_attempts=1)
    result = asyncio.run(loop.arun_turn("make prog.py"))
    assert result.stop_reason == "completed"
    transcript = "\n".join(m.text or "" for m in loop.session.messages)
    assert _REVIEW_PHRASE not in transcript  # approved → no send-back nudge


def test_completion_review_critic_fails_open(tmp_path: Path) -> None:
    # A critic that cannot return a valid verdict (here: prose, not JSON) FAILS OPEN — it must
    # never trap a turn that is genuinely finished.
    run_cmd = resolve_run_command("prog.py")
    if run_cmd is None:
        pytest.skip("no python interpreter available")
    write = LLMResult(tool_calls=[_c("w1", "write_file", path="prog.py", content="print('hi')\n")])
    run = LLMResult(tool_calls=[_c("r1", "bash", command=run_cmd)])
    done = LLMResult(text="All done!")
    junk = LLMResult(text="looks good to me, ship it!")  # not a JSON verdict
    provider = _ScriptedProvider([write, run, done, junk])
    loop = _loop(provider, tmp_path, completion_review_attempts=1)
    result = asyncio.run(loop.arun_turn("make prog.py"))
    assert result.stop_reason == "completed"  # fail-open → finished, not trapped
    transcript = "\n".join(m.text or "" for m in loop.session.messages)
    assert _REVIEW_PHRASE not in transcript


def test_completion_review_off_by_default(tmp_path: Path) -> None:
    # Default (attempts=0): no review nudge — byte-identical to before this feature.
    run_cmd = resolve_run_command("prog.py")
    if run_cmd is None:
        pytest.skip("no python interpreter available")
    write = LLMResult(tool_calls=[_c("w1", "write_file", path="prog.py", content="print('hi')\n")])
    run = LLMResult(tool_calls=[_c("r1", "bash", command=run_cmd)])
    done = LLMResult(text="All done!")
    provider = _ScriptedProvider([write, run, done])
    loop = _loop(provider, tmp_path)  # completion_review_attempts defaults to 0
    result = asyncio.run(loop.arun_turn("make prog.py"))
    assert result.stop_reason == "completed"
    assert provider.calls == 3  # write, run, done — no review round
    transcript = "\n".join(m.text or "" for m in loop.session.messages)
    assert _REVIEW_PHRASE not in transcript


def test_completion_review_inert_when_no_code_changed(tmp_path: Path) -> None:
    # A turn that changed NO code (no runnable write) is never sent back, even when enabled —
    # the gate is about verifying produced code, not gating a plain answer.
    provider = _ScriptedProvider([LLMResult(text="2 + 2 = 4")])
    loop = _loop(provider, tmp_path, completion_review_attempts=2)
    result = asyncio.run(loop.arun_turn("what is 2+2?"))
    assert result.stop_reason == "completed"
    assert provider.calls == 1
    transcript = "\n".join(m.text or "" for m in loop.session.messages)
    assert _REVIEW_PHRASE not in transcript


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


def test_extract_acceptance_rejects_output_filenames_generically() -> None:
    # Generic name.ext rejection (not a hardcoded list): .csv/.xml/.log/.ini/.dat all go.
    for fn in ("result.csv", "data.xml", "report.log", "config.ini", "out.dat"):
        assert extract_acceptance(f"it prints `{fn}`") is None, fn


def test_extract_acceptance_keeps_numeric_literals() -> None:
    # A numeric literal is NOT a filename (its "extension" is digits) and is kept.
    assert extract_acceptance("it prints `3.14`") == "3.14"
    assert extract_acceptance("it prints `v1.2`") == "v1.2"


def test_extract_acceptance_rejects_cli_flags() -> None:
    # A CLI option flag reads like an expected literal in "Print the top 10 ... `--top N` flag",
    # but it is a usage token, not stdout — reject it (else the recipe gate would demand the
    # program print `--top N`, which it never will → a FALSE recipe_stalled, the exact bench
    # 01-wordfreq over-fire). (item 3)
    assert extract_acceptance("Print the top 10 by default. Support a `--top N` flag.") is None
    assert extract_acceptance("it prints `-v`") is None
    assert extract_acceptance("prints `--help`") is None
    # ...but a genuine negative-number output is NOT a flag (dash then digit) and is kept.
    assert extract_acceptance("it prints `-5`") == "-5"
    assert extract_acceptance("it prints `-3.14`") == "-3.14"


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


def test_nudge_is_language_correct_not_hardcoded_python() -> None:
    # review2 #1: the gate now arms on .js/.sh/.rb/.ps1 too — the nudge must cite the RIGHT
    # interpreter for the pending target, never the Python `py` launcher for a non-.py file.
    expect = {"app.js": "node", "build.sh": "bash", "task.rb": "ruby", "deploy.ps1": "pwsh"}
    for fname, interp in expect.items():
        c = RecipeCursor(enabled=True)
        c.observe([_c("w", "write_file", path=fname)], [_r("w", path=fname)])
        msg = c.nudge()
        assert fname in msg, fname
        # The hint cites this file's interpreter (when on PATH) and never the Python launcher.
        assert "`py " not in msg and "py <file>" not in msg, msg
        if shutil.which(interp):
            assert interp in msg, (fname, msg)


def test_nudge_for_python_still_cites_python() -> None:
    c = RecipeCursor(enabled=True)
    c.observe([_c("w", "write_file", path="prog.py")], [_r("w", path="prog.py")])
    msg = c.nudge()
    assert "prog.py" in msg
    # resolve_run_command always resolves for .py (sys.executable fallback), so a concrete
    # python run command is cited.
    assert "prog.py" in (resolve_run_command("prog.py") or "")


def test_recipe_acceptance_stalls_on_wrong_output(tmp_path: Path) -> None:
    if resolve_run_command("p.py") is None:
        pytest.skip("no python interpreter available")
    # The harness runs the file (exit 0) but its output lacks "pong": acceptance fails, so
    # the gate is not satisfied and the turn stalls after the cap.
    write = LLMResult(tool_calls=[_c("w1", "write_file", path="p.py", content="print('nope')\n")])
    done = LLMResult(text="done")
    provider = _ScriptedProvider([write, done])
    loop = _loop(provider, tmp_path, attempt_cap=1)
    result = asyncio.run(loop.arun_turn("create p.py that prints `pong`"))
    assert result.stop_reason == "recipe_stalled"  # ran, but output lacked "pong"


def test_recipe_acceptance_completes_on_right_output(tmp_path: Path) -> None:
    if resolve_run_command("p.py") is None:
        pytest.skip("no python interpreter available")
    write = LLMResult(tool_calls=[_c("w1", "write_file", path="p.py", content="print('pong')\n")])
    done = LLMResult(text="done")
    provider = _ScriptedProvider([write, done])
    loop = _loop(provider, tmp_path)
    result = asyncio.run(loop.arun_turn("create p.py that prints `pong`"))
    assert result.stop_reason == "completed"  # harness ran it; output contained "pong"


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


def test_resolve_run_command() -> None:
    cmd = resolve_run_command("/tmp/x.py")
    assert cmd is not None  # a python interpreter exists in the test env
    assert "/tmp/x.py" in cmd


def test_resolve_run_command_picks_interpreter_by_extension() -> None:
    # The interpreter is chosen from the extension; an unknown extension resolves to None.
    # A .py always resolves to SOME python (a PATH `py`/`python*`, or the sys.executable
    # fallback whose basename is python.exe) — so just assert "py" appears, not a prefix.
    py = resolve_run_command("x.py")
    assert py is not None and "py" in py.lower()
    assert resolve_run_command("notes.txt") is None  # not a runnable extension


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
    if resolve_run_command("x.py") is None:
        pytest.skip("no python interpreter available")
    write = LLMResult(tool_calls=[_c("w1", "write_file", path="p.py", content="print('pong')\n")])
    done = LLMResult(text="done")  # the model NEVER runs it
    provider = _ScriptedProvider([write, done])
    loop = _loop(provider, tmp_path)
    result = asyncio.run(loop.arun_turn("make p.py"))
    assert result.stop_reason == "completed"  # the HARNESS ran it
    transcript = "\n".join(m.text or "" for m in loop.session.messages)
    assert "[harness]" in transcript


# ── issue #33: the harness-issued run is shell-aware (bash, else powershell) ───


def test_harness_shell_call_prefers_bash_verbatim(tmp_path: Path) -> None:
    # No policy -> the harness gate is unsuppressed: bash is the first registered shell and
    # its command is passed through verbatim (its cmd.exe / POSIX-sh quoting matches the run cmd).
    loop = _loop(_ScriptedProvider([LLMResult(text="noop")]), tmp_path)
    call = loop._harness_shell_call('py "x.py"', "verify_run_0")
    assert call is not None
    assert call.name == "bash"
    assert call.arguments["command"] == 'py "x.py"'  # no `&` prefix on the bash form
    assert call.id == "verify_run_0"


def test_harness_shell_call_falls_back_to_powershell_with_call_operator(tmp_path: Path) -> None:
    # A powershell-preferred host: the operator granted `powershell` but not `bash`, so the
    # synthetic bash would prompt. The harness falls back to powershell, and the command is
    # prefixed with the call operator `&` so a *quoted* exe path executes (issue #33).
    policy = PermissionPolicy(PermissionMode.ASK)
    policy._session_allow.add("powershell")  # granted powershell, NOT bash
    loop = _loop(_ScriptedProvider([LLMResult(text="noop")]), tmp_path, permission_policy=policy)
    call = loop._harness_shell_call('"C:\\py\\python.exe" "x.py"', "recipe_run_0")
    assert call is not None
    assert call.name == "powershell"
    assert call.arguments["command"] == '& "C:\\py\\python.exe" "x.py"'


def test_harness_shell_call_returns_none_when_no_shell_auto_allows(tmp_path: Path) -> None:
    # ASK mode, neither shell granted: every synthetic run would prompt, so the harness declines
    # (None) and the caller falls back to nudging the model — the pre-#33 powershell-host behavior.
    loop = _loop(
        _ScriptedProvider([LLMResult(text="noop")]),
        tmp_path,
        permission_policy=PermissionPolicy(PermissionMode.ASK),
    )
    assert loop._harness_shell_call("py x.py", "verify_run_0") is None


def test_harness_run_is_bounded_on_a_broken_file(tmp_path: Path) -> None:
    if resolve_run_command("x.py") is None:
        pytest.skip("no python interpreter available")
    # 1/0 compiles (passes the firewall) but errors at runtime -> the harness run never
    # verifies; it must still stall gracefully after the cap, not loop forever.
    write = LLMResult(tool_calls=[_c("w1", "write_file", path="p.py", content="1 / 0\n")])
    done = LLMResult(text="done")
    provider = _ScriptedProvider([write, done])
    loop = _loop(provider, tmp_path, attempt_cap=2)
    result = asyncio.run(loop.arun_turn("make p.py"))
    assert result.stop_reason == "recipe_stalled"


# ── audit #9: the streaming gate path (astream_turn — the REPL's real path) ────


def _drain_stream(loop: AgentLoop, user_text: str) -> list[Any]:
    async def run() -> list[Any]:
        return [ev async for ev in loop.astream_turn(user_text)]

    return asyncio.run(run())


def test_recipe_streaming_gate_completes(tmp_path: Path) -> None:
    run_cmd = resolve_run_command("prog.py")
    if run_cmd is None:
        pytest.skip("no python interpreter available")
    write = LLMResult(tool_calls=[_c("w1", "write_file", path="prog.py", content="print('hi')\n")])
    done = LLMResult(text="All done!")
    run = LLMResult(tool_calls=[_c("r1", "bash", command=run_cmd)])
    provider = _ScriptedProvider([write, done, run, done])
    events = _drain_stream(_loop(provider, tmp_path), "make prog.py")
    done_ev = next(e for e in events if isinstance(e, AgentDone))
    assert done_ev.stop_reason == "completed"


def test_recipe_streaming_gate_stalls_and_emits_status(tmp_path: Path) -> None:
    # acceptEdits arms the gate (write auto-allows) but suppresses the harness run (a shell
    # run would prompt), forcing the model-nudge path; the model never runs it -> stall with
    # the streaming-only "could not verify" status.
    write = LLMResult(tool_calls=[_c("w1", "write_file", path="p.py", content="print('x')\n")])
    done = LLMResult(text="done")  # never runs the file
    provider = _ScriptedProvider([write, done])
    loop = _loop(
        provider,
        tmp_path,
        attempt_cap=1,
        permission_policy=PermissionPolicy(PermissionMode.ACCEPT_EDITS),
    )
    events = _drain_stream(loop, "make p.py")
    done_ev = next(e for e in events if isinstance(e, AgentDone))
    assert done_ev.stop_reason == "recipe_stalled"
    statuses = [e.message for e in events if isinstance(e, AgentStatus)]
    assert any("could not verify" in m for m in statuses)  # the streaming-only stall status


def test_recipe_streaming_harness_run_emits_status(tmp_path: Path) -> None:
    if resolve_run_command("x.py") is None:
        pytest.skip("no python interpreter available")
    write = LLMResult(tool_calls=[_c("w1", "write_file", path="p.py", content="print('pong')\n")])
    done = LLMResult(text="done")  # the model never runs it; the harness does
    provider = _ScriptedProvider([write, done])
    loop = _loop(provider, tmp_path)
    events = _drain_stream(loop, "make p.py")
    done_ev = next(e for e in events if isinstance(e, AgentDone))
    assert done_ev.stop_reason == "completed"
    statuses = [e.message for e in events if isinstance(e, AgentStatus)]
    assert any("ran the file to verify" in m for m in statuses)
    # audit2 #9: the harness-issued bash run is surfaced on the live stream like any tool.
    harness_calls = [e for e in events if isinstance(e, AgentToolCall) and e.name == "bash"]
    assert harness_calls and "p.py" in harness_calls[0].arguments["command"]
    assert any(isinstance(e, AgentToolResult) for e in events)


def test_recipe_nudge_over_empty_completion_refunds_budget(tmp_path: Path) -> None:
    # audit2 #14: a recipe nudge over an EMPTY completion did no work, so it must refund the
    # shared-budget unit (matching the non-recipe empty-completion path) — else a stalling
    # recipe turn drains a shared delegation pool faster than budget.py promises. acceptEdits
    # arms the gate (write auto-allows) while suppressing the harness run (a shell run would
    # prompt), which forces the model-nudge path this test exercises.
    budget = IterationBudget(10)
    write = LLMResult(tool_calls=[_c("w1", "write_file", path="p.py", content="print('x')\n")])
    empty = LLMResult(text="")  # empty completion: no text, no tool calls
    provider = _ScriptedProvider([write, empty, empty])
    loop = _loop(
        provider,
        tmp_path,
        attempt_cap=1,
        budget=budget,
        permission_policy=PermissionPolicy(PermissionMode.ACCEPT_EDITS),
    )
    result = asyncio.run(loop.arun_turn("make p.py"))
    assert result.stop_reason == "recipe_stalled"
    # write (1) consumed; the empty nudge iteration was refunded; the stall iteration (1)
    # consumed and terminal → 2 retained of 10. Without the refund it would be 7 remaining.
    assert budget.remaining == 8


# ── audit #10: harness-run under a REAL permission policy (never an uninitiated prompt) ──


def test_harness_run_suppressed_when_run_would_prompt(tmp_path: Path) -> None:
    if resolve_run_command("x.py") is None:
        pytest.skip("no python interpreter available")
    # acceptEdits auto-allows the WRITE (so the gate arms) but a shell run would still
    # prompt: the harness run MUST be suppressed (fall back to a nudge) and the turn
    # stalls -- it must never auto-run shell behind an uninitiated prompt.
    write = LLMResult(tool_calls=[_c("w1", "write_file", path="p.py", content="print('pong')\n")])
    done = LLMResult(text="done")
    provider = _ScriptedProvider([write, done])
    loop = _loop(
        provider,
        tmp_path,
        attempt_cap=1,
        permission_policy=PermissionPolicy(PermissionMode.ACCEPT_EDITS),
    )
    result = asyncio.run(loop.arun_turn("make p.py"))
    assert result.stop_reason == "recipe_stalled"
    transcript = "\n".join(m.text or "" for m in loop.session.messages)
    # The run was never issued (it would have prompted). Match the harness-RUN message
    # specifically — since ADR-0021 every injected nudge also carries the [harness]
    # provenance tag, so the bare tag no longer discriminates run-happened from nudged.
    assert "[harness] I ran" not in transcript
    assert "run it now" in transcript  # the fallback nudge is what the model got instead


def test_harness_run_fires_under_allow_policy(tmp_path: Path) -> None:
    if resolve_run_command("x.py") is None:
        pytest.skip("no python interpreter available")
    # ALLOW mode would not prompt, so the harness run is permitted and verifies the file.
    write = LLMResult(tool_calls=[_c("w1", "write_file", path="p.py", content="print('pong')\n")])
    done = LLMResult(text="done")
    provider = _ScriptedProvider([write, done])
    loop = _loop(
        provider,
        tmp_path,
        permission_policy=PermissionPolicy(PermissionMode.ALLOW),
    )
    result = asyncio.run(loop.arun_turn("make p.py"))
    assert result.stop_reason == "completed"
    transcript = "\n".join(m.text or "" for m in loop.session.messages)
    assert "[harness]" in transcript  # the harness auto-ran it


# ── usage-refusal verification (2026-08-25) ───────────────────────────────────


def test_usage_refusal_verifies_an_args_required_script() -> None:
    """A no-args run that exits nonzero with a leading Usage: synopsis verifies the
    script — it parsed, ran, and correctly demanded arguments the harness cannot invent."""
    c = RecipeCursor(enabled=True)
    c.observe([_c("w", "write_file", path="fetch.sh")], [_r("w", path="fetch.sh")])
    assert c.needs_verification() is True
    c.observe(
        [_c("r", "bash", command="bash fetch.sh")],
        [_r("r", is_error=True, output="Usage: fetch.sh <file_id>")],
    )
    assert c.needs_verification() is False


def test_generic_failure_still_verifies_nothing() -> None:
    c = RecipeCursor(enabled=True)
    c.observe([_c("w", "write_file", path="fetch.sh")], [_r("w", path="fetch.sh")])
    c.observe(
        [_c("r", "bash", command="bash fetch.sh")],
        [_r("r", is_error=True, output="Traceback (most recent call last):\n  boom")],
    )
    assert c.needs_verification() is True


def test_buried_usage_word_does_not_verify() -> None:
    """Only a LEADING usage/synopsis line counts — 'usage' deep in a real failure never does."""
    c = RecipeCursor(enabled=True)
    c.observe([_c("w", "write_file", path="fetch.sh")], [_r("w", path="fetch.sh")])
    c.observe(
        [_c("r", "bash", command="bash fetch.sh")],
        [_r("r", is_error=True, output="error: bad flag\nsee usage below\nUsage: fetch.sh")],
    )
    assert c.needs_verification() is True


def test_usage_refusal_does_not_satisfy_an_acceptance_literal() -> None:
    """An explicit expected-output literal still demands a real green run."""
    c = RecipeCursor(enabled=True, acceptance="FETCHED")
    c.observe([_c("w", "write_file", path="fetch.sh")], [_r("w", path="fetch.sh")])
    c.observe(
        [_c("r", "bash", command="bash fetch.sh")],
        [_r("r", is_error=True, output="Usage: fetch.sh <file_id>")],
    )
    assert c.needs_verification() is True


# ── ADR-0114: verification credit that matches how Python actually runs ────────


def test_extract_acceptance_rejects_format_templates() -> None:
    """'as "word count" lines' describes the SHAPE of every output line, not one exact stdout
    string. Extracting it demands the program print the literal `word count`, which no run
    can, and the gate stalls a fully green turn (measured 2026-09-05 on coach's local model).
    """
    request = (
        "create wordstats/cli.py with a main() that reads a file and prints the five most "
        'common words as "word count" lines; then run python3 -m pytest -q tests'
    )
    assert extract_acceptance(request) is None
    assert extract_acceptance('print each result as "name: score" lines') is None
    assert extract_acceptance('output rows in the form "id,total"') is None
    assert extract_acceptance("prints records like `key=value`") is None
    assert extract_acceptance('displays them e.g. "3 apples"') is None
    # ...while a stated exact output still extracts, including one followed by ordinary prose.
    assert extract_acceptance('it should print "pong"') == "pong"
    assert extract_acceptance('prints "done" then exits') == "done"
    assert extract_acceptance("write a script that prints `Hello, World!`") == "Hello, World!"


def test_executed_targets_credit_module_path_runs() -> None:
    from zakcode.agent.recipe import _executed_targets

    targets = {"cli.py", "core.py"}
    executed = _executed_targets('cd "/w" && python3 -m wordstats.cli sample.txt --json', targets)
    assert executed == {"cli.py"}
    assert _executed_targets("py -m wordstats", {"wordstats.py"}) == {"wordstats.py"}
    assert (
        _executed_targets("python -m pytest -q tests", targets) == set()
    )  # a runner, not a target
    assert _executed_targets("echo -m wordstats.cli", targets) == set()  # not an interpreter head


def test_cursor_verified_by_module_path_run() -> None:
    c = RecipeCursor(enabled=True)
    path = "/w/wordstats/cli.py"
    c.observe([_c("w", "write_file", path=path)], [_r("w", path=path)])
    c.observe(
        [_c("r", "bash", command='cd "/w" && python3 -m wordstats.cli sample.txt --json')],
        [_r("r", output='{"hello": 3}\n[exit code: 0]')],
    )
    assert c.needs_verification() is False


def test_plumbing_files_do_not_arm_the_gate() -> None:
    """__init__.py runs only by import and conftest.py only under pytest: a write of either arms
    nothing, so the harness never 'verifies' one by running it (exit 0, having done nothing).
    """
    for path in ("/w/wordstats/__init__.py", "/w/conftest.py"):
        c = RecipeCursor(enabled=True)
        c.observe([_c("w", "write_file", path=path)], [_r("w", path=path)])
        assert c.needs_verification() is False, path
    # ...and they do not linger as unverifiable targets beside a real module.
    c = RecipeCursor(enabled=True)
    c.observe(
        [
            _c("w1", "write_file", path="/w/pkg/__init__.py"),
            _c("w2", "write_file", path="/w/pkg/core.py"),
        ],
        [_r("w1", path="/w/pkg/__init__.py"), _r("w2", path="/w/pkg/core.py")],
    )
    c.observe([_c("r", "bash", command='cd "/w" && python3 -m pkg.core')], [_r("r")])
    assert c.needs_verification() is False


def test_resolve_run_command_uses_module_form_inside_a_package(tmp_path: Path) -> None:
    pkg = tmp_path / "wordstats"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    target = pkg / "cli.py"
    target.write_text("from wordstats import core\n")
    cmd = resolve_run_command(str(target))
    assert cmd is not None
    assert f'cd "{tmp_path}"' in cmd and cmd.rstrip().endswith("-m wordstats.cli")
    assert str(target) not in cmd  # never `py "pkg/cli.py"`: that run cannot import its package
    # Nested packages resolve the whole dotted path from the first non-package ancestor.
    sub = pkg / "inner"
    sub.mkdir()
    (sub / "__init__.py").write_text("")
    (sub / "job.py").write_text("")
    nested = resolve_run_command(str(sub / "job.py"))
    assert nested is not None and nested.endswith("-m wordstats.inner.job")
    # A plain script outside any package keeps the direct form.
    script = tmp_path / "tool.py"
    script.write_text("print('ok')\n")
    plain = resolve_run_command(str(script))
    assert plain is not None and str(script) in plain and "-m" not in plain


def test_resolve_run_command_routes_test_modules_to_the_runner(tmp_path: Path) -> None:
    if not (shutil.which("pytest") or importlib.util.find_spec("pytest")):
        pytest.skip("no pytest available to route to")
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    target = tests_dir / "test_core.py"
    target.write_text("def test_x():\n    assert True\n")
    cmd = resolve_run_command(str(target))
    assert cmd is not None
    assert f'cd "{tmp_path}"' in cmd  # the parent of tests/, where the package under test lives
    assert "pytest -q" in cmd and str(target) in cmd
    # The runner form is credited both ways the gate can be satisfied.
    from zakcode.agent.recipe import _executed_targets, _runs_test_suite

    assert _runs_test_suite(cmd) is True
    assert _executed_targets(cmd, {"test_core.py"}) == {"test_core.py"}
