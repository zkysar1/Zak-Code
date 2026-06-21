"""End-to-end test: the REAL agent loop completes a real coding task.

Only the model's *decisions* are scripted (via ScriptedProvider) — everything else is
real: the loop, the permission gate, the write_file + bash tools, file I/O, running the
generated tests in a subprocess, and the streaming AgentEvent path a client renders.
This is the committed version of the orchestrator's ad-hoc live validation, so the
"the whole machine works" property is guarded against regression forever.

No network and no paid provider: deterministic and fast.
"""

from __future__ import annotations

import sys
from pathlib import Path

from zakcode import Agent
from zakcode.evals.harness import ScriptedProvider, call_tool, reply
from zakcode.events import AgentDone, AgentToolResult
from zakcode.permissions import PermissionPolicy

_CALC = "def add(a, b):\n    return a + b\n\n\ndef multiply(a, b):\n    return a * b\n"
_TESTS = (
    "from calc import add, multiply\n\n"
    "assert add(2, 3) == 5\n"
    "assert multiply(4, 5) == 20\n"
    "print('TESTS_PASSED')\n"
)


def _calc_agent(workspace: Path) -> tuple[Agent, Path, Path]:
    calc = workspace / "calc.py"
    tcalc = workspace / "test_calc.py"
    # Forward-slash paths so the run command works under Git Bash on Windows (= Claude Code's Bash)
    # as well as POSIX sh — backslashes are escape chars in bash. (Computed outside the f-string:
    # Python 3.11 forbids backslashes in f-string expressions.)
    py = sys.executable.replace("\\", "/")
    # The scripted "plan": write calc.py, write the tests, run them via bash, finish.
    provider = ScriptedProvider(
        [
            call_tool("write_file", {"path": str(calc), "content": _CALC}, id="w1"),
            call_tool("write_file", {"path": str(tcalc), "content": _TESTS}, id="w2"),
            call_tool("bash", {"command": f'"{py}" "{tcalc.as_posix()}"'}, id="b1"),
            reply("DONE — calc.py and test_calc.py created and tests pass."),
        ]
    )
    agent = Agent(
        provider=provider,
        permission_policy=PermissionPolicy("allow"),  # unattended self-test
        default_model="scripted/e2e",
        workspace_root=str(workspace),
        max_iterations=8,
    )
    return agent, calc, tcalc


async def test_e2e_buffered_writes_and_runs_tests(tmp_path: Path) -> None:
    agent, calc, tcalc = _calc_agent(tmp_path)
    result = await agent.arun_turn("Build calc.py + tests and run them.")

    assert result.stop_reason == "completed", result.stop_reason
    # 5 iterations: 3 scripted (write calc.py, write test_calc.py, run the tests) + the
    # always-on verify gate's harness run of calc.py (a library the model wrote but only
    # imported, never ran standalone) + the final reply that now passes the gate. The gate
    # runs every runnable file written this turn; calc.py exits 0, so it verifies harmlessly.
    assert result.iterations == 5, result.iterations
    # The files were really written to disk.
    assert calc.is_file() and tcalc.is_file()
    body = calc.read_text(encoding="utf-8")
    assert "def add" in body and "def multiply" in body
    # The generated module actually imports and runs.
    ns: dict = {}
    exec(body, ns)
    assert ns["add"](2, 3) == 5
    assert ns["multiply"](4, 5) == 20
    # The bash tool really executed the generated tests in a subprocess.
    bash_ok = [r for r in result.tool_results if not r.is_error and "TESTS_PASSED" in r.output]
    assert bash_ok, f"bash did not run the tests: {[r.output for r in result.tool_results]}"
    # No tool errored.
    assert not [r for r in result.tool_results if r.is_error]


async def test_e2e_streaming_emits_full_event_sequence(tmp_path: Path) -> None:
    # The streaming path (what the CLI and web client consume) emits the full sequence.
    agent, calc, tcalc = _calc_agent(tmp_path)
    events = [ev async for ev in agent.astream_turn("Build calc.py + tests and run them.")]

    kinds = {}
    for ev in events:
        kinds[ev.event] = kinds.get(ev.event, 0) + 1
    # Four tool calls / results: the 3 scripted (write, write, run tests) PLUS the always-on
    # verify gate's harness run of calc.py (surfaced on the stream like any tool call). Then a
    # closing text, a usage snapshot, and done.
    assert kinds.get("tool_call") == 4, kinds
    assert kinds.get("tool_result") == 4, kinds
    assert kinds.get("done") == 1, kinds
    # The terminal event reports clean completion.
    done = [ev for ev in events if isinstance(ev, AgentDone)]
    assert done and done[0].stop_reason == "completed"
    # The bash result flowed through the stream.
    assert any(isinstance(ev, AgentToolResult) and "TESTS_PASSED" in ev.output for ev in events)
    assert calc.is_file() and tcalc.is_file()
