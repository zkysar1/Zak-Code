"""Tests for write-grounding (Slice 1 of the Recipe Cursor).

After a successful write/edit the harness reads the file back and injects the real
content + a syntax check, so a weak model cannot hallucinate what a file contains.
"""

from __future__ import annotations

from pathlib import Path

from zakcode.agent.grounding import build_write_grounding, syntax_note
from zakcode.messages import ToolResultBlock
from zakcode.providers.base import ToolCall


def _call(call_id: str, name: str, **args: object) -> ToolCall:
    return ToolCall(id=call_id, name=name, arguments=dict(args))


def _result(
    tool_use_id: str, *, path: str | None = None, is_error: bool = False
) -> ToolResultBlock:
    data = {"path": path} if path else None
    return ToolResultBlock(tool_use_id=tool_use_id, output="ok", is_error=is_error, data=data)


def test_syntax_note() -> None:
    assert "OK" in syntax_note("a.py", "x = 1\n")
    assert "FAIL" in syntax_note("a.py", "def f(:\n")
    assert syntax_note("a.txt", "def f(:") == ""  # only .py is checked
    assert syntax_note("a.py", "") == ""  # empty is not flagged


def test_grounding_reads_written_file_back(tmp_path: Path) -> None:
    f = tmp_path / "fizzbuzz.py"
    f.write_text("print('Fizz')\n")
    calls = [_call("c1", "write_file", path=str(f), content="print('Fizz')\n")]
    results = [_result("c1", path=str(f))]
    msg = build_write_grounding(calls, results)
    assert msg is not None
    assert "print('Fizz')" in msg.text  # the REAL content is echoed back
    assert "[syntax: OK]" in msg.text
    assert str(f) in msg.text


def test_grounding_none_when_no_writes() -> None:
    calls = [_call("c1", "read_file", path="a.py")]
    results = [_result("c1")]
    assert build_write_grounding(calls, results) is None


def test_grounding_skips_errored_write(tmp_path: Path) -> None:
    calls = [_call("c1", "write_file", path="nope.py", content="x")]
    results = [_result("c1", path=str(tmp_path / "nope.py"), is_error=True)]
    assert build_write_grounding(calls, results) is None


def test_grounding_flags_broken_python(tmp_path: Path) -> None:
    f = tmp_path / "broken.py"
    f.write_text("def f(:\n")  # invalid python on disk (written directly here)
    calls = [_call("c1", "edit_file", path=str(f))]
    results = [_result("c1", path=str(f))]
    msg = build_write_grounding(calls, results)
    assert msg is not None
    assert "FAIL" in msg.text  # the syntax problem is surfaced to the model


def test_grounding_caps_large_content(tmp_path: Path) -> None:
    f = tmp_path / "big.txt"
    f.write_text("x" * 10000)
    calls = [_call("c1", "write_file", path=str(f), content="...")]
    results = [_result("c1", path=str(f))]
    msg = build_write_grounding(calls, results, max_chars=100)
    assert msg is not None
    assert "truncated" in msg.text
    assert len(msg.text) < 1000  # the 10k file did not blow up the message
