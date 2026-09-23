"""The web client draws a turn the way the terminal does (ADR-0219).

The page's ``terminal-grammar`` block is a line-for-line port of the terminal renderer's
receipts, preview rows, plan line, call-line arguments and footer. These tests run THAT
block, cut from the shipped page and never copied, under node, and compare every case with
what :mod:`zakcode.cli.render` produces for the same input. A web receipt that says less
than the terminal's, or a cap that drifts, fails here instead of on someone's screen.

Only the outcome mark differs, by design: the terminal spells success ``✓``, the page's
card dot carries it, so a success receipt is compared without its first two characters.
"""

from __future__ import annotations

import io
import json
import re
import shutil
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from rich.console import Console
from rich.text import Text

from zakcode.cli._theme import ZAK_THEME
from zakcode.cli.render import (
    _STOP_LABEL,
    StreamRenderer,
    _condense_args,
    _fmt_cost,
    _fmt_duration,
    _humanize_tokens,
    _plan_key,
)
from zakcode.events import AgentDone
from zakcode.usage import Usage

_INDEX = (
    Path(__file__).resolve().parents[1] / "src" / "zakcode" / "server" / "static" / "index.html"
)
_NODE = shutil.which("node")

pytestmark = pytest.mark.skipif(_NODE is None, reason="needs node on PATH (CI runners carry it)")

#: Appended to the extracted block: read [function, args] calls as JSON on stdin, answer each.
_HARNESS = """
const calls = JSON.parse(require("fs").readFileSync(0, "utf8"));
const fns = {
  toolView, condenseArgs, stopLabel, fmtTokens, fmtCost, fmtDur, cachePercent, footerText,
  planKey: (text) => planKey(splitLines(text)),
};
process.stdout.write(JSON.stringify(calls.map(([fn, args]) => fns[fn](...args))));
"""

#: The terminal's row styles, by the class the page gives the same row.
_ROW_CLASS = {
    "result.output": "out",
    "result.more": "more",
    "err.body": "err",
    "diff.add": "add",
    "diff.del": "del",
    "diff.meta": "dmeta",
    "diff.ctx": "ctx",
    "todo.done": "todo-done",
    "todo.open": "todo-open",
}


@pytest.fixture(scope="module")
def web(tmp_path_factory: pytest.TempPathFactory) -> Callable[[list[Any]], list[Any]]:
    """Run calls against the page's own grammar block under node."""
    html = _INDEX.read_text(encoding="utf-8")
    block = re.search(r"// BEGIN terminal-grammar\n(.*?)// END terminal-grammar", html, re.S)
    assert block, "the page must carry its terminal-grammar block"
    script = tmp_path_factory.mktemp("grammar") / "grammar.js"
    script.write_text(block.group(1) + _HARNESS, encoding="utf-8")

    def run(calls: list[Any]) -> list[Any]:
        assert _NODE is not None
        proc = subprocess.run(
            [_NODE, str(script)],
            input=json.dumps(calls).encode("utf-8"),
            capture_output=True,
            timeout=60,
            check=False,
        )
        assert proc.returncode == 0, proc.stderr.decode("utf-8", "replace")
        answers: list[Any] = json.loads(proc.stdout.decode("utf-8"))
        assert len(answers) == len(calls)
        return answers

    return run


@pytest.fixture
def terminal(monkeypatch: pytest.MonkeyPatch) -> StreamRenderer:
    monkeypatch.delenv("ZAKCODE_ASCII", raising=False)  # the page draws the unicode glyphs
    console = Console(
        file=io.StringIO(), force_terminal=False, width=300, no_color=True, theme=ZAK_THEME
    )
    return StreamRenderer(console=console, clock=lambda: 0.0, wall=lambda: 0.0)


def _row(text: Text) -> list[str]:
    # A row built with a style carries it as its base style; a diff row carries it on its one
    # span. Only an EMPTY diff row has neither, and the diff preview paints that as context.
    style = str(text.style) if text.style else (str(text.spans[0].style) if text.spans else "")
    return [_ROW_CLASS[style or "diff.ctx"], text.plain]


def _lines(n: int) -> str:
    return "\n".join(f"line {i}" for i in range(1, n + 1))


_DIFF = "--- a/x.py\n+++ b/x.py\n@@ -1,3 +1,3 @@\n ctx\n-old\n+new\n\n ctx2"
_LONG_DIFF = "@@ -1,20 +1,20 @@\n" + "\n".join(f"{'+-'[i % 2]}line {i}" for i in range(20))
_PLAN_PARTIAL = (
    "Current plan (1/3 steps done):\n  [x] 1 design — done\n"
    "  [~] 2 build the parser — note  <- current\n  [ ] 3 test"
)
_PLAN_NO_MARKER = "  [x] 1 a\n  [~] 2 b\n  [ ] 3 c"
_PLAN_DONE = (
    "Current plan (2/2 steps done):\n  [x] 1 a\n  [-] 2 b\n"
    "[plan-completion-verdict] all steps closed"
)
_PLAN_LONG_STEP = "Current plan (0/1 steps done):\n  [ ] 1 " + "x" * 120 + "  <- current"
_TODO_FLAT = "[x] write tests\n[ ] ship\nnotes line\n"

_OUTPUTS = [
    "",
    "one",
    _lines(3),
    _lines(5),
    _lines(6),
    _lines(12),
    _lines(13),
    _lines(40),
    "a\r\nb\r\n",
    "- a markdown bullet\n+ not a diff\n",
    "\n\nx\n",
    _DIFF,
    _LONG_DIFF,
    _PLAN_PARTIAL,
    _PLAN_NO_MARKER,
    _PLAN_DONE,
    _PLAN_LONG_STEP,
    _TODO_FLAT,
]
_NAMES = ["Read", "List", "Fetch", "Run", "Search", "Glob", "Edit", "Write", "Todo", "Skill"]
_ERRORS = ["", "exited with code 1", _lines(9), _lines(10), _lines(30)]


def test_every_receipt_and_preview_row_matches_the_terminal(
    web: Callable[[list[Any]], list[Any]], terminal: StreamRenderer
) -> None:
    cases = [(name, out, False) for name in _NAMES for out in _OUTPUTS]
    cases += [(name, out, True) for name in ("Run", "Edit") for out in _ERRORS]
    got = web([["toolView", list(case)] for case in cases])
    for (name, output, is_error), view in zip(cases, got, strict=True):
        summary, rows = terminal._synthesize_result(name, output.splitlines(), is_error=is_error)
        if not is_error:
            assert summary.plain.startswith("✓ "), "positive control: a success opens with ✓"
        receipt = summary.plain if is_error else summary.plain.removeprefix("✓ ")
        expected = {"receipt": receipt, "rows": [_row(r) for r in rows]}
        assert {
            "receipt": view["receipt"],
            "rows": [[r["cls"], r["text"]] for r in view["rows"]],
        } == expected, (name, output, is_error)
        # `whole` (the page's alone): the receipt and rows already show every output line,
        # so the card has nothing more to open onto.
        shown = sum(1 for cls, _ in expected["rows"] if cls != "more")
        shown += 1 if is_error and output.splitlines() else 0
        assert view["whole"] == (shown == len(output.splitlines())), (name, output, is_error)


def test_the_call_line_argument_matches_the_terminal(
    web: Callable[[list[Any]], list[Any]], terminal: StreamRenderer
) -> None:
    cases: list[dict[str, Any]] = [
        {"command": "pytest -q tests/test_x.py"},
        {"command": "echo " + "a" * 300},
        {"command": "first line\nsecond line"},
        {"command": "", "path": "src/app.py"},
        {"pattern": "def main", "path": "src"},
        {"pattern": "TODO", "glob": "*.py", "path": "src"},
        {"pattern": "x" * 200},
        {"path": "/very/" + "deep/" * 40 + "file.py"},
        {"file_path": "C:\\Users\\zak\\" + "a" * 120 + ".txt"},
        {"query": "how do I " + "q" * 150},
        {"url": "https://example.com/" + "p" * 100},
        {"name": "aspirations"},
        {"other": "free text value"},
        {"count": 3, "flag": True},
        {},
        {"path": "\N{FILE FOLDER}/" * 60 + "é.txt"},  # characters, never UTF-16 halves
    ]
    got = web([["condenseArgs", [args]] for args in cases])
    for args, text in zip(cases, got, strict=True):
        is_command, condensed = _condense_args(args, terminal._g)
        assert text == ("$ " if is_command else "") + condensed, args


def test_the_footer_label_and_state_match_the_terminal(
    web: Callable[[list[Any]], list[Any]], terminal: StreamRenderer
) -> None:
    reasons = [*_STOP_LABEL, "interrupted", "some_new_reason"]
    cases = [
        (reason, degraded, open_steps, error)
        for reason in reasons
        for degraded in (False, True)
        for open_steps in (0, 3)
        for error in ("", "rate limited\nretry after 30s")
    ]
    got = web([["stopLabel", list(case)] for case in cases])
    for (reason, degraded, open_steps, error), answer in zip(cases, got, strict=True):
        done = AgentDone(
            stop_reason=reason,
            iterations=1,
            degraded=degraded,
            open_steps=open_steps,
            error=error,
        )
        assert [answer["label"], answer["tone"]] == list(terminal._stop_label(done)), done


def test_the_footer_numbers_match_the_terminal(web: Callable[[list[Any]], list[Any]]) -> None:
    tokens = [0, 1, 999, 1000, 1049, 1050, 1250, 1350, 1750, 2250, 12345, 999_999]
    costs = [0.0, 0.0004, 0.023, 0.0230, 1.5, 0.12345, 12.3456789]
    seconds = [0.0, 0.04, 1.23, 59.94, 59.96, 60.0, 61.5, 3599.9]
    shares = [(1, 8), (3, 8), (1, 3), (2, 3), (50, 100), (12_000, 18_000)]
    calls = [["fmtTokens", [n]] for n in tokens]
    calls += [["fmtCost", [c]] for c in costs]
    calls += [["fmtDur", [s]] for s in seconds]
    calls += [["cachePercent", [c, p]] for c, p in shares]
    expected = [_humanize_tokens(n) for n in tokens]
    expected += [_fmt_cost(c) for c in costs]
    expected += [_fmt_duration(s) for s in seconds]
    expected += [round(100 * c / p) for c, p in shares]
    assert web(calls) == expected


def test_the_whole_footer_line_matches_the_terminal(
    web: Callable[[list[Any]], list[Any]], terminal: StreamRenderer
) -> None:
    usages = [
        Usage(prompt_tokens=18_000, completion_tokens=234, total_tokens=18_234, cost_usd=0.0213),
        Usage(
            prompt_tokens=18_000,
            completion_tokens=234,
            total_tokens=18_234,
            cost_usd=0.0213,
            cache_read_tokens=12_000,
        ),
    ]
    for usage in usages:
        done = AgentDone(stop_reason="completed", iterations=6, usage=usage)
        terminal._clock = lambda: 75.3  # the turn started at 0.0: 1m 15s
        terminal._turn_start = 0.0
        buffer = terminal.console.file
        assert isinstance(buffer, io.StringIO)
        buffer.seek(0)
        buffer.truncate()
        terminal._print_footer(done)
        drawn = buffer.getvalue().strip()
        assert drawn.startswith("● ")
        body = drawn.removeprefix("● ").rsplit(" · ", 1)[0]  # the page stamps HH:MM itself
        label, _ = terminal._stop_label(done)
        usage_json = usage.model_dump(mode="json")
        [text] = web([["footerText", [label, done.iterations, usage_json, 75.3]]])
        assert text == body


def test_a_plan_is_the_same_plan_on_both_sides(
    web: Callable[[list[Any]], list[Any]],
) -> None:
    plans = [_PLAN_PARTIAL, _PLAN_NO_MARKER, _PLAN_DONE, _PLAN_LONG_STEP, _TODO_FLAT, ""]
    got = web([["planKey", [plan]] for plan in plans])
    assert got == [_plan_key(plan.splitlines()) for plan in plans]
