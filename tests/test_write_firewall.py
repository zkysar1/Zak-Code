"""Tests for the deterministic write firewall (Slice 0 of the Recipe Cursor design).

The firewall refuses, before any bytes land, content that is plainly a shell command
the model expected to be substituted (the exact 3B corruption seen live: ``$(cat
other.py)`` written as a file body), and ``.py`` content that does not compile.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from zakcode.tools.base import ToolContext
from zakcode.tools.builtins import _safety
from zakcode.tools.builtins._safety import (
    check_literal_content,
    check_python_syntax,
    check_skill_claims,
    diagnose_python_syntax,
    skill_hosts,
)
from zakcode.tools.builtins.edit import EditFileTool
from zakcode.tools.builtins.write_file import WriteFileTool

# ── pure helpers ──────────────────────────────────────────────────────────────


def test_literal_content_rejects_whole_command_substitution() -> None:
    assert check_literal_content("$(cat temp_fizzbuzz.py)") is not None
    assert check_literal_content("  `ls -la`  ") is not None


def test_literal_content_allows_real_content() -> None:
    assert check_literal_content("print('cost is $(x)')") is None  # contains, not whole
    assert check_literal_content("x = 1\ny = 2\n") is None
    assert check_literal_content("") is None
    assert check_literal_content("cat sat on the mat") is None  # prose starting with 'cat'


def test_python_syntax_rejects_invalid_py() -> None:
    assert check_python_syntax("a.py", "def f(:\n    pass") is not None
    assert check_python_syntax("a.py", "$(cat x.py)") is not None  # also not valid python


def test_python_syntax_allows_valid_and_nonpy() -> None:
    assert check_python_syntax("a.py", "def f():\n    return 1\n") is None
    assert check_python_syntax("notes.txt", "def f(:") is None  # only .py is checked
    assert check_python_syntax("a.py", "") is None  # empty allowed


# ── through the tools ─────────────────────────────────────────────────────────


async def test_write_file_refuses_command_substitution(tmp_path: Path) -> None:
    ctx = ToolContext(workspace_root=tmp_path)
    res = await WriteFileTool().execute(
        {"path": "fizzbuzz.py", "content": "$(cat temp_fizzbuzz.py)"}, ctx
    )
    assert res.is_error
    assert "literally" in res.output.lower()
    assert not (tmp_path / "fizzbuzz.py").exists()  # no corrupt bytes landed


async def test_write_file_refuses_invalid_python(tmp_path: Path) -> None:
    ctx = ToolContext(workspace_root=tmp_path)
    res = await WriteFileTool().execute({"path": "broken.py", "content": "def f(:\n"}, ctx)
    assert res.is_error
    assert not (tmp_path / "broken.py").exists()


async def test_write_file_allows_valid_python(tmp_path: Path) -> None:
    ctx = ToolContext(workspace_root=tmp_path)
    res = await WriteFileTool().execute({"path": "ok.py", "content": "print('ok')\n"}, ctx)
    assert not res.is_error, res.output
    assert (tmp_path / "ok.py").read_text() == "print('ok')\n"


async def test_edit_refuses_change_that_breaks_python(tmp_path: Path) -> None:
    ctx = ToolContext(workspace_root=tmp_path)
    (tmp_path / "m.py").write_text("x = 1\n")
    res = await EditFileTool().execute(
        {"path": "m.py", "old_string": "x = 1", "new_string": "x = ("}, ctx
    )
    assert res.is_error
    assert (tmp_path / "m.py").read_text() == "x = 1\n"  # original preserved on a bad edit


# ── self-diagnosing refusals (ADR-0118) ───────────────────────────────────────
# Field incident 2026-09-09 (serene): every write of a mangled .py was refused with a bare
# "unterminated string literal (line 47)". The model could not see its own content, read
# the refusal as "the tool succeeded and then the environment complained", and handed the
# user a one-line fix to apply by hand. The refusal now shows the line, names the likely
# cause, states that nothing was written, and says whose problem it is.


def test_refusal_quotes_the_offending_line_with_context_and_names_the_line_once() -> None:
    r = diagnose_python_syntax("a.py", "x = 1\ny = 2\ndef f()\n    return 1\nz = 3\n")
    assert r is not None
    assert r.cause == "syntax" and r.lineno == 3
    assert "expected ':' (line 3)" in r.message
    assert "> 3 | def f()" in r.message
    assert "  2 | y = 2" in r.message and "  4 |     return 1" in r.message
    assert "The file was NOT changed" in r.message
    assert "detected at line" not in r.message  # the parser's own suffix is folded away
    assert "not the file or the environment" in r.fix
    assert "ask the user to apply the change by hand" in r.fix


def test_refusal_classifies_content_cut_off_at_its_last_line_as_truncated() -> None:
    for content in ("def f():\n    x = 'abc\n", "def f():\n    return foo(1,\n", "if x:\n"):
        r = diagnose_python_syntax("a.py", content)
        assert r is not None, content
        assert r.cause == "truncated", (content, r.message)
        assert "was cut off" in r.fix and "edit_file" in r.fix


def test_refusal_classifies_a_real_newline_inside_a_string() -> None:
    r = diagnose_python_syntax("a.py", "def f():\n    s = 'a\nb'\n    return s\n")
    assert r is not None
    assert r.cause == "newline_in_string" and r.lineno == 2
    assert "a real line break landed inside the string" in r.fix
    assert "backslash-n" in r.fix and "triple quotes" in r.fix


def test_check_python_syntax_still_returns_the_message_string() -> None:
    msg = check_python_syntax("a.py", "def f(:\n    pass")
    assert msg is not None and msg.startswith("Refusing to write invalid Python to a.py")


async def test_write_file_refusal_carries_the_fix_rail_and_the_refusal_tag(
    tmp_path: Path,
) -> None:
    ctx = ToolContext(workspace_root=tmp_path)
    res = await WriteFileTool().execute(
        {"path": "broken.py", "content": "def f():\n    x = 'abc\n"}, ctx
    )
    assert res.is_error
    assert res.data == {"refusal": "python_syntax", "cause": "truncated", "line": 2}
    assert res.fix is not None and "was cut off" in res.fix
    assert "> 2 |     x = 'abc" in res.output
    assert not (tmp_path / "broken.py").exists()
    lit = await WriteFileTool().execute({"path": "x.py", "content": "$(cat y.py)"}, ctx)
    assert lit.is_error and lit.data == {"refusal": "literal_content"}


async def test_edit_refusal_says_the_edit_would_break_the_file(tmp_path: Path) -> None:
    ctx = ToolContext(workspace_root=tmp_path)
    (tmp_path / "m.py").write_text("x = 1\ny = 2\n")
    res = await EditFileTool().execute(
        {"path": "m.py", "old_string": "y = 2", "new_string": "y = ("}, ctx
    )
    assert res.is_error
    assert res.data is not None and res.data["refusal"] == "python_syntax"
    assert "This edit would break the file, so it was NOT applied" in res.output
    assert "> 2 | y = (" in res.output
    assert res.fix is not None
    assert (tmp_path / "m.py").read_text() == "x = 1\ny = 2\n"


async def test_edit_of_an_already_broken_file_is_applied_with_a_parse_note(
    tmp_path: Path,
) -> None:
    # A repair is made one edit at a time: refusing every edit that does not fix the
    # whole file at once would leave a broken file unfixable through edit_file.
    ctx = ToolContext(workspace_root=tmp_path)
    # bytes, not text mode: Windows would translate "\n" to "\r\n" and break old_string
    (tmp_path / "b.py").write_bytes(b"def f()\n    return 1\nz = 1\n")
    res = await EditFileTool().execute(
        {"path": "b.py", "old_string": "z = 1", "new_string": "z = 2"}, ctx
    )
    assert not res.is_error, res.output
    assert (tmp_path / "b.py").read_bytes() == b"def f()\n    return 1\nz = 2\n"
    assert "still does not parse" in res.output and "> 1 | def f()" in res.output
    # …and the edit that fixes it is plain success, no note.
    fixed = await EditFileTool().execute(
        {"path": "b.py", "old_string": "def f()\n", "new_string": "def f():\n"}, ctx
    )
    assert not fixed.is_error and "does not parse" not in fixed.output


async def test_edit_old_string_misses_are_tagged_as_refusals(tmp_path: Path) -> None:
    ctx = ToolContext(workspace_root=tmp_path)
    (tmp_path / "t.txt").write_text("a a\n")
    miss = await EditFileTool().execute(
        {"path": "t.txt", "old_string": "zzz", "new_string": "y"}, ctx
    )
    assert miss.is_error and miss.data == {"refusal": "old_string_missing"}
    assert miss.fix is not None and "re-read the file" in miss.fix
    ambiguous = await EditFileTool().execute(
        {"path": "t.txt", "old_string": "a", "new_string": "b"}, ctx
    )
    assert ambiguous.is_error and ambiguous.data == {"refusal": "old_string_ambiguous"}


# ── the skill-claim gate (ADR-0126) ───────────────────────────────────────────

_FAKE_DNS = {
    "fantasysports.yahooapis.com": "ok",
    "github.com": "ok",
    "api.fantasy.yahoo.com": "nxdomain",  # the host coach invented, eleven times
    "slow.example-partner.io": "unknown",  # a lookup that could not complete
}
_FABRICATED = (
    "---\nname: waiver-wire\ndescription: weekly pickups\n---\n"
    "GET https://api.fantasy.yahoo.com/v3/fantasy/league/{league_id}/freeagents\n"
)


def _resolve(host: str) -> str:
    return _FAKE_DNS.get(host, "nxdomain")


def test_skill_hosts_skips_placeholders_locals_ips_and_marked_lines() -> None:
    body = (
        "https://api.example.com/x https://localhost:8000/y http://10.0.0.250:9090/v1\n"
        "https://{host}/templated https://svc.internal/z https://github.com/a\n"
        "https://made.up.host/q  <!-- unverified -->\n"
        "https://fantasysports.yahooapis.com/fantasy/v2/league\n"
    )
    assert skill_hosts(body) == ["github.com", "fantasysports.yahooapis.com"]


def test_a_skill_naming_a_host_that_does_not_exist_is_refused_with_the_remedy() -> None:
    hit = check_skill_claims(".zakcode/skills/waiver-wire/SKILL.md", _FABRICATED, resolve=_resolve)
    assert hit is not None
    message, bad = hit
    assert bad == ["api.fantasy.yahoo.com"]
    assert "api.fantasy.yahoo.com" in message and "does not exist" in message
    assert "web_search" in message and "unverified" in message  # both remedies named
    assert "Never invent a host" in message


def test_the_gate_is_scoped_to_skill_files_and_definitive_answers() -> None:
    # The same content anywhere else is not a skill and is not checked.
    assert check_skill_claims("notes/api-ideas.md", _FABRICATED, resolve=_resolve) is None
    assert check_skill_claims("skills/README.md", _FABRICATED, resolve=_resolve) is None
    # A real host passes; a lookup that could not complete never counts against the author.
    real = _FABRICATED.replace("api.fantasy.yahoo.com", "fantasysports.yahooapis.com")
    assert check_skill_claims("skills/w/SKILL.md", real, resolve=_resolve) is None
    slow = _FABRICATED.replace("api.fantasy.yahoo.com", "slow.example-partner.io")
    assert check_skill_claims("skills/w/SKILL.md", slow, resolve=_resolve) is None
    # Windows separators are the same path.
    assert check_skill_claims("skills\\w\\SKILL.md", _FABRICATED, resolve=_resolve) is not None
    # Saying "unverified" on the line is the honest form, and it passes.
    marked = _FABRICATED.replace("freeagents", "freeagents   (unverified — check the docs)")
    assert check_skill_claims("skills/w/SKILL.md", marked, resolve=_resolve) is None


async def test_write_file_refuses_a_fabricated_skill_as_the_models_own_content(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(_safety, "resolve_host", _resolve)
    ctx = ToolContext(workspace_root=tmp_path)
    res = await WriteFileTool().execute(
        {"path": ".zakcode/skills/waiver-wire/SKILL.md", "content": _FABRICATED}, ctx
    )
    assert res.is_error
    assert res.data is not None and res.data["refusal"] == "skill_claims"  # a content refusal
    assert res.data["hosts"] == ["api.fantasy.yahoo.com"]
    assert not (tmp_path / ".zakcode/skills/waiver-wire/SKILL.md").exists()  # nothing landed
    # The honest form is written.
    ok = await WriteFileTool().execute(
        {
            "path": ".zakcode/skills/waiver-wire/SKILL.md",
            "content": _FABRICATED.replace("freeagents", "freeagents  (unverified)"),
        },
        ctx,
    )
    assert not ok.is_error


# ── a skill no session could load is refused with the header (ADR-0131) ──────────────────


def test_a_skill_without_frontmatter_is_refused_with_its_header() -> None:
    from zakcode.tools.builtins._safety import check_skill_format

    msg = check_skill_format(".zakcode/skills/free-agent-scan/SKILL.md", "# Free Agent Scan\n")
    assert msg is not None and "no session could load it" in msg
    assert "frontmatter fence" in msg  # the parser's own reason
    assert "---\nname: free-agent-scan\ndescription: <what it does" in msg
    # A loadable skill, and any non-skill file, pass untouched.
    assert check_skill_format("skills/w/SKILL.md", "---\nname: w\n---\nBody\n") is None
    assert check_skill_format("notes/SKILL.md.txt", "# no fence here") is None
    assert check_skill_format("README.md", "# no fence here") is None
    # A fence with no name is as unloadable as no fence.
    assert check_skill_format("skills/w/SKILL.md", "---\ndescription: x\n---\nBody\n") is not None


async def test_write_file_refuses_an_unloadable_skill_before_reading_its_claims(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(_safety, "resolve_host", _resolve)
    ctx = ToolContext(workspace_root=tmp_path)
    path = ".zakcode/skills/free-agent-scan/SKILL.md"
    res = await WriteFileTool().execute(
        {"path": path, "content": "# Free Agent Scan\nGET https://api.fantasy.yahoo.com/v3\n"},
        ctx,
    )
    assert res.is_error and res.data is not None and res.data["refusal"] == "skill_format"
    assert not (tmp_path / path).exists()
    ok = await WriteFileTool().execute(
        {"path": path, "content": "---\nname: free-agent-scan\ndescription: scan\n---\nBody\n"},
        ctx,
    )
    assert not ok.is_error and (tmp_path / path).exists()


async def test_edit_refuses_only_the_edit_that_makes_a_skill_unloadable(tmp_path: Path) -> None:
    ctx = ToolContext(workspace_root=tmp_path)
    skill = tmp_path / ".zakcode/skills/w/SKILL.md"
    skill.parent.mkdir(parents=True)
    skill.write_bytes(b"---\nname: w\n---\nBody\n")
    res = await EditFileTool().execute(
        {
            "path": ".zakcode/skills/w/SKILL.md",
            "old_string": "---\nname: w\n---\n",
            "new_string": "",
        },
        ctx,
    )
    assert res.is_error and res.data is not None and res.data["refusal"] == "skill_format"
    assert skill.read_bytes().startswith(b"---\nname: w")  # untouched
    # A file that never parsed may be edited — repairing it is what the refusal asks for.
    skill.write_bytes(b"# Free Agent Scan\nBody\n")
    res = await EditFileTool().execute(
        {
            "path": ".zakcode/skills/w/SKILL.md",
            "old_string": "# Free Agent Scan\n",
            "new_string": "---\nname: w\ndescription: scan\n---\n# Free Agent Scan\n",
        },
        ctx,
    )
    assert not res.is_error and skill.read_bytes().startswith(b"---\nname: w")


async def test_edit_refuses_only_the_edit_that_introduces_a_dead_host(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(_safety, "resolve_host", _resolve)
    ctx = ToolContext(workspace_root=tmp_path)
    skill = tmp_path / ".zakcode/skills/w/SKILL.md"
    skill.parent.mkdir(parents=True)
    skill.write_bytes(b"---\nname: w\n---\nGET https://github.com/api\n")
    res = await EditFileTool().execute(
        {
            "path": ".zakcode/skills/w/SKILL.md",
            "old_string": "https://github.com/api",
            "new_string": "https://api.fantasy.yahoo.com/v3",
        },
        ctx,
    )
    assert res.is_error and res.data is not None and res.data["refusal"] == "skill_claims"
    assert b"github.com/api" in skill.read_bytes()  # untouched
    # A file that ALREADY names the dead host may still be edited elsewhere: the
    # pre-existing claim is not this edit's doing (mirrors the parse_note rule, ADR-0118).
    skill.write_bytes(b"---\nname: w\n---\nGET https://api.fantasy.yahoo.com/v3\nStep two\n")
    res = await EditFileTool().execute(
        {"path": ".zakcode/skills/w/SKILL.md", "old_string": "Step two", "new_string": "Step 2"},
        ctx,
    )
    assert not res.is_error and b"Step 2" in skill.read_bytes()
