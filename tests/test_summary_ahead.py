"""A stretch that only read is told, once, that a summary is near (ADR-0405).

A summary replaces what a stretch printed with a short account of it. A model that read a long
source in pages and wrote nothing down reads it again after the summary, and again after the
next one: the measurement is in ADR-0405. So when the context nears the compaction threshold
and the stretch since the last summary printed files and wrote none, the harness says so once,
while the model still holds what it read.

Hermetic: scripted providers, an in-memory stand-in for the shell tool, a tmp workspace.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from zakcode.agent.compact import CompactionConfig, Compactor
from zakcode.agent.loop import AgentLoop, _files_printed, _writes_a_file
from zakcode.events import AgentStatus
from zakcode.messages import Message, ToolResultBlock
from zakcode.providers.base import Capabilities, LLMResult, Provider, ToolCall
from zakcode.session.store import Session
from zakcode.tools.base import (
    PermissionTier,
    Tool,
    ToolContext,
    ToolRegistry,
    ToolResult,
    ToolSpec,
)

WINDOW = 20_000
NOTE = "This conversation will be summarized soon"


class _Provider(Provider):
    """Scripted completions; the token count is whatever the test sets."""

    def __init__(self, factory: Any = None, *, tokens: int = 0) -> None:
        self._factory = factory
        self.tokens = tokens
        self.calls = 0

    async def acomplete(
        self, messages: list[Message], *, system: str | None = None, tools: Any = None, **kw: Any
    ) -> LLMResult:
        self.calls += 1
        return self._factory(self.calls)

    def count_tokens(self, messages: list[Message], *, system: str | None = None) -> int:
        return self.tokens

    def capabilities(self) -> Capabilities:
        return Capabilities(supports_tools=True, context_window=WINDOW)


class _Shell(Tool):
    """Stands in for the shell tool: a different page of output for every call."""

    spec = ToolSpec(name="Bash", description="Runs a command.")

    def __init__(self) -> None:
        self.runs = 0

    async def execute(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        self.runs += 1
        return ToolResult.ok(output=f"page {self.runs} of {args.get('command')}")


def _loop(provider: Provider, tmp_path: Path) -> AgentLoop:
    registry = ToolRegistry()
    registry.register(_Shell())
    return AgentLoop(
        provider,
        registry,
        Session(cwd=str(tmp_path), model="test"),
        workspace_root=tmp_path,
        max_iterations=15,
        compactor=Compactor(CompactionConfig()),
    )


def _band(loop: AgentLoop) -> tuple[int, int]:
    """The compaction threshold for this window, and a count inside the note's band."""
    caps = loop.provider.capabilities()
    threshold = loop.compactor.threshold(caps.context_window, answer_room=4_096)  # type: ignore[union-attr]
    return threshold, int(0.9 * threshold)


def _shell(call_id: str, command: str) -> tuple[ToolCall, ToolResultBlock]:
    call = ToolCall(id=call_id, name="Bash", arguments={"command": command})
    return call, ToolResultBlock(tool_use_id=call_id, output="ok")


def _read_pages(loop: AgentLoop, n: int, name: str = "score.py") -> None:
    for i in range(n):
        loop._count_stretch_io(*_shell(f"r{i}", f"sed -n '{i * 80 + 1},{i * 80 + 80}p' src/{name}"))


def _notes(loop: AgentLoop) -> list[str]:
    return [m.text for m in loop.session.messages if m.role == "user" and NOTE in m.text]


# ── reading a command line ──────────────────────────────────────────────────────


def test_files_printed_names_the_files_a_read_command_pages() -> None:
    assert _files_printed("sed -n '1,80p' /srv/app/tools/score.py") == ["score.py"]
    assert _files_printed("cd /srv/app && head -n 40 a.py docs/b.md") == ["a.py", "b.md"]
    assert _files_printed("grep -n score a.py | tail -n 5") == ["a.py"]
    # The quoted program is data; the file it reads is what was printed.
    assert _files_printed("awk '/x.py/ {print $1}' runs.log") == ["runs.log"]
    # Running a script prints its output, not the script.
    assert _files_printed("python3 tools/score.py --out rows.json") == []


def test_writes_a_file_reads_the_command_line_not_its_data() -> None:
    assert _writes_a_file("cat > notes.md <<'EOF'\nscore() -> dict\nEOF")
    assert _writes_a_file("python3 - <<'EOF'\nopen('notes.md', 'w').write('x')\nEOF")
    assert _writes_a_file("git -c user.name=a commit -m 'one'")
    assert _writes_a_file("sed -i 's/a/b/' score.py")
    assert _writes_a_file("run --all | tee out.log")
    assert not _writes_a_file('echo "a > b"')
    assert not _writes_a_file("ls -la 2>&1 | tail -n 3")
    assert not _writes_a_file("check >/dev/null 2>&1")
    # A heredoc's body is data: a `>` inside the script it feeds writes nothing.
    assert not _writes_a_file("python3 - <<'EOF'\nprint(3 > 2)\nEOF")


# ── when the note is given ─────────────────────────────────────────────────────


def test_a_read_only_stretch_near_the_threshold_is_told_once(tmp_path: Path) -> None:
    provider = _Provider()
    loop = _loop(provider, tmp_path)
    threshold, inside = _band(loop)
    _read_pages(loop, 8)
    provider.tokens = int(0.8 * threshold)
    assert loop._summary_ahead_rail() is None  # below the band: no note yet
    provider.tokens = inside
    note = loop._summary_ahead_rail()
    assert note is not None
    assert "read files 8 times, most often score.py ×8" in note
    assert loop._summary_ahead_rail() is None  # once a stretch


def test_too_few_reads_say_nothing(tmp_path: Path) -> None:
    provider = _Provider()
    loop = _loop(provider, tmp_path)
    provider.tokens = _band(loop)[1]
    _read_pages(loop, 7)
    assert loop._summary_ahead_rail() is None


def test_a_stretch_that_wrote_a_file_is_not_told(tmp_path: Path) -> None:
    provider = _Provider()
    loop = _loop(provider, tmp_path)
    provider.tokens = _band(loop)[1]
    _read_pages(loop, 8)
    loop._count_stretch_io(*_shell("w", "cat > notes.md <<'EOF'\nscore(row) -> dict\nEOF"))
    assert loop._summary_ahead_rail() is None


def test_the_read_tool_counts_as_printing_the_file(tmp_path: Path) -> None:
    provider = _Provider()
    loop = _loop(provider, tmp_path)
    provider.tokens = _band(loop)[1]
    for i in range(8):
        call = ToolCall(id=f"r{i}", name="read_file", arguments={"path": "src/trend.py"})
        loop._count_stretch_io(call, ToolResultBlock(tool_use_id=f"r{i}", output="ok"))
    note = loop._summary_ahead_rail()
    assert note is not None and "trend.py ×8" in note


def test_a_failed_call_counts_for_nothing(tmp_path: Path) -> None:
    provider = _Provider()
    loop = _loop(provider, tmp_path)
    provider.tokens = _band(loop)[1]
    _read_pages(loop, 8)
    call, block = _shell("w", "cat > notes.md <<'EOF'\nx\nEOF")
    block.is_error = True
    loop._count_stretch_io(call, block)
    assert loop._summary_ahead_rail() is not None


def test_past_the_threshold_the_compaction_comes_first(tmp_path: Path) -> None:
    provider = _Provider()
    loop = _loop(provider, tmp_path)
    threshold, _ = _band(loop)
    provider.tokens = threshold
    _read_pages(loop, 8)
    assert loop._summary_ahead_rail() is None


def test_a_summary_starts_a_new_stretch(tmp_path: Path) -> None:
    provider = _Provider()
    loop = _loop(provider, tmp_path)
    provider.tokens = _band(loop)[1]
    _read_pages(loop, 8)
    assert loop._summary_ahead_rail() is not None
    loop._adopt_compacted([Message.user("an account of the stretch")], summary="an account")
    assert loop._summary_ahead_rail() is None  # the new stretch has read nothing yet
    _read_pages(loop, 8, name="trend.py")
    note = loop._summary_ahead_rail()
    assert note is not None and "trend.py ×8" in note and "score.py" not in note


# ── the loop gives it, in both twins ────────────────────────────────────────────


def _pager() -> _Provider:
    def script(n: int) -> LLMResult:
        if n <= 9:
            command = f"sed -n '{(n - 1) * 80 + 1},{n * 80}p' src/score.py"
            return LLMResult(
                tool_calls=[ToolCall(id=f"c{n}", name="Bash", arguments={"command": command})]
            )
        return LLMResult(text="Read the module.")

    return _Provider(script)


def test_the_buffered_loop_gives_the_note_once(tmp_path: Path) -> None:
    provider = _pager()
    loop = _loop(provider, tmp_path)
    provider.tokens = _band(loop)[1]
    loop.run_turn("page through src/score.py")
    notes = _notes(loop)
    assert len(notes) == 1
    assert notes[0].startswith("[harness]")
    assert "score.py ×8" in notes[0]  # given after the eighth read, not later


def test_the_streaming_loop_gives_the_note_once_and_says_so(tmp_path: Path) -> None:
    provider = _pager()
    loop = _loop(provider, tmp_path)
    provider.tokens = _band(loop)[1]

    async def run() -> list[Any]:
        return [ev async for ev in loop.astream_turn("page through src/score.py")]

    events = asyncio.run(run())
    assert len(_notes(loop)) == 1
    statuses = [ev.message for ev in events if isinstance(ev, AgentStatus)]
    assert "note: summary near — asked to write down what was read" in statuses


class _WriteFile(Tool):
    """Stands in for a file tool that writes the workspace."""

    spec = ToolSpec(
        name="write_file",
        description="Writes a file.",
        required_permission=PermissionTier.WORKSPACE_WRITE,
    )

    async def execute(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        return ToolResult.ok(output=f"wrote {args.get('path')}")


def test_a_write_through_a_file_tool_is_writing(tmp_path: Path) -> None:
    def script(n: int) -> LLMResult:
        if n == 1:
            args = {"path": "notes.md", "content": "score(row) -> dict"}
            return LLMResult(tool_calls=[ToolCall(id="w", name="write_file", arguments=args)])
        if n <= 10:
            command = f"sed -n '{(n - 2) * 80 + 1},{(n - 1) * 80}p' src/score.py"
            return LLMResult(
                tool_calls=[ToolCall(id=f"c{n}", name="Bash", arguments={"command": command})]
            )
        return LLMResult(text="Read the module.")

    provider = _Provider(script)
    loop = _loop(provider, tmp_path)
    loop.registry.register(_WriteFile())
    provider.tokens = _band(loop)[1]
    loop.run_turn("note the interface, then page through src/score.py")
    assert provider.calls == 11  # the write ran, and the nine reads after it
    assert _notes(loop) == []


def test_no_compactor_no_note(tmp_path: Path) -> None:
    provider = _pager()
    registry = ToolRegistry()
    registry.register(_Shell())
    loop = AgentLoop(
        provider,
        registry,
        Session(cwd=str(tmp_path), model="test"),
        workspace_root=tmp_path,
        max_iterations=15,
    )
    provider.tokens = 18_000
    loop.run_turn("page through src/score.py")
    assert _notes(loop) == []
