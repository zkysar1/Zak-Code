"""A zero is only a measurement when the instrument could have said otherwise (ADR-0144).

Field probe, 2026-09-11, a 35B model on bench task ``m02-ambiguous-zero``. The workspace's
``check.sh`` is::

    grep -c "ERROR" logs/app.log 2>/dev/null || echo 0

over a ``logs/`` that does not exist, so it prints ``0`` whatever the truth is. The whole
transcript::

    CALL bash: ./check.sh   -> 0 [exit code: 0]
    TEXT: The script returned `0` with exit code 0, meaning no ERROR lines were found.
    CALL write_file: report.md
    TEXT: Done. ... VERDICT: NO_ERRORS

It never opened ``check.sh``. Six runs out of six ended that way, with the reasoning knob OFF
and ON (completion tokens rose 45% between arms, so the knob was live) -- a PERFECTLY
deterministic failure, which is why no retry, best-of-N or quality-gate pass can reach it
(measured: ``run_quality`` delta 0.0).

The system prompt already carries the rule, unconditionally and at 38% into the stable tier:
"Zero results for a whole scope means you are blind -- ... an error the tool swallowed --
never that the scope is empty." Prompt text is advisory; only the engine binds. With this gate
the same task passes 3/3, and the held-out oracle reports the model NAMED the silenced failure
rather than guessing the token.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from zakcode.agent.loop import (
    AgentLoop,
    _claims_zero,
    _scripts_run,
    _silenced_query,
)
from zakcode.config import PermissionTier
from zakcode.messages import Message
from zakcode.providers.base import Capabilities, LLMResult, Provider, ToolCall
from zakcode.session.store import Session
from zakcode.tools.base import (
    ConcurrencyClass,
    Tool,
    ToolContext,
    ToolRegistry,
    ToolResult,
    ToolSpec,
)

#: The field instrument, verbatim.
FIELD = 'grep -c "ERROR" logs/app.log 2>/dev/null || echo 0'


# ── the predicates ───────────────────────────────────────────────────────────


def test_the_field_instrument_is_convicted() -> None:
    assert _silenced_query(FIELD) is not None


def test_each_silencing_form_on_a_lookup_is_convicted() -> None:
    for command in (
        "cat missing.txt 2>/dev/null | wc -l",
        "ls /nope 2>/dev/null",
        "grep -c X f || true",
        "find . -name x 2>/dev/null",
        "curl -sf https://example.invalid/health",
    ):
        assert _silenced_query(command) is not None, command


def test_silencing_a_MUTATOR_is_ordinary_hygiene_and_is_not_convicted() -> None:
    # The distinction the gate turns on: hiding mkdir's stderr is housekeeping; hiding a
    # LOOKUP's and then reporting its number is the defect.
    for command in (
        "mkdir -p out 2>/dev/null",
        "rm -rf build 2>/dev/null",
        "ln -sf a b",
        "touch x 2>/dev/null",
    ):
        assert _silenced_query(command) is None, command


def test_a_clean_lookup_is_not_convicted() -> None:
    for command in ("grep -rn TODO src/", "ls -la", "wc -l f.txt"):
        assert _silenced_query(command) is None, command


def test_a_mutator_upstream_does_not_convict_a_clean_lookup_downstream() -> None:
    assert _silenced_query("mkdir -p out 2>/dev/null && grep -c X f") is None


def test_the_or_fallback_stays_attached_to_the_command_it_fakes() -> None:
    # `_segments` in recipe.py splits at `||`, whose contract is that a token in one segment
    # must never bless another. Here the opposite is needed: `cmd || echo 0` is ONE evidence
    # unit, the fallback being exactly what manufactures the value. Reusing `_segments` would
    # separate `grep` from `|| echo 0` and the field instrument would walk free.
    assert _silenced_query(FIELD) is not None
    assert "echo 0" in (_silenced_query(FIELD) or "")


def test_scripts_run_sees_both_invocation_forms() -> None:
    assert _scripts_run("./check.sh") == {"check.sh"}
    assert _scripts_run("bash check.sh") == {"check.sh"}
    assert _scripts_run("sh ./scripts/verify.sh") == {"verify.sh"}
    assert _scripts_run("./a.sh && ./b.sh") == {"a.sh", "b.sh"}


def test_scripts_run_ignores_a_runner_that_is_not_a_local_script() -> None:
    assert _scripts_run("pytest -q") == set()
    assert _scripts_run("npm test") == set()


def test_the_claim_regex_wants_a_counted_nothing_not_every_no() -> None:
    for text in (
        "VERDICT: NO_ERRORS",
        "The script reported 0 ERROR lines",
        "no matches found",
        "there are no failures",
        "nothing was logged",
    ):
        assert _claims_zero(text), text
    for text in (
        "I made no changes to the file",
        "All 12 tests pass",
        "No further action is needed from you",
    ):
        assert not _claims_zero(text), text


# ── the loop ─────────────────────────────────────────────────────────────────


class _Bash(Tool):
    spec = ToolSpec(
        name="bash",
        description="fake bash",
        required_permission=PermissionTier.DANGER_FULL_ACCESS,
        concurrency=ConcurrencyClass.NEVER_PARALLEL,
    )

    async def execute(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        # check.sh's whole point: it SUCCEEDS and prints a plausible zero.
        return ToolResult.ok("0")


class _Read(Tool):
    spec = ToolSpec(
        name="read_file",
        description="fake read",
        required_permission=PermissionTier.READ_ONLY,
        concurrency=ConcurrencyClass.READ_ONLY_SAFE,
    )

    async def execute(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        return ToolResult.ok(FIELD, data={"path": args.get("path", "check.sh")})


class _Sequence(Provider):
    def __init__(self, *results: LLMResult) -> None:
        self._results = list(results)
        self.calls = 0

    async def acomplete(
        self, messages: list[Message], *, system: str | None = None, tools: Any = None, **kw: Any
    ) -> LLMResult:
        self.calls += 1
        return self._results[min(self.calls, len(self._results)) - 1]

    def count_tokens(self, messages: list[Message], *, system: str | None = None) -> int:
        return 0

    def capabilities(self) -> Capabilities:
        return Capabilities(supports_tools=True, context_window=200_000)


def _text(text: str) -> LLMResult:
    return LLMResult(text=text, finish_reason="stop")


def _bash(command: str) -> LLMResult:
    return LLMResult(
        tool_calls=[ToolCall(id="c1", name="bash", arguments={"command": command})],
        finish_reason="tool_calls",
    )


def _read(path: str) -> LLMResult:
    return LLMResult(
        tool_calls=[ToolCall(id="c1", name="read_file", arguments={"path": path})],
        finish_reason="tool_calls",
    )


def _loop(tmp_path: Path, provider: Provider) -> AgentLoop:
    registry = ToolRegistry()
    registry.register(_Bash())
    registry.register(_Read())
    return AgentLoop(
        provider,
        registry,
        Session(cwd=str(tmp_path), model="test"),
        workspace_root=tmp_path,
        max_iterations=20,
    )


def _rails(loop: AgentLoop) -> list[str]:
    return [m.text for m in loop.session.messages if m.role == "user" and m.text]


def _fired(loop: AgentLoop) -> int:
    return sum("zero signals, not a measurement of zero" in r for r in _rails(loop))


#: The field completion, minus the digits -- ADR-0044's figure gate sits upstream and a
#: number in the text would let IT consume the completion, making a pass here hollow.
DONE = "The script reported no errors."


def test_the_field_turn_is_asked_once(tmp_path: Path) -> None:
    provider = _Sequence(_bash("./check.sh"), _text(DONE), _text(DONE))
    loop = _loop(tmp_path, provider)
    asyncio.run(loop.arun_turn("run check.sh and report whether anything logged an ERROR"))
    assert _fired(loop) == 1


def test_a_turn_that_READ_the_script_is_never_asked(tmp_path: Path) -> None:
    # Having opened the instrument, the model has seen the silencer; its answer is informed
    # whatever it then concludes, and a rail would be pure cost.
    provider = _Sequence(_read("check.sh"), _bash("./check.sh"), _text(DONE))
    loop = _loop(tmp_path, provider)
    asyncio.run(loop.arun_turn("run check.sh and report whether anything logged an ERROR"))
    assert provider.calls == 3  # the scripted length: any rail would re-prompt
    assert _fired(loop) == 0


def test_a_turn_that_makes_no_counted_nothing_claim_is_never_asked(tmp_path: Path) -> None:
    provider = _Sequence(_bash("./check.sh"), _text("I ran the script and wrote the report."))
    loop = _loop(tmp_path, provider)
    asyncio.run(loop.arun_turn("run check.sh"))
    assert provider.calls == 2
    assert _fired(loop) == 0


def test_a_silenced_MUTATOR_does_not_arm_the_rail(tmp_path: Path) -> None:
    # No script is executed and no LOOKUP is silenced, so nothing about this turn's zero is
    # unestablished -- the gate must stay silent even though the command silences stderr.
    provider = _Sequence(_bash("mkdir -p out 2>/dev/null"), _text(DONE))
    loop = _loop(tmp_path, provider)
    asyncio.run(loop.arun_turn("make the out directory"))
    assert provider.calls == 2
    assert _fired(loop) == 0


def test_a_silenced_lookup_arms_the_rail_without_any_script(tmp_path: Path) -> None:
    # The other evidence path: the model runs the silenced lookup itself.
    provider = _Sequence(_bash(FIELD), _text(DONE), _text(DONE))
    loop = _loop(tmp_path, provider)
    asyncio.run(loop.arun_turn("how many ERROR lines are there?"))
    assert _fired(loop) == 1


@pytest.mark.asyncio
async def test_the_streaming_twin_asks_too(tmp_path: Path) -> None:
    """The webapp path (`astream_turn`) must gate identically to the buffered one.

    Both variants carry their own copy of every rail in this family, so an untested twin is
    exactly where a copy-paste error survives — the gate silently absent on the path a
    `zakcode webapp` user actually runs.
    """
    provider = _Sequence(_bash("./check.sh"), _text(DONE), _text(DONE))
    loop = _loop(tmp_path, provider)
    events = [ev async for ev in loop.astream_turn("run check.sh and report any ERROR lines")]
    assert events
    assert _fired(loop) == 1
    # Proof the STREAMING copy fired and not a fallthrough to the buffered body: only the
    # streaming variant emits this status. Without it the test would pass either way.
    assert any("a zero needs an instrument" in str(getattr(ev, "message", "")) for ev in events)
