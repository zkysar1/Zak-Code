"""Missing-conclusion gate, contested-claim rail, apology-spiral discard (ADR-0040).

Two field transcripts, 2026-08-27, same day:

* "the script could not be found in the workspace … Could you please provide the correct
  path" — twice — with no content search ever run; "you can't grep it?" → seven hits.
* "there is no way it is this big already, go actually try to fetch some of those" → "You're
  absolutely right, my apologies …" ×9, then "I am a large language model" ×40, discarded,
  retried into another apology, done — struggled. Nothing was ever re-measured.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from zakcode.agent.loop import (
    _APOLOGY_NUDGE,
    _CHALLENGE_RAIL,
    _MISSING_NUDGE,
    AgentLoop,
    _apology_spiral,
    _claims_missing,
    _contests_prior_claim,
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

APOLOGY = (
    "You're absolutely right, my apologies. I made an incorrect assumption about the size of "
    "the tree. I apologize for the error. I am still learning and I will do better. Please "
    "disregard the previous response."
)


# ── predicates ───────────────────────────────────────────────────────────────


def test_claims_missing_matches_the_field_phrasings_only() -> None:
    assert _claims_missing("the script 'google-drive-list' could not be found in the workspace.")
    assert _claims_missing("I could not find the google-drive-list script.")
    assert _claims_missing("I still cannot locate the script. Please provide the path.")
    assert _claims_missing("That file does not exist.")
    assert not _claims_missing("I can find it under tools/.")
    assert not _claims_missing("Found it: .zakcode/skills/google-drive-list/SKILL.md")


def test_contests_prior_claim_is_disbelief_not_any_request() -> None:
    assert _contests_prior_claim(
        "go actually try to fetch some of those, there is no way it is this big already"
    )
    assert _contests_prior_claim("are you sure? that can't be right")
    assert _contests_prior_claim("I don't believe that, prove it")
    assert not _contests_prior_claim("go check the logs and fix the failing test")
    assert not _contests_prior_claim("how big is the knowledge tree?")


def test_apology_spiral_needs_three_markers() -> None:
    assert _apology_spiral(APOLOGY)
    assert not _apology_spiral("Sorry — re-measured: the tree has 1,510 nodes.")
    assert not _apology_spiral("The count is 1,510 nodes (tree stats, just now).")


# ── the loop ─────────────────────────────────────────────────────────────────


class _Grep(Tool):
    spec = ToolSpec(
        name="grep",
        description="search",
        required_permission=PermissionTier.READ_ONLY,
        concurrency=ConcurrencyClass.READ_ONLY_SAFE,
    )

    async def execute(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        return ToolResult.ok("(no matches)")


class _Stats(Tool):
    """A measurement tool whose successive outputs the test scripts (ADR-0044 figure gate:
    a figure the model states must have come from a tool, so the re-measure IS a call)."""

    outputs: list[str] = []

    spec = ToolSpec(
        name="tree_stats",
        description="stats",
        required_permission=PermissionTier.READ_ONLY,
        concurrency=ConcurrencyClass.READ_ONLY_SAFE,
    )

    async def execute(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        outputs = type(self).outputs
        return ToolResult.ok(outputs.pop(0) if len(outputs) > 1 else outputs[0])


def _call(name: str) -> LLMResult:
    return LLMResult(
        tool_calls=[ToolCall(id="c1", name=name, arguments={})], finish_reason="tool_calls"
    )


class _Sequence(Provider):
    """Plays back scripted completions in order; repeats the last one forever."""

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


def _loop(tmp_path: Path, provider: Provider) -> AgentLoop:
    registry = ToolRegistry()
    registry.register(_Grep())
    registry.register(_Stats())
    return AgentLoop(
        provider,
        registry,
        Session(cwd=str(tmp_path), model="test"),
        workspace_root=tmp_path,
        max_iterations=20,
    )


def _rails(loop: AgentLoop) -> list[str]:
    return [m.text for m in loop.session.messages if m.role == "user" and m.text]


def test_could_not_find_without_a_search_is_asked_for_the_grep_once(tmp_path: Path) -> None:
    provider = _Sequence(
        _text(
            "The script 'google-drive-list' could not be found in the workspace. I need its path."
        ),
        _text("Still could not find it; nothing more to do."),
    )
    loop = _loop(tmp_path, provider)
    asyncio.run(loop.arun_turn("review the python script that lists the drive files"))
    assert provider.calls == 2  # nudged once, then the turn ended on the second miss
    rails = _rails(loop)
    assert any(_MISSING_NUDGE in r for r in rails)
    assert sum(_MISSING_NUDGE in r for r in rails) == 1


def test_a_search_this_turn_earns_the_conclusion(tmp_path: Path) -> None:
    provider = _Sequence(
        LLMResult(
            tool_calls=[ToolCall(id="c1", name="grep", arguments={"pattern": "google-drive-list"})],
            finish_reason="tool_calls",
        ),
        _text(
            "Searched every file: google-drive-list could not be found anywhere in the workspace."
        ),
    )
    loop = _loop(tmp_path, provider)
    asyncio.run(loop.arun_turn("find the drive lister"))
    assert provider.calls == 2
    assert not any(_MISSING_NUDGE in r for r in _rails(loop))


def test_a_challenge_opens_with_the_re_measure_rail(tmp_path: Path) -> None:
    provider = _Sequence(
        _text("The knowledge tree has 10,892 nodes."),
        _text("Re-measured with tree stats just now: 1,510 nodes. The earlier figure was wrong."),
    )
    loop = _loop(tmp_path, provider)
    asyncio.run(loop.arun_turn("how big is the knowledge tree?"))
    asyncio.run(
        loop.arun_turn("go actually try to fetch some of those, there is no way it is this big")
    )
    messages = loop.session.messages
    idx = next(i for i, m in enumerate(messages) if m.text and "no way it is this big" in m.text)
    assert messages[idx + 1].role == "user" and _CHALLENGE_RAIL in (messages[idx + 1].text or "")


def test_a_plain_request_gets_no_challenge_rail(tmp_path: Path) -> None:
    provider = _Sequence(_text("Done."))
    loop = _loop(tmp_path, provider)
    asyncio.run(loop.arun_turn("how big is the tree?"))
    asyncio.run(loop.arun_turn("go check the logs and fix the failing test"))
    assert not any(_CHALLENGE_RAIL in r for r in _rails(loop))


def test_first_turn_challenge_has_nothing_to_contest(tmp_path: Path) -> None:
    provider = _Sequence(_text("Done."))
    loop = _loop(tmp_path, provider)
    asyncio.run(loop.arun_turn("no way, are you sure?"))
    assert not any(_CHALLENGE_RAIL in r for r in _rails(loop))


def test_an_apology_spiral_is_discarded_once_and_the_measurement_demanded(
    tmp_path: Path,
) -> None:
    # Both figures come from the measurement tool: the first turn reads a (wrong) 10,892
    # from it, the re-measure after the challenge reads 1,510 — so neither trips the
    # unsourced-figure gate (ADR-0044) and the apology discard is the only rail in play.
    _Stats.outputs = ["nodes: 10,892", "nodes: 1,510"]
    provider = _Sequence(
        _call("tree_stats"),
        _text("The knowledge tree has 10,892 nodes."),
        _text(APOLOGY),
        _call("tree_stats"),
        _text("Re-measured: 1,510 nodes."),
    )
    loop = _loop(tmp_path, provider)
    asyncio.run(loop.arun_turn("how big is the tree?"))
    result = asyncio.run(loop.arun_turn("there is no way it is this big"))
    assert provider.calls == 5  # stats, answer | apology (discarded), stats, answer
    assert result.stop_reason == "completed"
    assistant_texts = [m.text for m in loop.session.messages if m.role == "assistant" and m.text]
    assert APOLOGY not in assistant_texts  # discarded, never transcribed
    assert "Re-measured: 1,510 nodes." in assistant_texts
    assert any(_APOLOGY_NUDGE in r for r in _rails(loop))
