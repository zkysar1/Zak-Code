"""The attribution gate: "that failure is pre-existing" needs the pre-change tree (ADR-0138).

Field transcript, 2026-09-11, a 35B model on a real repo. The task was to add
``truncate_middle`` to ``src/zakcode/providers/text_tools.py``. It did so correctly, with ten
tests. It then ran the full suite, saw one red -- in a test covering that very module -- and
closed the turn:

    "The one failure in the full suite (test_textify_assistant_tool_use_becomes_text) is
     pre-existing and unrelated to this change."

Verification commands run in seventeen iterations: **zero**. No ``git stash``, no ``git diff``,
no isolated re-run. The claim was true (the red was planted in HEAD), and that is precisely the
danger -- the same sentence is how a genuine regression ships, and nothing in the turn could
tell the two apart. Unlike the other conclusions in this family, this one is mechanically
checkable: the pre-change tree is one ``git stash`` away.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from zakcode.agent.loop import (
    _ATTRIB_NUDGE,
    _GIT_BASELINE_RE,
    AgentLoop,
    _attributes_failure_away,
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

#: The field claim, verbatim.
FIELD = (
    "I added truncate_middle to src/zakcode/providers/text_tools.py with ten tests, all "
    "passing. The one failure in the full suite (test_textify_assistant_tool_use_becomes_text) "
    "is pre-existing and unrelated to this change."
)


# ── predicates ───────────────────────────────────────────────────────────────


def test_the_field_claim_is_recognized() -> None:
    assert _attributes_failure_away(FIELD)


def test_the_phrasings_a_model_reaches_for() -> None:
    assert _attributes_failure_away("That test was already failing before I started.")
    assert _attributes_failure_away("The failure is not caused by my change.")
    assert _attributes_failure_away("test_foo fails on main too, so I left it.")
    assert _attributes_failure_away("One pre-existing failure remains in test_bar.")
    assert _attributes_failure_away("The red test has nothing to do with this change.")
    assert _attributes_failure_away("This is an existing failure, unrelated to the refactor.")


def test_an_attribution_phrase_alone_is_not_a_verdict_about_a_red_test() -> None:
    # No failure anywhere near: ordinary prose about scope, or about old code.
    assert not _attributes_failure_away("The docs update is unrelated to this refactor.")
    assert not _attributes_failure_away("I kept the pre-existing naming convention.")
    assert not _attributes_failure_away("All 3638 tests pass; nothing else changed.")
    assert not _attributes_failure_away("The helper is unrelated to the parser.")


def test_proximity_is_required_so_a_distant_failure_word_does_not_convict() -> None:
    far = (
        "Two tests failed at first and I fixed both. "
        + "I also tidied the imports while I was there. " * 8
        + "The README wording is unrelated to this change."
    )
    assert not _attributes_failure_away(far)


def test_the_verdict_is_read_from_the_tail_like_its_siblings() -> None:
    # An attribution sentence buried far above the tail is history, not this turn's verdict.
    assert not _attributes_failure_away(
        "Earlier I thought the failure was pre-existing. " + "Then I measured it. " * 120
    )


def test_baseline_probes_are_the_commands_that_move_the_tree_back() -> None:
    assert _GIT_BASELINE_RE.search("git stash && uv run pytest -q tests/test_text_tools.py")
    assert _GIT_BASELINE_RE.search("git stash -u")
    assert _GIT_BASELINE_RE.search("git worktree add /tmp/base origin/main")
    assert _GIT_BASELINE_RE.search("git -C /repo checkout -- tests/test_text_tools.py")
    assert _GIT_BASELINE_RE.search("git switch --detach origin/main")
    assert _GIT_BASELINE_RE.search("git bisect start")
    # Reporting what YOU changed is not establishing what the tree DID. Measured 2026-09-11:
    # the model edited the failing test's own file six times, ran `git diff --stat`, and still
    # closed with "not caused by my changes" -- the diff was evidence AGAINST the claim.
    assert not _GIT_BASELINE_RE.search("git diff --stat")
    assert not _GIT_BASELINE_RE.search("git --no-pager diff HEAD")
    assert not _GIT_BASELINE_RE.search("git log --oneline -5 -- tests/test_text_tools.py")
    assert not _GIT_BASELINE_RE.search("git blame tests/test_text_tools.py")
    assert not _GIT_BASELINE_RE.search("git show HEAD:tests/test_text_tools.py")
    # Never evidence of anything.
    assert not _GIT_BASELINE_RE.search("git status --porcelain")
    assert not _GIT_BASELINE_RE.search("git add -A && git commit -m wip")
    # Re-running the failing test alone still carries this turn's edits -- the whole problem.
    assert not _GIT_BASELINE_RE.search("uv run pytest -q tests/test_text_tools.py::test_textify")
    # One shell segment only: a later `checkout` word is not the git subcommand.
    assert not _GIT_BASELINE_RE.search("git status --porcelain; echo checkout")


# ── the loop ─────────────────────────────────────────────────────────────────


class _Bash(Tool):
    spec = ToolSpec(
        name="bash",
        description="fake bash",
        required_permission=PermissionTier.DANGER_FULL_ACCESS,
        concurrency=ConcurrencyClass.NEVER_PARALLEL,
    )

    async def execute(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        return ToolResult.ok("3637 passed, 1 failed")


class _Write(Tool):
    spec = ToolSpec(
        name="write_file",
        description="fake write",
        required_permission=PermissionTier.WORKSPACE_WRITE,
        concurrency=ConcurrencyClass.NEVER_PARALLEL,
    )

    async def execute(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        # notes.md, deliberately not runnable: this exercises the attribution gate, not the
        # recipe gate, which would demand a verifying run of anything executable.
        return ToolResult.ok("written", data={"path": args.get("path", "notes.md")})


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


def _bash(command: str) -> LLMResult:
    return LLMResult(
        tool_calls=[ToolCall(id="c1", name="bash", arguments={"command": command})],
        finish_reason="tool_calls",
    )


def _write() -> LLMResult:
    return LLMResult(
        tool_calls=[ToolCall(id="c1", name="write_file", arguments={"path": "notes.md"})],
        finish_reason="tool_calls",
    )


def _loop(tmp_path: Path, provider: Provider) -> AgentLoop:
    registry = ToolRegistry()
    registry.register(_Bash())
    registry.register(_Write())
    return AgentLoop(
        provider,
        registry,
        Session(cwd=str(tmp_path), model="test"),
        workspace_root=tmp_path,
        max_iterations=20,
    )


def _rails(loop: AgentLoop) -> list[str]:
    return [m.text for m in loop.session.messages if m.role == "user" and m.text]


#: The verbatim FIELD text also reports file work ("I added ... to src/...py"), which the
#: claim-vs-action guard (ADR-0033) answers FIRST when no write tool ran. The loop tests below
#: therefore use the attribution sentence alone, so what they measure is this gate and not a
#: sibling upstream of it -- and the negative cases assert NO rail at all, so a pass cannot be
#: hollow.
CLAIM = (
    "The full suite has one failure, test_textify_assistant_tool_use_becomes_text. "
    "It is pre-existing and unrelated to this change."
)


def test_the_claim_without_a_look_is_asked_for_the_stash_and_re_run_once(tmp_path: Path) -> None:
    provider = _Sequence(_text(CLAIM), _text("The suite is otherwise green."))
    loop = _loop(tmp_path, provider)
    asyncio.run(loop.arun_turn("add truncate_middle to the text tools"))
    assert provider.calls == 2  # nudged once; the second answer ends the turn
    assert sum(_ATTRIB_NUDGE in r for r in _rails(loop)) == 1


def test_the_field_turn_shape_is_nudged_after_its_own_edits(tmp_path: Path) -> None:
    # The field ordering: edit, then the verdict. The write satisfies the ADR-0033 guard, so
    # this gate is the one that answers -- exactly as it would have on 2026-09-11.
    provider = _Sequence(_write(), _text(CLAIM), _text("The suite is otherwise green."))
    loop = _loop(tmp_path, provider)
    asyncio.run(loop.arun_turn("add truncate_middle to the text tools"))
    assert sum(_ATTRIB_NUDGE in r for r in _rails(loop)) == 1


def test_a_stash_this_turn_earns_the_verdict(tmp_path: Path) -> None:
    provider = _Sequence(
        _bash("git stash && uv run pytest -q tests/test_text_tools.py; git stash pop"),
        _text(CLAIM),
    )
    loop = _loop(tmp_path, provider)
    asyncio.run(loop.arun_turn("add truncate_middle to the text tools"))
    # provider.calls == the scripted length is what makes this non-hollow: ANY gate firing
    # would have re-prompted and raised it. So the turn ended on the verdict, unchallenged.
    assert provider.calls == 2
    assert not any(_ATTRIB_NUDGE in r for r in _rails(loop))


def test_a_suite_run_before_the_first_edit_is_itself_the_baseline(tmp_path: Path) -> None:
    provider = _Sequence(_bash("uv run pytest -q"), _write(), _text(CLAIM))
    loop = _loop(tmp_path, provider)
    asyncio.run(loop.arun_turn("add truncate_middle to the text tools"))
    assert provider.calls == 3  # no gate re-prompted (see the stash test's note)
    assert not any(_ATTRIB_NUDGE in r for r in _rails(loop))


def test_a_suite_run_after_an_edit_is_not_a_baseline(tmp_path: Path) -> None:
    # The same command, the other way round: this suite saw the edits, so it says nothing
    # about the tree before them.
    provider = _Sequence(_write(), _bash("uv run pytest -q"), _text(CLAIM), _text("Green."))
    loop = _loop(tmp_path, provider)
    asyncio.run(loop.arun_turn("add truncate_middle to the text tools"))
    assert sum(_ATTRIB_NUDGE in r for r in _rails(loop)) == 1


def test_a_completion_that_blames_nothing_is_never_nudged(tmp_path: Path) -> None:
    provider = _Sequence(_write(), _text("Added the helper and its tests; the suite is green."))
    loop = _loop(tmp_path, provider)
    asyncio.run(loop.arun_turn("add truncate_middle to the text tools"))
    assert not any(_ATTRIB_NUDGE in r for r in _rails(loop))
