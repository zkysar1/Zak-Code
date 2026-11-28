"""ADR-0420: ``read_rule`` answers with a rule's current file, not only the copy it loaded.

The registry is read once, when the process starts, and the prompt is rendered from it. A rule
edited on disk after that used to reach a running session only at its next restart: a pinned
rule got the pointer to the old copy, and a rule read before got the pointer to the old text.
Now the tool re-reads the file first. An edited rule's new text goes out, marked as newer than
every earlier copy, and the registry keeps it. An unchanged, vanished, emptied or renamed file
changes nothing.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from zakcode.agent.loop import AgentLoop
from zakcode.agent.prompt import SystemPromptBuilder
from zakcode.messages import ToolResultBlock
from zakcode.providers.base import Capabilities, LLMResult, Provider, ToolCall
from zakcode.rules import RuleRegistry, discover_rules, reread_rule
from zakcode.session.store import Session
from zakcode.tools.base import ToolContext, ToolRegistry
from zakcode.tools.builtins.read_rule import (
    ALREADY_READ,
    EDITED_HEAD,
    JUST_ASKED,
    PINNED,
    ReadRuleTool,
    rule_text,
    would_point,
)
from zakcode.usage import Usage

_OLD = "# Errors\nAfter a remote call, run the check named in the settings file."
_NEW = "# Errors\nAfter a remote call, run: tool check-errors."


def _write(root: Path, name: str, body: str, *, pinned: bool) -> Path:
    rules = root / ".claude" / "rules"
    rules.mkdir(parents=True, exist_ok=True)
    head = "---\nalwaysApply: true\n---\n" if pinned else ""
    path = rules / f"{name}.md"
    path.write_text(f"{head}{body}\n", encoding="utf-8")
    return path


def _registry(root: Path, *, pinned: bool = True) -> tuple[RuleRegistry, Path]:
    path = _write(root, "error-response", _OLD, pinned=pinned)
    registry, errors = discover_rules(root)
    assert not errors
    return registry, path


def _ctx(
    registry: RuleRegistry, prompt: str, held: tuple[str, ...] = (), last: str | None = None
) -> ToolContext:
    """A context whose system prompt was rendered when the session loaded its rules."""
    return ToolContext(
        workspace_root=Path("/tmp"),
        rule_registry=registry,
        in_context=lambda text: any(text in output for output in held),
        last_tool_output=lambda: last,
        in_prompt=lambda text: text in prompt,
    )


def _ask(ctx: ToolContext, **args: object):
    return asyncio.run(ReadRuleTool().execute({"name": "error-response", **args}, ctx))


def _edited(body: str = _NEW) -> str:
    return (
        EDITED_HEAD.format(name="error-response")
        + f"# error-response\n{body}\n"
        + ("[end of rule 'error-response']")
    )


# ── reread_rule ──────────────────────────────────────────────────────────────


def test_an_unchanged_file_is_no_reread(tmp_path: Path) -> None:
    registry, _ = _registry(tmp_path)
    rule = registry.get("error-response")
    assert rule is not None and reread_rule(rule) is None


def test_an_edited_file_is_reread_as_the_same_rule_marked_edited(tmp_path: Path) -> None:
    registry, path = _registry(tmp_path)
    rule = registry.get("error-response")
    assert rule is not None and rule.edited is False
    _write(tmp_path, "error-response", _NEW, pinned=True)
    fresh = reread_rule(rule)
    assert fresh is not None
    assert (fresh.name, fresh.content, fresh.path, fresh.always_apply) == (
        "error-response",
        _NEW,
        path,
        True,
    )
    assert fresh.edited is True
    assert rule.content == _OLD  # the loaded copy is not changed in place


@pytest.mark.parametrize("change", ["deleted", "emptied", "renamed", "unreadable"])
def test_a_file_that_cannot_stand_for_the_rule_keeps_the_loaded_copy(
    tmp_path: Path, change: str
) -> None:
    registry, path = _registry(tmp_path)
    rule = registry.get("error-response")
    assert rule is not None
    if change == "deleted":
        path.unlink()
    elif change == "emptied":
        path.write_text("---\nalwaysApply: true\n---\n\n", encoding="utf-8")
    elif change == "renamed":
        path.write_text(f"---\nname: another-rule\n---\n{_NEW}\n", encoding="utf-8")
    else:
        path.unlink()
        path.mkdir()  # reading a directory raises, as an unreadable file does
    assert reread_rule(rule) is None


# ── the tool ─────────────────────────────────────────────────────────────────


def test_a_pinned_rule_edited_after_loading_sends_its_new_text(tmp_path: Path) -> None:
    registry, _ = _registry(tmp_path)
    prompt = registry.render_index()
    assert _ask(_ctx(registry, prompt)).output == PINNED.format(n=1, name="error-response", step="")
    _write(tmp_path, "error-response", _NEW, pinned=True)
    res = _ask(_ctx(registry, prompt))
    assert res.output == _edited()
    assert res.verbatim and not res.is_error
    held = registry.get("error-response")
    assert held is not None and held.content == _NEW and held.edited


def test_the_ask_after_the_new_text_is_a_pointer(tmp_path: Path) -> None:
    """The new text is now the one held: a repeat is pointed at it, then the firmer line."""
    registry, _ = _registry(tmp_path)
    prompt = registry.render_index()
    _write(tmp_path, "error-response", _NEW, pinned=True)
    sent = _ask(_ctx(registry, prompt)).output
    again = _ask(_ctx(registry, prompt, held=(sent,), last=sent)).output
    summary = " Its summary: Errors"
    assert again == ALREADY_READ.format(n=1, name="error-response", step="", summary=summary)
    third = _ask(_ctx(registry, prompt, held=(sent, again), last=again)).output
    assert third == JUST_ASKED.format(n=2, name="error-response", step="")


def test_a_rule_read_before_its_edit_is_sent_again(tmp_path: Path) -> None:
    """Not pinned: the old text sits in an earlier tool result, which used to earn a pointer."""
    registry, _ = _registry(tmp_path, pinned=False)
    prompt = registry.render_index()
    first = _ask(_ctx(registry, prompt)).output
    old = rule_text("error-response", registry.get("error-response"))
    assert first == old
    _write(tmp_path, "error-response", _NEW, pinned=False)
    res = _ask(_ctx(registry, prompt, held=(first,), last="other tool output"))
    assert res.output == _edited()


def test_a_rule_read_before_and_not_edited_still_gets_the_pointer(tmp_path: Path) -> None:
    registry, _ = _registry(tmp_path, pinned=False)
    prompt = registry.render_index()
    first = _ask(_ctx(registry, prompt)).output
    res = _ask(_ctx(registry, prompt, held=(first,), last="other tool output"))
    summary = " Its summary: Errors"
    assert res.output == ALREADY_READ.format(n=1, name="error-response", step="", summary=summary)


def test_the_new_text_keeps_its_mark_when_it_goes_out_again(tmp_path: Path) -> None:
    """A compaction can take the tool result away; the next copy still says it is the newer one."""
    registry, _ = _registry(tmp_path)
    prompt = registry.render_index()
    _write(tmp_path, "error-response", _NEW, pinned=True)
    assert _ask(_ctx(registry, prompt)).output == _edited()
    assert _ask(_ctx(registry, prompt)).output == _edited()


def test_would_point_follows_the_file_and_changes_nothing(tmp_path: Path) -> None:
    registry, _ = _registry(tmp_path)
    prompt = registry.render_index()
    ctx = _ctx(registry, prompt)
    assert would_point({"name": "error-response"}, ctx) is True
    _write(tmp_path, "error-response", _NEW, pinned=True)
    assert would_point({"name": "error-response"}, ctx) is False
    held = registry.get("error-response")
    assert held is not None and held.content == _OLD  # asked, never applied


# ── through the loop ─────────────────────────────────────────────────────────


class _EditingProvider(Provider):
    """Scripted replies; before the reply at ``edit_at`` the rule's file is rewritten."""

    def __init__(self, results: list[LLMResult], root: Path, edit_at: int) -> None:
        self._results = results
        self._root = root
        self._edit_at = edit_at
        self.calls = 0

    async def acomplete(self, messages, *, system=None, tools=None, **kw) -> LLMResult:
        if self.calls == self._edit_at:
            _write(self._root, "error-response", _NEW, pinned=True)
        i = min(self.calls, len(self._results) - 1)
        self.calls += 1
        return self._results[i]

    def count_tokens(self, messages, *, system=None) -> int:
        return 0

    def capabilities(self) -> Capabilities:
        return Capabilities(context_window=32768)


def _read(call_id: str) -> LLMResult:
    call = ToolCall(id=call_id, name="read_rule", arguments={"name": "error-response"})
    return LLMResult(text="", tool_calls=[call], usage=Usage(total_tokens=1))


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True])
async def test_the_loop_delivers_a_rule_edited_mid_turn(tmp_path: Path, streaming: bool) -> None:
    registry, _ = _registry(tmp_path)
    tools = ToolRegistry()
    tools.register(ReadRuleTool())
    script = [_read("r1"), _read("r2"), LLMResult(text="done", usage=Usage(total_tokens=1))]
    loop = AgentLoop(
        _EditingProvider(script, tmp_path, edit_at=1),
        tools,
        Session(cwd=str(tmp_path), model="t"),
        prompt_builder=SystemPromptBuilder(rules=registry.render_index()),
        rule_registry=registry,
    )
    if streaming:
        async for _event in loop.astream_turn("go"):
            pass
    else:
        await loop.arun_turn("go")
    outputs = [
        block.output
        for message in loop.session.messages
        for block in message.blocks
        if isinstance(block, ToolResultBlock)
    ]
    assert outputs == [PINNED.format(n=1, name="error-response", step=""), _edited()]
