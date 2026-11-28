"""ADR-0387: ``read_rule`` stops sending a text the model already holds.

A rule pinned whole in the system prompt (``alwaysApply``, ADR-0105) gets a numbered line that
says where it is there, and an ask right after a rule's own pointer gets a firmer line that says
where the text is, instead of the text again (ADR-0381). A rule the model does not hold, a pinned
rule the prompt cut or dropped, and ``refresh=true`` still get the text.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from zakcode.agent.loop import AgentLoop
from zakcode.agent.prompt import SystemPromptBuilder
from zakcode.messages import ToolResultBlock
from zakcode.providers.base import Capabilities, LLMResult, Provider, ToolCall
from zakcode.rules import MAX_RULE_FILE_CHARS, Rule, RuleRegistry
from zakcode.session.store import Session
from zakcode.tasks import Task, TaskNetwork
from zakcode.tools.base import ToolContext, ToolRegistry
from zakcode.tools.builtins.read_rule import JUST_ASKED, PINNED, ReadRuleTool, rule_text
from zakcode.usage import Usage

_PINNED_BODY = "# Errors\nAfter any infrastructure call, check the error alerts."


def _registry(*extra: Rule) -> RuleRegistry:
    reg = RuleRegistry()
    reg.add(Rule("alpha-rule", "# Alpha\nAlways do the alpha thing.", Path("/x/alpha-rule.md")))
    reg.add(Rule("error-response", _PINNED_BODY, Path("/x/error-response.md"), always_apply=True))
    for rule in extra:
        reg.add(rule)
    return reg


def _ctx(
    registry: RuleRegistry,
    *,
    prompt: str | None = None,
    held: tuple[str, ...] = (),
    last: str | None = None,
    plan: TaskNetwork | None = None,
) -> ToolContext:
    """A context whose system prompt carries ``prompt`` (the lean index by default)."""
    rules = registry.render_index() if prompt is None else prompt
    return ToolContext(
        workspace_root=Path("/tmp"),
        rule_registry=registry,
        in_context=lambda text: any(text in output for output in held),
        last_tool_output=lambda: last,
        in_prompt=lambda text: text in rules,
        task_network=plan,
    )


def _ask(ctx: ToolContext, name: str = "error-response", **args: object):
    return asyncio.run(ReadRuleTool().execute({"name": name, **args}, ctx))


def _pinned(n: int = 1, name: str = "error-response") -> str:
    return PINNED.format(n=n, name=name, step="")


def _text(registry: RuleRegistry, name: str) -> str:
    text = rule_text(name, registry.get(name))
    assert text is not None
    return text


def test_a_pinned_rule_gets_where_it_is_in_the_prompt_not_its_text() -> None:
    """Measured: three pinned rules sent 13,834 characters the prompt already held."""
    res = _ask(_ctx(_registry()))
    assert res.output == _pinned()
    assert "check the error alerts" not in res.output
    assert not res.verbatim
    assert not res.is_error


def test_the_pinned_line_is_counted_like_the_pointer() -> None:
    """Right after its own line, too: a pinned rule never alternates line and text."""
    res = _ask(_ctx(_registry(), held=(_pinned(1),), last=_pinned(1)))
    assert res.output == _pinned(2)


def test_the_pinned_line_sends_the_model_back_to_its_step() -> None:
    """ADR-0423: it says the ask changed nothing and names the step to go on with, and no longer
    tells the model to apply the rule there. Measured on one worker session: a pinned rule was
    asked 84 times in two hours, most asks answered by a line saying to apply it to the step."""
    plan = TaskNetwork(tasks=[Task(title="Write the tests")])
    res = _ask(_ctx(_registry(), plan=plan))
    assert res.output == PINNED.format(
        n=1, name="error-response", step=" to your current step, 'Write the tests'"
    )
    assert (
        "so this ask changed nothing. Go back to your current step, 'Write the tests' and carry "
        "on;" in res.output
    )
    assert "Apply" not in res.output
    assert "so this ask changed nothing. Go back and carry on;" in _ask(_ctx(_registry())).output


def test_refresh_still_returns_a_pinned_rules_text() -> None:
    registry = _registry()
    res = _ask(_ctx(registry), refresh=True)
    assert res.output == _text(registry, "error-response")
    assert res.verbatim


def test_a_pinned_rule_cut_at_the_per_rule_cap_still_gets_its_text() -> None:
    """Its heading is in the prompt, but not all of it, so the prompt is not where it is."""
    big = Rule(
        "big-pinned", "x" * (MAX_RULE_FILE_CHARS + 500), Path("/x/big.md"), always_apply=True
    )
    registry = _registry(big)
    assert "## big-pinned\n" in registry.render_index()
    assert _ask(_ctx(registry), "big-pinned").output == _text(registry, "big-pinned")


def test_a_pinned_rule_the_budget_dropped_still_gets_its_text() -> None:
    pinned = [
        Rule(f"p{i}", f"rule {i}\n" + "y" * 8000, Path(f"/x/p{i}.md"), always_apply=True)
        for i in range(5)
    ]
    registry = _registry(*pinned)
    dropped = [rule.name for rule in pinned if f"## {rule.name}\n" not in registry.render_index()]
    assert dropped
    assert _ask(_ctx(registry), dropped[0]).output == _text(registry, dropped[0])


def test_a_rule_that_is_not_pinned_gets_its_text_though_the_prompt_holds_it() -> None:
    """The full render holds every rule; only a pinned one is answered with where it is."""
    registry = _registry()
    full = registry.render()
    assert "## alpha-rule\n# Alpha\nAlways do the alpha thing." in full
    assert _ask(_ctx(registry, prompt=full), "alpha-rule").output == _text(registry, "alpha-rule")


def test_a_bare_context_delivers_a_pinned_rules_text() -> None:
    registry = _registry()
    ctx = ToolContext(workspace_root=Path("/tmp"), rule_registry=registry)
    assert _ask(ctx).output == _text(registry, "error-response")


def test_the_firmer_line_names_the_lines_the_text_opens_and_closes_with() -> None:
    registry = _registry()
    first, *_, last = _text(registry, "alpha-rule").splitlines()
    line = JUST_ASKED.format(n=2, name="alpha-rule", step="")
    assert f'"{first}"' in line
    assert f'"{last}"' in line


class _Scripted(Provider):
    def __init__(self, results: list[LLMResult]) -> None:
        self._results = results
        self.calls = 0

    async def acomplete(self, messages, *, system=None, tools=None, **kw) -> LLMResult:
        i = min(self.calls, len(self._results) - 1)
        self.calls += 1
        return self._results[i]

    def count_tokens(self, messages, *, system=None) -> int:
        return 0

    def capabilities(self) -> Capabilities:
        return Capabilities(context_window=32768)


def _read(call_id: str, name: str) -> LLMResult:
    return LLMResult(
        text="",
        tool_calls=[ToolCall(id=call_id, name="read_rule", arguments={"name": name})],
        usage=Usage(total_tokens=1),
    )


async def _outputs_of(loop: AgentLoop, *, streaming: bool) -> list[str]:
    if streaming:
        async for _event in loop.astream_turn("go"):
            pass
    else:
        await loop.arun_turn("go")
    return [
        block.output
        for message in loop.session.messages
        for block in message.blocks
        if isinstance(block, ToolResultBlock)
    ]


def _loop(tmp_path: Path, registry: RuleRegistry, prompt_rules: str | None) -> AgentLoop:
    tools = ToolRegistry()
    tools.register(ReadRuleTool())
    asks = [_read("r1", "error-response"), _read("r2", "alpha-rule")]
    return AgentLoop(
        _Scripted([*asks, LLMResult(text="done", usage=Usage(total_tokens=1))]),
        tools,
        Session(cwd=str(tmp_path), model="t"),
        prompt_builder=SystemPromptBuilder(rules=prompt_rules),
        rule_registry=registry,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True])
async def test_the_loop_answers_a_pinned_rule_from_its_own_prompt(
    tmp_path: Path, streaming: bool
) -> None:
    """Both turn paths: the loop's prompt carries the lean index, so the pinned rule gets where it
    is and the other rule its text."""
    registry = _registry()
    outputs = await _outputs_of(
        _loop(tmp_path, registry, registry.render_index()), streaming=streaming
    )
    assert outputs == [_pinned(), _text(registry, "alpha-rule")]


@pytest.mark.asyncio
async def test_a_loop_whose_prompt_has_no_rules_sends_the_pinned_text(tmp_path: Path) -> None:
    registry = _registry()
    outputs = await _outputs_of(_loop(tmp_path, registry, None), streaming=False)
    assert outputs == [_text(registry, "error-response"), _text(registry, "alpha-rule")]
