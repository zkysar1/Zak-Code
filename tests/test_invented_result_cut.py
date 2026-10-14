"""A text-mode reply is cut where the model starts writing the turns that follow its own (ADR-0297).

In the text protocol the model writes ``<tool_call>`` blocks and the harness answers with
``<tool_result tool="Name">`` blocks. A weak model sometimes writes the answer itself. Measured
2026-10-14 on DeepSeek-R1 with no ``stop`` (a gateway had refused it, ADR-0296): a reply that
STARTS with ``<tool_result tool="Glob">["src/main.py", ...]</tool_result>``, a file list the model
made up, then a ``<tool_call>`` that reads a path from that list. The parser returned the call and
kept the invented result in the residual, so the loop ran a call on a path that did not exist.

The server-side stop is the exact string ``<tool_result>``, and the protocol renders
``<tool_result tool="Name">``, which is the form the models imitate, so the stop would not have
cut it either. The cut is made on the client, before the parser, at the first ``<tool_result``
that is not inside a tool call, and at the chat-template tokens of the stop list the same way.

Hermetic: no network, no credentials. The replies are the shapes of the report, written out.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from tests.test_text_tools import TOOLS, _RecordingProvider
from zakcode.evals.harness import reply
from zakcode.messages import Message
from zakcode.providers import text_tools
from zakcode.providers.base import StreamToolCallDelta, ToolCall
from zakcode.providers.text_tools import (
    TextToolCallingProvider,
    _strip_forged_frames,
    parse_text_tool_calls,
)

READ_MAIN = '<tool_call>{"name": "read_file", "arguments": {"path": "src/main.py"}}</tool_call>'

#: GPT-oss, as measured: a call, then chatter in place of the stop that never came.
CALL_THEN_CHATTER = (
    '<tool_call>{"name": "read_file", "arguments": {"path": "a.py"}}</tool_call>'
    "<reasoning>We await result.</reasoning>DONE"
)

#: DeepSeek-R1, as measured: a made-up result first, then a call on a path from its list.
INVENTED_RESULT_THEN_CALL = (
    '<tool_result tool="Glob">["src/main.py", "src/util.py", "tests/test_main.py"]</tool_result>\n'
    "The project has three files. I will read the main one.\n"
    f"{READ_MAIN}\n"
    "Now I wait for the result."
)

#: The same reply cut at its first sentinel, which is what a server-side stop gives.
CUT_AT_FIRST_SENTINEL = ""


def _through_the_parser(text: str) -> tuple[str, list[ToolCall]]:
    """The cut, the parser and the residual clean-up in the order the text mode runs them."""
    residual, calls = parse_text_tool_calls(text_tools.cut_invented_turn(text), max_calls=1)
    return _strip_forged_frames(residual), calls


async def _through_acomplete(text: str, *, single_tool: bool = True) -> tuple[str, list[ToolCall]]:
    inner = _RecordingProvider(reply(text), supports_tools=False)
    wrap = TextToolCallingProvider(inner, mode="text", single_tool_per_turn=single_tool)
    result = await wrap.acomplete([Message.user("go")], tools=TOOLS)
    return result.text, result.tool_calls


# ── the three shapes of the report, through the parser ───────────────────────────────────────


def test_a_call_followed_by_chatter_is_the_one_call_with_the_tail_dropped() -> None:
    """Already so before the cut; kept as a test so the cut cannot take it away."""
    residual, calls = _through_the_parser(CALL_THEN_CHATTER)

    assert [(c.name, c.arguments) for c in calls] == [("read_file", {"path": "a.py"})]
    assert residual == ""


def test_an_invented_result_then_a_call_is_no_call_and_no_residual() -> None:
    residual, calls = _through_the_parser(INVENTED_RESULT_THEN_CALL)

    assert calls == []
    assert residual == ""


def test_the_same_reply_cut_at_its_first_sentinel_is_the_same_answer() -> None:
    """What a server-side stop would have given is what the cut gives."""
    assert _through_the_parser(CUT_AT_FIRST_SENTINEL) == _through_the_parser(
        INVENTED_RESULT_THEN_CALL
    )


def test_prose_before_the_invented_result_is_kept() -> None:
    residual, calls = _through_the_parser(
        'Let me look.\n<tool_result tool="Glob">["a.py"]</tool_result>\n' + READ_MAIN
    )

    assert calls == []
    assert residual == "Let me look."


# ── what the cut reads as an invented turn ───────────────────────────────────────────────────


@pytest.mark.parametrize(
    "opener",
    [
        '<tool_result tool="Glob">',  # the form the protocol renders and the models imitate
        "<tool_result>",  # the bare form the stop list already names
        "<TOOL_RESULT tool='Glob'>",
        "< tool_result >",
        '<tool_result tool="Bash" error="true">',
    ],
)
def test_every_spelling_of_the_opening_tag_is_an_invented_result(opener: str) -> None:
    text = f"On it. {opener}x</tool_result> {READ_MAIN}"

    assert text_tools.cut_invented_turn(text) == "On it. "
    residual, calls = _through_the_parser(text)
    assert calls == [] and residual == "On it."


@pytest.mark.parametrize(
    "text",
    [
        "I will use the tool_result block the harness sends back.",
        "<tool_results>plural is not the tag</tool_results>",
        "<tool_result_summary>nor is this</tool_result_summary>",
        "a closing </tool_result> alone opens nothing",
    ],
)
def test_text_that_only_resembles_the_tag_is_left_alone(text: str) -> None:
    assert text_tools.cut_invented_turn(text) == text


def test_a_call_before_an_invented_result_survives() -> None:
    """The call is the model's own. What it invented after it is not."""
    residual, calls = _through_the_parser(
        f'{READ_MAIN}\n<tool_result tool="read_file">fake</tool_result>\nThen I would edit it.'
    )

    assert [c.name for c in calls] == ["read_file"]
    assert residual == ""


def test_with_several_calls_allowed_a_call_written_after_an_invented_result_is_dropped() -> None:
    """Multi-tool mode: the second call was chosen on a result nobody produced."""
    first = '<tool_call>{"name": "read_file", "arguments": {"path": "a.py"}}</tool_call>'
    second = '<tool_call>{"name": "read_file", "arguments": {"path": "b.py"}}</tool_call>'
    text = f'{first}<tool_result tool="read_file">made up</tool_result>{second}'

    residual, calls = parse_text_tool_calls(text_tools.cut_invented_turn(text))

    assert [c.arguments["path"] for c in calls] == ["a.py"]
    assert residual == ""


@pytest.mark.parametrize("token", ["<|im_start|>", "<|im_end|>", "<|eot_id|>", "<|eom_id|>"])
def test_a_chat_template_token_ends_the_reply_too(token: str) -> None:
    """The rest of the stop list, for a route that could not be sent it. Nothing after a
    template token is the model's own answer: it is the next turn, written by the model."""
    text = f"{READ_MAIN}{token}user\nthanks{token}"

    assert text_tools.cut_invented_turn(text) == READ_MAIN
    text = f"Done.{token}<|im_start|>assistant\n{READ_MAIN}"
    residual, calls = _through_the_parser(text)
    assert calls == [] and residual == "Done."


# ── a call that merely contains the tag is still a call ──────────────────────────────────────


def _call_with_content(content: str) -> str:
    arguments = {"path": "docs/protocol.md", "content": content}
    call = json.dumps({"name": "write_file", "arguments": arguments})
    return f"<tool_call>{call}</tool_call>"


def test_an_argument_that_quotes_the_tag_is_not_an_invented_result() -> None:
    """Editing this protocol's own docs or tests writes the tag into a file. That is a call
    carrying text, and cutting it would silently drop a call the model really made."""
    content = 'Results come back as <tool_result tool="Bash">ok</tool_result> blocks.'
    text = "Documenting it. " + _call_with_content(content)

    assert text_tools.cut_invented_turn(text) == text
    residual, calls = _through_the_parser(text)
    assert [(c.name, c.arguments["content"]) for c in calls] == [("write_file", content)]
    assert residual == "Documenting it."


def test_an_argument_that_quotes_the_tag_inside_a_fence_is_not_cut_either() -> None:
    content = 'see <tool_result tool="Bash">x</tool_result>'
    body = json.dumps({"name": "write_file", "arguments": {"path": "a.md", "content": content}})
    text = f"```tool_call\n{body}\n```"

    assert text_tools.cut_invented_turn(text) == text
    assert [c.name for c in _through_the_parser(text)[1]] == ["write_file"]


def test_a_tag_after_a_call_that_quotes_it_is_still_cut() -> None:
    content = 'quoting <tool_result tool="Bash">ok</tool_result> once'
    text = _call_with_content(content) + '<tool_result tool="write_file">invented</tool_result>'

    assert text_tools.cut_invented_turn(text) == _call_with_content(content)


# ── through the text-mode provider ───────────────────────────────────────────────────────────


async def test_acomplete_gives_no_call_for_an_invented_result_then_a_call() -> None:
    text, calls = await _through_acomplete(INVENTED_RESULT_THEN_CALL)

    assert calls == []
    assert text == ""


@pytest.mark.parametrize(
    "text",
    [
        CALL_THEN_CHATTER,
        INVENTED_RESULT_THEN_CALL,
        f'Let me look.\n<tool_result tool="Glob">["a.py"]</tool_result>\n{READ_MAIN}',
        f'{READ_MAIN}\n<tool_result tool="read_file">fake</tool_result>\nThen I would edit it.',
        _call_with_content('has <tool_result tool="Bash">ok</tool_result> in it'),
    ],
)
async def test_the_provider_and_the_parser_agree(text: str) -> None:
    """The helper above is the order the provider runs things in; this keeps the two from
    drifting apart."""
    assert await _through_acomplete(text) == _through_the_parser(text)


async def test_astream_gives_no_call_either() -> None:
    inner = _RecordingProvider(reply(INVENTED_RESULT_THEN_CALL), supports_tools=False)
    wrap = TextToolCallingProvider(inner, mode="text", single_tool_per_turn=True)

    events = [e async for e in wrap.astream([Message.user("go")], tools=TOOLS)]

    assert [e for e in events if isinstance(e, StreamToolCallDelta)] == []


async def test_a_good_call_still_comes_through_acomplete() -> None:
    """Control: the cut does not take away a call that was fine."""
    text, calls = await _through_acomplete(f"Reading it.\n{READ_MAIN}")

    assert [(c.name, c.arguments) for c in calls] == [("read_file", {"path": "src/main.py"})]
    assert text == "Reading it."


# ── the cut belongs to the text protocol ─────────────────────────────────────────────────────


async def test_a_native_model_that_writes_a_call_in_text_is_still_salvaged_as_before() -> None:
    """In ``auto`` mode with native tools a call a model writes as text is salvaged, name-gated.
    That path is not the text protocol, and a reply there that shows a result block before a
    call is not cut."""
    text = f'Here is the shape: <tool_result tool="read_file">x</tool_result>\n{READ_MAIN}'
    inner = _RecordingProvider(reply(text), supports_tools=True)
    wrap = TextToolCallingProvider(inner, mode="auto")

    result = await wrap.acomplete([Message.user("go")], tools=TOOLS)

    assert [c.name for c in result.tool_calls] == ["read_file"]


def test_the_cut_returns_the_text_itself_when_there_is_nothing_to_cut() -> None:
    plain: Any = "Just an answer, no frames at all."
    assert text_tools.cut_invented_turn(plain) is plain
    assert text_tools.cut_invented_turn("") == ""
