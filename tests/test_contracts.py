"""Tests for the M0 shared contracts: messages, usage, provider value objects, tools."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from zakcode.config import PermissionTier
from zakcode.messages import (
    Message,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
)
from zakcode.providers.base import Capabilities, LLMResult, ToolCall
from zakcode.tools.base import (
    ConcurrencyClass,
    Tool,
    ToolContext,
    ToolRegistry,
    ToolResult,
    ToolSpec,
)
from zakcode.usage import Usage, UsageTracker

# ── messages ─────────────────────────────────────────────────────────────────


def test_message_constructors_and_accessors() -> None:
    assert Message.user("hi").text == "hi"
    assert Message.system("sys").role == "system"
    msg = Message(
        role="assistant",
        blocks=[
            TextBlock(text="let me look"),
            ToolUseBlock(id="t1", name="read_file", input={"path": "a"}),
        ],
    )
    assert msg.text == "let me look"
    assert [b.name for b in msg.tool_uses] == ["read_file"]
    assert msg.tool_uses[0].input == {"path": "a"}


def test_message_discriminated_union_roundtrip() -> None:
    """A message with mixed block types survives JSON serialize/deserialize losslessly."""
    original = Message(
        role="assistant",
        blocks=[
            TextBlock(text="done"),
            ToolUseBlock(id="t1", name="bash", input={"command": "ls"}),
        ],
    )
    restored = Message.model_validate_json(original.model_dump_json())
    assert restored == original
    assert isinstance(restored.blocks[0], TextBlock)
    assert isinstance(restored.blocks[1], ToolUseBlock)


def test_tool_results_message() -> None:
    msg = Message.tool_results([ToolResultBlock(tool_use_id="t1", output="42")])
    assert msg.role == "tool"
    assert isinstance(msg.blocks[0], ToolResultBlock)
    assert msg.blocks[0].output == "42"


# ── usage ────────────────────────────────────────────────────────────────────


def test_usage_addition_and_tracker() -> None:
    a = Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15, cost_usd=0.01)
    b = Usage(prompt_tokens=3, completion_tokens=2, total_tokens=5, cost_usd=0.002)
    assert (a + b).total_tokens == 20
    tracker = UsageTracker()
    tracker.add(a)
    total = tracker.add(b)
    assert total.prompt_tokens == 13
    assert total.cost_usd == pytest.approx(0.012)


# ── provider value objects ───────────────────────────────────────────────────


def test_llmresult_has_tool_calls() -> None:
    assert not LLMResult(text="hello").has_tool_calls
    result = LLMResult(tool_calls=[ToolCall(id="1", name="glob", arguments={"pattern": "*.py"})])
    assert result.has_tool_calls
    assert result.tool_calls[0].arguments == {"pattern": "*.py"}


def test_capabilities_defaults() -> None:
    caps = Capabilities(context_window=128_000)
    assert caps.supports_tools is True
    assert caps.context_window == 128_000


# ── tools ────────────────────────────────────────────────────────────────────


def test_toolspec_to_openai_shape() -> None:
    spec = ToolSpec(
        name="read_file",
        description="Read a file",
        parameters={"type": "object", "properties": {"path": {"type": "string"}}},
        required_permission=PermissionTier.READ_ONLY,
        concurrency=ConcurrencyClass.READ_ONLY_SAFE,
    )
    wire = spec.to_openai()
    assert wire["type"] == "function"
    assert wire["function"]["name"] == "read_file"
    assert wire["function"]["parameters"]["properties"]["path"]["type"] == "string"


def test_tool_result_helpers() -> None:
    ok = ToolResult.ok("yay", data={"n": 1})
    assert not ok.is_error and ok.data == {"n": 1}
    err = ToolResult.error("boom")
    assert err.is_error and err.output == "boom"


class _EchoTool(Tool):
    spec = ToolSpec(name="echo", description="echo the message back")

    async def execute(self, args: dict, ctx: ToolContext) -> ToolResult:
        return ToolResult.ok(str(args.get("message", "")))


class _BoomTool(Tool):
    spec = ToolSpec(name="boom", description="always raises")

    async def execute(self, args: dict, ctx: ToolContext) -> ToolResult:
        raise RuntimeError("kaboom")


def _ctx() -> ToolContext:
    return ToolContext(workspace_root=Path.cwd())


async def test_registry_dispatch_and_aliases() -> None:
    reg = ToolRegistry()
    reg.register(_EchoTool(), aliases=["say"])
    assert reg.names() == ["echo"]
    assert reg.get("say") is reg.get("echo")
    result = await reg.execute("say", {"message": "hello"}, _ctx())
    assert result.output == "hello" and not result.is_error


async def test_registry_unknown_tool_is_error_not_exception() -> None:
    result = await ToolRegistry().execute("nope", {}, _ctx())
    assert result.is_error
    assert "unknown tool" in result.output


async def test_registry_wraps_handler_exceptions() -> None:
    reg = ToolRegistry()
    reg.register(_BoomTool())
    result = await reg.execute("boom", {}, _ctx())
    assert result.is_error
    assert "RuntimeError" in result.output and "kaboom" in result.output


def test_registry_rejects_duplicate_registration() -> None:
    reg = ToolRegistry()
    reg.register(_EchoTool())
    with pytest.raises(ValueError, match="already registered"):
        reg.register(_EchoTool())


def test_registry_definitions_filter() -> None:
    reg = ToolRegistry()
    reg.register(_EchoTool(), aliases=["say"])
    reg.register(_BoomTool())
    assert len(reg.definitions()) == 2
    only = reg.definitions(allowed=["say"])  # alias resolves to echo
    assert len(only) == 1 and only[0]["function"]["name"] == "echo"


# ── vendor-agnostic import ban ───────────────────────────────────────────────
# The boundary the in-repo zds-llm-provider package used to express as a package
# split (see ADR-0005) is now enforced HERE, as a contract test: the engine stays
# vendor-agnostic by construction, not by packaging.

_SRC_ROOT = Path(__file__).resolve().parents[1] / "src" / "zakcode"

#: The only modules allowed to import litellm: the litellm-backed provider and the
#: capability registry's guarded best-effort metadata lookup.
_LITELLM_ALLOWED = {
    _SRC_ROOT / "providers" / "litellm_provider.py",
    _SRC_ROOT / "providers" / "registry.py",
}

_VENDOR_IMPORT_RE = re.compile(
    r"^\s*(?:import|from)\s+(litellm|openai|anthropic|groq|mistralai|cohere)\b",
    re.MULTILINE,
)


def test_no_vendor_sdk_imports_outside_provider_layer() -> None:
    """litellm may be imported only by its two named provider modules; no other
    vendor SDK may be imported anywhere in the engine. Keeps the loop, tools, and
    session vendor-agnostic by construction (CLAUDE.md rule 2 / GUARDRAILS)."""
    violations: list[str] = []
    for path in sorted(_SRC_ROOT.rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        for match in _VENDOR_IMPORT_RE.finditer(text):
            if match.group(1) == "litellm" and path in _LITELLM_ALLOWED:
                continue
            rel = path.relative_to(_SRC_ROOT)
            violations.append(f"{rel}: {match.group(0).strip()}")
    assert not violations, "vendor SDK imported outside the provider layer:\n" + "\n".join(
        violations
    )
