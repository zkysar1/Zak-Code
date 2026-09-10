"""ADR-0116: credential-shaped tokens are scrubbed from tool output at the execution seam.

A tool output is the one string that reaches the model's context, the CLI transcript, AND
the on-disk session file at once, so a token printed there (``gcloud auth
print-access-token`` echoed by a model, 2026-09-08) leaks three ways. The seam masks the
provider-prefixed token SHAPES and tells the model the discipline in a rail.
"""

from __future__ import annotations

from typing import Any

import pytest

from zakcode.agent.loop import AgentLoop
from zakcode.config import PermissionTier
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
from zakcode.usage import Usage

_TOKEN = "ya29.c.c0AZ4bNp" + "Q" * 80


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
        return Capabilities(context_window=8192)


class _LeakyTool(Tool):
    spec = ToolSpec(
        name="leaky",
        description="prints a token",
        required_permission=PermissionTier.READ_ONLY,
        concurrency=ConcurrencyClass.READ_ONLY_SAFE,
    )

    async def execute(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        return ToolResult.ok(f"Obtained token:\n{_TOKEN}\nnow use it")


@pytest.mark.asyncio
async def test_token_in_tool_output_is_masked_before_model_transcript_and_session() -> None:
    provider = _Scripted(
        [
            LLMResult(
                text="",
                tool_calls=[ToolCall(id="t1", name="leaky", arguments={})],
                usage=Usage(total_tokens=1),
            ),
            LLMResult(text="done", tool_calls=[], usage=Usage(total_tokens=1)),
        ]
    )
    registry = ToolRegistry()
    registry.register(_LeakyTool())
    session = Session(cwd="/tmp", model="t/m")
    loop = AgentLoop(provider, registry, session, max_iterations=5)
    result = await loop.arun_turn("get me a token")

    assert result.stop_reason == "completed"
    block = result.tool_results[0]
    assert _TOKEN not in block.output
    assert "[REDACTED]" in block.output and "now use it" in block.output
    assert "never print a secret" in block.output  # the rail names the discipline
    assert "{{secret:NAME}}" in block.output  # the placeholder syntax survives str.format
    # Nothing persisted carries the value either.
    assert all(_TOKEN not in m.text for m in session.messages)
    kinds = (
        [(n.kind, n.detail) for n in result.trace.notes] if hasattr(result.trace, "notes") else []
    )
    assert not kinds or any(k == "redaction" for k, _ in kinds)


_OAUTH = "wDIrZA" + "kQ9x" * 56
_CRED_FILE = (
    '{"access_token": "' + _OAUTH + '", "refresh_token": "AEFMlG7pZq7pZq7pZq7pZq", '
    '"token_type": "bearer", "expires_at": 1756570800}'
)


class _CatTool(Tool):
    spec = ToolSpec(
        name="cat_token_file",
        description="reads a credential file verbatim",
        required_permission=PermissionTier.READ_ONLY,
        concurrency=ConcurrencyClass.READ_ONLY_SAFE,
    )

    async def execute(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        return ToolResult.ok(_CRED_FILE)


@pytest.mark.asyncio
async def test_a_credential_file_read_verbatim_is_scrubbed_at_the_seam() -> None:
    """ADR-0125. Measured on the coach rig 2026-09-10: a skill told the model to check the
    token file, the model ran ``cat .yahoo_token.json``, and a 230-char OAuth token reached
    the model, the CLI log and the session store — the seam knew only provider prefixes.
    Weeks of earlier logs held the same value once we looked."""
    provider = _Scripted(
        [
            LLMResult(
                text="",
                tool_calls=[ToolCall(id="c1", name="cat_token_file", arguments={})],
                usage=Usage(total_tokens=1),
            ),
            LLMResult(text="done", tool_calls=[], usage=Usage(total_tokens=1)),
        ]
    )
    registry = ToolRegistry()
    registry.register(_CatTool())
    session = Session(cwd="/tmp", model="t/m")
    loop = AgentLoop(provider, registry, session, max_iterations=5)
    result = await loop.arun_turn("check the token file")

    block = result.tool_results[0]
    assert _OAUTH not in block.output and "AEFMlG" not in block.output
    assert '"access_token": "[REDACTED]"' in block.output  # structure kept, value gone
    assert '"token_type": "bearer"' in block.output  # the model can still read the file
    assert "2 credential-shaped value(s) were redacted" in block.output
    assert all(_OAUTH not in m.text for m in session.messages)
