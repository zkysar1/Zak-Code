"""Shared test fixtures for zds-llm-provider."""
from __future__ import annotations

from typing import Any

from zds_llm_provider.messages import Message
from zds_llm_provider.types import Capabilities, LLMResult, Provider


class StubProvider(Provider):
    """Minimal concrete Provider for testing the ABC contract."""

    def __init__(
        self,
        result: LLMResult | None = None,
        caps: Capabilities | None = None,
    ) -> None:
        self._result = result or LLMResult(text="stub response")
        self._caps = caps or Capabilities()

    async def acomplete(
        self,
        messages: list[Message],
        *,
        system: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> LLMResult:
        return self._result

    def count_tokens(self, messages: list[Message], *, system: str | None = None) -> int:
        total = sum(len(m.text) for m in messages)
        if system:
            total += len(system)
        return total // 4

    def capabilities(self) -> Capabilities:
        return self._caps
