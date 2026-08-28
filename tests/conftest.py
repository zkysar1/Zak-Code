"""Shared pytest configuration and fixtures.

The FastAPI server is an **optional** extra (``zakcode[server]``). Its test modules
import ``fastapi`` / ``httpx`` at module scope, so when the extra is not installed
(a plain ``uv sync && pytest``, or a ``uv run pytest`` that did not carry
``--extra server``) they would fail at *collection* with ``ModuleNotFoundError``
rather than skipping. Ignore those modules when the extra is absent — they run
normally whenever it is present (CI installs it; see ``.github/workflows/ci.yml``).
"""

from __future__ import annotations

import importlib.util
from typing import Any

from zakcode.messages import Message
from zakcode.providers.base import Capabilities, LLMResult, Provider


class StubProvider(Provider):
    """Minimal concrete Provider for testing the ABC contract."""

    def __init__(
        self,
        result: LLMResult | None = None,
        caps: Capabilities | None = None,
    ) -> None:
        self._result = result or LLMResult(text="stub response")
        self._caps = caps or Capabilities(context_window=8192)

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


#: Server test modules that import the optional ``server`` extra at module scope.
_SERVER_TEST_MODULES = [
    "test_server_app.py",
    "test_server_client.py",
    "test_server_webclient.py",
    "test_server_ws.py",
]

collect_ignore: list[str] = []
if importlib.util.find_spec("fastapi") is None:
    collect_ignore += _SERVER_TEST_MODULES
