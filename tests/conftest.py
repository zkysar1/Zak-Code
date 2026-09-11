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
import re
from pathlib import Path
from typing import Any

import pytest

from zakcode.messages import Message
from zakcode.providers.base import Capabilities, LLMResult, Provider


@pytest.fixture(autouse=True)
def _isolated_config_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Give every test its own config home, so the suite cannot read the machine's.

    ``config.config_home()`` resolves ``~/.zakcode`` unless ``ZAKCODE_HOME`` is set --
    an override that exists, in its own words, for "tests / portable installs".
    Nothing was setting it, so the suite inherited whatever Zak Code config the box
    running it happened to have: on a machine where Zak Code is INSTALLED the tests
    read its real settings, skills and endpoints. Measured on such a box (zc-03, same
    tree and commit): **154 failed / 3467 passed** with the ambient config against
    **0 failed / 3621 passed** with this override -- 154 failures that are purely the
    developer's own installation, in the suite they would run to check their change.

    Per-test rather than per-session: a shared home would let one test's writes reach
    the next. A test that needs a particular home still sets its own; this runs first.
    """
    monkeypatch.setenv("ZAKCODE_HOME", str(tmp_path / ".zakcode"))


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


#: Packages that ONLY the optional ``server`` extra installs. ``httpx`` is
#: deliberately absent: it ships with the ``web`` extra, so its presence says
#: nothing about whether the server extra is installed.
_SERVER_EXTRA_IMPORT = re.compile(
    r"^(?:import|from)\s+(?:fastapi|starlette|sse_starlette|uvicorn|websockets)\b",
    re.MULTILINE,
)


def _modules_needing_the_server_extra() -> list[str]:
    """Test modules that import a ``server``-extra package at MODULE scope.

    DERIVED, never hand-listed. The hand-listed form went stale in both
    directions: it named four modules, two of which no longer exist, while
    seventeen import the extra -- so the plain ``uv sync && pytest`` this block
    exists to protect died with collection errors in the fourteen it missed.
    A filename glob would not fix it either, because one of the seventeen
    (``test_sdk_iface_parity.py``) is not named ``test_server_*``; the import is
    the only reliable signal.

    The line anchor is load-bearing: an import nested inside a function or a
    fixture is lazy and cannot break *collection*, which is the only failure
    this guards against.
    """
    here = Path(__file__).parent
    return sorted(
        path.name
        for path in here.glob("test_*.py")
        if _SERVER_EXTRA_IMPORT.search(path.read_text(encoding="utf-8", errors="replace"))
    )


collect_ignore: list[str] = []
if importlib.util.find_spec("fastapi") is None:
    collect_ignore += _modules_needing_the_server_extra()
