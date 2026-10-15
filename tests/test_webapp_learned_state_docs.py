"""Under ``zakcode webapp`` what a provider learns from a refusal lasts one turn, and the docs say
so (ADR-0317).

The provider learns from a route's refusals (ADR-0181, ADR-0188, ADR-0296, ADR-0303): the switch
to the text tool protocol, ``reasoning_effort`` set to ``none`` for a call with tools, a dropped
``temperature`` or ``stop``, a stop length limit, a field refused by name. ``docs/CONFIG.md``
said these hold "for the rest of the session", which is true of the
CLI. The web server builds a new agent, and so a new provider, for every turn: each
``POST /chat`` and ``POST /chat/stream``, each say a served run takes, and each ``POST /complete``
builds its own provider. So the next turn learns the same refusal again, one refused request each.

The first test reads the server section of ``docs/CONFIG.md``. The others hold that section to
the code: if the server ever keeps a provider across turns, they fail and the docs must change.
Hermetic: no network, no credentials, no user home.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from zakcode.config import Settings
from zakcode.providers.litellm_provider import LiteLLMProvider
from zakcode.providers.text_tools import TextToolCallingProvider
from zakcode.server.app import _default_agent_factory, _default_provider_factory, create_app
from zakcode.session.store import Session, SessionStore

_CONFIG = Path(__file__).resolve().parent.parent / "docs" / "CONFIG.md"


def _server_section() -> str:
    text = _CONFIG.read_text(encoding="utf-8")
    start = text.index("## HTTP server (`zakcode webapp`)")
    end = text.find("\n## ", start + 1)
    return text[start:] if end == -1 else text[start:end]


def _paragraph() -> str:
    section = _server_section()
    start = section.index("**Learned refusals last one turn here.**")
    end = section.find("\n\n", start)
    return section[start:] if end == -1 else section[start:end]


def test_the_server_section_says_learned_refusals_last_one_turn() -> None:
    paragraph = " ".join(_paragraph().split())

    assert "new provider, for each turn" in paragraph
    assert "the CLI" in paragraph
    for adr in ("ADR-0181", "ADR-0188", "ADR-0296", "ADR-0303"):
        assert adr in paragraph
    # The setting it names for skipping the tool-use refusal is a setting.
    assert "`tool_calling_mode` set to `text`" in paragraph
    assert "tool_calling_mode" in Settings.model_fields


def test_the_routes_the_paragraph_names_are_the_servers(tmp_path: Path) -> None:
    named = set(re.findall(r"`POST (/[a-z/]+)`", _paragraph()))
    app = create_app(
        settings=Settings(
            default_model="scripted/test", context_window=8192, workspace_root=tmp_path
        ),
        store=SessionStore(base_dir=tmp_path / "sessions"),
    )
    routes = {getattr(route, "path", None) for route in app.routes}

    assert named == {"/chat", "/chat/stream", "/complete"}
    assert named <= routes


@pytest.fixture
def settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Settings:
    monkeypatch.setattr(Path, "home", lambda: tmp_path / "home")  # ignore the real user home
    return Settings(
        default_model="openai/route-a",
        api_base="https://gateway.invalid/v1",
        api_key="sk-test",
        context_window=131_072,
        workspace_root=tmp_path,
    )


def test_each_turn_of_a_session_gets_a_provider_that_learned_nothing(
    settings: Settings, tmp_path: Path
) -> None:
    factory = _default_agent_factory(settings, SessionStore(base_dir=tmp_path / "sessions"))
    session = Session(cwd=str(tmp_path), model="openai/route-a")

    first = factory(session, None, None).provider
    assert isinstance(first, TextToolCallingProvider)
    assert isinstance(first.inner, LiteLLMProvider)
    # What the first turn's provider learned from three refusals.
    first._tools_refused = True
    first.inner.rejected_request_fields += ["temperature", "stop"]
    first.inner.max_stop_chars = 10

    second = factory(session, None, None).provider
    assert isinstance(second, TextToolCallingProvider)
    assert isinstance(second.inner, LiteLLMProvider)

    assert second is not first
    assert second._tools_refused is False
    assert second.inner.rejected_request_fields == []
    assert second.inner.max_stop_chars is None


def test_each_complete_request_gets_its_own_provider(settings: Settings) -> None:
    factory = _default_provider_factory(settings)

    first = factory(None)
    assert isinstance(first, LiteLLMProvider)
    first.rejected_request_fields.append("temperature")

    second = factory(None)
    assert isinstance(second, LiteLLMProvider)

    assert second is not first
    assert second.rejected_request_fields == []
