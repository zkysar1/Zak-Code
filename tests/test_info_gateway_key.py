"""``zakcode info`` says where the gateway key came from (ADR-0299).

The panel listed ``OPENAI_API_KEY``, ``ANTHROPIC_API_KEY`` and ``TAVILY_API_KEY``, and on a
machine that reaches its models through an OpenAI-compatible gateway all three read ``not
set``. The key such a machine actually uses is ``ZAKCODE_API_KEY``, which the panel never
named, so there was no way to see whether it had resolved, or from which file. The row reports
presence and provenance only. The value is never printed.

The last four tests pin the premise behind the word ``empty``: what litellm puts on the wire for
a blank key.

Hermetic: a throwaway config home and working directory hold the ``.env`` files, and every
``ZAKCODE_*`` variable is cleared first. The wire is a stubbed transport, so no request leaves
the process, and nothing is read from the real ``~/.zakcode``.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import httpx
import pytest
from typer.testing import CliRunner

import zakcode.config as cfg
from zakcode.cli import app, build_info_lines
from zakcode.config import Settings, load_settings
from zakcode.messages import Message
from zakcode.providers.base import ProviderUnavailable
from zakcode.providers.litellm_provider import LiteLLMProvider

LABEL = "ZAKCODE_API_KEY"
#: Shaped like a real key and obviously not one. It only has to be findable in any output.
SECRET = "sk-test-gateway-key-0123456789abcdef"
OTHER_SECRET = "sk-test-the-other-file-fedcba9876543210"


@pytest.fixture(autouse=True)
def _isolated(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A clean environment, an empty config home and an empty working directory."""
    for name in [n for n in os.environ if n.startswith("ZAKCODE_")]:
        monkeypatch.delenv(name, raising=False)
    home = tmp_path / "confighome"
    home.mkdir()
    work = tmp_path / "work"
    work.mkdir()
    monkeypatch.setenv("ZAKCODE_HOME", str(home))
    monkeypatch.chdir(work)
    # ``env_source`` keeps process-global provenance; one test's load must not leak into the next.
    monkeypatch.setattr(cfg, "_ENV_SOURCES", {})
    monkeypatch.setattr(cfg, "_DOTENV_EXPORTED", {})


def _write_env(directory: Path, text: str) -> None:
    (directory / ".env").write_text(text, encoding="utf-8")


def _row() -> str:
    return dict(build_info_lines(load_settings()))[LABEL]


def test_the_panel_names_the_gateway_key_when_it_is_missing() -> None:
    assert _row() == "not set"


def test_a_key_in_the_environment_is_reported_as_the_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(LABEL, SECRET)

    assert _row() == "set (env)"


def test_a_key_in_the_user_home_is_reported_as_the_user_file(tmp_path: Path) -> None:
    _write_env(tmp_path / "confighome", f"{LABEL}={SECRET}\n")

    assert _row() == "set (user .env)"


def test_a_key_in_the_working_directory_is_reported_as_the_workspace_file(
    tmp_path: Path,
) -> None:
    _write_env(tmp_path / "work", f"{LABEL}={SECRET}\n")

    assert _row() == "set (workspace .env)"


def test_the_row_names_the_file_that_won_when_two_files_hold_a_key(tmp_path: Path) -> None:
    """The workspace file shadows the user one: the row says which value is in force."""
    _write_env(tmp_path / "confighome", f"{LABEL}={OTHER_SECRET}\n")
    _write_env(tmp_path / "work", f"{LABEL}={SECRET}\n")

    assert _row() == "set (workspace .env)"


def test_the_real_environment_outranks_both_files(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _write_env(tmp_path / "confighome", f"{LABEL}={OTHER_SECRET}\n")
    _write_env(tmp_path / "work", f"{LABEL}={OTHER_SECRET}\n")
    monkeypatch.setenv(LABEL, SECRET)

    assert _row() == "set (env)"


@pytest.mark.parametrize("written", ["", '"   "'], ids=["nothing", "spaces"])
def test_a_blank_key_is_not_a_resolved_key(tmp_path: Path, written: str) -> None:
    """``ZAKCODE_API_KEY=`` left empty in a file reads as set to a presence check and is still
    not a key: the premises at the end show what goes on the wire, so the panel must not call
    it one."""
    _write_env(tmp_path / "confighome", f"{LABEL}={written}\n")

    assert _row() == "empty (user .env)"


def test_a_key_given_to_settings_directly_has_no_file_behind_it() -> None:
    """The SDK and the tests build ``Settings`` themselves. The key is there, and there is
    nothing to name as its source, which must not read ``set (not set)``."""
    rows = dict(build_info_lines(Settings(api_key=SECRET)))

    assert rows[LABEL] == "set (in process)"
    assert SECRET not in rows[LABEL]


def test_the_three_cloud_keys_are_still_listed_beside_it() -> None:
    rows = dict(build_info_lines(load_settings()))

    assert {"OPENAI_API_KEY", "ANTHROPIC_API_KEY", "TAVILY_API_KEY"} <= set(rows)
    assert LABEL in rows


def test_no_row_shows_the_value(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv(LABEL, SECRET)
    _write_env(tmp_path / "confighome", f"OPENAI_API_KEY={OTHER_SECRET}\n")

    rows = build_info_lines(load_settings())

    assert [value for label, value in rows if label == LABEL] == ["set (env)"]
    assert all(SECRET not in label and SECRET not in value for label, value in rows)
    assert all(OTHER_SECRET not in label and OTHER_SECRET not in value for label, value in rows)


def test_the_printed_panel_names_the_key_and_never_shows_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The rendered panel, not only the rows: the annotation must survive rich's markup and
    the value must not appear in anything the command prints."""
    monkeypatch.setenv(LABEL, SECRET)

    result = CliRunner().invoke(app, ["info"])

    assert result.exit_code == 0
    assert LABEL in result.output
    assert "set (env)" in result.output
    assert SECRET not in result.output


# --- the premise behind the word ``empty``: a blank key is not the key a gateway receives -------


@pytest.fixture
def authorization(monkeypatch: pytest.MonkeyPatch) -> list[str | None]:
    """The Authorization header of every request litellm's own client sends, answered from a
    canned body. Nothing leaves the process."""
    seen: list[str | None] = []

    async def send(self: httpx.AsyncClient, request: httpx.Request, **kw: Any) -> httpx.Response:
        seen.append(request.headers.get("authorization"))
        answer = {
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "created": 1,
            "model": "alias",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "ready"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }
        return httpx.Response(200, json=answer, request=request)

    monkeypatch.setattr(httpx.AsyncClient, "send", send)
    return seen


async def _ask_through_a_gateway(api_key: str | None) -> None:
    provider = LiteLLMProvider(
        Settings(
            default_model="openai/alias",
            api_base="https://gateway.invalid/v1",
            api_key=api_key,
            context_window=131_072,
            _env_file=None,
        )
    )
    await provider.acomplete([Message.user("hi")], system="s")


async def test_premise_the_configured_key_is_what_the_gateway_receives(
    monkeypatch: pytest.MonkeyPatch, authorization: list[str | None]
) -> None:
    """The control: a real key wins over ``OPENAI_API_KEY``, so the tests below show a
    difference and not a route that ignores keys."""
    monkeypatch.setenv("OPENAI_API_KEY", OTHER_SECRET)

    await _ask_through_a_gateway(SECRET)

    assert authorization == [f"Bearer {SECRET}"]


async def test_premise_an_empty_key_is_read_as_no_key_and_the_environment_one_is_sent(
    monkeypatch: pytest.MonkeyPatch, authorization: list[str | None]
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", OTHER_SECRET)

    await _ask_through_a_gateway("")

    assert authorization == [f"Bearer {OTHER_SECRET}"]


async def test_premise_an_empty_key_with_no_other_key_sends_no_request(
    monkeypatch: pytest.MonkeyPatch, authorization: list[str | None]
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(ProviderUnavailable):
        await _ask_through_a_gateway("")

    assert authorization == []


async def test_premise_a_key_of_spaces_goes_out_as_a_blank_credential(
    monkeypatch: pytest.MonkeyPatch, authorization: list[str | None]
) -> None:
    """Unlike the empty string, spaces are not read as no key, so the environment key is not
    sent in their place: the gateway receives a blank credential."""
    monkeypatch.setenv("OPENAI_API_KEY", OTHER_SECRET)

    await _ask_through_a_gateway("   ")

    assert len(authorization) == 1
    assert authorization[0] is not None
    assert authorization[0].startswith("Bearer")
    assert authorization[0].removeprefix("Bearer").strip() == ""
