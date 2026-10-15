"""A display-only name for the provider token (ADR-0314).

A model id is ``<provider>/<model>``, and through an OpenAI-compatible gateway the provider token
is the protocol, not the vendor: ``openai/route-a`` behind a LiteLLM 1.9x gateway in front of
Bedrock/Vertex sends nothing to OpenAI, yet "openai" is what every line a person reads said. The
``provider_labels`` setting (``ZAKCODE_PROVIDER_LABELS``, a JSON object read the way
``ZAKCODE_EXTRA_BODY`` is) maps a token to the name a person should read: ``{"openai":
"Gateway"}`` here.

Each display site is checked with the setting and without it, and without it the text is
today's. What a machine reads keeps the token: the model string on the wire, the ``source`` of a
zakpick entry, the session record and the log file.

Hermetic: no network, no credentials. Every ``ZAKCODE_*`` variable is cleared, the config home
is a temporary folder and the working directory an empty one.
"""

from __future__ import annotations

import ast
import asyncio
import json
import logging
import os
import re
from collections.abc import AsyncIterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from typer.testing import CliRunner

import zakcode
import zakcode.config as cfg
import zakcode.providers.litellm_provider as lp
from zakcode.agent.loop import AgentLoop
from zakcode.cli import app, build_info_lines
from zakcode.cli.logsink import log_file_path, transcript_line
from zakcode.config import Settings, load_settings
from zakcode.events import AgentDone, AgentStatus, AgentTextDelta
from zakcode.hooks import HookManager
from zakcode.messages import Message
from zakcode.permissions import PermissionPolicy
from zakcode.providers.base import (
    Capabilities,
    LLMResult,
    Provider,
    ProviderStreamEvent,
    StreamDone,
    StreamTextDelta,
)
from zakcode.providers.litellm_provider import LiteLLMProvider
from zakcode.providers.routing import describe_zakpick
from zakcode.providers.text_tools import TextToolCallingProvider
from zakcode.session.store import Session
from zakcode.tools.base import ToolRegistry
from zakcode.usage import Usage

LABELS = {"openai": "Gateway"}
ROUTE = "openai/route-a"
OTHER = "openai/route-b"
GATEWAY = "https://gateway.invalid/v1"
WINDOW = 131_072

TEMPERATURE_SENTENCE = (
    "This model doesn't support the temperature field. Remove temperature and try again."
)
TOOL_USE_SENTENCE = "This model doesn't support tool use."
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a file.",
            "parameters": {"type": "object", "properties": {"path": {"type": "string"}}},
        },
    }
]
FELL_BACK = {
    "llm_provider-x-litellm-attempted-fallbacks": "1",
    "llm_provider-x-litellm-model-group": "route-b",
    "llm_provider-x-litellm-model-name": "vendor/route-b-1",
}


@pytest.fixture(autouse=True)
def _isolated(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """No ``ZAKCODE_*`` variable, an empty config home, an empty working directory."""
    for name in [n for n in os.environ if n.startswith("ZAKCODE_")]:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("ZAKCODE_HOME", str(tmp_path / "home"))
    work = tmp_path / "work"
    work.mkdir()
    monkeypatch.chdir(work)
    # ``env_source`` keeps process-global provenance; one test's load must not leak into the next.
    monkeypatch.setattr(cfg, "_ENV_SOURCES", {})
    monkeypatch.setattr(cfg, "_DOTENV_EXPORTED", {})
    # The provider spaces request starts a second apart; that sleep is not under test.
    monkeypatch.setattr(lp, "_MIN_REQUEST_INTERVAL_S", 0.0)


# ── the helper ───────────────────────────────────────────────────────────────────────────────


def test_a_label_replaces_the_leading_provider_token_of_a_model_id() -> None:
    from zakcode.providers.labels import display_model_id, provider_label

    assert display_model_id(ROUTE, LABELS) == "Gateway/route-a"
    assert provider_label("openai", LABELS) == "Gateway"


@pytest.mark.parametrize(
    "model_id",
    [
        "bedrock/converse/openai.route-c-1:0",  # the token is bedrock; openai is inside
        "openai-compatible/route-a",
        "notopenai/route-a",
        "route-a/openai",
        "openai",  # no provider token at all
    ],
)
def test_a_label_never_replaces_a_substring(model_id: str) -> None:
    from zakcode.providers.labels import display_model_id, provider_label

    assert display_model_id(model_id, LABELS) == model_id
    assert provider_label("openai-compatible", LABELS) == "openai-compatible"


def test_without_labels_every_name_is_shown_as_it_is() -> None:
    from zakcode.providers.labels import display_model_id, provider_label

    for labels in ({}, None):
        assert display_model_id(ROUTE, labels) == ROUTE
        assert provider_label("openai", labels) == "openai"


# ── the setting ──────────────────────────────────────────────────────────────────────────────


def test_the_setting_is_read_from_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ZAKCODE_PROVIDER_LABELS", json.dumps(LABELS))

    assert load_settings().provider_labels == LABELS


@pytest.mark.parametrize("raw", ["notjson", "[1, 2]"])
def test_a_malformed_value_fails_the_way_extra_body_does(
    monkeypatch: pytest.MonkeyPatch, raw: str
) -> None:
    monkeypatch.setenv("ZAKCODE_EXTRA_BODY", raw)
    with pytest.raises(ValueError) as body_error:
        load_settings()
    monkeypatch.delenv("ZAKCODE_EXTRA_BODY")
    monkeypatch.setenv("ZAKCODE_PROVIDER_LABELS", raw)

    with pytest.raises(ValueError) as labels_error:
        load_settings()

    assert type(labels_error.value) is type(body_error.value)
    assert "provider_labels" in str(labels_error.value)


def test_a_label_that_is_not_text_is_refused_naming_the_setting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ZAKCODE_PROVIDER_LABELS", '{"openai": 5}')

    with pytest.raises(ValueError, match="provider_labels"):
        load_settings()


# ── the info panel and /model ────────────────────────────────────────────────────────────────


def _zakpick(**kw: Any) -> Settings:
    return Settings(
        default_model="zakpick",
        workspace_root=".",
        zakpick_models={
            "quick_code": {"model": "route-a", "source": "openai", "context_window": WINDOW}
        },
        **kw,
    )


def _info(monkeypatch: pytest.MonkeyPatch, settings: Settings) -> dict[str, str]:
    # The context-window rows probe the server's model listing; not under test, never sent.
    monkeypatch.setattr("zakcode.cli._window_rows", lambda _settings: [])
    return dict(build_info_lines(settings))


def test_the_zakpick_summary_shows_the_label_for_the_source() -> None:
    out = describe_zakpick(_zakpick(provider_labels=LABELS))

    assert "route-a (Gateway)" in out
    assert "(openai)" not in out


def test_without_labels_the_zakpick_summary_is_unchanged() -> None:
    out = describe_zakpick(_zakpick())

    assert "route-a (openai)" in out
    assert "Gateway" not in out


def test_info_labels_the_model_row_and_shows_the_setting(monkeypatch: pytest.MonkeyPatch) -> None:
    rows = _info(monkeypatch, _zakpick(provider_labels=LABELS))
    assert "route-a (Gateway)" in rows["Model"]
    assert rows["Provider labels"] == '{"openai": "Gateway"} (display only)'

    plain = _info(
        monkeypatch, Settings(default_model=ROUTE, workspace_root=".", provider_labels=LABELS)
    )
    assert plain["Model"] == "Gateway/route-a"
    # The provider row is the configured token, read beside the row that names the labels.
    assert plain["Provider"] == "openai"


def test_info_labels_the_auto_pick(monkeypatch: pytest.MonkeyPatch) -> None:
    pick = SimpleNamespace(model=ROUTE, reason="first viable")
    resolver = SimpleNamespace(resolve=lambda **_kw: pick)
    monkeypatch.setattr("zakcode.providers.resolve.resolver_for", lambda *_a, **_kw: resolver)

    labelled = _info(
        monkeypatch, Settings(default_model="auto", workspace_root=".", provider_labels=LABELS)
    )
    plain = _info(monkeypatch, Settings(default_model="auto", workspace_root="."))

    assert labelled["Model"] == "auto → Gateway/route-a (first viable)"
    assert plain["Model"] == "auto → openai/route-a (first viable)"


def test_without_labels_info_is_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    rows = _info(monkeypatch, _zakpick())
    assert "route-a (openai)" in rows["Model"]
    assert "Provider labels" not in rows

    plain = _info(monkeypatch, Settings(default_model=ROUTE, workspace_root="."))
    assert plain["Model"] == ROUTE
    assert plain["Provider"] == "openai"


@pytest.mark.parametrize("probe", ["resolved", "failed"])
def test_info_labels_the_context_window_rows(monkeypatch: pytest.MonkeyPatch, probe: str) -> None:
    from zakcode.cli import _window_rows

    def _resolve(*_a: Any, **_kw: Any) -> Any:
        if probe == "failed":
            raise OSError("no listing")
        return SimpleNamespace(window=WINDOW, source="config", describe=lambda: "131,072 (config)")

    # The rows probe the server's model listing; a stand-in answers, nothing is sent.
    monkeypatch.setattr(lp, "resolve_context_window", _resolve)
    tail = "131,072 (config)" if probe == "resolved" else "(could not resolve)"

    labelled = _window_rows(
        Settings(default_model=ROUTE, workspace_root=".", provider_labels=LABELS)
    )
    plain = _window_rows(Settings(default_model=ROUTE, workspace_root="."))

    assert [text for _, text in labelled] == [f"Gateway/route-a: {tail}"]
    assert [text for _, text in plain] == [f"{ROUTE}: {tail}"]


# ── the chat: the banner, /model, /cost and the grey call row ────────────────────────────────


class _FakeAgent:
    """Stands in for ``zakcode.Agent`` in the REPL: two models' usage on the session, and a turn
    that logs litellm's per-call line, the record the grey call row is made from."""

    def __init__(self, **_overrides: object) -> None:
        self.settings = load_settings()
        self.session = Session(cwd=".", model=self.settings.default_model)
        for model in (ROUTE, OTHER):
            self.session.add_usage(
                Usage(prompt_tokens=1, completion_tokens=1, total_tokens=2, model=model)
            )
        self.permission_policy = PermissionPolicy(self.settings.permission_mode)
        self.hook_manager = HookManager()

    def astream_turn(self, text: str) -> AsyncIterator[Any]:
        async def _turn() -> AsyncIterator[Any]:
            logging.getLogger("LiteLLM").info(
                "LiteLLM completion() model= route-a; provider = openai"
            )
            yield AgentTextDelta(text="ok\n")
            yield AgentDone(stop_reason="completed", iterations=1, usage=Usage())

        return _turn()


def _chat(
    monkeypatch: pytest.MonkeyPatch, labels: dict[str, str] | None, agent: type = _FakeAgent
) -> str:
    monkeypatch.setenv("ZAKCODE_DEFAULT_MODEL", ROUTE)
    monkeypatch.setenv("ZAKCODE_API_BASE", GATEWAY)
    if labels is None:
        monkeypatch.delenv("ZAKCODE_PROVIDER_LABELS", raising=False)
    else:
        monkeypatch.setenv("ZAKCODE_PROVIDER_LABELS", json.dumps(labels))
    monkeypatch.setattr(zakcode, "Agent", agent)
    result = CliRunner().invoke(app, ["cli"], input="hello\n/model\n/cost\n/exit\n")
    assert result.exit_code == 0, result.stdout
    return result.stdout


def _row(out: str, *needles: str) -> str:
    """The first line holding every needle, without its indent or the banner box's borders."""
    line = next(line for line in out.splitlines() if all(n in line for n in needles))
    return line.strip().strip("│|").strip()


def test_the_chat_labels_the_banner_model_cost_and_call_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    out = _chat(monkeypatch, LABELS)

    banner = _row(out, "model", "route-a")
    assert "Gateway" in banner and "openai" not in banner
    assert "Gateway/route-a" in [line.strip() for line in out.splitlines()]  # /model
    assert _row(out, "route-a", "tok").startswith("Gateway/route-a")  # /cost, by model
    assert _row(out, "route-b", "tok").startswith("Gateway/route-b")
    call = _row(out, "call route-a")
    assert call.endswith("Gateway") and "openai" not in call
    assert "openai" not in out.replace(GATEWAY, "")  # nowhere else either


def test_without_labels_the_chat_is_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    out = _chat(monkeypatch, None)

    assert _row(out, "model", "route-a").endswith("openai")
    assert ROUTE in [line.strip() for line in out.splitlines()]
    assert _row(out, "route-a", "tok").startswith(ROUTE)
    assert _row(out, "call route-a").endswith("openai")
    assert "Gateway" not in out


def test_an_empty_mapping_prints_what_no_setting_prints(monkeypatch: pytest.MonkeyPatch) -> None:
    def _stable(text: str) -> str:
        text = re.sub(r"\(\d\d:\d\d\)", "(hh:mm)", text)
        return re.sub(r"\b[0-9a-f]{32}\b", "<session>", text)

    assert _stable(_chat(monkeypatch, {})) == _stable(_chat(monkeypatch, None))


class _ZakpickAgent(_FakeAgent):
    """Logs its startup model while it is built, as the real Agent does on zakpick."""

    def __init__(self, **overrides: object) -> None:
        from zakcode.providers.labels import ModelRef

        logging.getLogger("zakcode").info(
            "zakpick: routing per task category; startup model %s", ModelRef(ROUTE)
        )
        super().__init__(**overrides)


def test_a_line_logged_while_the_agent_is_built_shows_the_label(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    out = _chat(monkeypatch, LABELS, agent=_ZakpickAgent)

    assert _row(out, "startup model").endswith("startup model Gateway/route-a")


def test_the_log_file_keeps_the_token(monkeypatch: pytest.MonkeyPatch) -> None:
    _chat(monkeypatch, LABELS)

    text = log_file_path().read_text(encoding="utf-8")
    assert "provider = openai" in text
    assert "Gateway" not in text


# ── the route status ─────────────────────────────────────────────────────────────────────────


class _Streamed(Provider):
    """One streamed answer, from a provider that names its model as a route would."""

    model = ROUTE

    async def acomplete(
        self, messages: list[Message], *, system: str | None = None, tools: Any = None, **kw: Any
    ) -> LLMResult:  # pragma: no cover - the streamed path only
        raise AssertionError("the buffered path must not run")

    async def astream(
        self, messages: list[Message], *, system: str | None = None, tools: Any = None, **kw: Any
    ) -> AsyncIterator[ProviderStreamEvent]:
        yield StreamTextDelta(text="Done: nothing else was needed.")
        yield StreamDone(finish_reason="stop")

    def count_tokens(self, messages: list[Message], *, system: str | None = None) -> int:
        return 0

    def capabilities(self) -> Capabilities:
        return Capabilities(supports_tools=True, context_window=WINDOW)


def _route_statuses(tmp_path: Path, labels: dict[str, str] | None) -> list[str]:
    provider = _Streamed()
    extra: dict[str, Any] = {} if labels is None else {"provider_labels": labels}
    loop = AgentLoop(
        provider,
        ToolRegistry(),
        Session(cwd=str(tmp_path), model=ROUTE),
        workspace_root=tmp_path,
        max_iterations=4,
        settings=Settings(workspace_root=str(tmp_path), **extra),
        main_provider_for=lambda category: provider,  # zakpick on: the route is decided
    )

    async def _collect() -> list[Any]:
        return [event async for event in loop.astream_turn("say hello")]

    return [
        event.message
        for event in asyncio.run(_collect())
        if isinstance(event, AgentStatus) and event.message.startswith("route: ")
    ]


def test_the_route_status_shows_the_label(tmp_path: Path) -> None:
    statuses = _route_statuses(tmp_path, LABELS)

    assert statuses and all(s.endswith(" → Gateway/route-a") for s in statuses)


def test_without_labels_the_route_status_is_unchanged(tmp_path: Path) -> None:
    statuses = _route_statuses(tmp_path, None)

    assert statuses and all(s.endswith(f" → {ROUTE}") for s in statuses)


# ── the provider's notices ───────────────────────────────────────────────────────────────────


def _answer(served: str = "route-a", headers: dict[str, str] | None = None) -> SimpleNamespace:
    message = SimpleNamespace(content="ok", tool_calls=None, reasoning_content=None)
    return SimpleNamespace(
        model=served,
        choices=[SimpleNamespace(message=message, finish_reason="stop")],
        usage=SimpleNamespace(prompt_tokens=3, completion_tokens=1, total_tokens=4),
        _hidden_params={"additional_headers": dict(headers)} if headers else {},
    )


def _refusal(sentence: str) -> Exception:
    """What litellm raises for a gateway 400 whose message is ``sentence``."""
    return lp.litellm.BadRequestError(
        message=f"OpenAIException - {sentence}", model="route-a", llm_provider="openai"
    )


def _scripted(*steps: Any) -> Any:
    queue = list(steps)

    async def _acompletion(**_kwargs: Any) -> Any:
        step = queue.pop(0) if len(queue) > 1 else queue[0]
        if isinstance(step, Exception):
            raise step
        return step

    return _acompletion


def _provider(temperature: float | None = None) -> LiteLLMProvider:
    return LiteLLMProvider(
        model=ROUTE,
        api_base=GATEWAY,
        api_key="sk-test",
        context_window=WINDOW,
        temperature=temperature,
    )


NOTICES = ["served by", "fallback", "temperature refused", "tool use refused"]


async def _notice(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, notice: str
) -> list[logging.LogRecord]:
    """Run the call that makes ``notice`` and return the records of the zakcode loggers that
    name the route."""
    caplog.set_level(logging.INFO, logger="zakcode.providers")
    hi = [Message.user("hi")]
    if notice == "served by":
        monkeypatch.setattr(lp.litellm, "acompletion", _scripted(_answer(served="route-b")))
        await _provider().acomplete(hi)
    elif notice == "fallback":
        monkeypatch.setattr(
            lp.litellm, "acompletion", _scripted(_answer(served="route-b", headers=FELL_BACK))
        )
        await _provider().acomplete(hi)
    elif notice == "temperature refused":
        monkeypatch.setattr(
            lp.litellm, "acompletion", _scripted(_refusal(TEMPERATURE_SENTENCE), _answer())
        )
        await _provider(temperature=0.2).acomplete(hi)
    else:
        monkeypatch.setattr(
            lp.litellm, "acompletion", _scripted(_refusal(TOOL_USE_SENTENCE), _answer())
        )
        wrapper = TextToolCallingProvider(_provider(), mode="native", single_tool_per_turn=True)
        await wrapper.acomplete(hi, tools=TOOLS)
    return [r for r in caplog.records if r.name.startswith("zakcode.") and ROUTE in r.getMessage()]


@pytest.mark.parametrize("notice", NOTICES)
async def test_a_notice_shows_the_label_on_screen(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, notice: str
) -> None:
    records = await _notice(monkeypatch, caplog, notice)

    assert records
    for record in records:
        screen = transcript_line(record, LABELS)
        assert screen is not None
        assert "Gateway/route-a" in screen and ROUTE not in screen
        assert ROUTE in record.getMessage()  # the record itself is left as it was


@pytest.mark.parametrize("notice", NOTICES)
async def test_a_notice_keeps_the_model_id_in_the_log_file_and_without_labels(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, notice: str
) -> None:
    records = await _notice(monkeypatch, caplog, notice)

    assert records
    for record in records:
        # The same record with each model id handed over as the plain string it was before.
        plain = logging.makeLogRecord(record.__dict__)
        plain.args = tuple(getattr(a, "model_id", a) for a in record.args or ())
        file_format = logging.Formatter("%(message)s")
        file_text = file_format.format(record)
        assert ROUTE in file_text and "Gateway" not in file_text
        assert file_text == file_format.format(plain)
        screen = transcript_line(record)
        assert screen is not None and ROUTE in screen
        assert screen == transcript_line(plain)


async def test_a_name_the_backend_reported_is_shown_as_it_came(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO, logger="zakcode.providers")
    monkeypatch.setattr(lp.litellm, "acompletion", _scripted(_answer(served=OTHER)))

    await _provider().acomplete([Message.user("hi")])

    [record] = [r for r in caplog.records if "served by" in r.getMessage()]
    screen = transcript_line(record, LABELS)
    assert screen is not None and screen.endswith(f"Gateway/route-a: served by {OTHER}")


def test_a_labelled_row_is_scrubbed_like_every_other_row() -> None:
    from zakcode.cli.logsink import RedactingFilter
    from zakcode.providers.labels import ModelRef

    # Shaped for the redactor at run time from a word and a counter; asserted on booleans only.
    probe = "sk-" + "-".join(["label"] * 4) + "-1"
    record = logging.LogRecord(
        "zakcode.providers",
        logging.INFO,
        __file__,
        1,
        "%s: served by %s",
        (ModelRef(ROUTE), "x"),
        None,
    )
    RedactingFilter().filter(record)

    screen = transcript_line(record, {"openai": probe}) or ""

    leaked = probe in screen
    assert not leaked
    assert screen.endswith("/route-a: served by x")


_LOG_METHODS = {"debug", "info", "warning", "error", "exception", "critical"}
#: The one log line that names models and is left as it is: the failover note is one string that a
#: log line and a status share, so it is labelled whole or not at all (ADR-0314).
_LEFT_AS_IS = ("model failover: ",)
#: A name the backend reported is shown as it came; the label is for the configured token.
_BACKEND_WORDS = ("served_model",)


def test_every_log_line_that_names_a_model_passes_it_through_the_label_helper() -> None:
    """Walks the sites: a log call whose argument is a model id hands it over as ``ModelRef``,
    which the log file renders as the id and the transcript as the labelled id. A new notice
    that names a model fails here until it does the same, which is one call."""
    root = Path(zakcode.__file__).parent
    bare: list[str] = []
    for path in sorted(root.rglob("*.py")):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if not (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr in _LOG_METHODS
                and node.args
                and "log" in ast.unparse(node.func.value).lower()
            ):
                continue
            fmt = node.args[0]
            if isinstance(fmt, ast.Constant) and str(fmt.value).startswith(_LEFT_AS_IS):
                continue
            for arg in node.args[1:]:
                text = ast.unparse(arg)
                if text.endswith(("model", "model_id()")) and not text.endswith(_BACKEND_WORDS):
                    bare.append(f"{path.relative_to(root).as_posix()}:{node.lineno} {text}")

    assert bare == []


# ── what a machine reads ─────────────────────────────────────────────────────────────────────


async def test_the_wire_model_and_the_stored_names_keep_the_token(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    sent: list[dict[str, Any]] = []

    async def _acompletion(**kwargs: Any) -> Any:
        sent.append(kwargs)
        return _answer()

    monkeypatch.setattr(lp.litellm, "acompletion", _acompletion)
    settings = Settings(
        default_model=ROUTE,
        api_base=GATEWAY,
        api_key="sk-test",
        context_window=WINDOW,
        workspace_root=str(tmp_path),
        provider_labels=LABELS,
        zakpick_models={
            "quick_code": {"model": "route-a", "source": "openai", "context_window": WINDOW}
        },
    )

    await LiteLLMProvider(settings).acomplete([Message.user("hi")])

    assert sent[0]["model"] == ROUTE
    entry = settings.zakpick_models["quick_code"]
    assert entry.source == "openai"
    assert entry.litellm_string == ROUTE
    stored = json.loads(Session(cwd=str(tmp_path), model=settings.default_model).model_dump_json())
    assert stored["model"] == ROUTE
