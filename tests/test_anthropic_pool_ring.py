"""ADR-0323: a provider on the Anthropic credit pool sends each request on its session's account.

These tests run the real :class:`LiteLLMProvider` and litellm's real HTTP stack against a loopback
stand-in for the Anthropic API that answers by the key it is sent, so what is asserted is what
went on the wire, not what a mock was called with: the key, the ``anthropic-workspace-id`` header,
the version header, and the order of the requests.

What is pinned:

* a session is seated on the lowest-ranked account with a key, and every request carries that
  account's key and workspace;
* a credit refusal (a 400 that says the balance is too low, and a 402 ``billing_error``) moves the
  session to the next account ONCE, the call is re-issued there and succeeds, and the session
  STAYS there on later calls, buffered and streamed;
* when every account is spent the turn ends on ``QuotaExhausted`` naming the accounts and why, and
  no key appears in the text;
* a 429 stays on its account and keeps its ``retry-after``; a 5xx stays; a refused key (401)
  marks the account invalid for the process and moves the session;
* two concurrent calls refused by the same account move the session once and both succeed;
* each successful call writes one ledger row whose token split matches the wire's usage, and a
  new session is seated off an account whose month total is at the steer-off line;
* the pool is off unless asked for, and never under ``hipaa_mode``, behind a gateway, or for a
  model the registry does not name.
"""

from __future__ import annotations

import asyncio
import json
import threading
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import litellm
import pytest
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler

from zakcode.config import Settings
from zakcode.messages import Message
from zakcode.providers.anthropic_pool import (
    DEFAULT_REGISTRY_PATH,
    refusal_facts,
    reset_runtimes,
    ring_for,
)
from zakcode.providers.base import (
    ProviderUnavailable,
    QuotaExhausted,
    RateLimited,
    StreamTextDelta,
)
from zakcode.providers.litellm_provider import LiteLLMProvider

_SPENT = (
    "Your credit balance is too low to access the Anthropic API. Please go to Plans & Billing "
    "to upgrade or purchase credits."
)
_ERRORS: dict[str, tuple[int, str, str]] = {
    # behaviour -> (status, Anthropic error type, message)
    "spent_400": (400, "invalid_request_error", _SPENT),
    "spent_402": (402, "billing_error", "Your account has a billing problem."),
    "revoked": (401, "authentication_error", "invalid x-api-key"),
    "forbidden": (403, "permission_error", "Your API key does not have permission to use this."),
    "throttled": (429, "rate_limit_error", "This request would exceed your rate limit."),
    "overloaded": (529, "overloaded_error", "Overloaded"),
}
_MODEL = "claude-haiku-5-5"


@dataclass
class _Anthropic:
    """A loopback Anthropic API. ``answers`` maps a key to what that key is told."""

    answers: dict[str, str]
    #: What each request carried, in arrival order.
    seen: list[dict[str, object]] = field(default_factory=list)
    #: Usage the "ok" answer reports for a key, when it is not the default.
    usage: dict[str, dict[str, int]] = field(default_factory=dict)
    base_url: str = ""
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def keys_seen(self) -> list[object]:
        return [row["x-api-key"] for row in self.seen]


def _handler(server: _Anthropic) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_POST(self) -> None:  # noqa: N802 - the http.server hook name
            body = json.loads(self.rfile.read(int(self.headers.get("content-length", 0))) or b"{}")
            key = self.headers.get("x-api-key", "")
            with server._lock:
                server.seen.append(
                    {
                        "x-api-key": key,
                        "anthropic-workspace-id": self.headers.get("anthropic-workspace-id"),
                        "anthropic-version": self.headers.get("anthropic-version"),
                        "stream": bool(body.get("stream")),
                        "path": self.path,
                    }
                )
            behaviour = server.answers.get(key, "revoked")
            if behaviour == "ok":
                self._ok(key, stream=bool(body.get("stream")))
            else:
                self._error(
                    *_ERRORS[behaviour], retry_after="7" if behaviour == "throttled" else None
                )

        def _send(
            self, status: int, payload: bytes, content_type: str, extra: dict[str, str]
        ) -> None:
            self.send_response(status)
            self.send_header("content-type", content_type)
            self.send_header("content-length", str(len(payload)))
            for name, value in extra.items():
                self.send_header(name, value)
            self.end_headers()
            self.wfile.write(payload)

        def _error(self, status: int, kind: str, message: str, retry_after: str | None) -> None:
            payload = json.dumps(
                {"type": "error", "error": {"type": kind, "message": message}}
            ).encode()
            extra = {"retry-after": retry_after} if retry_after else {}
            self._send(status, payload, "application/json", extra)

        def _ok(self, key: str, *, stream: bool) -> None:
            usage = {
                "input_tokens": 10,
                "output_tokens": 2,
                "cache_creation_input_tokens": 0,
                "cache_read_input_tokens": 0,
                **server.usage.get(key, {}),
            }
            if not stream:
                message = {
                    "id": "msg_1",
                    "type": "message",
                    "role": "assistant",
                    "model": _MODEL,
                    "content": [{"type": "text", "text": "ok"}],
                    "stop_reason": "end_turn",
                    "stop_sequence": None,
                    "usage": usage,
                }
                self._send(200, json.dumps(message).encode(), "application/json", {})
                return
            start = {
                "type": "message_start",
                "message": {
                    "id": "msg_1",
                    "type": "message",
                    "role": "assistant",
                    "model": _MODEL,
                    "content": [],
                    "stop_reason": None,
                    "stop_sequence": None,
                    "usage": {**usage, "output_tokens": 1},
                },
            }
            events = [
                ("message_start", start),
                (
                    "content_block_start",
                    {
                        "type": "content_block_start",
                        "index": 0,
                        "content_block": {"type": "text", "text": ""},
                    },
                ),
                (
                    "content_block_delta",
                    {
                        "type": "content_block_delta",
                        "index": 0,
                        "delta": {"type": "text_delta", "text": "ok"},
                    },
                ),
                ("content_block_stop", {"type": "content_block_stop", "index": 0}),
                (
                    "message_delta",
                    {
                        "type": "message_delta",
                        "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                        "usage": {"output_tokens": usage["output_tokens"]},
                    },
                ),
                ("message_stop", {"type": "message_stop"}),
            ]
            payload = "".join(
                f"event: {name}\ndata: {json.dumps(data)}\n\n" for name, data in events
            ).encode()
            self._send(200, payload, "text/event-stream", {})

        def log_message(self, *args: object) -> None:
            pass

    return Handler


@pytest.fixture
def anthropic() -> Iterator[_Anthropic]:
    stub = _Anthropic(answers={})
    http = ThreadingHTTPServer(("127.0.0.1", 0), _handler(stub))
    stub.base_url = f"http://127.0.0.1:{http.server_address[1]}"
    thread = threading.Thread(target=http.serve_forever, daemon=True)
    thread.start()
    try:
        yield stub
    finally:
        http.shutdown()
        http.server_close()


@pytest.fixture(autouse=True)
def _fresh_runtimes(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Each test starts with no pool runtime and leaves no HTTP connection of litellm's behind.

    Each test runs on an event loop of its own, and litellm keeps clients, and the connections
    they pool, from one test to the next: a streamed call goes through one module-level client,
    and a buffered call's client is cached under the ``id`` of its loop. A later request then
    makes the pool close sockets an earlier, closed loop opened, and fails with "Event loop is
    closed" (measured: the first test to run after the pooled connections had idled out). A test
    gets a client of its own for the streamed path and drops what it cached for the buffered one.
    """
    cache = litellm.in_memory_llm_clients_cache
    before = set(cache.cache_dict)
    monkeypatch.setattr(
        litellm,
        "module_level_aclient",
        AsyncHTTPHandler(timeout=litellm.request_timeout, client_alias="pool test aclient"),
    )
    reset_runtimes()
    yield
    reset_runtimes()
    for key in [key for key in cache.cache_dict if key not in before]:
        cache._remove_key(key)


@pytest.fixture
def accounts(monkeypatch: pytest.MonkeyPatch) -> None:
    """Three accounts hold keys and workspaces; no other slot does."""
    for slot in range(11):
        for kind in ("API_KEY", "WORKSPACE_ID", "ORG_ID"):
            monkeypatch.delenv(f"ANTHROPIC_ACCT_{slot:02d}_{kind}", raising=False)
    for slot in ("02", "03", "04"):
        monkeypatch.setenv(f"ANTHROPIC_ACCT_{slot}_API_KEY", f"key-{slot}")
        monkeypatch.setenv(f"ANTHROPIC_ACCT_{slot}_WORKSPACE_ID", f"wrkspc-{slot}")


def _registry(tmp_path: Path, base_url: str) -> Path:
    registry = json.loads(DEFAULT_REGISTRY_PATH.read_text(encoding="utf-8"))
    registry["base_url"] = base_url
    path = tmp_path / "registry.json"
    path.write_text(json.dumps(registry), encoding="utf-8")
    return path


def _provider(
    tmp_path: Path, stub: _Anthropic, *, pool: bool = True, **settings: object
) -> LiteLLMProvider:
    values: dict[str, object] = {
        "default_model": f"anthropic/{_MODEL}",
        "anthropic_pool": pool,
        "anthropic_pool_registry": _registry(tmp_path, stub.base_url),
        "anthropic_pool_ledger_dir": tmp_path / "ledger",
    }
    values.update(settings)
    return LiteLLMProvider(Settings(**values), context_window=200_000)  # type: ignore[arg-type]


async def _text(provider: LiteLLMProvider, mode: str) -> str:
    messages = [Message.user("hi")]
    if mode == "buffered":
        return (await provider.acomplete(messages)).text
    parts: list[str] = []
    async for event in provider.astream(messages):
        if isinstance(event, StreamTextDelta):
            parts.append(event.text)
    return "".join(parts)


def _ledger_rows(directory: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for path in sorted(directory.glob("anthropic-pool-ledger-*.jsonl")):
        rows.extend(json.loads(line) for line in path.read_text(encoding="utf-8").splitlines())
    return rows


MODES = pytest.mark.parametrize("mode", ["buffered", "streamed"])


@MODES
async def test_a_request_carries_the_seated_accounts_key_and_workspace(
    tmp_path: Path, anthropic: _Anthropic, accounts: None, mode: str
) -> None:
    anthropic.answers = {"key-02": "ok"}
    provider = _provider(tmp_path, anthropic)
    assert await _text(provider, mode) == "ok"
    assert anthropic.seen == [
        {
            "x-api-key": "key-02",
            "anthropic-workspace-id": "wrkspc-02",
            "anthropic-version": "2023-06-01",
            "stream": mode == "streamed",
            "path": "/v1/messages",
        }
    ]
    assert provider._ring is not None
    assert provider._ring.member == "acct-02"


@MODES
@pytest.mark.parametrize("refusal", ["spent_400", "spent_402"])
async def test_a_credit_refusal_moves_the_session_once_and_it_stays(
    tmp_path: Path, anthropic: _Anthropic, accounts: None, mode: str, refusal: str
) -> None:
    anthropic.answers = {"key-02": refusal, "key-03": "ok", "key-04": "ok"}
    provider = _provider(tmp_path, anthropic)

    assert await _text(provider, mode) == "ok"
    assert anthropic.keys_seen() == ["key-02", "key-03"]
    assert provider._ring is not None
    assert provider._ring.member == "acct-03"

    # The next calls do not go back to the spent account, and do not move again.
    assert await _text(provider, mode) == "ok"
    assert await _text(provider, mode) == "ok"
    assert anthropic.keys_seen() == ["key-02", "key-03", "key-03", "key-03"]
    assert [row["anthropic-workspace-id"] for row in anthropic.seen] == [
        "wrkspc-02",
        "wrkspc-03",
        "wrkspc-03",
        "wrkspc-03",
    ]

    # The pool is shared by the process: a session that starts now skips the spent account.
    later = _provider(tmp_path, anthropic)
    assert await _text(later, mode) == "ok"
    assert anthropic.keys_seen()[-1] == "key-03"
    assert later._ring is not None
    assert later._ring.member == "acct-03"


@MODES
async def test_every_account_spent_ends_the_turn_naming_accounts_and_no_key(
    tmp_path: Path, anthropic: _Anthropic, accounts: None, mode: str
) -> None:
    anthropic.answers = {"key-02": "spent_400", "key-03": "spent_402", "key-04": "spent_400"}
    provider = _provider(tmp_path, anthropic)

    with pytest.raises(QuotaExhausted) as raised:
        await _text(provider, mode)

    # One refused call per account, then it stops: no loop.
    assert anthropic.keys_seen() == ["key-02", "key-03", "key-04"]
    text = str(raised.value)
    for account in ("acct-02", "acct-03", "acct-04"):
        assert account in text
    assert "out of credit until" in text
    for secret in ("key-02", "key-03", "key-04", "wrkspc"):
        assert secret not in text


@MODES
async def test_a_429_stays_on_its_account_and_keeps_its_retry_after(
    tmp_path: Path, anthropic: _Anthropic, accounts: None, mode: str
) -> None:
    anthropic.answers = {"key-02": "throttled", "key-03": "ok"}
    provider = _provider(tmp_path, anthropic)

    with pytest.raises(RateLimited) as raised:
        await _text(provider, mode)

    assert raised.value.retry_after == 7.0
    assert anthropic.keys_seen() == ["key-02"]
    assert provider._ring is not None
    assert provider._ring.member == "acct-02"


@MODES
async def test_a_529_stays_on_its_account(
    tmp_path: Path, anthropic: _Anthropic, accounts: None, mode: str
) -> None:
    anthropic.answers = {"key-02": "overloaded", "key-03": "ok"}
    provider = _provider(tmp_path, anthropic)

    with pytest.raises(ProviderUnavailable):
        await _text(provider, mode)

    assert anthropic.keys_seen() == ["key-02"]
    assert provider._ring is not None
    assert provider._ring.member == "acct-02"


@MODES
@pytest.mark.parametrize("refusal", ["revoked", "forbidden"])
async def test_a_refused_key_marks_the_account_invalid_and_moves_the_session(
    tmp_path: Path,
    anthropic: _Anthropic,
    accounts: None,
    mode: str,
    refusal: str,
    caplog: pytest.LogCaptureFixture,
) -> None:
    anthropic.answers = {"key-02": refusal, "key-03": "ok"}
    provider = _provider(tmp_path, anthropic)

    with caplog.at_level("ERROR", logger="zakcode.providers.anthropic_pool"):
        assert await _text(provider, mode) == "ok"

    assert anthropic.keys_seen() == ["key-02", "key-03"]
    alerts = [r.getMessage() for r in caplog.records if "ALERT" in r.getMessage()]
    assert len(alerts) == 1
    assert "acct-02" in alerts[0]
    assert "key-02" not in alerts[0]

    # Invalid until restart, for every session of the process.
    later = _provider(tmp_path, anthropic)
    assert await _text(later, mode) == "ok"
    assert anthropic.keys_seen() == ["key-02", "key-03", "key-03"]


async def test_two_concurrent_calls_refused_by_one_account_move_the_session_once(
    tmp_path: Path, anthropic: _Anthropic, accounts: None
) -> None:
    anthropic.answers = {"key-02": "spent_400", "key-03": "ok", "key-04": "ok"}
    provider = _provider(tmp_path, anthropic)

    first, second = await asyncio.gather(_text(provider, "buffered"), _text(provider, "buffered"))

    assert (first, second) == ("ok", "ok")
    keys = anthropic.keys_seen()
    assert keys.count("key-02") == 2  # each call was addressed to the account it started on
    assert keys.count("key-03") == 2  # and each re-issued on the one the session moved to
    assert "key-04" not in keys  # the second refusal did not move the session a second time
    assert provider._ring is not None
    assert provider._ring.member == "acct-03"


@MODES
async def test_each_call_writes_a_ledger_row_whose_split_matches_the_wire(
    tmp_path: Path, anthropic: _Anthropic, accounts: None, mode: str
) -> None:
    anthropic.answers = {"key-02": "ok"}
    anthropic.usage = {
        "key-02": {
            "input_tokens": 20,
            "cache_read_input_tokens": 100,
            "cache_creation_input_tokens": 50,
            "output_tokens": 7,
        }
    }
    provider = _provider(tmp_path, anthropic)
    assert await _text(provider, mode) == "ok"
    assert await _text(provider, mode) == "ok"

    rows = _ledger_rows(tmp_path / "ledger")
    assert len(rows) == 2
    for row in rows:
        assert row["member"] == "acct-02"
        assert row["model"] == _MODEL
        assert row["input_uncached"] == 20
        assert row["cache_read"] == 100
        assert row["cache_write"] == 50
        assert row["output"] == 7
        assert isinstance(row["est_usd"], float)
        assert 0 < row["est_usd"] < 0.01
        assert "unpriced" not in row
    month = datetime.now(UTC).strftime("%Y-%m")
    assert (tmp_path / "ledger" / f"anthropic-pool-ledger-{month}.jsonl").is_file()


async def test_a_new_session_is_seated_off_an_account_at_the_steer_off_line(
    tmp_path: Path, anthropic: _Anthropic, accounts: None
) -> None:
    """The month's ledger total, written by any process, decides where a NEW session goes."""
    anthropic.answers = {"key-02": "ok", "key-03": "ok"}
    ledger = tmp_path / "ledger"
    ledger.mkdir()
    now = datetime.now(UTC)
    row = {"member": "acct-02", "est_usd": 190.0}  # 95% of the 200 dollars of credit
    (ledger / f"anthropic-pool-ledger-{now:%Y-%m}.jsonl").write_text(
        json.dumps(row) + "\nnot json\n", encoding="utf-8"
    )

    provider = _provider(tmp_path, anthropic)
    assert await _text(provider, "buffered") == "ok"

    assert anthropic.keys_seen() == ["key-03"]


async def test_no_key_at_all_ends_the_turn_before_any_request(
    tmp_path: Path, anthropic: _Anthropic, monkeypatch: pytest.MonkeyPatch
) -> None:
    for slot in range(11):
        monkeypatch.delenv(f"ANTHROPIC_ACCT_{slot:02d}_API_KEY", raising=False)
    provider = _provider(tmp_path, anthropic)

    with pytest.raises(QuotaExhausted) as raised:
        await _text(provider, "buffered")

    assert anthropic.seen == []
    assert "no key in the environment (slot 02)" in str(raised.value)


async def test_a_prod_run_finds_no_account_until_the_card_account_has_a_cap(
    tmp_path: Path, anthropic: _Anthropic, accounts: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ANTHROPIC_ACCT_00_API_KEY", "key-00")
    provider = _provider(tmp_path, anthropic, anthropic_pool_use="prod")

    with pytest.raises(QuotaExhausted) as raised:
        await _text(provider, "buffered")

    assert anthropic.seen == []
    assert "acct-00 (slot 00): card billing with no monthly cap set" in str(raised.value)


def test_the_pool_is_off_unless_asked_for(tmp_path: Path, anthropic: _Anthropic) -> None:
    provider = _provider(tmp_path, anthropic, pool=False)
    assert provider._ring is None
    kwargs = provider._build_kwargs([{"role": "user", "content": "hi"}], None)
    assert "api_key" not in kwargs
    assert "anthropic-workspace-id" not in (kwargs.get("extra_headers") or {})


def test_the_pool_does_not_apply_where_it_must_not(accounts: None) -> None:
    on = Settings(default_model=f"anthropic/{_MODEL}", anthropic_pool=True)
    env = {"ANTHROPIC_ACCT_02_API_KEY": "key-02"}
    assert ring_for(on, model=f"anthropic/{_MODEL}", via_gateway=False, env=env) is not None
    assert ring_for(on, model=_MODEL, via_gateway=False, env=env) is not None
    # a model the registry does not name
    assert ring_for(on, model="anthropic/claude-opus-5-5", via_gateway=False, env=env) is None
    # a route behind a gateway, which holds its own credentials
    assert ring_for(on, model=f"anthropic/{_MODEL}", via_gateway=True, env=env) is None
    # a home that handles patient data
    hipaa = Settings(default_model=f"anthropic/{_MODEL}", anthropic_pool=True, hipaa_mode=True)
    assert ring_for(hipaa, model=f"anthropic/{_MODEL}", via_gateway=False, env=env) is None
    # no settings at all
    assert ring_for(None, model=f"anthropic/{_MODEL}", via_gateway=False, env=env) is None


def test_refusal_facts_reads_status_type_and_text() -> None:
    class Failure(Exception):
        def __init__(self, text: str, status_code: int | None, body: object = None) -> None:
            super().__init__(text)
            self.status_code = status_code
            self.body = body

    body = '{"type":"error","error":{"type":"billing_error","message":"x"}}'
    plain = refusal_facts(Failure(f"AnthropicException - {body}", 402))
    assert (plain.status, plain.error_type) == (402, "billing_error")

    quoted = refusal_facts(Failure(json.dumps(f"AnthropicException - {body}"), 402))
    assert quoted.error_type == "billing_error"

    from_body = refusal_facts(
        Failure("opaque", 402, body={"type": "error", "error": {"type": "billing_error"}})
    )
    assert from_body.error_type == "billing_error"

    wrapper = Failure("midstream", 503)
    wrapper.original_exception = Failure(f"inner {body}", 402)  # type: ignore[attr-defined]
    assert refusal_facts(wrapper).status == 402

    # litellm has no branch for a 402 or a 403: it raises its generic connection error, status
    # 500, with the body only as text. The error type is one-to-one with the status, so it is
    # read back; a real 5xx keeps its status because its types are different ones.
    forbidden = '{"type":"error","error":{"type":"permission_error","message":"x"}}'
    assert refusal_facts(Failure(f"AnthropicException - {body}", 500)).status == 402
    assert refusal_facts(Failure(f"AnthropicException - {forbidden}", 500)).status == 403
    assert refusal_facts(Failure(f"AnthropicException - {body}", None)).status == 402
    for kind, status in (("api_error", 500), ("overloaded_error", 529)):
        real = f'{{"type":"error","error":{{"type":"{kind}","message":"x"}}}}'
        assert refusal_facts(Failure(f"AnthropicException - {real}", status)).status == status
    # a status litellm did keep is not second-guessed by the text
    assert refusal_facts(Failure(f"AnthropicException - {body}", 429)).status == 429

    assert refusal_facts(Failure("no status", None)).status is None
    assert refusal_facts(ValueError("not an http failure")).status is None
    assert refusal_facts(Failure("bool is not a status", True)).status is None  # type: ignore[arg-type]
