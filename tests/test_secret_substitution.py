"""Named-secrets substitution — provider, web_fetch integration, secret_names tool.

Hermetic (no network): web_fetch is driven by the fake httpx from test_web_tools, and
IP-literal URLs keep the SSRF guard off DNS.

The through-line assertion, made in both directions wherever it applies:

* REQUEST side (positive control): the substituted VALUE is present in what left the
  tool — the captured outbound URL / headers. Without this, a substitution layer that
  silently did nothing would pass every hygiene test below.
* MODEL side: the value appears in NO model-facing string — output, data, hint, fix —
  on success or on any error path. The placeholder form is what comes back.
"""

from __future__ import annotations

import json
from pathlib import Path

from tests.test_web_tools import _FakeResponse, _patch_httpx
from zakcode.config import Settings
from zakcode.tools.base import ToolContext
from zakcode.tools.builtins._secrets import (
    SecretsProvider,
    UnknownSecretError,
)
from zakcode.tools.builtins.default_registry import default_registry
from zakcode.tools.builtins.secret_names import SecretNamesTool
from zakcode.tools.builtins.web_fetch import WebFetchTool

SECRET_VALUE = "sk-live-9f8e7d6c5b4a3210"


def _ctx(tmp_path: Path) -> ToolContext:
    return ToolContext(workspace_root=tmp_path)


def _provider(tmp_path: Path, secrets: dict[str, str] | None = None, **kw) -> SecretsProvider:
    path = tmp_path / "secrets.json"
    if secrets is not None:
        path.write_text(json.dumps(secrets), encoding="utf-8")
    return SecretsProvider(path if secrets is not None else None, **kw)


def _dump(result) -> str:
    """Every model-facing string of a ToolResult, as one haystack."""
    return json.dumps(result.model_dump())


# ── SecretsProvider ──────────────────────────────────────────────────────────────────


def test_provider_names_sorted_and_filtered(tmp_path: Path) -> None:
    provider = _provider(
        tmp_path,
        {
            "WEATHER_API_KEY": SECRET_VALUE,
            "A_KEY": "value-a",
            "lowercase_bad": "x",  # bad charset: filtered
            "9STARTS_WITH_DIGIT": "x",  # bad charset: filtered
            "EMPTY_VALUE": "",  # empty value: filtered
        },
    )
    assert provider.names() == ["A_KEY", "WEATHER_API_KEY"]


def test_provider_empty_when_missing_malformed_or_oversized(tmp_path: Path) -> None:
    assert SecretsProvider(None).names() == []
    assert SecretsProvider(tmp_path / "absent.json").names() == []

    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    assert SecretsProvider(bad).names() == []

    listy = tmp_path / "list.json"
    listy.write_text('["A", "B"]', encoding="utf-8")
    assert SecretsProvider(listy).names() == []

    huge = tmp_path / "huge.json"
    huge.write_text(json.dumps({"A_KEY": "x" * (70 * 1024)}), encoding="utf-8")
    assert SecretsProvider(huge).names() == []


def test_provider_resolve_substitutes_and_reports_names(tmp_path: Path) -> None:
    provider = _provider(tmp_path, {"A_KEY": "aaa-value", "B_KEY": "bbb-value"})
    resolved, used = provider.resolve(
        "https://api.example/x?a={{secret:A_KEY}}&b={{secret:B_KEY}}&a2={{secret:A_KEY}}"
    )
    assert resolved == "https://api.example/x?a=aaa-value&b=bbb-value&a2=aaa-value"
    assert used == {"A_KEY", "B_KEY"}


def test_provider_resolve_unknown_name_is_names_only(tmp_path: Path) -> None:
    provider = _provider(tmp_path, {"A_KEY": SECRET_VALUE})
    try:
        provider.resolve("x={{secret:MISSING_KEY}}")
        raise AssertionError("expected UnknownSecretError")
    except UnknownSecretError as exc:
        assert "MISSING_KEY" in str(exc)
        assert SECRET_VALUE not in str(exc)


def test_provider_scrub_folds_values_back_and_skips_short(tmp_path: Path) -> None:
    provider = _provider(tmp_path, {"LONG_KEY": SECRET_VALUE, "TINY": "12345"})
    scrubbed = provider.scrub(f"the api said: your key {SECRET_VALUE} is invalid; 12345 too")
    assert SECRET_VALUE not in scrubbed
    assert "{{secret:LONG_KEY}}" in scrubbed
    # 5 chars is below the scrub floor: replacing it would mangle ordinary text.
    assert "12345" in scrubbed


def test_provider_scrub_replaces_longer_value_first(tmp_path: Path) -> None:
    # OUTER's value contains INNER's value; longer-first keeps the fold correct.
    provider = _provider(tmp_path, {"INNER": "abcdef", "OUTER": "xxabcdefyy"})
    assert provider.scrub("token=xxabcdefyy") == "token={{secret:OUTER}}"


def test_provider_records_usage_names_only(tmp_path: Path) -> None:
    usage = tmp_path / "usage.jsonl"
    provider = _provider(tmp_path, {"A_KEY": SECRET_VALUE}, usage_path=usage)
    provider.record_use({"A_KEY"})
    content = usage.read_text(encoding="utf-8")
    row = json.loads(content.strip())
    assert row["name"] == "A_KEY"
    assert "used_at" in row
    assert SECRET_VALUE not in content  # the whole point of names-only

    # No usage path configured: a silent no-op, never an error.
    _provider(tmp_path, {"A_KEY": SECRET_VALUE}).record_use({"A_KEY"})


def test_provider_rereads_file_each_call(tmp_path: Path) -> None:
    path = tmp_path / "secrets.json"
    path.write_text(json.dumps({"A_KEY": "first"}), encoding="utf-8")
    provider = SecretsProvider(path)
    assert provider.names() == ["A_KEY"]
    path.write_text(json.dumps({"A_KEY": "first", "B_KEY": "second"}), encoding="utf-8")
    # A secret saved after startup is usable without a restart.
    assert provider.names() == ["A_KEY", "B_KEY"]


# ── web_fetch integration (fake httpx; IP literals, no DNS) ──────────────────────────


def _tool(tmp_path: Path, secrets: dict[str, str], **kw) -> WebFetchTool:
    return WebFetchTool(secrets=_provider(tmp_path, secrets, **kw))


async def test_url_substitution_outbound_yes_model_no(tmp_path: Path, monkeypatch) -> None:
    fake = _patch_httpx(
        monkeypatch,
        lambda url: _FakeResponse(headers={"content-type": "text/plain"}, body=b"hello"),
    )
    tool = _tool(tmp_path, {"API_KEY": SECRET_VALUE})
    res = await tool.execute(
        {"url": "http://93.184.216.34/data?key={{secret:API_KEY}}"}, _ctx(tmp_path)
    )
    assert not res.is_error
    # REQUEST side: the real value went out.
    assert SECRET_VALUE in fake.calls[0]["url"]
    # MODEL side: no value anywhere in the result; the placeholder form survives.
    assert SECRET_VALUE not in _dump(res)
    assert "{{secret:API_KEY}}" in res.data["final_url"]


async def test_header_substitution_outbound_yes_model_no(tmp_path: Path, monkeypatch) -> None:
    fake = _patch_httpx(
        monkeypatch,
        lambda url: _FakeResponse(headers={"content-type": "text/plain"}, body=b"ok"),
    )
    tool = _tool(tmp_path, {"API_KEY": SECRET_VALUE})
    res = await tool.execute(
        {
            "url": "http://93.184.216.34/data",
            "headers": {"Authorization": "Bearer {{secret:API_KEY}}"},
        },
        _ctx(tmp_path),
    )
    assert not res.is_error
    assert fake.calls[0]["headers"]["Authorization"] == f"Bearer {SECRET_VALUE}"
    assert SECRET_VALUE not in _dump(res)


async def test_body_echo_is_scrubbed_to_placeholder(tmp_path: Path, monkeypatch) -> None:
    # An API that echoes the request (or an error page embedding the auth header) must
    # not hand the value back into the model's context.
    echo = f"you sent key={SECRET_VALUE} which is expired".encode()
    _patch_httpx(
        monkeypatch,
        lambda url: _FakeResponse(headers={"content-type": "text/plain"}, body=echo),
    )
    tool = _tool(tmp_path, {"API_KEY": SECRET_VALUE})
    res = await tool.execute(
        {"url": "http://93.184.216.34/data?key={{secret:API_KEY}}"}, _ctx(tmp_path)
    )
    assert not res.is_error
    assert SECRET_VALUE not in res.output
    assert "{{secret:API_KEY}}" in res.output


async def test_transport_error_embedding_resolved_url_is_scrubbed(
    tmp_path: Path, monkeypatch
) -> None:
    # Transport exceptions commonly embed the URL they failed against — which is the
    # RESOLVED url. The error path must fold the value back into its placeholder.
    def _route(url: str):
        return RuntimeError(f"connect failed for {url}")

    _patch_httpx(monkeypatch, _route)
    tool = _tool(tmp_path, {"API_KEY": SECRET_VALUE})
    res = await tool.execute(
        {"url": "http://93.184.216.34/data?key={{secret:API_KEY}}"}, _ctx(tmp_path)
    )
    assert res.is_error
    assert SECRET_VALUE not in _dump(res)
    assert "{{secret:API_KEY}}" in res.output


async def test_unknown_secret_never_reaches_network(tmp_path: Path, monkeypatch) -> None:
    fake = _patch_httpx(
        monkeypatch,
        lambda url: _FakeResponse(headers={"content-type": "text/plain"}, body=b"x"),
    )
    tool = _tool(tmp_path, {"API_KEY": SECRET_VALUE})
    res = await tool.execute(
        {"url": "http://93.184.216.34/x?k={{secret:NOT_A_KEY}}"}, _ctx(tmp_path)
    )
    assert res.is_error
    assert "NOT_A_KEY" in res.output
    assert SECRET_VALUE not in _dump(res)
    assert res.fix and "secret_names" in res.fix
    assert fake.calls == []  # a half-substituted request must never leave the box


async def test_forbidden_and_malformed_headers_refused(tmp_path: Path, monkeypatch) -> None:
    fake = _patch_httpx(
        monkeypatch,
        lambda url: _FakeResponse(headers={"content-type": "text/plain"}, body=b"x"),
    )
    tool = _tool(tmp_path, {"API_KEY": SECRET_VALUE})
    for headers in (
        {"Host": "evil.example"},  # transport-controlled (carries the SSRF pin)
        {"Accept-Encoding": "gzip"},  # decompression-bomb defense
        {"X-Bad\nName": "v"},  # header-injection shape
        {"X-Ok": 7},  # non-string value
    ):
        res = await tool.execute(
            {"url": "http://93.184.216.34/x", "headers": headers}, _ctx(tmp_path)
        )
        assert res.is_error, headers
    assert fake.calls == []


async def test_ssrf_guard_sees_the_resolved_url(tmp_path: Path) -> None:
    # Adversarial: a secret whose VALUE is an internal host must not smuggle the fetch
    # past the guard — the RESOLVED form is what gets validated.
    tool = _tool(tmp_path, {"SNEAKY_HOST": "127.0.0.1"})
    res = await tool.execute({"url": "http://{{secret:SNEAKY_HOST}}/x"}, _ctx(tmp_path))
    assert res.is_error
    assert "refusing to fetch" in res.output


async def test_usage_recorded_on_fetch_not_on_unknown(tmp_path: Path, monkeypatch) -> None:
    _patch_httpx(
        monkeypatch,
        lambda url: _FakeResponse(headers={"content-type": "text/plain"}, body=b"ok"),
    )
    usage = tmp_path / "usage.jsonl"
    tool = _tool(tmp_path, {"API_KEY": SECRET_VALUE}, usage_path=usage)

    res = await tool.execute(
        {"url": "http://93.184.216.34/x?k={{secret:NOT_A_KEY}}"}, _ctx(tmp_path)
    )
    assert res.is_error
    assert not usage.exists()  # nothing was released, so nothing was "used"

    res = await tool.execute({"url": "http://93.184.216.34/x?k={{secret:API_KEY}}"}, _ctx(tmp_path))
    assert not res.is_error
    rows = [json.loads(line) for line in usage.read_text(encoding="utf-8").splitlines()]
    assert [r["name"] for r in rows] == ["API_KEY"]


async def test_no_placeholders_is_the_old_behavior(tmp_path: Path, monkeypatch) -> None:
    # A plain fetch with no provider configured must be byte-for-byte the old tool.
    fake = _patch_httpx(
        monkeypatch,
        lambda url: _FakeResponse(headers={"content-type": "text/plain"}, body=b"plain"),
    )
    res = await WebFetchTool().execute({"url": "http://93.184.216.34/x"}, _ctx(tmp_path))
    assert not res.is_error
    assert res.output.startswith("plain")
    assert fake.calls[0]["url"].startswith("http://93.184.216.34")


# ── secret_names tool + settings wiring ──────────────────────────────────────────────


async def test_secret_names_lists_names_never_values(tmp_path: Path) -> None:
    tool = SecretNamesTool(_provider(tmp_path, {"WEATHER_API_KEY": SECRET_VALUE}))
    res = await tool.execute({}, _ctx(tmp_path))
    assert not res.is_error
    assert "WEATHER_API_KEY" in res.output
    assert SECRET_VALUE not in _dump(res)
    assert res.data["names"] == ["WEATHER_API_KEY"]


async def test_secret_names_empty_is_clean(tmp_path: Path) -> None:
    res = await SecretNamesTool().execute({}, _ctx(tmp_path))
    assert not res.is_error
    assert res.data["names"] == []


async def test_registry_wires_settings_paths_through(tmp_path: Path) -> None:
    secrets_file = tmp_path / "secrets.json"
    secrets_file.write_text(json.dumps({"MAPS_KEY": "maps-value-123"}), encoding="utf-8")
    settings = Settings(secrets_file=secrets_file)
    registry = default_registry(settings)
    res = await registry.get("secret_names").execute({}, _ctx(tmp_path))
    assert res.data["names"] == ["MAPS_KEY"]
    # And the alias route resolves to the same tool.
    assert registry.get("list_secrets").spec.name == "secret_names"


def test_settings_reads_paths_from_env(monkeypatch) -> None:
    monkeypatch.setenv("ZAKCODE_SECRETS_FILE", "/tmp/does-not-matter/secrets.json")
    monkeypatch.setenv("ZAKCODE_SECRETS_USAGE_FILE", "/tmp/does-not-matter/usage.jsonl")
    settings = Settings()
    assert settings.secrets_file == Path("/tmp/does-not-matter/secrets.json")
    assert settings.secrets_usage_file == Path("/tmp/does-not-matter/usage.jsonl")
