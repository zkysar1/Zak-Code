"""Durable OKF bundle publish (PEARL §10.5) — the box side of the storage route.

The properties worth pinning here are the ones whose violation is SILENT:

* **Unconfigured and partially-configured boxes must no-op, and that no-op must
  be a SUCCESS.** A fleet scheduler calls this on every box; if an unconfigured
  box exited non-zero, the alert channel would be permanently red and the real
  failures would be invisible inside the noise.
* **A publish failure must never propagate.** Publishing is an enhancement —
  a box whose PUT fails must keep serving ``/knowledge/*``. Every failure path
  below asserts a returned report, never a raised exception.
* **The credential must not leak into the URL.** The URL reaches access logs;
  the header does not. Asserted as a derived boolean so the test output can
  never contain the secret either.

No test here touches the network: every case drives a recording stub, so a
regression that starts making real requests fails on the stub's absence of a
call rather than hanging.
"""

from __future__ import annotations

import pytest

from zakcode.config import Settings
from zakcode.server.knowledge_publish import (
    KNOWLEDGE_PREFIX,
    PublishResult,
    publish_bundle,
    publish_url,
)

KEY = "vin_secretkeyvalue0123456789"
BASE = "https://gateway.example.test"
ENV = "env-abc123"

BUNDLE = {
    "index.md": '---\ntype: "index"\n---\n\n# Knowledge base\n',
    "nodes/biosignatures.md": '---\ntype: "node"\n---\n\n# Biosignatures\n',
}


class _Resp:
    def __init__(self, status_code: int) -> None:
        self.status_code = status_code


class _StubClient:
    """Records every PUT. Raises nothing unless told to."""

    def __init__(self, status: int = 200, raises: Exception | None = None) -> None:
        self.calls: list[dict] = []
        self.closed = False
        self._status = status
        self._raises = raises

    def put(self, url: str, *, content: bytes, headers: dict) -> _Resp:
        self.calls.append({"url": url, "content": content, "headers": headers})
        if self._raises is not None:
            raise self._raises
        return _Resp(self._status)

    def close(self) -> None:
        self.closed = True


def _settings(**over) -> Settings:
    base = {
        "default_model": "scripted/test",
        "knowledge_publish_url": BASE,
        "knowledge_publish_env_id": ENV,
        "knowledge_publish_key": KEY,
    }
    base.update(over)
    return Settings(**base)


# ── no-op contract ───────────────────────────────────────────────────────────


def test_unconfigured_box_no_ops_and_reports_success() -> None:
    s = Settings(default_model="scripted/test")
    stub = _StubClient()
    result = publish_bundle(s, BUNDLE, client=stub)
    assert result.ok, "an unconfigured box must be a SUCCESS, not a failure"
    assert result.skipped_reason
    assert result.published == [] and result.failed == []
    assert stub.calls == [], "must not attempt a request when unconfigured"


@pytest.mark.parametrize(
    "missing",
    ["knowledge_publish_url", "knowledge_publish_env_id", "knowledge_publish_key"],
)
def test_partial_configuration_no_ops_rather_than_half_publishing(missing: str) -> None:
    # The discriminating case: two of three set is NOT "nearly ready". Attempting
    # a PUT with a blank env id would target /storage//knowledge/... — a path the
    # route rejects, per file, on every scheduled run.
    result = publish_bundle(_settings(**{missing: None}), BUNDLE, client=_StubClient())
    assert result.ok and result.skipped_reason
    assert result.published == []


def test_blank_string_counts_as_unset() -> None:
    # An env var exported as "" is the common way a box is half-provisioned.
    result = publish_bundle(_settings(knowledge_publish_env_id="   "), BUNDLE, client=_StubClient())
    assert result.skipped_reason


def test_empty_bundle_is_skipped_not_published() -> None:
    stub = _StubClient()
    result = publish_bundle(_settings(), {}, client=stub)
    assert result.ok and result.skipped_reason == "bundle is empty"
    assert stub.calls == []


# ── the happy path ───────────────────────────────────────────────────────────


def test_every_bundle_file_is_put_under_the_knowledge_prefix() -> None:
    stub = _StubClient()
    result = publish_bundle(_settings(), BUNDLE, client=stub)
    assert result.ok
    assert sorted(result.published) == sorted(BUNDLE)
    urls = {c["url"] for c in stub.calls}
    assert urls == {
        f"{BASE}/v1/vinheim/storage/{ENV}/{KNOWLEDGE_PREFIX}/index.md",
        f"{BASE}/v1/vinheim/storage/{ENV}/{KNOWLEDGE_PREFIX}/nodes/biosignatures.md",
    }


def test_content_is_sent_as_utf8_markdown() -> None:
    stub = _StubClient()
    publish_bundle(_settings(), {"nodes/n.md": "# Café ☕\n"}, client=stub)
    call = stub.calls[0]
    assert call["content"] == "# Café ☕\n".encode()
    assert call["headers"]["content-type"].startswith("text/markdown")


def test_trailing_slash_on_base_url_does_not_double_up() -> None:
    stub = _StubClient()
    publish_bundle(_settings(knowledge_publish_url=BASE + "/"), {"index.md": "x"}, client=stub)
    assert "//v1/vinheim" not in stub.calls[0]["url"]


def test_publish_url_is_stable_and_prefixed() -> None:
    url = publish_url(_settings(), "nodes/a.md")
    assert url == f"{BASE}/v1/vinheim/storage/{ENV}/{KNOWLEDGE_PREFIX}/nodes/a.md"


# ── failures are reported, never raised ──────────────────────────────────────


def test_http_error_is_collected_not_raised() -> None:
    result = publish_bundle(_settings(), BUNDLE, client=_StubClient(status=403))
    assert not result.ok
    assert len(result.failed) == len(BUNDLE)
    assert all("403" in reason for _, reason in result.failed)
    assert result.published == []


def test_transport_exception_is_collected_not_raised() -> None:
    # Must not propagate: a network blip cannot be allowed to kill the caller.
    result = publish_bundle(_settings(), BUNDLE, client=_StubClient(raises=OSError("boom")))
    assert not result.ok
    assert len(result.failed) == len(BUNDLE)
    assert all(reason == "OSError" for _, reason in result.failed)


def test_one_failure_does_not_abort_the_remaining_files() -> None:
    class _FlakyClient(_StubClient):
        def put(self, url: str, *, content: bytes, headers: dict) -> _Resp:
            self.calls.append({"url": url, "content": content, "headers": headers})
            return _Resp(500 if "index.md" in url else 200)

    result = publish_bundle(_settings(), BUNDLE, client=_FlakyClient())
    assert result.published == ["nodes/biosignatures.md"]
    assert [p for p, _ in result.failed] == ["index.md"]


# ── secret hygiene ───────────────────────────────────────────────────────────


def test_key_travels_in_the_header_and_never_in_the_url() -> None:
    stub = _StubClient()
    publish_bundle(_settings(), BUNDLE, client=stub)
    # Derived booleans only — never print or assert on the value itself, so a
    # failure message cannot carry the credential (guard-1270).
    assert all(KEY not in c["url"] for c in stub.calls)
    assert all(c["headers"]["authorization"].endswith(KEY) for c in stub.calls)


def test_key_is_excluded_from_model_dump() -> None:
    dumped = _settings().model_dump()
    assert "knowledge_publish_key" not in dumped
    # The non-secret half must still be visible — otherwise `GET /config` cannot
    # show an operator whether publishing is even pointed anywhere.
    assert dumped["knowledge_publish_url"] == BASE
    assert dumped["knowledge_publish_env_id"] == ENV


def test_readiness_predicate_matches_publish_behavior() -> None:
    assert _settings().knowledge_publish_ready is True
    assert Settings(default_model="scripted/test").knowledge_publish_ready is False
    assert _settings(knowledge_publish_key=None).knowledge_publish_ready is False


# ── client lifecycle ─────────────────────────────────────────────────────────


def test_injected_client_is_not_closed_by_the_callee() -> None:
    # The caller owns a client it passed in; closing it would break connection
    # reuse for a caller publishing several workspaces in one process.
    stub = _StubClient()
    publish_bundle(_settings(), BUNDLE, client=stub)
    assert stub.closed is False


def test_result_summary_is_readable_in_both_states() -> None:
    assert "skipped" in PublishResult(skipped_reason="nope").summary()
    assert PublishResult(published=["a"], failed=[("b", "HTTP 500")]).summary() == (
        "published 1, failed 1"
    )
