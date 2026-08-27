"""Tests for the `zakcode webapp` CLI command (workspace pointer wiring).

Hermetic: ``uvicorn.Server`` and ``create_app`` are monkeypatched so nothing binds a port
or builds a real app — the test only asserts that --workspace threads the workspace root
into create_app, and that the bounded-run callback can bring the server down.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("uvicorn")

from typer.testing import CliRunner  # noqa: E402

from zakcode.cli import app  # noqa: E402

runner = CliRunner()


def _patch(monkeypatch: pytest.MonkeyPatch, captured: dict[str, Any]) -> None:
    import uvicorn

    import zakcode.server.app as server_app

    def fake_create_app(*, settings: Any = None, **kw: Any) -> object:
        captured["settings"] = settings
        captured["on_run_end"] = kw.get("on_run_end")
        return object()  # dummy ASGI app; the server is stubbed so it is never served

    class _FakeServer:
        """Stands in for uvicorn's Server: records itself, never binds a port.

        ``serve`` constructs a Server rather than calling ``uvicorn.run`` so a bounded
        run can ask it to exit (ADR-0039), so THIS is the seam the tests must stub —
        patching ``uvicorn.run`` would no longer intercept anything and the test would
        really serve.
        """

        def __init__(self, config: Any) -> None:
            self.config = config
            self.should_exit = False
            captured["server"] = self

        def run(self) -> None:
            captured["ran"] = True

    monkeypatch.setattr(server_app, "create_app", fake_create_app)
    monkeypatch.setattr(uvicorn, "Server", _FakeServer)


def test_webapp_workspace_threads_workspace_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, Any] = {}
    _patch(monkeypatch, captured)

    result = runner.invoke(app, ["webapp", "--workspace", str(tmp_path)])

    assert result.exit_code == 0, result.stdout
    assert captured["settings"] is not None
    assert str(captured["settings"].workspace_root) == str(tmp_path)


def test_webapp_without_workspace_uses_configured_default(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}
    _patch(monkeypatch, captured)

    result = runner.invoke(app, ["webapp"])

    assert result.exit_code == 0, result.stdout
    # create_app() is called with no settings (server resolves them from env/.env).
    assert captured["settings"] is None


# ── non-loopback bind guard (unauthenticated exposure) ─────────────────────────────


def test_webapp_refuses_non_loopback_without_auth(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}
    _patch(monkeypatch, captured)
    monkeypatch.delenv("ZAKCODE_AUTH_TOKEN", raising=False)

    result = runner.invoke(app, ["webapp", "--host", "0.0.0.0"])

    assert result.exit_code == 1
    assert "refusing" in result.stdout.lower()
    assert "settings" not in captured  # never reached create_app/uvicorn.run


def test_webapp_non_loopback_allowed_with_insecure(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}
    _patch(monkeypatch, captured)
    monkeypatch.delenv("ZAKCODE_AUTH_TOKEN", raising=False)

    result = runner.invoke(app, ["webapp", "--host", "0.0.0.0", "--insecure"])

    assert result.exit_code == 0, result.stdout
    assert "settings" in captured  # the guard let it through to create_app


def test_webapp_non_loopback_allowed_with_auth_token(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}
    _patch(monkeypatch, captured)
    monkeypatch.setenv("ZAKCODE_AUTH_TOKEN", "a-token")

    result = runner.invoke(app, ["webapp", "--host", "0.0.0.0"])

    assert result.exit_code == 0, result.stdout
    assert "settings" in captured


# ── bounded runs: the callback has to actually stop the server (ADR-0039) ──────────


def test_run_end_callback_asks_the_server_to_exit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The cap only saves money if the vessel STOPS.

    Stopping the turn loop while uvicorn keeps serving leaves the host idle and still
    billing — the cap would buy nothing. This pins the wiring end to end: serve hands
    create_app an ``on_run_end``, and awaiting it sets ``should_exit`` on the very
    server that was started.
    """
    import asyncio

    captured: dict[str, Any] = {}
    _patch(monkeypatch, captured)

    result = runner.invoke(app, ["webapp", "--workspace", str(tmp_path)])
    assert result.exit_code == 0, result.stdout
    assert captured["ran"] is True

    on_run_end = captured["on_run_end"]
    assert on_run_end is not None, "serve must hand create_app a run-end callback"
    assert captured["server"].should_exit is False  # control: not stopped by merely serving

    asyncio.run(on_run_end("duration_cap"))
    assert captured["server"].should_exit is True
