"""Tests for the `zakcode serve` CLI command (workspace pointer wiring).

Hermetic: uvicorn.run and create_app are monkeypatched so nothing binds a port or builds a
real app — the test only asserts that --workspace threads the workspace root into create_app.
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

    def fake_create_app(*, settings: Any = None, **_kw: Any) -> object:
        captured["settings"] = settings
        return object()  # dummy ASGI app; uvicorn.run is mocked so it is never served

    monkeypatch.setattr(server_app, "create_app", fake_create_app)
    monkeypatch.setattr(uvicorn, "run", lambda *a, **k: None)


def test_serve_workspace_threads_workspace_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, Any] = {}
    _patch(monkeypatch, captured)

    result = runner.invoke(app, ["serve", "--workspace", str(tmp_path)])

    assert result.exit_code == 0, result.stdout
    assert captured["settings"] is not None
    assert str(captured["settings"].workspace_root) == str(tmp_path)


def test_serve_without_workspace_uses_configured_default(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}
    _patch(monkeypatch, captured)

    result = runner.invoke(app, ["serve"])

    assert result.exit_code == 0, result.stdout
    # create_app() is called with no settings (server resolves them from env/.env).
    assert captured["settings"] is None


# ── non-loopback bind guard (unauthenticated exposure) ─────────────────────────────


def test_serve_refuses_non_loopback_without_auth(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}
    _patch(monkeypatch, captured)
    monkeypatch.delenv("ZAKCODE_AUTH_TOKEN", raising=False)

    result = runner.invoke(app, ["serve", "--host", "0.0.0.0"])

    assert result.exit_code == 1
    assert "refusing" in result.stdout.lower()
    assert "settings" not in captured  # never reached create_app/uvicorn.run


def test_serve_non_loopback_allowed_with_insecure(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}
    _patch(monkeypatch, captured)
    monkeypatch.delenv("ZAKCODE_AUTH_TOKEN", raising=False)

    result = runner.invoke(app, ["serve", "--host", "0.0.0.0", "--insecure"])

    assert result.exit_code == 0, result.stdout
    assert "settings" in captured  # the guard let it through to create_app


def test_serve_non_loopback_allowed_with_auth_token(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}
    _patch(monkeypatch, captured)
    monkeypatch.setenv("ZAKCODE_AUTH_TOKEN", "a-token")

    result = runner.invoke(app, ["serve", "--host", "0.0.0.0"])

    assert result.exit_code == 0, result.stdout
    assert "settings" in captured
