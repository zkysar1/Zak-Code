"""Tests for the sidecar read-only surface: GET /workspace/summary (P0-4) + /sidecar/health (P0-5).

Both endpoints read artifacts a SEPARATE component writes into the loop workspace
(research/journal.md, research/findings*, .current-session) and MUST degrade gracefully when
they are absent — the surface is safe to poll before the agent loop has produced anything.
These are plain JSON GETs (no streaming), so Starlette's TestClient drives them directly (unlike
the /watch SSE tail, which needs a real server). Auth: _AUTH_EXEMPT_PATHS is the EXACT string
"/health", so BOTH endpoints — including /sidecar/health — require the bearer token when one is
configured; only the unauth liveness /health is exempt.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

from zakcode.config import Settings
from zakcode.server.app import create_app
from zakcode.session.store import Session, SessionStore


class _FakeAgent:
    """Minimal AgentLike — never invoked here (no turn runs on these read-only routes)."""

    def __init__(self, session: Session) -> None:
        self.session = session


def _factory(session: Session, model: str | None, prompter: object = None) -> _FakeAgent:  # noqa: ARG001
    return _FakeAgent(session)


def _make_app(workspace: Path, *, auth_token: str | None = None) -> FastAPI:
    settings = Settings(
        default_model="scripted/test", workspace_root=workspace, auth_token=auth_token
    )
    store = SessionStore(base_dir=workspace / "sessions")
    return create_app(settings=settings, store=store, agent_factory=_factory)


def _client(workspace: Path, *, auth_token: str | None = None) -> TestClient:
    return TestClient(_make_app(workspace, auth_token=auth_token))


def _seed_research(workspace: Path, journal: str, findings: list[str]) -> None:
    research = workspace / "research"
    research.mkdir(parents=True, exist_ok=True)
    (research / "journal.md").write_text(journal, encoding="utf-8")
    (research / "findings.md").write_text("\n".join(f"- {f}" for f in findings), encoding="utf-8")


def test_workspace_summary_returns_journal_findings_and_session(tmp_path: Path) -> None:
    _seed_research(tmp_path, "## Journal\nline one\n", ["alpha finding", "beta finding"])
    (tmp_path / ".current-session").write_text("sess-123\n", encoding="utf-8")
    body = _client(tmp_path).get("/workspace/summary").json()
    assert body["journal"].startswith("## Journal")
    assert body["finding_count"] == 2
    assert body["session_id"] == "sess-123"


def test_workspace_summary_degrades_when_nothing_written(tmp_path: Path) -> None:
    # No research/ dir, no .current-session — safe to poll before the loop's first turn.
    body = _client(tmp_path).get("/workspace/summary").json()
    assert body == {"journal": "", "finding_count": 0, "session_id": None}


def test_workspace_summary_caps_journal_at_5000_chars(tmp_path: Path) -> None:
    research = tmp_path / "research"
    research.mkdir()
    (research / "journal.md").write_text("x" * 12000, encoding="utf-8")
    body = _client(tmp_path).get("/workspace/summary").json()
    assert len(body["journal"]) == 5000


def test_finding_count_counts_directory_files_excluding_dotfiles(tmp_path: Path) -> None:
    findings = tmp_path / "research" / "findings"
    findings.mkdir(parents=True)
    (findings / "f1.md").write_text("one", encoding="utf-8")
    (findings / "f2.md").write_text("two", encoding="utf-8")
    (findings / ".gitkeep").write_text("", encoding="utf-8")  # dotfile — not a finding
    body = _client(tmp_path).get("/workspace/summary").json()
    assert body["finding_count"] == 2


def test_sidecar_health_reports_active_session(tmp_path: Path) -> None:
    (tmp_path / ".current-session").write_text("sess-777\n", encoding="utf-8")
    body = _client(tmp_path).get("/sidecar/health").json()
    assert body == {"status": "ok", "active_session_id": "sess-777"}


def test_sidecar_health_active_session_none_before_first_turn(tmp_path: Path) -> None:
    body = _client(tmp_path).get("/sidecar/health").json()
    assert body == {"status": "ok", "active_session_id": None}


def test_sidecar_endpoints_require_bearer_when_auth_configured(tmp_path: Path) -> None:
    token = "sidecar-secret"
    (tmp_path / ".current-session").write_text("sess-9\n", encoding="utf-8")
    client = _client(tmp_path, auth_token=token)
    # /sidecar/health is NOT the exempt /health — no token → 401 (proves the distinction).
    assert client.get("/sidecar/health").status_code == 401
    assert client.get("/workspace/summary").status_code == 401
    auth = {"Authorization": f"Bearer {token}"}
    assert client.get("/sidecar/health", headers=auth).status_code == 200
    assert client.get("/workspace/summary", headers=auth).status_code == 200
    assert client.get("/health").status_code == 200  # exempt liveness — still no token needed
