"""Tests for the Pearl watch-surface Layer-4 filter (zakcode.server.safe_projection).

The projection is the ONLY thing standing between a raw agent event and a kid's browser,
so these tests pin two properties hard: (1) whitelist-by-construction — every raw event type
maps to an explicit allow-listed shape or is DROPPED, and an UNKNOWN type is always dropped
(never passed through); (2) extended redaction removes credential-shaped strings, exact env
secret values, workspace paths, and high-entropy tokens from every text field that escapes.
Bravo audit g-335-41 F2 is the reason for the unknown-type test: a blacklist would fail open
the moment the SDK adds a field or event type.
"""

from __future__ import annotations

from types import SimpleNamespace

from zakcode.events import (
    AgentDone,
    AgentStatus,
    AgentTaskUpdate,
    AgentTextDelta,
    AgentToolCall,
    AgentToolResult,
    AgentUsage,
)
from zakcode.secrets import redact_secrets
from zakcode.server.safe_projection import (
    SafeDone,
    SafeEventProjection,
    SafeSessionRotated,
    SafeStatus,
    SafeTaskUpdate,
    SafeText,
    SafeToolSummary,
    redact_secrets_extended,
)

GSK = "gsk_ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789ab"  # Groq-shaped
VIN = "vin_" + "a1b2c3d4" * 6  # Vinheim-shaped (vin_ + 48 hex)
SK = "sk-ABCDEFGHIJKLMNOP1234567890"  # OpenAI-shaped


def _proj(**kw: object) -> SafeEventProjection:
    # Deterministic projection: no ambient env secrets unless a test passes them.
    kw.setdefault("env", {})
    return SafeEventProjection(**kw)  # type: ignore[arg-type]


# ── Whitelist by construction: the 7 event types + the unknown "+1" ──────────


def test_text_event_is_redacted() -> None:
    p = _proj()
    out = p.project(AgentTextDelta(text=f"my key is {GSK} keep it safe"))
    assert isinstance(out, SafeText)
    assert GSK not in out.text
    # gsk_ is caught by the base redact_secrets (uppercase [REDACTED]); the extended
    # layers use lowercase [redacted]/[path] — both are valid markers, so match either.
    assert "[redacted]" in out.text.lower() and "keep it safe" in out.text


def test_status_event_passes_through_redacted() -> None:
    out = _proj().project(AgentStatus(message="starting iteration 3"))
    assert isinstance(out, SafeStatus)
    assert out.message == "starting iteration 3"


def test_tool_call_strips_arguments_entirely() -> None:
    out = _proj().project(
        AgentToolCall(id="c1", name="read_file", arguments={"path": "/etc/shadow", "token": SK})
    )
    assert isinstance(out, SafeToolSummary)
    assert out.name == "read_file" and out.status == "running"
    dumped = out.model_dump_json()
    assert "/etc/shadow" not in dumped and SK not in dumped and "arguments" not in dumped


def test_tool_result_strips_output_and_reports_status() -> None:
    ok = _proj().project(AgentToolResult(tool_use_id="t1", output=f"secret {SK}", is_error=False))
    assert isinstance(ok, SafeToolSummary) and ok.status == "completed"
    assert SK not in ok.model_dump_json() and "output" not in ok.model_dump_json()
    err = _proj().project(AgentToolResult(tool_use_id="t2", output="boom", is_error=True))
    assert isinstance(err, SafeToolSummary) and err.status == "failed"


def test_task_update_allowlists_description_and_status() -> None:
    out = _proj().project(
        AgentTaskUpdate(
            tasks=[
                {
                    "id": "1",
                    "title": "run the thing",
                    "status": "running",
                    "kind": "task",
                    "note": f"internal secret {VIN}",
                    "children": [{"id": "1a", "title": "leak me"}],
                }
            ]
        )
    )
    assert isinstance(out, SafeTaskUpdate)
    assert out.tasks == [{"description": "run the thing", "status": "running"}]
    dumped = out.model_dump_json()
    assert VIN not in dumped and "leak me" not in dumped and "note" not in dumped


def test_usage_event_is_dropped() -> None:
    assert _proj().project(AgentUsage()) is None


def test_done_strips_internals() -> None:
    out = _proj().project(
        AgentDone(stop_reason="completed", iterations=3, error=f"trace with {SK}", degraded=True)
    )
    assert isinstance(out, SafeDone) and out.stop_reason == "completed"
    dumped = out.model_dump_json()
    assert (
        SK not in dumped
        and "error" not in dumped
        and "trace" not in dumped
        and "degraded" not in dumped
    )


def test_unknown_event_type_is_dropped_not_passed_through() -> None:
    # Bravo F2: the whitelist MUST drop anything it does not recognize — a future SDK event
    # type or the permission-approval flow must never reach the wire.
    assert _proj().project(SimpleNamespace(event="action_required", prompt="approve?")) is None
    assert _proj().project(SimpleNamespace(event="some_future_event_type", secret=SK)) is None
    assert _proj().project(SimpleNamespace(event=None)) is None


def test_session_rotated_marker_projects_to_safe_form_and_redacts() -> None:
    # The watch meta-event (a driver session rotation) projects to its allow-listed
    # SafeSessionRotated form; its reason is secret-redacted like every escaping text field.
    out = _proj().project(
        SimpleNamespace(event="session_rotated", reason=f"daemon restarted with {GSK}")
    )
    assert isinstance(out, SafeSessionRotated)
    assert out.event == "session_rotated"
    assert GSK not in out.reason and "daemon restarted" in out.reason


def test_session_rotated_marker_carries_no_ids_or_extra_fields() -> None:
    # Whitelist by construction: even if the raw marker smuggles a session id or other field,
    # only the allow-listed {event, reason} escape — the projection reads nothing else.
    out = _proj().project(
        SimpleNamespace(event="session_rotated", reason="rotated", new_sid="sess-secret", cursor=9)
    )
    assert isinstance(out, SafeSessionRotated)
    dumped = out.model_dump()
    assert dumped == {"event": "session_rotated", "reason": "rotated"}
    assert "new_sid" not in dumped and "cursor" not in dumped


# ── Extended redaction (redact_secrets_extended + P0-6 base) ─────────────────


def test_p0_6_base_redact_secrets_now_catches_gsk_and_vin() -> None:
    # Regression for P0-6: the base regex missed gsk_/vin_ before this change.
    assert (
        redact_secrets(f"here {GSK} ok")[1] >= 1 and GSK not in redact_secrets(f"here {GSK} ok")[0]
    )
    assert (
        redact_secrets(f"here {VIN} ok")[1] >= 1 and VIN not in redact_secrets(f"here {VIN} ok")[0]
    )


def test_extended_redacts_all_token_shapes() -> None:
    for tok in (GSK, VIN, SK):
        assert tok not in redact_secrets_extended(f"leading {tok} trailing")


def test_extended_matches_exact_env_values() -> None:
    # A secret VALUE that is NOT credential-shaped is still caught by value-matching.
    p = _proj(env={"MY_API_KEY": "plainlookingbutsecret123"})
    out = p.redact("the value plainlookingbutsecret123 appears here")
    assert "plainlookingbutsecret123" not in out and "[redacted]" in out


def test_extended_strips_workspace_paths() -> None:
    p = _proj(workspace_root="/home/agent/loopws")
    out = p.redact("reading /home/agent/loopws/research/journal.md now")
    assert "/home/agent/loopws" not in out and "[path]" in out


def test_high_entropy_token_redacted_but_normal_text_survives() -> None:
    secret = "Xq7Zp2Rk9Vt4Bn6Ls8Wc3Yd5Ef1Gh0Ij"  # 33 chars, random -> high entropy
    out = redact_secrets_extended(f"opaque {secret} value")
    assert secret not in out and "[redacted]" in out
    # A git SHA (hex, entropy ~4.0) and ordinary prose must NOT be redacted.
    sha = "9f8e7d6c5b4a3f2e1d0c9b8a7f6e5d4c3b2a1f0e"
    prose = redact_secrets_extended(f"commit {sha} fixes the login bug in the handler")
    assert sha in prose and "[redacted]" not in prose


def test_redaction_never_raises_on_empty() -> None:
    assert redact_secrets_extended("") == ""
    assert _proj().project(AgentTextDelta(text="")) == SafeText(text="")
