"""Contract test (M10-2): the bundled web client agrees with the server's wire.

The web client is a thin renderer of the AgentEvent stream. These tests guard the
JS↔server contract from drifting: the renderer's declared event-type set must equal
the server's, and the client must speak the WebSocket protocol verbs (input /
approval) the server's permission bridge expects. Pure string/JSON inspection of the
shipped asset — no browser needed.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from zakcode.server.wire import event_type_names

_STATIC = Path(__file__).resolve().parents[1] / "src" / "zakcode" / "server" / "static"
_INDEX = _STATIC / "index.html"


def _html() -> str:
    return _INDEX.read_text(encoding="utf-8")


def test_index_exists_and_is_html() -> None:
    assert _INDEX.is_file(), f"missing bundled web client at {_INDEX}"
    html = _html()
    assert "<!doctype html>" in html.lower()
    assert "websocket" in html.lower()


def test_event_types_match_server() -> None:
    # Extract the EVENT_TYPES = [ ... ] array literal from the JS and compare to the
    # server's authoritative discriminator set. This is the core anti-drift check.
    html = _html()
    match = re.search(r"EVENT_TYPES\s*=\s*\[([^\]]*)\]", html)
    assert match, "web client must declare an EVENT_TYPES array"
    declared = sorted(re.findall(r'"([a-z_]+)"', match.group(1)))
    assert declared == event_type_names(), (declared, event_type_names())


def test_client_handles_every_event_type() -> None:
    # Each server event type must have a matching `case "<type>":` in renderEvent.
    html = _html()
    for name in event_type_names():
        assert f'case "{name}":' in html, f"renderer has no case for event '{name}'"


def test_client_speaks_ws_protocol_verbs() -> None:
    # The client must send the exact WS client-message shapes the server parses:
    # {type:"input"} to start a turn and {type:"approval", outcome:...} for the
    # permission bridge. (Asserted as substrings of the shipped JS.)
    html = _html()
    assert '"type": "input"' in html or 'type: "input"' in html
    assert '"type": "approval"' in html or 'type: "approval"' in html
    # And it consumes the server's permission prompt frame.
    assert "action_required" in html
    # The approval outcomes it sends are valid PermissionOutcome values.
    from zakcode.permissions import PermissionOutcome

    valid = {o.value for o in PermissionOutcome}
    sent = set(re.findall(r'sendApproval\("([a-z_]+)"\)', html))
    assert sent, "client must send at least one approval outcome"
    assert sent <= valid, (sent, valid)


def test_client_can_interrupt_a_turn() -> None:
    # The Stop button sends the WS interrupt frame the server's bridge parses.
    html = _html()
    assert '"type": "interrupt"' in html or 'type: "interrupt"' in html


def test_client_surfaces_unknown_events_and_frames() -> None:
    # The literal default: arms are the contract-drift tripwire: an unknown event or
    # control frame must surface visibly in the transcript, never drop silently.
    html = _html()
    assert html.count("default:") >= 2
    assert "[unknown event: " in html
    assert "[unknown frame: " in html


def test_client_handles_every_control_frame_literally() -> None:
    # onFrame dispatches the server's control frames with literal case arms (no
    # programmatic dispatch — the regex-able source IS the contract surface).
    html = _html()
    for arm in ('case "action_required":', 'case "error":', 'case "status":'):
        assert arm in html, f"onFrame has no literal arm {arm}"


def test_client_sends_exactly_the_three_approval_outcomes() -> None:
    # The three approval buttons map 1:1 onto the outcomes the bridge accepts.
    html = _html()
    sent = set(re.findall(r'sendApproval\("([a-z_]+)"\)', html))
    assert sent == {"allow_once", "allow_session", "deny_once"}, sent


def test_dom_writes_are_textcontent_only() -> None:
    # Server/model data must never hit innerHTML (markup-injection surface).
    assert "innerhtml" not in _html().lower()


def test_client_is_self_contained() -> None:
    # No CDN, no external fonts/scripts/styles: the bundled asset works offline.
    html = _html().lower()
    assert "<link" not in html
    assert "<script src" not in html
    assert "@import" not in html


def test_client_creates_session_via_rest() -> None:
    # The thin client bootstraps a session through the documented REST endpoint.
    html = _html()
    assert '"/sessions"' in html
    assert 'method: "POST"' in html


def test_no_agent_logic_leaked_into_client() -> None:
    # A thin renderer must not embed provider/model/agent-loop logic. Guard against
    # the obvious leaks (it should only ever talk to the server over HTTP/WS).
    html = _html().lower()
    for banned in ("litellm", "openai", "api_key", "system prompt", "tool registry"):
        assert banned not in html, f"web client must not contain '{banned}'"


def test_events_schema_is_valid_json_schema_shape() -> None:
    # Sanity: the published schema (what a generated client would consume) round-trips
    # as JSON and is non-trivial.
    from zakcode.server.wire import events_schema

    schema = events_schema()
    json.dumps(schema)  # must be JSON-serializable
    assert schema  # non-empty
