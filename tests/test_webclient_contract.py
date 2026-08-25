"""Contract test (M10-2): the bundled web client agrees with the server's wire.

The web client is a pure viewer + say-writer: it renders the ``?full=1`` watch
stream and writes input — messages, y/a/n permission answers, interrupts — through
the say contract (``POST /say`` / ``POST /interrupt``). These tests guard the
JS↔server contract from drifting: the renderer's declared event-type set must equal
the server's, and the client must speak exactly those contract endpoints. Pure
string/JSON inspection of the shipped asset — no browser needed.
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
    assert "websocket" not in html.lower()  # the WS private pipe is gone — say contract only


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


def test_client_renders_artifact_download_links() -> None:
    html = _html()
    assert "artifactUrl" in html
    assert "/artifacts/" in html
    assert ".download" in html


def test_client_can_upload_files_to_session() -> None:
    html = _html()
    assert 'id="file-input"' in html
    assert 'id="attach"' in html
    assert "/uploads" in html
    assert "readAsDataURL" in html
    assert "uploadFiles" in html


def test_client_speaks_the_say_contract() -> None:
    # Input and approvals both go through POST /say — the ONE contract every
    # surface writes; the page renders its own user rows from the bus's
    # ``user_message`` frames, never from a local echo.
    html = _html()
    assert '"/say"' in html
    assert 'case "user_message":' in html
    # It consumes the announced permission prompt frame from the full watch.
    assert "action_required" in html
    # And answers with the say grammar the server's SayInboxPrompter parses.
    match = re.search(r"OUTCOME_KEYS\s*=\s*\{([^}]*)\}", html)
    assert match, "client must map approval outcomes to say answers"
    from zakcode.permissions import parse_permission_answer

    mapping = dict(re.findall(r'(\w+):\s*"(\w)"', match.group(1)))
    assert mapping, "OUTCOME_KEYS must not be empty"
    for outcome_name, answer in mapping.items():
        parsed = parse_permission_answer(answer)
        assert parsed is not None and parsed.value == outcome_name, (outcome_name, answer)


def test_sse_parser_normalizes_crlf_framing() -> None:
    # sse-starlette's wire default is CRLF line endings (DEFAULT_SEPARATOR), so the
    # page's frame scan on "\n\n" finds nothing unless the buffer is normalized
    # first — the parser rendered ZERO frames until it was (fresh-eyes F-1). httpx's
    # aiter_lines normalizes endings, so only this source-level pin guards the page.
    html = _html()
    assert 'buf = buf.replace(/\\r\\n/g, "\\n");' in html


def test_client_can_interrupt_a_turn() -> None:
    # The Stop button POSTs /interrupt — the contract's sibling control file.
    html = _html()
    assert '"/interrupt"' in html


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
    # the obvious leaks (it should only ever talk to the server over HTTP/SSE).
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
