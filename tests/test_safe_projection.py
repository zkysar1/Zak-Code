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

import io
import re
from types import SimpleNamespace

import pytest
from pydantic import BaseModel
from rich.console import Console

from zakcode.agent.loop import AgentLoop
from zakcode.cli._theme import ZAK_THEME
from zakcode.cli.render import _STOP_LABEL, StreamRenderer, _display_name
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
    SafeUserMessage,
    redact_secrets_extended,
)
from zakcode.tasks import Task, TaskNetwork

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
            request=f"fix the loader for {VIN}",
            tasks=[
                {
                    "id": "1",
                    "title": "run the thing",
                    "status": "running",
                    "kind": "task",
                    "note": f"internal secret {VIN}",
                    "children": [{"id": "1a", "title": "leak me"}],
                }
            ],
        )
    )
    assert isinstance(out, SafeTaskUpdate)
    assert out.tasks == [{"description": "run the thing", "status": "running"}]
    # ADR-0113: the request the plan serves travels, redacted like everything else.
    assert out.request.startswith("fix the loader for ")
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


def test_user_message_marker_projects_to_safe_form_and_redacts() -> None:
    # The watch/talk unification meta-event (the question the driver consumed) projects to
    # its allow-listed SafeUserMessage form; the text is redacted as defense in depth even
    # though it already crossed the gateway's sanitization boundary.
    out = _proj().project(
        SimpleNamespace(event="user_message", text=f"what is this key {GSK} for?")
    )
    assert isinstance(out, SafeUserMessage)
    assert out.event == "user_message"
    assert GSK not in out.text and "what is this key" in out.text


def test_user_message_marker_carries_only_the_text_field() -> None:
    # Whitelist by construction, same property as the rotation marker: only {event, text}
    # escape, whatever else rides on the raw object.
    out = _proj().project(
        SimpleNamespace(event="user_message", text="hi", sender="kid-7", session="sess-1")
    )
    assert isinstance(out, SafeUserMessage)
    dumped = out.model_dump()
    assert dumped == {"event": "user_message", "text": "hi"}
    assert "sender" not in dumped and "session" not in dumped


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


# ── g-366-05: named-vault layer — usage NAMES surfaced, stored VALUES scrubbed ─


def test_tool_call_surfaces_placeholder_names_only() -> None:
    out = _proj().project(
        AgentToolCall(
            id="c2",
            name="web_fetch",
            arguments={
                "url": "https://api.example.com?key={{secret:WEATHER_API_KEY}}",
                "headers": {"Authorization": "Bearer {{secret:API_TOKEN_A}}"},
                "retries": [1, "then {{secret:WEATHER_API_KEY}} again"],
            },
        )
    )
    assert isinstance(out, SafeToolSummary)
    # Sorted, deduplicated NAMES — the "agent used WEATHER_API_KEY" watch signal.
    assert out.used_secrets == ["API_TOKEN_A", "WEATHER_API_KEY"]
    # Arguments themselves still never escape.
    dumped = out.model_dump_json()
    assert "arguments" not in dumped and "api.example.com" not in dumped


def test_tool_call_without_placeholders_has_empty_used_secrets() -> None:
    out = _proj().project(
        # Lowercase name is OUTSIDE the placeholder grammar — must not match.
        AgentToolCall(id="c3", name="bash", arguments={"cmd": "echo {{secret:weather}}"})
    )
    assert isinstance(out, SafeToolSummary) and out.used_secrets == []
    result = _proj().project(AgentToolResult(tool_use_id="c3", output="x", is_error=False))
    assert isinstance(result, SafeToolSummary) and result.used_secrets == []


def test_vault_value_is_scrubbed_from_text_events(tmp_path) -> None:  # type: ignore[no-untyped-def]
    vault = tmp_path / "secrets.json"
    vault.write_text('{"WEATHER_API_KEY": "wombat-wombat-secret-01"}', encoding="utf-8")
    p = _proj(secrets_file=vault)
    out = p.project(AgentTextDelta(text="calling with wombat-wombat-secret-01 now"))
    assert isinstance(out, SafeText)
    # Low-entropy value (below the layer-4 catch-all) — only the vault layer can remove it,
    # folding it back into its placeholder so the watcher still sees WHICH secret.
    assert "wombat-wombat-secret-01" not in out.text
    assert "{{secret:WEATHER_API_KEY}}" in out.text


def test_vault_value_saved_after_init_is_still_scrubbed(tmp_path) -> None:  # type: ignore[no-untyped-def]
    vault = tmp_path / "secrets.json"
    vault.write_text("{}", encoding="utf-8")
    p = _proj(secrets_file=vault)
    # Pre-save: the value passes through untouched (proves the scrub below is the vault's).
    before = p.project(AgentTextDelta(text="value wombat-wombat-secret-02 here"))
    assert isinstance(before, SafeText) and "wombat-wombat-secret-02" in before.text
    # The operator saves a NEW secret while the server is running (no restart, no re-init).
    vault.write_text('{"ROTATED_KEY": "wombat-wombat-secret-02"}', encoding="utf-8")
    after = p.project(AgentTextDelta(text="value wombat-wombat-secret-02 here"))
    assert isinstance(after, SafeText)
    assert "wombat-wombat-secret-02" not in after.text
    assert "{{secret:ROTATED_KEY}}" in after.text


def test_vault_missing_file_is_harmless(tmp_path) -> None:  # type: ignore[no-untyped-def]
    p = _proj(secrets_file=tmp_path / "never-created.json")
    out = p.project(AgentTextDelta(text="ordinary sentence survives"))
    assert isinstance(out, SafeText) and out.text == "ordinary sentence survives"


def test_placeholder_names_survive_redaction_in_text() -> None:
    # The name is the watch stream's public currency — layer 1's assignment heuristic
    # ("secret:NAME" reads as key/value) must not mangle it to {{secret:[REDACTED]}}.
    out = _proj().project(AgentTextDelta(text="calling with {{secret:WEATHER_API_KEY}} now"))
    assert isinstance(out, SafeText)
    assert out.text == "calling with {{secret:WEATHER_API_KEY}} now"


def test_placeholder_shell_cannot_smuggle_an_env_value() -> None:
    # An env secret VALUE that happens to fit the name grammar, wrapped in placeholder
    # syntax, must NOT ride the name-protection through redaction (fail-closed lift guard).
    p = _proj(env={"SNEAKY_KEY": "UPPERCASEVALUE99"})
    out = p.redact("x {{secret:UPPERCASEVALUE99}} y")
    assert "UPPERCASEVALUE99" not in out


# ── ADR-0220: the terminal's sentences, built from counts ────────────────────
#
# Each case below pins a frame against the terminal renderer's OWN function for the same event,
# and plants _MARK in every output, title and note the frame must not carry. The positive
# controls show the terminal does carry it, so a clean frame is a real result, not a blind test.

#: Everything a watcher may be told about a finished tool: a fixed sentence around integers.
#: A receipt outside this grammar is a leak by definition, whatever built it.
_RECEIPT_GRAMMAR = re.compile(
    r"Read \d+ lines?|Listed \d+ entr(?:y|ies)|Fetched (?:\d+ lines?|no output)"
    r"|Ran · (?:\d+ lines?|no output)|Found \d+ (?:match|matches|files?)"
    r"|Updated \+\d+ -\d+|Edited · \d+ lines?|Written|failed"
    r"|Plan complete · \d+ steps?|Plan · (?:\d+/\d+ steps|\d+ steps?|\d+ items?)"
    r"|[A-Z][A-Za-z0-9]* · \d+ lines?"
)
_MARK = "zebra-marmalade"


@pytest.fixture
def terminal(monkeypatch: pytest.MonkeyPatch) -> StreamRenderer:
    monkeypatch.delenv("ZAKCODE_ASCII", raising=False)  # the stream carries the unicode glyphs
    console = Console(
        file=io.StringIO(), force_terminal=False, width=300, no_color=True, theme=ZAK_THEME
    )
    return StreamRenderer(console=console, clock=lambda: 0.0, wall=lambda: 0.0)


def _marked(n: int) -> str:
    return "\n".join(f"{_MARK} line {i}" for i in range(1, n + 1))


_OUTPUTS = [
    "",
    _marked(1),
    _marked(3),
    _marked(13),
    f"--- a/{_MARK}.py\n+++ b/{_MARK}.py\n@@ -1,2 +1,2 @@\n-{_MARK} old\n+{_MARK} new\n ctx",
    f"- {_MARK} bullet\n+ {_MARK} not a diff",
    f"Current plan (1/3 steps done):\n  [x] 1 {_MARK} — done\n"
    f"  [~] 2 {_MARK} — note  <- current\n  [ ] 3 {_MARK}",
    f"Current plan (2/2 steps done):\n  [x] 1 {_MARK}\n  [-] 2 {_MARK}",
    f"Plan updated: 1/3 steps done · current: 2 {_MARK}",
    f"[x] {_MARK}\n[ ] {_MARK}\n",
]
#: One canonical name per display name the terminal knows, and two it title-cases.
_TOOLS = [
    "Read",
    "LS",
    "WebFetch",
    "Bash",
    "Grep",
    "Glob",
    "Edit",
    "Write",
    "update_plan",
    "Skill",
    "mcp__srv__fetch_all",
]


def test_every_receipt_is_the_terminals_counted_never_copied(terminal: StreamRenderer) -> None:
    p = _proj()
    for tool in _TOOLS:
        for output in _OUTPUTS:
            frame = p.project(AgentToolResult(tool_use_id="t", name=tool, output=output))
            assert isinstance(frame, SafeToolSummary) and frame.status == "completed"
            summary, _ = terminal._synthesize_result(
                _display_name(tool), output.splitlines(), is_error=False
            )
            assert summary.plain.startswith("✓ "), "positive control: a success opens with ✓"
            # The terminal's sentence without its ✓, and for a checklist without the
            # "current:" tail, which is the step's row read out of the output.
            expected = summary.plain.removeprefix("✓ ").split(" · current: ")[0]
            assert frame.receipt == expected, (tool, output)
            assert _RECEIPT_GRAMMAR.fullmatch(frame.receipt), frame.receipt
            assert _MARK not in frame.model_dump_json(), (tool, output)
    # Positive control: the terminal's checklist receipt does carry the output's text.
    plan = _OUTPUTS[6].splitlines()
    assert _MARK in terminal._synthesize_result("Todo", plan, is_error=False)[0].plain


def test_a_failed_result_says_failed_and_nothing_of_its_output(terminal: StreamRenderer) -> None:
    for tool in ("Bash", "Edit", "mcp__srv__fetch_all"):
        frame = _proj().project(
            AgentToolResult(tool_use_id="t", name=tool, output=_marked(12), is_error=True)
        )
        assert isinstance(frame, SafeToolSummary)
        assert (frame.status, frame.receipt) == ("failed", "failed")
        assert _MARK not in frame.model_dump_json()
        # Positive control: the terminal's failure receipt IS the output's first line.
        lines = _marked(12).splitlines()
        assert _MARK in terminal._synthesize_result(tool, lines, is_error=True)[0].plain


def test_call_and_result_carry_the_terminals_display_name() -> None:
    p = _proj()
    cases = [
        ("Bash", "Run"),
        ("Grep", "Search"),
        ("read_file", "Read"),
        ("update_plan", "Todo"),
        ("mcp__srv__fetch_all", "McpSrvFetchAll"),
    ]
    for tool, display in cases:
        assert _display_name(tool) == display  # the terminal's own map, not a copy
        call = p.project(AgentToolCall(id="c", name=tool, arguments={"command": _MARK}))
        result = p.project(AgentToolResult(tool_use_id="c", name=tool, output=_MARK))
        assert isinstance(call, SafeToolSummary) and isinstance(result, SafeToolSummary)
        assert (call.name, call.display_name, call.receipt) == (tool, display, "")
        assert (result.name, result.display_name) == (tool, display)
        assert _MARK not in call.model_dump_json() + result.model_dump_json()


def test_a_result_that_names_no_tool_is_receipted_as_the_terminal_does(
    terminal: StreamRenderer,
) -> None:
    # A producer that predates AgentToolResult.name: the terminal calls it "Tool" too.
    frame = _proj().project(AgentToolResult(tool_use_id="t", output=_marked(3)))
    assert isinstance(frame, SafeToolSummary)
    assert (frame.name, frame.display_name, frame.receipt) == ("", "", "Tool · 3 lines")
    lines = _marked(3).splitlines()
    assert terminal._synthesize_result("Tool", lines, is_error=False)[0].plain == (
        "✓ Tool · 3 lines"
    )


def _plan_event(*steps: Task) -> AgentTaskUpdate:
    """The loop's own ``task_update`` for a plan (``AgentLoop._task_update_event``): the bytes
    a served session publishes, never a hand-written render (rb-11490)."""
    network = TaskNetwork()
    network.replace_from_author(list(steps))
    holder = SimpleNamespace(session=SimpleNamespace(task_network=network))
    event = AgentLoop._task_update_event(holder)  # type: ignore[arg-type]
    assert event is not None
    return event


def test_the_plan_line_names_a_top_level_step_in_hand(terminal: StreamRenderer) -> None:
    event = _plan_event(
        Task(title="design", status="done", outcome=f"{_MARK} decided"),
        Task(title="build the parser", status="in_progress", note=f"{_MARK} tests pass"),
        Task(title="test", status="pending"),
    )
    frame = _proj().project(event)
    assert isinstance(frame, SafeTaskUpdate)
    assert (frame.finished, frame.total) == (1, 3)
    assert frame.receipt == "Plan · 1/3 steps · current: 2 build the parser"
    assert _MARK not in frame.model_dump_json()
    # The terminal names the same step by its whole row, note included; the frame keeps
    # its id and title, which ``tasks`` already carries.
    summary = terminal._synthesize_result("Todo", event.plan.splitlines(), is_error=False)[0]
    assert summary.plain == f"✓ Plan · 1/3 steps · current: 2 build the parser — {_MARK} tests pass"


def test_the_plan_line_never_names_a_child_step(terminal: StreamRenderer) -> None:
    event = _plan_event(
        Task(title="design", status="done"),
        Task(
            title="build",
            kind="compound",
            children=[
                Task(title=f"leak me {_MARK}", status="in_progress"),
                Task(title="wire", status="pending"),
            ],
        ),
    )
    frame = _proj().project(event)
    assert isinstance(frame, SafeTaskUpdate)
    assert (frame.finished, frame.total, frame.receipt) == (1, 3, "Plan · 1/3 steps")
    dumped = frame.model_dump_json()
    assert _MARK not in dumped and "leak me" not in dumped
    # Positive control: the terminal's line names the child, which this frame must not.
    assert (
        "leak me"
        in terminal._synthesize_result("Todo", event.plan.splitlines(), is_error=False)[0].plain
    )


def test_a_finished_plan_reads_complete(terminal: StreamRenderer) -> None:
    event = _plan_event(Task(title="a", status="done"), Task(title="b", status="cancelled"))
    frame = _proj().project(event)
    assert isinstance(frame, SafeTaskUpdate)
    summary = terminal._synthesize_result("Todo", event.plan.splitlines(), is_error=False)[0]
    assert frame.receipt == summary.plain.removeprefix("✓ ") == "Plan complete · 2 steps"
    assert (frame.finished, frame.total) == (2, 2)


def test_the_done_label_is_the_terminals_less_what_the_frame_withholds(
    terminal: StreamRenderer,
) -> None:
    for reason in [*_STOP_LABEL, "some_new_reason"]:
        for open_steps in (0, 3):
            for degraded in (False, True):
                event = AgentDone(
                    stop_reason=reason,
                    iterations=4,
                    open_steps=open_steps,
                    degraded=degraded,
                    error=f"{_MARK} rate limited",
                )
                frame = _proj().project(event)
                assert isinstance(frame, SafeDone) and frame.iterations == 4
                # The terminal's label for the same stop with neither withheld field in play.
                bare = AgentDone(stop_reason=reason, iterations=4, open_steps=open_steps)
                assert frame.label == terminal._stop_label(bare)[0], event
                assert _MARK not in frame.model_dump_json()
    # Positive controls: with them in play the terminal's label says more, and this does not.
    struggled = AgentDone(stop_reason="completed", iterations=1, degraded=True, open_steps=2)
    assert terminal._stop_label(struggled)[0] == "done — struggled — 2 plan step(s) left open"
    failed = AgentDone(stop_reason="provider_error", iterations=1, error=f"{_MARK} down")
    assert _MARK in terminal._stop_label(failed)[0]


def test_each_safe_frame_carries_exactly_its_allow_listed_fields() -> None:
    # Widening a public frame is a decision with an ADR (ADR-0220 is the last). This pin makes
    # the next one deliberate: a field added to a Safe model fails here until it is listed.
    models: list[type[BaseModel]] = [
        SafeText,
        SafeStatus,
        SafeToolSummary,
        SafeTaskUpdate,
        SafeDone,
        SafeSessionRotated,
        SafeUserMessage,
    ]
    assert {m.__name__: sorted(m.model_fields) for m in models} == {
        "SafeText": ["event", "text"],
        "SafeStatus": ["event", "message"],
        "SafeToolSummary": [
            "display_name",
            "event",
            "name",
            "receipt",
            "status",
            "used_secrets",
        ],
        "SafeTaskUpdate": ["event", "finished", "receipt", "request", "tasks", "total"],
        "SafeDone": ["event", "iterations", "label", "stop_reason"],
        "SafeSessionRotated": ["event", "reason"],
        "SafeUserMessage": ["event", "text"],
    }
