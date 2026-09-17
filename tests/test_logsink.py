"""ADR-0186: log records are transcript lines, never raw output — and never carry a key."""

from __future__ import annotations

import io
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

import pytest
from rich.console import Console

from zakcode.cli import _configure_logging
from zakcode.cli._theme import ZAK_THEME
from zakcode.cli.logsink import (
    RedactingFilter,
    TranscriptLogHandler,
    install_transcript_logging,
    log_file_path,
    transcript_line,
)

_HTTPX_LINE = (
    "HTTP Request: POST https://us-central1-aiplatform.googleapis.com/v1/projects/my-proj/"
    "locations/us-central1/publishers/google/models/gemini-3.8-flash:streamGenerateContent"
    '?alt=sse&key=AIzaSyFAKEFAKEFAKEFAKEFAKEFAKE "HTTP/1.1 200 OK"'
)
# litellm's per-call record, verbatim (a leading newline, then the line) — reproduced with a
# mocked completion: it is what repeated under every call in the cockpit.
_LITELLM_CALL = "\nLiteLLM completion() model= gemini-3.8-flash; provider = vertex_ai"
_LITELLM_CHATTER = "Wrapper: Completed Call, calling success_handler"


def _record(name: str, msg: str, level: int = logging.INFO, *args: object) -> logging.LogRecord:
    return logging.LogRecord(name, level, __file__, 1, msg, args, None)


@pytest.fixture
def _restore_root_logging():
    """Snapshot/restore the root logger: both entry points below replace its handlers."""
    root = logging.getLogger()
    saved_handlers = root.handlers[:]
    saved_level = root.level
    try:
        yield
    finally:
        for handler in root.handlers[:]:
            if handler not in saved_handlers:
                root.removeHandler(handler)
                handler.close()
        root.handlers[:] = saved_handlers
        root.setLevel(saved_level)


def test_the_filter_scrubs_a_key_carried_in_a_logged_url() -> None:
    record = _record("httpx", _HTTPX_LINE)
    assert RedactingFilter().filter(record) is True
    assert "AIzaSy" not in record.getMessage()
    assert "key=***" in record.getMessage()
    assert "gemini-3.8-flash:streamGenerateContent" in record.getMessage()  # the endpoint survives
    # idempotent: the same record passes a second handler's filter unchanged
    again = record.getMessage()
    RedactingFilter().filter(record)
    assert record.getMessage() == again


def test_the_filter_scrubs_arguments_in_place_and_keeps_their_shape() -> None:
    record = _record(
        "zakcode.x",
        "token=%s for %s (%d)",
        logging.INFO,
        "sk-FAKEFAKE12345",
        "u",
        3,
    )
    RedactingFilter().filter(record)
    assert "sk-FAKEFAKE12345" not in record.getMessage()
    assert record.getMessage().endswith(" for u (3)")
    # caplog-style readers still see the arguments (a test elsewhere asserts on them)
    assert len(record.args) == 3 and record.args[1:] == ("u", 3)
    mapping = _record("zakcode.x", "url=%(url)s", logging.INFO, {"url": "https://h/p?key=SECRET9"})
    RedactingFilter().filter(mapping)
    assert mapping.getMessage() == "url=https://h/p?key=***"


def test_compact_lines_keep_the_model_call_and_drop_the_noise() -> None:
    # the per-call line the operator asked to keep: model and provider, on ONE line
    assert transcript_line(_record("LiteLLM", _LITELLM_CALL)) == (
        "call gemini-3.8-flash · vertex_ai"
    )
    assert transcript_line(_record("httpx", _HTTPX_LINE)) == (
        "POST gemini-3.8-flash:streamGenerateContent · 200 OK"
    )
    # third-party INFO chatter is file-only; its WARNINGs still print
    assert transcript_line(_record("LiteLLM", _LITELLM_CHATTER)) is None
    assert transcript_line(_record("httpx", "something else")) is None
    assert transcript_line(_record("LiteLLM", "rate limited", logging.WARNING)) == (
        "LiteLLM: rate limited"
    )
    # zakcode's own records print, prefix dropped, whitespace collapsed
    own = _record("zakcode.agent.loop", "turn stopped: budget exhausted (cost=$0.0100, tokens=9)")
    assert transcript_line(own) == (
        "agent.loop: turn stopped: budget exhausted (cost=$0.0100, tokens=9)"
    )
    assert transcript_line(_record("root", "plain")) == "plain"
    assert transcript_line(_record("zakcode.x", "first\n   second")) == "x: first second"


def test_records_print_as_grey_rows_at_the_receipt_column() -> None:
    buffer = io.StringIO()
    console = Console(file=buffer, force_terminal=False, width=100, no_color=True, theme=ZAK_THEME)
    handler = TranscriptLogHandler(console)
    handler.addFilter(RedactingFilter())
    handler.emit(_record("LiteLLM", _LITELLM_CALL))
    handler.emit(_record("LiteLLM", _LITELLM_CHATTER))
    handler.emit(_record("httpx", _HTTPX_LINE))
    handler.emit(_record("zakcode.x", "careful", logging.WARNING))
    out = buffer.getvalue().splitlines()
    assert out[0].startswith("    · call gemini-3.8-flash · vertex_ai")
    assert out[1].startswith("    · POST gemini-3.8-flash:streamGenerateContent · 200 OK")
    assert out[2].startswith("    · x: careful")
    assert len(out) == 3, out  # the chatter never printed
    assert "AIzaSy" not in buffer.getvalue()
    assert "INFO" not in buffer.getvalue() and "2026-" not in buffer.getvalue()


def test_records_are_styled_by_level_not_dimmed() -> None:
    buffer = io.StringIO()
    console = Console(
        file=buffer, force_terminal=True, color_system="256", width=100, theme=ZAK_THEME
    )
    handler = TranscriptLogHandler(console)
    handler.emit(_record("zakcode.x", "quiet"))
    handler.emit(_record("zakcode.x", "loud", logging.ERROR))
    lines = buffer.getvalue().splitlines()
    assert "\x1b[38;5;242m" in lines[0]  # grey by index
    assert "\x1b[2m" not in lines[0]  # never the dim attribute
    assert "\x1b[1;31m" in lines[1] or "\x1b[31m" in lines[1]  # err is red


def test_the_interactive_cli_routes_logging_into_the_transcript_and_a_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, _restore_root_logging: None
) -> None:
    monkeypatch.setenv("ZAKCODE_HOME", str(tmp_path))
    monkeypatch.delenv("ZAKCODE_LOG_LEVEL", raising=False)
    _configure_logging()  # the entry point's daemon shape: a stdout handler
    buffer = io.StringIO()
    console = Console(file=buffer, force_terminal=False, width=100, no_color=True, theme=ZAK_THEME)
    install_transcript_logging(console)
    root = logging.getLogger()
    kinds = {type(h) for h in root.handlers}
    assert kinds == {RotatingFileHandler, TranscriptLogHandler}, kinds
    assert root.level == logging.INFO
    logging.getLogger("LiteLLM").info(_LITELLM_CALL)
    logging.getLogger("LiteLLM").info(_LITELLM_CHATTER)
    logging.getLogger("zakcode.providers.test").info(_HTTPX_LINE)
    transcript = buffer.getvalue()
    assert "    · call gemini-3.8-flash · vertex_ai" in transcript
    assert "Wrapper" not in transcript
    assert "AIzaSy" not in transcript and "key=***" in transcript
    for handler in root.handlers:
        handler.flush()
    assert log_file_path() == tmp_path / "logs" / "zakcode.log"
    logged = log_file_path().read_text(encoding="utf-8")
    assert "key=***" in logged and "AIzaSy" not in logged
    assert "INFO LiteLLM" in logged and _LITELLM_CHATTER in logged  # the file keeps everything


def test_the_daemon_stdout_handler_scrubs_keys_too(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch, _restore_root_logging: None
) -> None:
    monkeypatch.delenv("ZAKCODE_LOG_LEVEL", raising=False)
    _configure_logging()
    # The daemon shape leaves per-logger levels alone; the scrub applies to whatever does
    # reach stdout.
    logging.getLogger("zakcode.providers.test").info(_HTTPX_LINE)
    out = capsys.readouterr().out
    assert "key=***" in out and "AIzaSy" not in out
