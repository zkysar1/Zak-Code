"""Where log records go when a transcript owns the terminal (ADR-0186).

``zakcode`` configures stdlib logging at INFO for every command, and the interactive CLI
used to inherit the daemon's shape: timestamped records straight to stdout. litellm's own
``LiteLLM`` logger propagates to the root (``LITELLM_LOG`` only levels litellm's OWN
handler), so every model call put a two-line record in the cockpit — ``INFO LiteLLM`` /
``LiteLLM completion() model= gemini-3.8-flash; provider = vertex_ai`` — and a real call adds
``Wrapper: Completed Call, calling success_handler``: full contrast, unindented, outside the
column grid. The operator asked to keep the per-call line (it says which model was called)
but wanted it quiet.

Here a record becomes a transcript line in the grid — a ``·`` row under the current block,
grey for INFO, ``warn`` / ``err`` for the rest, credentials scrubbed — and the full record
goes to a rotating file under ``~/.zakcode/logs`` instead of stdout. Third-party INFO
chatter that is not the per-call line stays in the file; zakcode's own records and every
WARNING and above print. The scrub matters even for loggers that are quiet today: httpx logs
the FULL request URL whenever it does log, and some providers carry the API key in it.
"""

from __future__ import annotations

import logging
import os
import re
from logging.handlers import RotatingFileHandler
from pathlib import Path
from urllib.parse import urlsplit

from rich.console import Console
from rich.text import Text

from zakcode.cli._glyphs import resolve_glyphs
from zakcode.cli._layout import block
from zakcode.config import zakcode_home
from zakcode.secrets import redact_secrets, redact_url_credentials

LOG_LEVEL_ENV = "ZAKCODE_LOG_LEVEL"

_HTTPX_REQUEST_RE = re.compile(
    r'^HTTP Request: (?P<method>[A-Z]+) (?P<url>\S+) "(?P<proto>HTTP/[\d.]+) '
    r'(?P<status>\d{3})(?: (?P<reason>[^"]*))?"'
)
_LITELLM_CALL_RE = re.compile(
    r"LiteLLM completion\(\) model= ?(?P<model>[^;\s]+); provider ?= ?(?P<provider>\S+)"
)


def log_level_from_env() -> int:
    """``ZAKCODE_LOG_LEVEL`` as a level number; an unrecognised name means INFO, never an error."""
    name = os.environ.get(LOG_LEVEL_ENV, "INFO").strip().upper()
    level = logging.getLevelName(name)
    return level if isinstance(level, int) else logging.INFO


def _scrub(text: str) -> str:
    return redact_secrets(redact_url_credentials(text))[0]


class RedactingFilter(logging.Filter):
    """Scrub credentials out of every record before any handler formats it.

    Rewrites the record in place so the file, stdout and transcript handlers all see the
    same scrubbed text. The format string and each string argument are scrubbed on their
    own first, which keeps the argument tuple's shape for other readers of the record
    (``caplog``, a structured handler). A credential that is only recognisable with its
    context — ``token=%s`` around a bare value — shows up once the record is rendered; then
    the rendered, scrubbed text becomes the message and the arguments are consumed.
    Idempotent for a record that passes through more than one handler.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        if isinstance(record.msg, str):
            record.msg = _scrub(record.msg)
        args = record.args
        if isinstance(args, tuple):
            record.args = tuple(_scrub(a) if isinstance(a, str) else a for a in args)
        elif isinstance(args, dict):
            record.args = {k: _scrub(v) if isinstance(v, str) else v for k, v in args.items()}
        try:
            rendered = record.getMessage()
        except Exception:  # noqa: BLE001 — a bad format string must not lose the record
            record.msg = _scrub(str(record.msg))
            record.args = ()
            return True
        scrubbed = _scrub(rendered)
        if scrubbed != rendered:
            record.msg = scrubbed
            record.args = ()
        return True


def _is_own(record: logging.LogRecord) -> bool:
    return record.name in ("root", "zakcode") or record.name.startswith("zakcode.")


def transcript_line(record: logging.LogRecord) -> str | None:
    """The compact one-line form of a record for the transcript, or None for file-only.

    litellm's call record reads ``call gemini-3.8-flash · vertex_ai`` — the model call the
    operator wanted to keep seeing, on one line. An httpx request (when that logger is not
    pinned quiet) reads ``POST gemini-3.8-flash:streamGenerateContent · 200 OK``. zakcode's
    own records read ``<logger>: <message>`` with the ``zakcode.`` prefix dropped (the root
    logger's carry no name), whitespace collapsed so no record breaks the grid. Any other
    third-party record prints only at WARNING and above: ``Wrapper: Completed Call, calling
    success_handler`` is chatter, and the file keeps it.
    """
    message = " ".join(record.getMessage().split())
    if record.name == "httpx":
        match = _HTTPX_REQUEST_RE.match(message)
        if match:
            parts = urlsplit(match["url"])
            leaf = parts.path.rstrip("/").rsplit("/", 1)[-1] or parts.netloc
            tail = f"{match['status']} {match['reason'] or ''}".strip()
            return f"{match['method']} {leaf} · {tail}"
    elif record.name.startswith("LiteLLM"):
        match = _LITELLM_CALL_RE.search(message)
        if match:
            return f"call {match['model']} · {match['provider']}"
    if not _is_own(record) and record.levelno < logging.WARNING:
        return None
    name = record.name
    if name.startswith("zakcode."):
        name = name[len("zakcode.") :]
    if not name or name in ("root", "zakcode"):
        return message
    return f"{name}: {message}"


class TranscriptLogHandler(logging.Handler):
    """Print each record that has a transcript form as a quiet ``·`` row at indent 4."""

    def __init__(self, console: Console) -> None:
        super().__init__()
        self.console = console
        self._glyphs = resolve_glyphs(console)

    def emit(self, record: logging.LogRecord) -> None:
        try:
            line = transcript_line(record)
            if line is None:
                return
            if record.levelno >= logging.ERROR:
                style = "err"
            elif record.levelno >= logging.WARNING:
                style = "warn"
            else:
                style = "log"
            line = line.replace(" · ", f" {self._glyphs['dot']} ")
            self.console.print(
                block(
                    self.console,
                    Text(line, style=style),
                    marker=self._glyphs["dot"],
                    marker_style=style,
                    indent=4,
                )
            )
        except Exception:  # noqa: BLE001 — logging must never take the transcript down
            self.handleError(record)


def log_file_path() -> Path:
    return zakcode_home() / "logs" / "zakcode.log"


def install_transcript_logging(console: Console) -> None:
    """Route this process's logging into the transcript and a rotating file.

    Replaces whatever the console-script entry configured (the daemon's stdout shape) with
    the two handlers a transcript wants: the full record, scrubbed, into
    ``~/.zakcode/logs/zakcode.log`` (2 MB × 3), and the compact scrubbed line into the
    console as a grey ``·`` row. A home directory that cannot be written costs the file,
    never the session. ``ZAKCODE_LOG_LEVEL=WARNING`` hides the per-call line.
    """
    root = logging.getLogger()
    root.setLevel(log_level_from_env())
    for handler in list(root.handlers):
        root.removeHandler(handler)
    scrub = RedactingFilter()
    try:
        path = log_file_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            path, maxBytes=2_000_000, backupCount=3, encoding="utf-8"
        )
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
        )
        file_handler.addFilter(scrub)
        root.addHandler(file_handler)
    except OSError:
        pass
    transcript = TranscriptLogHandler(console)
    transcript.addFilter(scrub)
    root.addHandler(transcript)


__all__ = [
    "LOG_LEVEL_ENV",
    "RedactingFilter",
    "TranscriptLogHandler",
    "install_transcript_logging",
    "log_file_path",
    "log_level_from_env",
    "transcript_line",
]
