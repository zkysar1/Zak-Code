"""Secret redaction at persistence boundaries.

``docs/GUARDRAILS.md`` §6 requires that credentials stay out of the model context
and out of long-lived state files. Cross-session memory is exactly such a state
file *and* it re-injects its contents into the prompt on recall, so anything stored
there must be scrubbed first. :func:`redact_secrets` is the small, dependency-free
guard applied at those boundaries (the ``remember`` write path and the recall
render path). It is intentionally conservative — it targets credential *shapes*
(API keys, AWS/GitHub/Slack tokens, PEM private keys, ``key = value`` secret
assignments), not arbitrary prose — so ordinary notes pass through untouched.

This is a defense-in-depth heuristic, not a vault: it reduces the chance a secret
is accidentally persisted/echoed, but the primary rule remains "never put a secret
in memory in the first place."
"""

from __future__ import annotations

import re

_REDACTED = "[REDACTED]"

# A whole PEM private-key block (header through footer), collapsed to a marker.
_PEM_RE = re.compile(
    r"-----BEGIN[^-]*PRIVATE KEY-----.*?-----END[^-]*PRIVATE KEY-----",
    re.DOTALL,
)

# Standalone high-signal credential tokens (provider-prefixed, so low false-positive).
_TOKEN_RE = re.compile(
    r"\b("
    r"sk-[A-Za-z0-9_-]{16,}"  # OpenAI-style
    r"|AKIA[0-9A-Z]{16}"  # AWS access key id
    r"|gh[pousr]_[A-Za-z0-9]{20,}"  # GitHub tokens
    r"|xox[baprs]-[A-Za-z0-9-]{10,}"  # Slack tokens
    r")\b"
)

# ``secret = value`` / ``api_key: value`` style assignments — keep the key, drop the value.
_ASSIGN_RE = re.compile(
    r"(?i)\b(api[_-]?key|secret|access[_-]?token|auth[_-]?token|token|password|passwd|bearer)\b"
    r"(\s*[:=]\s*)"
    r"(['\"]?)([A-Za-z0-9._\-/+]{8,})\3"
)


def redact_secrets(text: str) -> tuple[str, int]:
    """Return ``(scrubbed_text, num_redactions)``.

    Replaces credential-shaped spans with a redaction marker, preserving the
    surrounding text (and, for assignments, the key name). Never raises; returns the
    input unchanged with a count of 0 when nothing matches.
    """
    if not text:
        return text, 0
    count = 0

    def _mark(_m: re.Match[str]) -> str:
        nonlocal count
        count += 1
        return _REDACTED

    def _mark_pem(_m: re.Match[str]) -> str:
        nonlocal count
        count += 1
        return "[REDACTED PRIVATE KEY]"

    def _mark_assign(m: re.Match[str]) -> str:
        nonlocal count
        count += 1
        return f"{m.group(1)}{m.group(2)}{_REDACTED}"

    text = _PEM_RE.sub(_mark_pem, text)
    text = _TOKEN_RE.sub(_mark, text)
    text = _ASSIGN_RE.sub(_mark_assign, text)
    return text, count


def strip_url_credentials(url: str | None) -> str | None:
    """Mask any ``user:password@`` userinfo in a URL's authority.

    So an endpoint URL with embedded credentials (RFC-3986 userinfo, e.g.
    ``https://user:TOKEN@host/v1``) is never displayed or serialized verbatim — the host
    and rest of the URL are preserved, only the credentials are masked to ``***@``. Returns
    the input unchanged when there is no userinfo or it cannot be parsed. (audit3 #7)
    """
    if not url or "@" not in url:
        return url
    try:
        from urllib.parse import urlsplit, urlunsplit

        parts = urlsplit(url)
        if "@" not in parts.netloc:
            return url
        host = parts.netloc.rsplit("@", 1)[1]
        return urlunsplit(parts._replace(netloc=f"***@{host}"))
    except Exception:  # noqa: BLE001 — redaction must never raise; fall back to the input
        return url


__all__ = ["redact_secrets", "strip_url_credentials"]
