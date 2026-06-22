"""Secret redaction at exposure boundaries.

``docs/GUARDRAILS.md`` §6 requires that credentials stay out of the model context,
out of logs, and out of long-lived state. :func:`redact_secrets` is the small,
dependency-free guard applied where untrusted-or-derived text crosses such a
boundary — e.g. a provider error message before it reaches the user/logs
(:mod:`zakcode.providers.litellm_provider`), or any host-supplied context a hook
folds into the prompt. It is intentionally conservative — it targets credential
*shapes* (API keys, AWS/GitHub/Slack tokens, PEM private keys, ``key = value``
secret assignments), not arbitrary prose — so ordinary text passes through untouched.

This is a defense-in-depth heuristic, not a vault: it reduces the chance a secret is
accidentally echoed or persisted, but the primary rule remains "never put a secret
into durable state in the first place."
"""

from __future__ import annotations

import os
import re
from collections.abc import Mapping

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

# URL userinfo: ``scheme://user:password@host`` — mask the credentials, keep scheme + host. So a
# provider error echoing a credentialed api_base (e.g. a gateway URL) can't leak the password.
_URL_CRED_RE = re.compile(r"://[^/\s:@]+:[^/\s@]+@")


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

    def _mark_url(_m: re.Match[str]) -> str:
        nonlocal count
        count += 1
        return "://***@"

    text = _PEM_RE.sub(_mark_pem, text)
    text = _URL_CRED_RE.sub(_mark_url, text)
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


__all__ = ["provider_key_env_names", "redact_secrets", "strip_url_credentials"]


# ── subprocess env hygiene (RISKS: provider keys reach subprocesses) ──────────

#: Exact provider/service key variable names scrubbed from subprocess environments,
#: in addition to the ``*_API_KEY`` suffix rule in :func:`provider_key_env_names`.
_PROVIDER_KEY_ENV_EXACT = frozenset(
    {"OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GROQ_API_KEY", "TAVILY_API_KEY"}
)


def provider_key_env_names(environ: Mapping[str, str] | None = None) -> list[str]:
    """Names of provider-credential variables present in ``environ`` (default os.environ).

    Matches the known exact names plus the ``*_API_KEY`` suffix convention, so a newly
    added provider's key is scrubbed without a code change. Used to build the env-scrub
    list handed to subprocess tools (GUARDRAILS §6; opt out via
    ``ZAKCODE_SUBPROCESS_INHERIT_PROVIDER_KEYS=true``).

    DELIBERATELY NARROW (stack review minor #3): workflow credentials such as
    ``AWS_SECRET_ACCESS_KEY``/``AWS_SESSION_TOKEN`` are NOT scrubbed. The scrub
    targets MODEL-provider keys whose presence in the env is zakcode's own doing
    (``.env`` loading) and which no agent-run script should need; AWS/cloud creds are
    operator-managed workflow credentials, and agent-run CLIs (``aws s3 ...``) using
    them is a first-class use case in this household — and the opt-out is global, so
    scrubbing them by default would force all-or-nothing. Tighten per-deployment with
    the egress controls instead; revisit if a per-name opt-out ever lands.
    """
    env = os.environ if environ is None else environ
    return sorted(n for n in env if n in _PROVIDER_KEY_ENV_EXACT or n.endswith("_API_KEY"))
