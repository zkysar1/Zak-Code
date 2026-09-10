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
    r"|gsk_[A-Za-z0-9_-]{16,}"  # Groq (prefix `gsk_`; the leading `g` breaks a bare `sk-` match)
    r"|vin_[A-Za-z0-9_-]{16,}"  # Vinheim product keys
    r"|AKIA[0-9A-Z]{16}"  # AWS access key id
    r"|gh[pousr]_[A-Za-z0-9]{20,}"  # GitHub tokens
    r"|xox[baprs]-[A-Za-z0-9-]{10,}"  # Slack tokens
    r"|ya29\.[A-Za-z0-9._-]{30,}"  # Google OAuth2 access token (gcloud auth print-access-token)
    r"|AIza[0-9A-Za-z_-]{35}"  # Google API key
    r"|eyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}"  # JWT (three b64url segments)
    r")\b"
)

# ``secret = value`` / ``api_key: value`` style assignments — keep the key, drop the value.
_ASSIGN_RE = re.compile(
    r"(?i)(['\"]?)\b(api[_-]?key|secret|access[_-]?token|auth[_-]?token|token|password|passwd|bearer)\b"
    r"\1(\s*[:=]\s*)"
    r"(['\"]?)([A-Za-z0-9._\-/+]{8,})\4"
)

# URL userinfo: ``scheme://user[:password]@host`` — mask the WHOLE userinfo (a user-only token,
# ``user:password``, or a password containing ``@``) up to the LAST @ before the host, keeping
# scheme + host. So a provider error echoing a credentialed api_base (gateway URL) can't leak a
# token. The greedy ``[^/\s]+`` stops at a path ``/`` or whitespace, so a non-credential ``@`` in a
# path or free-form prose (no ``://``) is left untouched.
_URL_CRED_RE = re.compile(r"://[^/\s]+@")

# ── the credential-VALUE layer (ADR-0125) ────────────────────────────────────────────
# The token layer above knows PROVIDER PREFIXES. Most credentials have none: an OAuth
# access/refresh token is an opaque 200-char string, a client secret is 32 hex chars. The
# most common way one enters a transcript is a credential FILE read verbatim —
# ``cat .yahoo_token.json`` — and measured 2026-09-10 that passed the seam untouched, into
# the model, the CLI log and the session file (weeks of prior runs, once we looked).
#
# ADR-0116 kept the ``key = value`` layer OFF the seam for a real reason: over source code it
# rewrites ``api_key = settings.api_key`` and the model can no longer edit the file it read.
# So this layer does not ask "is there a value after a secret key" — it asks whether the
# value is SHAPED LIKE A CREDENTIAL rather than like an identifier or an expression:
#   * a QUOTED literal after a secret key (``"access_token": "…"``, ``token = '…'``) that is
#     16+ chars and carries a digit — a string literal after a secret key is a secret;
#   * an UNQUOTED value (``.env``, YAML) 20+ chars that no identifier could be: base64/uuid
#     punctuation (``-/+=``), or mixed case with several digits, or a hex string 32+ long,
#     or a digit-heavy lowercase run with no underscore.
# ``settings.api_key`` (16 chars, expression), ``os.environ["TOKEN"]`` (a bracket), a
# ``{{secret:NAME}}`` placeholder, ``[REDACTED]`` and ``changeme`` all fall through.
_CRED_KEY = (
    r"api[_-]?key|api[_-]?secret|secret[_-]?key|client[_-]?secret|access[_-]?token"
    r"|refresh[_-]?token|id[_-]?token|auth[_-]?token|session[_-]?token|private[_-]?key"
    r"|access[_-]?key|secret|token|password|passwd|bearer|authorization|credentials?"
)
# The key may be the SUFFIX of an env-style name (``YAHOO_CLIENT_SECRET``, ``TAVILY_API_KEY``):
# a leading ``\b`` alone never matches after the ``_``, so the prefix is consumed explicitly.
_CRED_ASSIGN_RE = re.compile(
    r"(?i)(?P<pre>(?:\\?[\"'])?\b(?:[A-Za-z0-9]+[_-])*(?:"
    + _CRED_KEY
    + r")\b(?:\\?[\"'])?\s*[:=]\s*(?P<quote>\\?[\"'])?)"
    r"(?P<value>[A-Za-z0-9._\-/+=]{16,})"
)
_HEX_RE = re.compile(r"[0-9a-f]{32,}|[0-9A-F]{32,}")


def _looks_like_credential(value: str, *, quoted: bool) -> bool:
    """The shape test that separates a secret from an identifier (ADR-0125)."""
    digits = sum(ch.isdigit() for ch in value)
    if not digits or not any(ch.isalpha() for ch in value):
        return False  # placeholders and words: ``changeme``, ``your_api_key_here``
    if quoted:
        return True  # a string literal after a secret key IS the secret
    if len(value) < 20:
        return False
    if any(ch in "-/+=" for ch in value):
        return True  # base64 / uuid punctuation — no identifier carries these
    if _HEX_RE.fullmatch(value):
        return True
    mixed = any(ch.isupper() for ch in value) and any(ch.islower() for ch in value)
    if mixed and digits >= 3:
        return True  # ``wDIrZA…`` — camelCase identifiers rarely carry three digits
    return "_" not in value and digits >= 4  # ``abcdef1234567890abcdef``: not snake_case


def redact_credential_values(text: str) -> tuple[str, int]:
    """Return ``(scrubbed_text, num_redactions)`` for credential-SHAPED values after a
    secret key, keeping the key and the quotes (ADR-0125). Never raises."""
    if not text:
        return text, 0
    count = 0

    def _mark(m: re.Match[str]) -> str:
        nonlocal count
        if not _looks_like_credential(m.group("value"), quoted=m.group("quote") is not None):
            return m.group(0)
        count += 1
        return m.group("pre") + _REDACTED

    return _CRED_ASSIGN_RE.sub(_mark, text), count


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
        return f"{m.group(1)}{m.group(2)}{m.group(1)}{m.group(3)}{_REDACTED}"

    def _mark_url(_m: re.Match[str]) -> str:
        nonlocal count
        count += 1
        return "://***@"

    text = _PEM_RE.sub(_mark_pem, text)
    text = _URL_CRED_RE.sub(_mark_url, text)
    text = _TOKEN_RE.sub(_mark, text)
    text = _ASSIGN_RE.sub(_mark_assign, text)
    # ADR-0125: the env-style and JSON-quoted keys the blanket layer's boundaries miss.
    text, values = redact_credential_values(text)
    return text, count + values


def redact_credential_tokens(text: str) -> tuple[str, int]:
    """Return ``(scrubbed_text, num_redactions)`` scrubbing ONLY the high-signal token shapes.

    The layer safe to run over arbitrary tool output (ADR-0116): provider-prefixed tokens
    (``sk-``, ``ya29.``, ``AIza``, ``AKIA``, GitHub/Slack, a JWT) and PEM private-key blocks.
    The blanket ``key = value`` layer of :func:`redact_secrets` is deliberately left out here
    — over source code it rewrites ``api_key = settings.api_key`` and breaks the file the
    model is reading. What runs instead is :func:`redact_credential_values` (ADR-0125): the
    same keys, but only a value SHAPED like a credential is touched, so a credential file
    read verbatim is scrubbed and an identifier never is. Never raises; returns the input
    unchanged with a count of 0 when nothing matches.
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

    text = _PEM_RE.sub(_mark_pem, text)
    text = _TOKEN_RE.sub(_mark, text)
    # ADR-0125: then the values that have no prefix to recognise — shape-tested, so an
    # identifier or expression after a secret key is left exactly as it was.
    text, values = redact_credential_values(text)
    return text, count + values


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


__all__ = [
    "provider_key_env_names",
    "redact_credential_tokens",
    "redact_secrets",
    "strip_url_credentials",
]


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
