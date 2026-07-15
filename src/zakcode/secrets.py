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

import math
import os
import re
from collections.abc import Iterable, Mapping

_REDACTED = "[REDACTED]"

# A whole PEM private-key block (header through footer), collapsed to a marker.
_PEM_RE = re.compile(
    r"-----BEGIN[^-]*PRIVATE KEY-----.*?-----END[^-]*PRIVATE KEY-----",
    re.DOTALL,
)

# Standalone high-signal credential tokens (provider-prefixed, so low false-positive).
# NOTE: ``gsk_`` (Groq) and ``vin_`` (Vinheim product keys) need their OWN alternatives —
# they are NOT caught by the ``sk-`` pattern: ``\bsk-`` cannot match inside ``gsk_`` (the ``g``
# leaves no word boundary before ``sk``, and the separator is ``_`` not ``-``). Added for the
# public watch surface, but a base fix that benefits every ``redact_secrets`` caller.
_TOKEN_RE = re.compile(
    r"\b("
    r"sk-[A-Za-z0-9_-]{16,}"  # OpenAI-style
    r"|gsk_[A-Za-z0-9_-]{16,}"  # Groq keys
    r"|vin_[A-Za-z0-9_-]{16,}"  # Vinheim product keys
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

# URL userinfo: ``scheme://user[:password]@host`` — mask the WHOLE userinfo (a user-only token,
# ``user:password``, or a password containing ``@``) up to the LAST @ before the host, keeping
# scheme + host. So a provider error echoing a credentialed api_base (gateway URL) can't leak a
# token. The greedy ``[^/\s]+`` stops at a path ``/`` or whitespace, so a non-credential ``@`` in a
# path or free-form prose (no ``://``) is left untouched.
_URL_CRED_RE = re.compile(r"://[^/\s]+@")


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


def _shannon_entropy(value: str) -> float:
    """Shannon entropy in bits/char of ``value`` (0.0 for empty).

    A heuristic signal for the high-entropy catch-all in
    :func:`redact_secrets_extended`: random credential material scores high
    (base64/hex secrets typically > 4.5), ordinary words and hex digests score low.
    """
    if not value:
        return 0.0
    counts: dict[str, int] = {}
    for ch in value:
        counts[ch] = counts.get(ch, 0) + 1
    n = len(value)
    return -sum((c / n) * math.log2(c / n) for c in counts.values())


#: A base64/hex/token-shaped run — the candidate span for the high-entropy catch-all.
_ENTROPY_CANDIDATE_RE = re.compile(r"[A-Za-z0-9+/=_-]{24,}")


def redact_secrets_extended(
    text: str,
    *,
    secret_values: Iterable[str] = (),
    workspace_paths: Iterable[str] = (),
    entropy_threshold: float = 4.5,
    min_value_len: int = 8,
) -> tuple[str, int]:
    """Aggressive redaction for a PUBLIC (kid-facing) watch surface.

    Layers four passes on top of :func:`redact_secrets` so credentials are stripped
    regardless of format, and filesystem structure is hidden:

    1. **Workspace-path stripping** — each path in ``workspace_paths`` (the workspace
       root and its parents) → ``[path]``. Longest first so a child path is masked
       before its parent prefix. Both ``/`` and ``\\`` separator forms are matched.
    2. **Exact secret-value match** — each value in ``secret_values`` (the process's
       ``*_KEY`` / ``*_TOKEN`` / ``*_SECRET`` env values, supplied by the caller) →
       ``[REDACTED]``, catching credentials of ANY format. Values shorter than
       ``min_value_len`` are skipped so ordinary short config values are not masked.
    3. **Shape-based redaction** — :func:`redact_secrets` (PEM blocks, URL userinfo,
       provider-prefixed tokens incl. ``gsk_`` / ``vin_``, ``key = value`` assignments).
    4. **High-entropy catch-all** — any remaining 24+ char base64/hex/token run whose
       Shannon entropy exceeds ``entropy_threshold`` → ``[redacted]``. Hex digests
       (git SHAs, uuids: ≤ 16 symbols → entropy ≤ 4.0) fall UNDER the threshold and
       survive, so session ids stay visible; random secrets do not.

    Returns ``(scrubbed_text, num_redactions)``. Never raises; returns the input
    unchanged with a count of 0 when nothing matches. Redaction happens at the SOURCE
    (inside the server, before serialization) — a whitelist projection strips tool
    arguments/output first, so this is defense-in-depth over already-narrowed fields.
    """
    if not text:
        return text, 0
    count = 0

    # 1. workspace-path stripping (longest first; both separator forms).
    path_variants: set[str] = set()
    for path in workspace_paths:
        if path:
            path_variants.update({path, path.replace("\\", "/"), path.replace("/", "\\")})
    for variant in sorted(path_variants, key=len, reverse=True):
        occurrences = text.count(variant)
        if occurrences:
            count += occurrences
            text = text.replace(variant, "[path]")

    # 2. exact secret-value match (longest first; skip short/empty values).
    for value in sorted(
        {v for v in secret_values if v and len(v) >= min_value_len}, key=len, reverse=True
    ):
        occurrences = text.count(value)
        if occurrences:
            count += occurrences
            text = text.replace(value, _REDACTED)

    # 3. shape-based redaction (the base guard).
    text, base_count = redact_secrets(text)
    count += base_count

    # 4. high-entropy catch-all.
    def _mark_entropy(m: re.Match[str]) -> str:
        nonlocal count
        token = m.group(0)
        if _shannon_entropy(token) > entropy_threshold:
            count += 1
            return "[redacted]"
        return token

    text = _ENTROPY_CANDIDATE_RE.sub(_mark_entropy, text)
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


__all__ = [
    "provider_key_env_names",
    "redact_secrets",
    "redact_secrets_extended",
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
