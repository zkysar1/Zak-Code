"""Named-secrets provider — ``{{secret:NAME}}`` substitution outside the model.

The contract, in one sentence: the model works with secret NAMES; the VALUES are
resolved into outbound requests at request-build time and are never part of any
model-facing string — not prompts, not tool output, not error messages, not logs.

How the pieces fit:

* An operator (or an orchestrating sidecar) writes a ``name -> value`` JSON object
  to a file and points ``ZAKCODE_SECRETS_FILE`` at it. Absent/empty = feature off.
* Names are env-var-shaped (``^[A-Z][A-Z0-9_]{0,63}$``) so a placeholder is
  unambiguous inside URLs and header values.
* Tools that perform HTTP (today: ``web_fetch``) call :meth:`SecretsProvider.resolve`
  on model-supplied strings just before the request is built, and route every
  model-facing string they produce through :meth:`SecretsProvider.scrub` — so a value
  that comes BACK (an API echoing the request, an exception string embedding the
  resolved URL) is folded back into its placeholder before the model can see it.
* Successful substitution appends a names-only usage event (JSONL) to
  ``ZAKCODE_SECRETS_USAGE_FILE`` so an orchestrator can surface "last used" to the
  human who saved the secret. Values never appear in the usage log.

Honest scope note (documented, not hidden): on a box where the agent also holds an
unrestricted shell tool under the same UID, a determined model could read the secrets
file directly. What this module enforces is CONTEXT HYGIENE — values stay out of the
model's context on every normal path — which is the property that keeps secrets out
of transcripts, traces, provider-side retention, and screen shares. OS-level
isolation from a hostile model is a deployment concern (file permissions, separate
UIDs), not something a tool-layer substitution can provide.

The file is re-read on every access (bounded by the 64 KiB cap below): a freshly
saved secret is usable on the next tool call without a daemon restart, and there is
no cache to invalidate.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path

#: Placeholder shape: ``{{secret:WEATHER_API_KEY}}``. The name charset is env-var
#: style — uppercase start, then uppercase/digits/underscore, max 64 chars.
SECRET_PLACEHOLDER_RE = re.compile(r"\{\{secret:([A-Z][A-Z0-9_]{0,63})\}\}")

#: Names must match this to be served at all (a malformed key in the secrets file is
#: skipped rather than half-working in the placeholder grammar).
SECRET_NAME_RE = re.compile(r"^[A-Z][A-Z0-9_]{0,63}$")

#: Refuse to read a secrets file larger than this — the file is a small name->value
#: map by contract; anything bigger is a misconfiguration, not a bigger map.
_MAX_SECRETS_FILE_BYTES = 64 * 1024

#: Values shorter than this are never scrubbed from outbound text: replacing every
#: "1" in a page because someone saved a one-character "secret" would mangle output
#: without protecting anything real.
_MIN_SCRUB_LEN = 6


class UnknownSecretError(Exception):
    """A placeholder referenced a name the provider does not hold.

    The message carries the NAME only — it is shown to the model as the tool error.
    """

    def __init__(self, name: str) -> None:
        self.name = name
        super().__init__(
            f"unknown secret {{{{secret:{name}}}}} — it is not in the configured secrets"
        )


class SecretsProvider:
    """Loads ``name -> value`` from a JSON file; substitutes, scrubs, and records use.

    A provider with no file (or a missing/empty/oversized/malformed one) behaves as
    EMPTY: ``names()`` is ``[]``, ``resolve`` raises :class:`UnknownSecretError` on any
    placeholder, ``scrub`` is the identity. Tools can therefore hold a provider
    unconditionally and never branch on "is the feature on".
    """

    def __init__(self, path: Path | None, *, usage_path: Path | None = None) -> None:
        self._path = path
        self._usage_path = usage_path

    def _load(self) -> dict[str, str]:
        """The current map, re-read from disk on every call (see module docstring)."""
        if self._path is None:
            return {}
        try:
            if self._path.stat().st_size > _MAX_SECRETS_FILE_BYTES:
                return {}
            raw = json.loads(self._path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}
        if not isinstance(raw, dict):
            return {}
        return {
            name: value
            for name, value in raw.items()
            if isinstance(name, str)
            and isinstance(value, str)
            and value
            and SECRET_NAME_RE.match(name)
        }

    def values_for(self, names: Iterable[str]) -> dict[str, str]:
        """The values held for ``names`` — the ONLY value-returning accessor.

        Deliberately name-driven rather than a ``dict`` dump: a caller must say which
        secrets it wants, so no code path can enumerate the vault's values by accident.
        Names absent from the file are simply omitted (no ``UnknownSecretError``) —
        callers here are asking "does the member happen to have saved this?", which is
        a legitimate no.

        Added for BYOK (g-369-11): provider-key overlay needs the VALUE of a specific
        well-known name, and every other reader of this file wants placeholders. Routing
        it through the same validated loader keeps the size cap, the name grammar and
        the malformed-file behaviour identical for both — a second reader with its own
        ``json.loads`` would drift from those three the first time any changed.

        Values returned here are NOT model-facing and must never be placed in a prompt,
        a tool result, or a log. The module contract (names in context, values in
        requests) is unchanged: this feeds an outbound Authorization header.
        """
        secrets = self._load()
        return {n: secrets[n] for n in names if n in secrets}

    def names(self) -> list[str]:
        """The names the model may reference, sorted. Never the values."""
        return sorted(self._load())

    def resolve(self, text: str) -> tuple[str, set[str]]:
        """Substitute every ``{{secret:NAME}}`` in ``text`` with its value.

        Returns ``(resolved_text, names_used)``. Raises :class:`UnknownSecretError`
        (names-only message) if any placeholder references a name not in the map —
        a half-substituted request must never leave the box.
        """
        secrets = self._load()
        used: set[str] = set()

        def _sub(match: re.Match[str]) -> str:
            name = match.group(1)
            if name not in secrets:
                raise UnknownSecretError(name)
            used.add(name)
            return secrets[name]

        return SECRET_PLACEHOLDER_RE.sub(_sub, text), used

    def scrub(self, text: str) -> str:
        """Fold any secret VALUE appearing in ``text`` back into its placeholder.

        The return-path half of the contract: applied to every model-facing string a
        substituting tool produces (body text, error messages, final URLs), so an API
        that echoes the request — or an exception that embeds the resolved URL —
        cannot carry a value into the model's context. Longer values are replaced
        first so a value that happens to contain another is folded correctly.
        """
        if not text:
            return text
        secrets = self._load()
        for name, value in sorted(secrets.items(), key=lambda kv: -len(kv[1])):
            if len(value) >= _MIN_SCRUB_LEN and value in text:
                text = text.replace(value, f"{{{{secret:{name}}}}}")
        return text

    def record_use(self, names: set[str]) -> None:
        """Append a names-only usage event per name (JSONL). Best-effort, never raises.

        The semantic recorded is "the value was released into a request attempt" —
        substitution succeeded and the request was handed to the transport, whatever
        the remote end then said.
        """
        if not names or self._usage_path is None:
            return
        stamp = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
        try:
            self._usage_path.parent.mkdir(parents=True, exist_ok=True)
            with self._usage_path.open("a", encoding="utf-8") as fh:
                for name in sorted(names):
                    fh.write(json.dumps({"name": name, "used_at": stamp}) + "\n")
        except OSError:
            return
