"""Automatic model resolution: the ``default_model: "auto"`` sentinel (PKG-AUTO, D19).

When the configured model is the string ``"auto"``, the resolver picks a concrete
model by **availability**: local Ollama if it is up and has a model, else the first
viable external provider in the configured preference order (default groq → openai →
anthropic). Viability is established with cheap **read-only** probes (``/api/tags``,
``/v1/models``) — never a chat call. Probe results are cached for the process and
re-probed on failure (the runtime failover path probes fresh).

Tool reliability is capability metadata, not a hardcoded sort (D21): when the
session has tools registered the resolver skips models whose
:class:`~zakcode.providers.base.Capabilities` say ``tools_unreliable`` — both
``groq/openai/gpt-oss-120b`` and ``groq/llama-3.3-70b-versatile`` are flagged (their native
tool calls hit Groq's ``tool_use_failed``), so ``groq/qwen/qwen3-32b`` is the groq tools pick,
and the rule keeps doing the right thing as models are added.

The resolver is a pluggable seam: anything matching :class:`ModelResolver` —
``resolve(task, require_tools=...) -> ResolvedModel | None`` — can replace the v1
:class:`AvailabilityResolver` later (Zachary's "zakpick" task-category routing)
without API breakage. v1 ignores ``task`` and resolves purely on availability.

Nothing viable is a LOUD failure: :class:`ModelResolutionError` carries a per-source
diagnosis (including each provider key's presence and provenance via
:func:`zakcode.config.env_source`) so the operator knows exactly what to set.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable, Collection
from dataclasses import dataclass
from typing import Protocol

from zakcode.providers.base import ProviderError
from zakcode.providers.registry import get_capabilities

logger = logging.getLogger("zakcode.providers")

#: The sentinel value of ``default_model`` that engages the availability resolver.
AUTO_SENTINEL = "auto"

#: The sentinel value of ``default_model`` that engages zakpick task-category routing (see
#: :mod:`zakcode.providers.routing`). Like ``auto`` it is resolved to a concrete startup model
#: at construction; per-call routing to the user's per-category model then happens at each site.
ZAKPICK_SENTINEL = "zakpick"

#: Read-only probe timeout. Startup cost is bounded by (sources tried) × this.
_PROBE_TIMEOUT_S = 2.0


class ModelResolutionError(ProviderError):
    """``default_model`` is ``"auto"`` but no model source is viable (loud failure)."""


@dataclass(frozen=True)
class ResolvedModel:
    """A concrete resolution: which model, from which source, and why."""

    model: str
    source: str  # "local" | "groq" | "openai" | "anthropic" | ...
    reason: str  # human-readable, shown in the info panel / status line


class ModelResolver(Protocol):
    """The pluggable resolution seam for ``default_model="auto"``.

    The ``exclude`` parameter is part of the contract (runtime failover drops the just-failed
    model and re-resolves); :class:`AvailabilityResolver` honors it. (zakpick takes a different,
    non-availability path — per-category user assignments in :mod:`zakcode.providers.routing` —
    so it does not implement this Protocol.)
    """

    def resolve(
        self,
        task: str | None = None,
        *,
        require_tools: bool = True,
        exclude: Collection[str] = (),
    ) -> ResolvedModel:
        """Return a concrete model or raise :class:`ModelResolutionError`."""
        ...


@dataclass(frozen=True)
class _ExternalSource:
    """One probeable external provider: its key, models endpoint, and candidates."""

    name: str
    key_env: str
    models_url: str
    #: Ordered model candidates; the first tools-viable one wins.
    candidates: tuple[str, ...]

    def headers(self, key: str) -> dict[str, str]:
        if self.name == "anthropic":
            return {"x-api-key": key, "anthropic-version": "2023-06-01"}
        return {"Authorization": f"Bearer {key}"}


#: Known external sources, keyed by the preference-list names.
_EXTERNAL_SOURCES: dict[str, _ExternalSource] = {
    "groq": _ExternalSource(
        name="groq",
        key_env="GROQ_API_KEY",
        models_url="https://api.groq.com/openai/v1/models",
        candidates=(
            "groq/openai/gpt-oss-120b",
            "groq/qwen/qwen3-32b",
            "groq/llama-3.3-70b-versatile",
        ),
    ),
    "openai": _ExternalSource(
        name="openai",
        key_env="OPENAI_API_KEY",
        models_url="https://api.openai.com/v1/models",
        candidates=("openai/gpt-4o-mini", "openai/gpt-4o"),
    ),
    "anthropic": _ExternalSource(
        name="anthropic",
        key_env="ANTHROPIC_API_KEY",
        models_url="https://api.anthropic.com/v1/models",
        candidates=("anthropic/claude-haiku-4-5", "anthropic/claude-sonnet-4-6"),
    ),
}

#: Process-lifetime probe cache: source name → (ok, detail). Cleared per-source on
#: runtime failure so a failed source is re-probed, per the spec.
_PROBE_CACHE: dict[str, tuple[bool, str]] = {}


def clear_probe_cache(source: str | None = None) -> None:
    """Forget cached probe results (all of them, or one source's)."""
    if source is None:
        _PROBE_CACHE.clear()
    else:
        _PROBE_CACHE.pop(source, None)


def _http_get(url: str, headers: dict[str, str]) -> tuple[int, str]:
    """One short read-only GET; returns (status_code, body_prefix). Never raises upward
    with vendor detail — callers turn exceptions into a not-viable reason string."""
    import httpx

    response = httpx.get(url, headers=headers, timeout=_PROBE_TIMEOUT_S)
    return response.status_code, response.text[:2000]


def _tools_viable(model: str, require_tools: bool) -> bool:
    if not require_tools:
        return True
    caps = get_capabilities(model)
    return caps.supports_tools and not caps.tools_unreliable


class AvailabilityResolver:
    """v1 resolver: local if viable, else first viable external by preference (D19).

    ``probe`` is injectable for tests (defaults to the real short-timeout GET).
    """

    def __init__(
        self,
        *,
        ollama_base_url: str,
        preference: list[str],
        probe: Callable[[str, dict[str, str]], tuple[int, str]] | None = None,
        use_cache: bool = True,
    ) -> None:
        self.ollama_base_url = ollama_base_url.rstrip("/")
        self.preference = preference
        # Late-bound default (module attr at call time, not def time) so tests that
        # monkeypatch ``_http_get`` are honored by resolver_for()-built instances.
        self._probe = probe if probe is not None else _http_get
        self._use_cache = use_cache

    # -- probes ---------------------------------------------------------------

    def _cached(self, source: str, run: Callable[[], tuple[bool, str]]) -> tuple[bool, str]:
        if self._use_cache and source in _PROBE_CACHE:
            return _PROBE_CACHE[source]
        result = run()
        _PROBE_CACHE[source] = result
        return result

    def _probe_local(self) -> tuple[bool, str]:
        """Ollama up + at least one model pulled → viable; detail names the first tag."""
        url = f"{self.ollama_base_url}/api/tags"

        def run() -> tuple[bool, str]:
            try:
                status, body = self._probe(url, {})
            except Exception as exc:  # noqa: BLE001 — a probe failure is a reason, not a crash
                return False, f"unreachable ({type(exc).__name__})"
            if status != 200:
                return False, f"HTTP {status}"
            import json

            try:
                tags = [m["name"] for m in json.loads(body).get("models", [])]
            except Exception:  # noqa: BLE001 — malformed body = not viable
                return False, "unparseable /api/tags response"
            if not tags:
                return False, "no models pulled"
            return True, tags[0]

        return self._cached("local", run)

    def _probe_external(self, source: _ExternalSource) -> tuple[bool, str]:
        key = os.environ.get(source.key_env, "")
        if not key:
            from zakcode.config import env_source

            return False, f"{source.key_env} not set ({env_source(source.key_env)})"

        def run() -> tuple[bool, str]:
            try:
                status, _body = self._probe(source.models_url, source.headers(key))
            except Exception as exc:  # noqa: BLE001 — a probe failure is a reason, not a crash
                return False, f"unreachable ({type(exc).__name__})"
            if status in (401, 403):
                return False, f"{source.key_env} present but rejected (HTTP {status})"
            if status != 200:
                return False, f"HTTP {status}"
            return True, "key valid, endpoint reachable"

        return self._cached(source.name, run)

    # -- resolution -----------------------------------------------------------

    def resolve(
        self,
        task: str | None = None,  # noqa: ARG002 — v1 is availability-only; zakpick consumes this
        *,
        require_tools: bool = True,
        exclude: Collection[str] = (),
    ) -> ResolvedModel:
        """Pick a concrete model, or raise :class:`ModelResolutionError` (loud).

        ``exclude`` removes specific model ids from consideration — the runtime
        failover path excludes the model that just failed.
        """
        diagnosis: list[str] = []

        # Local first: a viable Ollama always wins (free, private, no key).
        ok, detail = self._probe_local()
        if ok:
            model = f"ollama_chat/{detail}"
            if model in exclude:
                diagnosis.append(f"local ollama: excluded (just failed: {model})")
            elif not _tools_viable(model, require_tools):
                diagnosis.append(f"local ollama: {model} is not tools-viable")
            else:
                return ResolvedModel(
                    model=model, source="local", reason=f"local ollama is up ({detail})"
                )
        else:
            diagnosis.append(f"local ollama ({self.ollama_base_url}): {detail}")

        for name in self.preference:
            source = _EXTERNAL_SOURCES.get(name)
            if source is None:
                diagnosis.append(f"{name}: unknown source (check the preference list)")
                continue
            ok, detail = self._probe_external(source)
            if not ok:
                diagnosis.append(f"{name}: {detail}")
                continue
            candidates = [
                m for m in source.candidates if m not in exclude and _tools_viable(m, require_tools)
            ]
            if not candidates:
                diagnosis.append(f"{name}: viable, but no tools-viable candidate remains")
                continue
            chosen = candidates[0]
            reason = f"{name} viable ({detail})"
            logger.info("auto model resolution: %s via %s — %s", chosen, name, reason)
            return ResolvedModel(model=chosen, source=name, reason=reason)

        lines = "\n".join(f"  - {line}" for line in diagnosis)
        raise ModelResolutionError(
            "default_model is 'auto' but no model source is viable:\n"
            f"{lines}\n"
            "Fix: pull a local model (`ollama pull llama3.1`), or set a provider key "
            "in ~/.zakcode/.env or the workspace .env (see docs/CONFIG.md), or set "
            "ZAKCODE_DEFAULT_MODEL to an explicit model."
        )


#: Provider-key variable names this module can source from the member's vault, derived
#: from :data:`_EXTERNAL_SOURCES` so a newly added provider is covered without a second
#: list to keep in sync.
VAULT_PROVIDER_KEY_NAMES: frozenset[str] = frozenset(
    src.key_env for src in _EXTERNAL_SOURCES.values()
)

#: The deployment's OWN provider keys, captured the first time each name is seen — before
#: any overlay has replaced it. Restoring from here is what makes a vault entry's REMOVAL
#: take effect: without it the first overlay would pin the member's key into the process
#: for its whole life and deleting the secret would appear to do nothing until a restart.
_PLATFORM_KEY_BASELINE: dict[str, str | None] = {}


def apply_vault_provider_keys(settings: object) -> list[str]:
    """Overlay member-supplied provider keys from the per-env vault into ``os.environ``.

    BYOK (g-369-11 outcome 3). A member saves e.g. ``ANTHROPIC_API_KEY`` in the vault they
    already have; inference for that environment then bills to their provider account and
    the meter's thinking line goes to ~0. Returns the names taken from the vault, sorted —
    NAMES only, never values, so the return is safe to log.

    WHY ``os.environ`` AND NOT AN EXPLICIT ``api_key``. For a bare cloud model
    ``LiteLLMProvider`` deliberately omits ``api_key`` from the call so litellm reads the
    standard variable itself — a stale explicit key used to shadow it and raise a spurious
    AuthError (audit3 #6). Passing one here would re-create exactly that. The environment
    is the seam litellm already reads, and writing to it so litellm can is an established
    pattern in that module (``OLLAMA_API_BASE``).

    WHY IT RUNS PER CONSTRUCTION RATHER THAN ONCE AT IMPORT. A long-lived process freezes
    its environment at exec time, so a member who saves a key after the runtime started
    would see nothing happen. The vault file is re-read on every call (its loader is
    re-read-per-access by contract, bounded by a 64 KiB cap), which is what lets a freshly
    saved key work without a restart — the property the vault was built for.

    FEATURE-OFF IS TOTAL: no ``secrets_file``, no vault entry under a recognised name, or
    an unreadable/malformed file all leave the environment exactly as found. Every
    deployment without a vault is byte-identical to before this function existed.
    """
    secrets_file = getattr(settings, "secrets_file", None)

    held: dict[str, str] = {}
    if secrets_file is not None:
        try:
            from zakcode.tools.builtins._secrets import SecretsProvider

            held = SecretsProvider(secrets_file).values_for(VAULT_PROVIDER_KEY_NAMES)
        except Exception:  # noqa: BLE001 — BYOK must never be able to stop a session starting
            logger.warning("vault provider-key overlay skipped (vault unreadable)")
            return []

    applied: list[str] = []
    for name in sorted(VAULT_PROVIDER_KEY_NAMES):
        if name not in _PLATFORM_KEY_BASELINE:
            _PLATFORM_KEY_BASELINE[name] = os.environ.get(name)
        value = held.get(name)
        if value:
            os.environ[name] = value
            applied.append(name)
        else:
            baseline = _PLATFORM_KEY_BASELINE[name]
            if baseline is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = baseline
    if applied:
        # Names only. A value must never reach a log line (GUARDRAILS §6).
        logger.info(
            "BYOK: provider key(s) sourced from the environment vault: %s", ", ".join(applied)
        )
    return applied


def resolver_for(settings: object, *, use_cache: bool = True) -> AvailabilityResolver:
    """Build the v1 resolver from Settings (typed loosely to avoid an import cycle)."""
    return AvailabilityResolver(
        ollama_base_url=getattr(settings, "ollama_base_url", "http://localhost:11434"),
        preference=list(
            getattr(settings, "auto_model_preference", ["groq", "openai", "anthropic"])
        ),
        use_cache=use_cache,
    )


def describe_resolution(settings: object) -> str:
    """One-line resolution summary for the info panel (never raises).

    Routes the ``zakpick`` sentinel to its own friendly description (a per-category table);
    the ``auto`` path resolves the single availability pick.
    """
    if getattr(settings, "default_model", None) == ZAKPICK_SENTINEL:
        # Lazy import: routing imports this module, so importing it at top would cycle.
        from zakcode.providers.routing import describe_zakpick

        return describe_zakpick(settings)
    try:
        resolved = resolver_for(settings).resolve(require_tools=True)
    except ModelResolutionError as exc:
        first = str(exc).splitlines()[0]
        return f"auto → UNRESOLVED ({first})"
    return f"auto → {resolved.model} ({resolved.reason})"
