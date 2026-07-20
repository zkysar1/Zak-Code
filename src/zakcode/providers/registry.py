"""Static capability registry for known models.

Maps model identifiers to :class:`Capabilities`. Lookups fall back to
litellm's model metadata and finally to a conservative default so that
unknown models still behave sanely.

This module imports litellm (allowed: it lives under ``providers/``), but only
for best-effort metadata lookups guarded against any failure.
"""

from __future__ import annotations

from zakcode.providers.base import Capabilities

# Data-driven table of well-known models. Keys are matched case-insensitively
# and also by suffix (so "openai/gpt-4o" matches a "gpt-4o" entry).
_CAPABILITIES: dict[str, Capabilities] = {
    # --- OpenAI ---
    "gpt-4o": Capabilities(
        supports_tools=True,
        supports_vision=True,
        supports_caching=True,
        context_window=128_000,
        max_output=16_384,
    ),
    "gpt-4o-mini": Capabilities(
        supports_tools=True,
        supports_vision=True,
        supports_caching=True,
        context_window=128_000,
        max_output=16_384,
    ),
    "gpt-4-turbo": Capabilities(
        supports_tools=True,
        supports_vision=True,
        supports_caching=False,
        context_window=128_000,
        max_output=4_096,
    ),
    "gpt-3.5-turbo": Capabilities(
        supports_tools=True,
        supports_vision=False,
        supports_caching=False,
        context_window=16_385,
        max_output=4_096,
    ),
    # --- Anthropic (Claude) ---
    # Context windows deliberately pin the STANDARD (no-beta-header) 200k window:
    # litellm's metadata DB advertises the 1M long-context beta for the 4.x family,
    # but this runtime sends no beta header — budgeting against 1M would let the
    # compactor fire too late and overflow the real request limit. A conservative
    # pin costs only earlier compaction. (Audit P0-1a; acceptance: test_anthropic_registry.)
    "claude-opus-4-8": Capabilities(
        supports_tools=True,
        supports_vision=True,
        supports_caching=True,
        context_window=200_000,
        # litellm advertises 128k output for opus-4-8, but that (like the 1M window)
        # rides a beta header this runtime does not send — pin the no-header limit
        # for the same consistency reason as the window. (stack review minor #4)
        max_output=64_000,
    ),
    "claude-sonnet-4-6": Capabilities(
        supports_tools=True,
        supports_vision=True,
        supports_caching=True,
        context_window=200_000,
        max_output=64_000,
    ),
    "claude-sonnet-4-20250514": Capabilities(
        supports_tools=True,
        supports_vision=True,
        supports_caching=True,
        context_window=200_000,
        max_output=64_000,
    ),
    "claude-haiku-4-5": Capabilities(
        supports_tools=True,
        supports_vision=True,
        supports_caching=True,
        context_window=200_000,
        max_output=64_000,
    ),
    # --- Groq (hosted) ---
    # Keyed WITH the ``groq/`` prefix (exact match fires before prefix-stripping in
    # ``_lookup_key``) because the bare names are Groq catalog ids, not portable model
    # names. Values mirror litellm's pricing/metadata DB, probed 2026-06-10, so they
    # hold offline. (Audit P0-1a; acceptance: test_groq_registry.)
    "groq/llama-3.3-70b-versatile": Capabilities(
        supports_tools=True,
        # Emits llama pseudo-XML tool calls (<function=name {json}</function>) that
        # Groq's strict tool parser rejects with `tool_use_failed` — near-
        # deterministically with a multi-tool schema (live-verified 2026-06-11,
        # PR #13 root cause). The auto-resolver skips it when tools are in play.
        tools_unreliable=True,
        supports_vision=False,
        supports_caching=False,
        context_window=128_000,
        max_output=32_768,
    ),
    "groq/llama-3.1-8b-instant": Capabilities(
        supports_tools=True,
        supports_vision=False,
        supports_caching=False,
        context_window=128_000,
        max_output=8_192,
    ),
    "groq/openai/gpt-oss-120b": Capabilities(
        supports_tools=True,
        # Nominal tool support, unreliable in PRACTICE — so the auto-resolver skips it when
        # tools are in play, exactly like llama-3.3-70b above. It is a REASONING model: under a
        # real multi-tool schema it emits malformed NATIVE tool calls that Groq's strict parser
        # rejects with `tool_use_failed` (near-deterministically), and in text mode its output
        # goes to the reasoning channel so `content` comes back empty — broken in BOTH modes.
        # (live-verified 2026-06-18; bench deep set 0/3 — see bench/README.md; PR #45 root
        # cause. Upstream: pydantic-ai #4350, OpenHands #10187.)
        tools_unreliable=True,
        supports_vision=False,
        supports_caching=False,
        context_window=131_072,
        max_output=32_766,
    ),
    "groq/openai/gpt-oss-20b": Capabilities(
        supports_tools=True,
        # The smaller gpt-oss sibling, same REASONING-model failure mode as the 120b above: its
        # native tool calls come back malformed and Groq rejects them with `tool_use_failed`
        # (live-observed this session — a real multi-tool turn died with provider_error). So the
        # auto-resolver skips it when tools are in play, and zakpick keeps it ONLY on no-tool
        # categories (summaries). The tool-using quick_code tier moved off it to qwen3-32b.
        tools_unreliable=True,
        supports_vision=False,
        supports_caching=False,
        context_window=131_072,
        max_output=32_768,
    ),
    "groq/qwen/qwen3-32b": Capabilities(
        supports_tools=True,
        supports_vision=False,
        supports_caching=False,
        context_window=131_000,
        # litellm's DB claims max_output == context_window (131k/131k) — almost
        # certainly a conflated field. Groq documents 40,960 max completion tokens
        # for qwen3-32b; pin that. (stack review minor #5; re-probed 2026-06-10)
        max_output=40_960,
    ),
    # 2026-07-19 re-pin target (g-335-172): Groq DECOMMISSIONED qwen3-32b (above); the
    # mind-sidecar drive's pinned default moved to qwen3.6-27b (Ayoai-Environment-Server
    # ops/mind-sidecar/scripts/bootstrap.sh). Same tool-capable tier — supports_tools with
    # NO tools_unreliable flag, so the auto-resolver keeps it for tool sessions.
    "groq/qwen/qwen3.6-27b": Capabilities(
        supports_tools=True,
        supports_vision=False,
        supports_caching=False,
        # context_window/max_output MIRRORED from qwen3-32b pending a Groq-spec re-probe
        # for qwen3.6-27b (Groq /v1/models was unreachable from the box that filed this).
        context_window=131_000,
        max_output=40_960,
    ),
    # --- Ollama (local) ---
    "ollama_chat/llama3.1": Capabilities(
        supports_tools=True,
        supports_vision=False,
        supports_caching=False,
        context_window=128_000,
        max_output=None,
    ),
    "ollama_chat/llama3": Capabilities(
        supports_tools=True,
        supports_vision=False,
        supports_caching=False,
        context_window=8_192,
        max_output=None,
    ),
    "ollama_chat/qwen2.5": Capabilities(
        supports_tools=True,
        supports_vision=False,
        supports_caching=False,
        context_window=32_768,
        max_output=None,
    ),
    "ollama_chat/mistral": Capabilities(
        supports_tools=True,
        supports_vision=False,
        supports_caching=False,
        context_window=32_768,
        max_output=None,
    ),
}

_DEFAULT = Capabilities(
    supports_tools=True,
    supports_vision=False,
    supports_caching=False,
    context_window=8_192,
    max_output=None,
)


def _strip_provider_prefix(model: str) -> str:
    """Return the portion of ``model`` after a single provider prefix.

    ``"openai/gpt-4o"`` -> ``"gpt-4o"``. ``ollama_chat/...`` is preserved
    because the table keys it with the prefix intact.
    """
    if model.startswith("ollama_chat/") or model.startswith("ollama/"):
        return model
    if "/" in model:
        return model.split("/", 1)[1]
    return model


def _lookup_key(key: str) -> Capabilities | None:
    if key in _CAPABILITIES:
        return _CAPABILITIES[key]
    lowered = key.lower()
    for name, caps in _CAPABILITIES.items():
        if name.lower() == lowered:
            return caps
    # Try with the provider prefix removed (e.g. "openai/gpt-4o" -> "gpt-4o").
    stripped = _strip_provider_prefix(key)
    if stripped != key:
        if stripped in _CAPABILITIES:
            return _CAPABILITIES[stripped]
        stripped_lower = stripped.lower()
        for name, caps in _CAPABILITIES.items():
            if name.lower() == stripped_lower:
                return caps
    # Ollama tags a model as ``name:size`` (e.g. "ollama_chat/qwen2.5:3b"); fall back to
    # the untagged base ("ollama_chat/qwen2.5") so size variants share a capability entry
    # instead of dropping to the conservative default window.
    if ":" in key:
        base = key.split(":", 1)[0]
        if base in _CAPABILITIES:
            return _CAPABILITIES[base]
        base_lower = base.lower()
        for name, caps in _CAPABILITIES.items():
            if name.lower() == base_lower:
                return caps
    return None


def _lookup_static(model: str) -> Capabilities | None:
    key = model.strip()
    result = _lookup_key(key)
    if result is not None:
        return result
    # The bare ``ollama/`` and ``ollama_chat/`` prefixes name the SAME backend model, but
    # the table is keyed on ``ollama_chat/``; resolve ``ollama/X`` against that key so both
    # forms get the same context window (else ``ollama/qwen2.5`` would silently fall to the
    # conservative default and get half the num_ctx of ``ollama_chat/qwen2.5``). (audit #11)
    if key.startswith("ollama/"):
        return _lookup_key("ollama_chat/" + key[len("ollama/") :])
    return None


def _from_litellm(model: str) -> Capabilities | None:
    """Best-effort capability lookup via litellm metadata.

    Any failure (offline, unknown model, missing keys) returns ``None`` so the
    caller can fall through to the safe default.
    """
    try:
        import litellm
    except ImportError:
        return None

    context_window: int | None = None
    max_output: int | None = None
    supports_tools = True
    supports_vision = False
    supports_caching = False

    try:
        info = litellm.get_model_info(model)
    except Exception:
        info = None

    if isinstance(info, dict):
        max_in = info.get("max_input_tokens")
        if isinstance(max_in, int):
            context_window = max_in
        max_out = info.get("max_output_tokens") or info.get("max_tokens")
        if isinstance(max_out, int):
            max_output = max_out
        if isinstance(info.get("supports_function_calling"), bool):
            supports_tools = bool(info["supports_function_calling"])
        if isinstance(info.get("supports_vision"), bool):
            supports_vision = bool(info["supports_vision"])
        if isinstance(info.get("supports_prompt_caching"), bool):
            supports_caching = bool(info["supports_prompt_caching"])

    if context_window is None:
        try:
            max_tokens = litellm.get_max_tokens(model)
        except Exception:
            max_tokens = None
        if isinstance(max_tokens, int):
            context_window = max_tokens

    if context_window is None:
        return None

    return Capabilities(
        supports_tools=supports_tools,
        supports_vision=supports_vision,
        supports_caching=supports_caching,
        context_window=context_window,
        max_output=max_output,
    )


def get_capabilities(model: str) -> Capabilities:
    """Return the capabilities for ``model``.

    Resolution order: static table -> litellm metadata -> safe default
    (``context_window=8192``, ``supports_tools=True``). This never raises: an
    empty/blank/non-string model, an unknown model, or any lookup failure all
    fall through to the safe default.
    """
    # Defensively coerce: callers should pass a str, but a None/odd value must
    # not blow up capability resolution. An empty/blank name skips lookups.
    if not isinstance(model, str) or not model.strip():
        return _DEFAULT.model_copy()

    try:
        static = _lookup_static(model)
        if static is not None:
            return static
        dynamic = _from_litellm(model)
        if dynamic is not None:
            return dynamic
    except Exception:
        return _DEFAULT.model_copy()
    return _DEFAULT.model_copy()
