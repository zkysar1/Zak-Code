"""HIPAA mode: which models a covered home may call.

``Settings.hipaa_mode`` narrows every model call to ``Settings.hipaa_allowed_models``. The one
predicate lives here so the startup sweep (``Agent._assert_hipaa_models``) and the request
builder (``LiteLLMProvider._build_kwargs``) cannot disagree about what a model name means, the
same reason ``endpoints`` holds both of ``local_only``'s predicates.

This module imports no vendor SDK, so ``config`` and the Agent can import it freely.
"""

from __future__ import annotations

from collections.abc import Sequence

#: The litellm prefix OpenAI's own API is addressed by. ``openai/gpt-4o`` and ``gpt-4o`` name
#: the same hosted model, so the allowlist reads them as one. No other prefix is stripped: a
#: named provider, or a generic-OpenAI base such as ``openai_like/``, is a different
#: destination and has to be listed by its full name.
_OPENAI_PREFIX = "openai/"


class HipaaModelViolation(RuntimeError):
    """A call was about to use a model that is not on the HIPAA allowlist.

    Deliberately **not** a :class:`~zakcode.providers.base.ProviderError`, for the reason
    :class:`~zakcode.providers.endpoints.LocalOnlyViolation` is not. Provider errors are
    recoverable and ride the loop's failover, which answers by trying a DIFFERENT model, the
    one reaction that would turn a single refusal into patient data reaching a model nobody
    confirmed. This is a configuration fault: it is fatal, it names the offending model, and
    nothing retries it.
    """


def normalize_model(model: str | None) -> str:
    """The comparable form of a model name: trimmed, lowercased, a leading ``openai/`` removed."""
    name = (model or "").strip().lower()
    return name[len(_OPENAI_PREFIX) :] if name.startswith(_OPENAI_PREFIX) else name


def check_model(model: str | None, allowed: Sequence[str] | None) -> tuple[bool, str]:
    """Whether ``model`` may be called under HIPAA mode, with an operator-actionable reason.

    Fail CLOSED: a blank model, or an empty allowlist, permits nothing. Matching is exact on
    the normalized name, so a dated snapshot (``gpt-4o-2024-08-06``) is a different model from
    ``gpt-4o`` and has to be listed on its own.
    """
    name = normalize_model(model)
    if not name:
        return False, "there is no model name to check"
    listed = sorted({entry for entry in (normalize_model(e) for e in (allowed or ())) if entry})
    if not listed:
        return False, (
            f"{model!r} cannot be used: hipaa_allowed_models is empty, and an empty "
            "allowlist permits no model"
        )
    if name in listed:
        return True, f"{model!r} is on the allowlist"
    return False, f"{model!r} is not on the allowlist (hipaa_allowed_models: {', '.join(listed)})"


__all__ = ["HipaaModelViolation", "check_model", "normalize_model"]
