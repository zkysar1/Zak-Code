"""Vendor-agnostic structured-output helpers.

A small, dependency-light layer for asking a model for JSON and getting back a
*validated* object. Two concerns, kept separate from any provider:

* :func:`make_response_format` — build the OpenAI-shaped ``response_format`` payload a
  provider passes to the backend (``json_object`` mode, or a named ``json_schema``).
  litellm maps this onto each backend (incl. Ollama's native ``format``); a backend that
  ignores it just returns prose, which is why correctness lives at the *call site*, not
  in the kwarg.
* :func:`extract_json` / :func:`coerce_structured` — pull the leading JSON value out of a
  model's text (tolerating a ```json fence or trailing prose) and, when a schema and
  ``jsonschema`` are available, validate it — raising :class:`StructuredValidationError`
  on failure. ``jsonschema`` is an optional extra; without it, validation degrades to
  extraction-only (the object is returned unverified rather than erroring), so the helper
  works in a minimal install.

This module depends only on pydantic + the stdlib (+ optional ``jsonschema``) and the
:class:`~zds_llm_provider.types.Provider` ABC; it imports no vendor/transport SDK, so any
provider or caller can reuse it.
"""

from __future__ import annotations

import json
import re
from typing import Any

from pydantic import BaseModel, Field

from zds_llm_provider.messages import Message
from zds_llm_provider.types import Provider, ProviderError
from zds_llm_provider.usage import Usage

try:  # optional: schema VALIDATION. Without it, coerce_structured degrades to extraction.
    import jsonschema
except ImportError:  # pragma: no cover - exercised by monkeypatching jsonschema=None
    jsonschema = None  # type: ignore[assignment]

#: A leading Markdown code fence a weak model often wraps JSON in (```json ... ``` / ``` ...).
_FENCE_OPEN_RE = re.compile(r"^```[A-Za-z0-9_-]*[ \t]*\r?\n?")


class StructuredValidationError(ProviderError):
    """The model output could not be parsed as JSON or failed schema validation.

    Carries the offending ``raw_text`` so a caller (e.g. the ``/complete`` endpoint) can
    surface it for debugging without re-deriving it.
    """

    def __init__(self, message: str, *, raw_text: str = "") -> None:
        super().__init__(message)
        self.raw_text = raw_text


def make_response_format(
    schema: dict[str, Any] | None = None, *, name: str = "output", strict: bool = True
) -> dict[str, Any]:
    """The ``response_format`` payload to request structured output from a backend.

    ``schema is None`` → plain JSON mode (``{"type": "json_object"}``); a schema →
    a named ``json_schema`` block (the OpenAI shape litellm maps per-backend). This only
    *requests* structure; always validate the result with :func:`coerce_structured`.
    """
    if schema is None:
        return {"type": "json_object"}
    return {
        "type": "json_schema",
        "json_schema": {"name": name, "schema": schema, "strict": strict},
    }


def extract_json(text: str) -> Any:
    """The leading JSON value of ``text``, or ``None`` if it does not start with one.

    Strips a leading ```json/``` code fence first, then tries a strict ``json.loads`` and
    falls back to ``raw_decode`` (which consumes only the leading value, tolerating trailing
    prose like ``{...} done``). Prose *before* the JSON yields ``None`` — the text is left
    for the caller to treat as a failure rather than mis-parsed. Mirrors the text-protocol
    tool-call parser's leading-object tolerance.
    """
    s = text.strip()
    if not s:
        return None
    if s.startswith("```"):
        s = _FENCE_OPEN_RE.sub("", s, count=1)
        if s.rstrip().endswith("```"):
            s = s.rstrip()[:-3]
        s = s.strip()
    if not s:
        return None
    try:
        return json.loads(s)
    except (json.JSONDecodeError, ValueError):
        pass
    try:
        obj, _end = json.JSONDecoder().raw_decode(s)
    except (json.JSONDecodeError, ValueError):
        return None
    return obj


def coerce_structured(text: str, *, schema: dict[str, Any] | None = None) -> Any:
    """Parse ``text`` into a JSON value and (when possible) validate it against ``schema``.

    Raises :class:`StructuredValidationError` when no JSON value can be extracted, or when a
    ``schema`` is supplied AND ``jsonschema`` is installed AND validation fails. When
    ``jsonschema`` is absent the schema check is skipped (extraction-only degradation), so a
    minimal install still returns the parsed object. Returns the parsed value on success.
    """
    data = extract_json(text)
    if data is None:
        raise StructuredValidationError("no JSON value found in model output", raw_text=text)
    if schema is not None and jsonschema is not None:
        try:
            jsonschema.validate(instance=data, schema=schema)
        except jsonschema.ValidationError as exc:  # type: ignore[misc]
            raise StructuredValidationError(
                f"output failed schema validation: {exc.message}", raw_text=text
            ) from exc
    return data


class StructuredResult(BaseModel):
    """The outcome of :func:`complete_structured`.

    ``data`` is the parsed + validated JSON (``None`` when no schema was requested, or when
    validation failed after the repair budget). ``text`` is the model's last raw output.
    ``valid`` is True when no schema was requested OR the output validated. ``repaired`` is
    True when at least one corrective retry was issued. ``usage`` accumulates across attempts.
    """

    data: Any | None = None
    text: str = ""
    usage: Usage = Field(default_factory=Usage)
    repaired: bool = False
    valid: bool = True


async def complete_structured(
    provider: Provider,
    messages: list[Message],
    *,
    system: str | None = None,
    schema: dict[str, Any] | None = None,
    max_repairs: int = 1,
) -> StructuredResult:
    """Run a completion and, when a ``schema`` is given, return schema-valid JSON.

    With ``schema=None`` this is a single plain completion (``data=None``, ``valid=True``,
    ``text`` = the raw output). With a schema it requests structured output (``response_format``
    from :func:`make_response_format`, ``temperature`` forced to 0 for determinism on the
    schema path), validates via :func:`coerce_structured`, and on failure issues up to
    ``max_repairs`` corrective retries. After the budget is exhausted it returns ``valid=False``
    with the last raw text — it NEVER raises for a validation failure; only a provider error
    propagates. Buffered (acomplete) only — never streams.
    """
    convo = list(messages)
    usage = Usage()
    attempts = 0
    last_text = ""
    while True:
        call_kwargs: dict[str, Any] = {}
        if schema is not None:
            call_kwargs["response_format"] = make_response_format(schema)
            call_kwargs["temperature"] = 0.0  # deterministic on the schema path
        result = await provider.acomplete(convo, system=system, **call_kwargs)
        usage = usage + result.usage
        last_text = result.text
        if schema is None:
            return StructuredResult(data=None, text=last_text, usage=usage, valid=True)
        try:
            data = coerce_structured(last_text, schema=schema)
        except StructuredValidationError as exc:
            if attempts >= max_repairs:
                return StructuredResult(
                    data=None, text=last_text, usage=usage, repaired=attempts > 0, valid=False
                )
            attempts += 1
            convo = [
                *convo,
                Message.assistant_text(last_text),
                Message.user(
                    "Your previous reply was not valid JSON for the required schema "
                    f"({exc}). Reply with ONLY a single JSON object that matches the schema — "
                    "no prose, no markdown code fence."
                ),
            ]
            continue
        return StructuredResult(
            data=data, text=last_text, usage=usage, repaired=attempts > 0, valid=True
        )


__all__ = [
    "StructuredResult",
    "StructuredValidationError",
    "coerce_structured",
    "complete_structured",
    "extract_json",
    "make_response_format",
]
