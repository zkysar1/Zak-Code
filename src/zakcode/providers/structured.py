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
:class:`~zakcode.providers.base.Provider` ABC; it imports no vendor/transport SDK, so any
provider or caller can reuse it.
"""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from zakcode.messages import Message
from zakcode.providers.base import Provider, ProviderError
from zakcode.usage import Usage

# Optional: schema VALIDATION. Without ``jsonschema``, coerce_structured degrades to extraction.
# Under TYPE_CHECKING mypy sees the real (typed) module — so its uses below need no per-line
# ``type: ignore`` — while at runtime the import is guarded and may resolve to ``None`` (every
# use is fenced behind ``jsonschema is not None``).
if TYPE_CHECKING:
    import jsonschema
else:
    try:
        import jsonschema
    except ImportError:  # pragma: no cover - exercised by monkeypatching jsonschema=None
        jsonschema = None

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
    schema: dict[str, Any] | None = None, *, name: str = "output", strict: bool = False
) -> dict[str, Any]:
    """The ``response_format`` payload to request structured output from a backend.

    ``schema is None`` → plain JSON mode (``{"type": "json_object"}``); a schema →
    a named ``json_schema`` block (the OpenAI shape litellm maps per-backend). This only
    *requests* structure; always validate the result with :func:`coerce_structured`.

    ``strict`` defaults to **False** to stay vendor-agnostic: OpenAI/Azure *strict* json_schema
    mode imposes a structural subset (every property ``required``, ``additionalProperties:false``,
    a limited keyword set) that a perfectly valid JSON Schema can violate, hard-failing the
    request — whereas a schema is only meta-checked against Draft 2020-12 before the call. Since
    correctness comes from the local :func:`coerce_structured` validation regardless of the
    backend flag, we request the lenient mode every backend can map. Pass ``strict=True`` only
    when you specifically target an OpenAI-compatible backend and want server-side enforcement.
    """
    if schema is None:
        return {"type": "json_object"}
    return {
        "type": "json_schema",
        "json_schema": {"name": name, "schema": schema, "strict": strict},
    }


def _extract(text: str) -> tuple[bool, Any]:
    """``(found, value)`` for the leading JSON value of ``text``.

    Distinguishes a model that emitted the literal token ``null`` (``found=True, value=None``)
    from one that produced no JSON at all (``found=False``) — so a nullable-root schema can
    accept ``null`` rather than it being mistaken for a parse failure. Strips a leading
    ```json/``` fence and tolerates trailing prose; leading prose yields ``found=False``.
    ``RecursionError`` from pathologically-nested input degrades to ``found=False`` (it must
    never escape — see :func:`coerce_structured`'s callers, which must not 500).
    """
    s = text.strip()
    if not s:
        return False, None
    if s.startswith("```"):
        s = _FENCE_OPEN_RE.sub("", s, count=1)
        if s.rstrip().endswith("```"):
            s = s.rstrip()[:-3]
        s = s.strip()
    if not s:
        return False, None
    try:
        return True, json.loads(s)
    except (json.JSONDecodeError, ValueError, RecursionError):
        pass
    try:
        obj, _end = json.JSONDecoder().raw_decode(s)
    except (json.JSONDecodeError, ValueError, RecursionError):
        return False, None
    return True, obj


def extract_json(text: str) -> Any:
    """The leading JSON value of ``text``, or ``None`` if it does not start with one.

    Strips a leading ```json/``` code fence, then parses (tolerating trailing prose); leading
    prose yields ``None``. NOTE ``None`` is also returned for a literal ``null`` — callers that
    must tell the two apart use the internal ``_extract`` (e.g. :func:`coerce_structured`).
    Never raises (even on pathological nesting).
    """
    found, value = _extract(text)
    return value if found else None


def coerce_structured(text: str, *, schema: dict[str, Any] | None = None) -> Any:
    """Parse ``text`` into a JSON value and (when possible) validate it against ``schema``.

    Raises :class:`StructuredValidationError` when no JSON value can be extracted, or when a
    ``schema`` is supplied AND ``jsonschema`` is installed AND validation fails OR the schema
    itself is unusable (a malformed schema / unresolvable ``$ref`` is reported as a validation
    FAILURE, never escaping as an unhandled 500). When ``jsonschema`` is absent the schema check
    is skipped (extraction-only degradation). A legitimately-parsed ``null`` is validated like
    any other value, not treated as a parse failure.
    """
    found, data = _extract(text)
    if not found:
        raise StructuredValidationError("no JSON value found in model output", raw_text=text)
    if schema is not None and jsonschema is not None:
        try:
            jsonschema.validate(instance=data, schema=schema)
        except jsonschema.ValidationError as exc:
            raise StructuredValidationError(
                f"output failed schema validation: {exc.message}", raw_text=text
            ) from exc
        except Exception as exc:  # noqa: BLE001 — a malformed schema / unresolvable $ref must
            # surface as a clean validation failure, never escape as an unhandled 500.
            raise StructuredValidationError(
                f"schema could not be applied: {exc}", raw_text=text
            ) from exc
    return data


def schema_error(schema: dict[str, Any]) -> str | None:
    """A message if ``schema`` is not a usable JSON Schema, else ``None``.

    Lets a caller (e.g. an HTTP endpoint) reject a malformed *client* schema as a 4xx BEFORE
    spending a model call. Returns ``None`` when ``jsonschema`` is unavailable (no meta-check is
    possible, so the schema is accepted and simply not enforced downstream).
    """
    if jsonschema is None:
        return None
    try:
        jsonschema.Draft202012Validator.check_schema(schema)
    except Exception as exc:  # noqa: BLE001 — any meta-schema failure means the schema is unusable
        return str(exc)
    return None


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
    "schema_error",
]
