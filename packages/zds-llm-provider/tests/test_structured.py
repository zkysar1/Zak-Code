"""Tests for the vendor-agnostic structured-output helpers."""

from __future__ import annotations

import pytest

from zds_llm_provider import structured
from zds_llm_provider.structured import (
    StructuredValidationError,
    coerce_structured,
    extract_json,
    make_response_format,
)

_OBJ_SCHEMA = {
    "type": "object",
    "required": ["a"],
    "additionalProperties": False,
    "properties": {"a": {"type": "string"}},
}


# ── extract_json ────────────────────────────────────────────────────────────────


def test_extract_json_pure_object() -> None:
    assert extract_json('{"a": "x"}') == {"a": "x"}


def test_extract_json_strips_code_fence() -> None:
    assert extract_json('```json\n{"a": "x"}\n```') == {"a": "x"}
    assert extract_json('```\n{"a": 1}\n```') == {"a": 1}


def test_extract_json_tolerates_trailing_prose() -> None:
    assert extract_json('{"a": "x"} and that is the answer') == {"a": "x"}


def test_extract_json_rejects_leading_prose() -> None:
    # Prose BEFORE the JSON yields None (not mis-parsed) — the caller treats it as failure.
    assert extract_json('here you go: {"a": "x"}') is None


def test_extract_json_non_json_is_none() -> None:
    assert extract_json("not json at all") is None
    assert extract_json("") is None
    assert extract_json("   ") is None


def test_extract_json_array() -> None:
    assert extract_json('["x", "y"]') == ["x", "y"]


# ── make_response_format ─────────────────────────────────────────────────────────


def test_make_response_format_json_object_when_no_schema() -> None:
    assert make_response_format(None) == {"type": "json_object"}


def test_make_response_format_json_schema_shape() -> None:
    rf = make_response_format(_OBJ_SCHEMA, name="lesson")
    assert rf["type"] == "json_schema"
    assert rf["json_schema"]["name"] == "lesson"
    assert rf["json_schema"]["schema"] is _OBJ_SCHEMA
    assert rf["json_schema"]["strict"] is True


# ── coerce_structured ────────────────────────────────────────────────────────────


def test_coerce_structured_no_schema_returns_data() -> None:
    assert coerce_structured('{"a": "x"}', schema=None) == {"a": "x"}


def test_coerce_structured_no_json_raises() -> None:
    with pytest.raises(StructuredValidationError) as exc:
        coerce_structured("sorry, I cannot", schema=None)
    assert exc.value.raw_text == "sorry, I cannot"


def test_coerce_structured_schema_pass() -> None:
    pytest.importorskip("jsonschema")
    assert coerce_structured('{"a": "x"}', schema=_OBJ_SCHEMA) == {"a": "x"}


def test_coerce_structured_schema_fail_raises() -> None:
    pytest.importorskip("jsonschema")
    with pytest.raises(StructuredValidationError):
        coerce_structured('{"a": 1}', schema=_OBJ_SCHEMA)  # a must be a string
    with pytest.raises(StructuredValidationError):
        coerce_structured('{"b": "x"}', schema=_OBJ_SCHEMA)  # missing required + extra key


def test_coerce_structured_degrades_without_jsonschema(monkeypatch: pytest.MonkeyPatch) -> None:
    # With jsonschema absent, the schema check is skipped: a schema-VIOLATING but valid JSON
    # object is returned unverified rather than raising.
    monkeypatch.setattr(structured, "jsonschema", None)
    assert coerce_structured('{"a": 1}', schema=_OBJ_SCHEMA) == {"a": 1}
    # ...but a non-JSON output still raises (extraction failure is independent of jsonschema).
    with pytest.raises(StructuredValidationError):
        coerce_structured("not json", schema=_OBJ_SCHEMA)
