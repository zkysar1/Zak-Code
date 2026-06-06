"""Tests for the vendor-agnostic structured-output helpers."""

from __future__ import annotations

from typing import Any

import pytest

from zds_llm_provider import structured
from zds_llm_provider.messages import Message
from zds_llm_provider.structured import (
    StructuredResult,
    StructuredValidationError,
    coerce_structured,
    complete_structured,
    extract_json,
    make_response_format,
)
from zds_llm_provider.types import Capabilities, LLMResult, Provider
from zds_llm_provider.usage import Usage

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


# ── complete_structured (the C<->P seam) ─────────────────────────────────────────


class _FakeProvider(Provider):
    """Returns canned texts in order (repeats the last once exhausted); records call kwargs."""

    def __init__(self, texts: list[str]) -> None:
        self._texts = list(texts)
        self._last = texts[0] if texts else ""
        self.calls: list[dict[str, Any]] = []

    async def acomplete(  # noqa: ANN001
        self, messages, *, system=None, tools=None, response_format=None, **kwargs
    ) -> LLMResult:
        self.calls.append(
            {"response_format": response_format, "temperature": kwargs.get("temperature")}
        )
        if self._texts:
            self._last = self._texts.pop(0)
        return LLMResult(text=self._last, usage=Usage(total_tokens=1))

    def count_tokens(self, messages, *, system=None) -> int:  # noqa: ANN001
        return 0

    def capabilities(self) -> Capabilities:
        return Capabilities()


async def test_complete_structured_no_schema_single_call() -> None:
    p = _FakeProvider(["just text"])
    res = await complete_structured(p, [Message.user("hi")])
    assert isinstance(res, StructuredResult)
    assert res.data is None and res.text == "just text"
    assert res.valid is True and res.repaired is False
    assert len(p.calls) == 1
    # No schema -> no response_format and no forced temperature.
    assert p.calls[0]["response_format"] is None
    assert p.calls[0]["temperature"] is None


async def test_complete_structured_valid_first_try() -> None:
    pytest.importorskip("jsonschema")
    p = _FakeProvider(['{"a": "x"}'])
    res = await complete_structured(p, [Message.user("hi")], schema=_OBJ_SCHEMA)
    assert res.valid is True and res.repaired is False and res.data == {"a": "x"}
    assert len(p.calls) == 1
    assert p.calls[0]["response_format"] is not None  # schema requested
    assert p.calls[0]["temperature"] == 0.0  # deterministic on the schema path


async def test_complete_structured_invalid_then_valid() -> None:
    pytest.importorskip("jsonschema")
    p = _FakeProvider(['{"a": 1}', '{"a": "x"}'])  # first violates (a must be str), second valid
    res = await complete_structured(p, [Message.user("hi")], schema=_OBJ_SCHEMA)
    assert res.valid is True and res.repaired is True and res.data == {"a": "x"}
    assert len(p.calls) == 2  # exactly one repair retry (max_repairs=1)


async def test_complete_structured_always_invalid_no_raise() -> None:
    pytest.importorskip("jsonschema")
    p = _FakeProvider(['{"a": 1}'])  # repeats the invalid output
    res = await complete_structured(p, [Message.user("hi")], schema=_OBJ_SCHEMA)
    assert res.valid is False and res.data is None and res.text == '{"a": 1}'
    assert res.repaired is True and len(p.calls) == 2


async def test_complete_structured_accumulates_usage() -> None:
    pytest.importorskip("jsonschema")
    p = _FakeProvider(['{"a": 1}', '{"a": "x"}'])  # two calls, 1 token each
    res = await complete_structured(p, [Message.user("hi")], schema=_OBJ_SCHEMA)
    assert res.usage.total_tokens == 2
