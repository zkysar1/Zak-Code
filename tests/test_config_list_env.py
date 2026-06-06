"""Tests for list-valued Settings fields parsed from environment variables.

Regression for the review finding: ``allowed_models`` / ``denied_commands`` are ``list[str]``;
without a NoDecode + splitting validator, pydantic-settings JSON-decodes the env value and a
plain (or comma-separated) value crashes ``load_settings`` — which aborts ``zakcode serve`` at
startup. ``denied_commands`` holds regexes (which may contain commas/spaces), so it must split
on NEWLINES only, never on commas/whitespace.
"""

from __future__ import annotations

import pytest

from zakcode.config import Settings, load_settings


def _clear(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ZAKCODE_ALLOWED_MODELS", raising=False)
    monkeypatch.delenv("ZAKCODE_DENIED_COMMANDS", raising=False)


# ── allowed_models (simple ids: comma / whitespace / JSON) ─────────────────────────


def test_allowed_models_unset_is_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear(monkeypatch)
    assert load_settings().allowed_models == []


def test_allowed_models_plain_single_value(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear(monkeypatch)
    monkeypatch.setenv("ZAKCODE_ALLOWED_MODELS", "openai/gpt-4o")
    assert load_settings().allowed_models == ["openai/gpt-4o"]


def test_allowed_models_comma_separated(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear(monkeypatch)
    monkeypatch.setenv("ZAKCODE_ALLOWED_MODELS", "openai/gpt-4o,ollama_chat/qwen2.5:3b")
    assert load_settings().allowed_models == ["openai/gpt-4o", "ollama_chat/qwen2.5:3b"]


def test_allowed_models_whitespace_separated(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear(monkeypatch)
    monkeypatch.setenv("ZAKCODE_ALLOWED_MODELS", "openai/gpt-4o  ollama_chat/qwen2.5:3b")
    assert load_settings().allowed_models == ["openai/gpt-4o", "ollama_chat/qwen2.5:3b"]


def test_allowed_models_json_array(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear(monkeypatch)
    monkeypatch.setenv("ZAKCODE_ALLOWED_MODELS", '["openai/gpt-4o", "x/y"]')
    assert load_settings().allowed_models == ["openai/gpt-4o", "x/y"]


# ── denied_commands (regexes: NEWLINE / JSON only — never comma/space split) ────────


def test_denied_commands_unset_is_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear(monkeypatch)
    assert load_settings().denied_commands == []


def test_denied_commands_single_regex_with_space_preserved(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear(monkeypatch)
    monkeypatch.setenv("ZAKCODE_DENIED_COMMANDS", "rm -rf /")
    # Must stay ONE regex — splitting on whitespace would corrupt it into 'rm' / '-rf' / '/'.
    assert load_settings().denied_commands == ["rm -rf /"]


def test_denied_commands_regex_with_comma_preserved(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear(monkeypatch)
    monkeypatch.setenv("ZAKCODE_DENIED_COMMANDS", "a{1,2}b")
    # A comma inside a regex quantifier must NOT split.
    assert load_settings().denied_commands == ["a{1,2}b"]


def test_denied_commands_multiple_via_newlines(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear(monkeypatch)
    monkeypatch.setenv("ZAKCODE_DENIED_COMMANDS", "rm -rf /\nsudo .*\ngit push --force")
    assert load_settings().denied_commands == ["rm -rf /", "sudo .*", "git push --force"]


def test_denied_commands_json_array(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear(monkeypatch)
    monkeypatch.setenv("ZAKCODE_DENIED_COMMANDS", '["rm -rf /", "a{1,2}"]')
    assert load_settings().denied_commands == ["rm -rf /", "a{1,2}"]


# ── programmatic construction still works (lists pass through unchanged) ────────────


def test_programmatic_lists_pass_through() -> None:
    s = Settings(allowed_models=["a/b"], denied_commands=["x .*y", "p{2,3}"])
    assert s.allowed_models == ["a/b"]
    assert s.denied_commands == ["x .*y", "p{2,3}"]
