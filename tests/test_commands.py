"""Tests for the slash-command registry (M6-1)."""

from __future__ import annotations

import pytest

from zakcode.commands import Command, CommandRegistry, CommandResult


def test_register_and_run() -> None:
    reg = CommandRegistry()
    reg.register("greet", lambda arg: CommandResult.ok(f"hi {arg}"), description="greet")
    result = reg.run("greet", "world")
    assert result is not None
    assert result.output == "hi world"
    assert result.is_error is False


def test_name_is_normalized() -> None:
    reg = CommandRegistry()
    reg.register("/Foo", lambda arg: CommandResult.ok("x"))
    # Registered without slash, lowercased; lookup works with or without the slash.
    assert reg.has("foo")
    assert reg.get("/foo") is not None
    assert reg.run("FOO") is not None


def test_run_unknown_returns_none() -> None:
    reg = CommandRegistry()
    assert reg.run("nope") is None  # caller falls through to its own handling


def test_duplicate_registration_refused() -> None:
    reg = CommandRegistry()
    reg.register("dup", lambda arg: CommandResult.ok("a"))
    with pytest.raises(ValueError):
        reg.register("dup", lambda arg: CommandResult.ok("b"))


def test_duplicate_with_replace_allowed() -> None:
    reg = CommandRegistry()
    reg.register("dup", lambda arg: CommandResult.ok("a"))
    reg.register("dup", lambda arg: CommandResult.ok("b"), replace=True)
    result = reg.run("dup")
    assert result is not None and result.output == "b"


def test_empty_name_rejected() -> None:
    reg = CommandRegistry()
    with pytest.raises(ValueError):
        reg.register("/", lambda arg: CommandResult.ok("x"))


def test_handler_exception_becomes_error_result() -> None:
    def boom(arg: str) -> CommandResult:
        raise RuntimeError("kaboom")

    reg = CommandRegistry()
    reg.register("boom", boom)
    result = reg.run("boom")
    assert result is not None
    assert result.is_error is True
    assert "kaboom" in result.output


def test_definitions_and_source() -> None:
    reg = CommandRegistry()
    reg.register("a", lambda arg: CommandResult.ok(""), description="cmd a", source="plug")
    reg.register("b", lambda arg: CommandResult.ok(""))
    defs = reg.definitions()
    assert [c.name for c in defs] == ["a", "b"]
    a = reg.get("a")
    assert isinstance(a, Command)
    assert a.source == "plug"
    assert a.description == "cmd a"


def test_names_preserve_order() -> None:
    reg = CommandRegistry()
    for n in ["z", "a", "m"]:
        reg.register(n, lambda arg: CommandResult.ok(""))
    assert reg.names() == ["z", "a", "m"]
