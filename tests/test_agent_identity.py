"""Tests that the Agent facade loads + threads the operator identity (self.md) into the prompt."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

import zakcode
from zakcode.messages import Message
from zakcode.providers.base import Capabilities, LLMResult, Provider


class _StubProvider(Provider):
    """A no-network provider so Agent construction + prompt building stay hermetic."""

    async def acomplete(
        self,
        messages: list[Message],
        *,
        system: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        response_format: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> LLMResult:
        return LLMResult(text="")

    def count_tokens(self, messages: list[Message], *, system: str | None = None) -> int:
        return 0

    def capabilities(self) -> Capabilities:
        return Capabilities()


def _fake_home(monkeypatch: pytest.MonkeyPatch, home: Path) -> None:
    # Keep the user-level ~/.config/zakcode/self.md candidate out of the picture.
    monkeypatch.setattr(Path, "home", lambda: home)


def _agent(ws: Path, **kw: Any) -> zakcode.Agent:
    return zakcode.Agent(provider=_StubProvider(), workspace_root=ws, **kw)


def test_enable_identity_surfaces_self_md(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_home(monkeypatch, tmp_path / "home")
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "self.md").write_text("You are Vinheim, a friendly guide.", encoding="utf-8")
    prompt = _agent(ws).loop._build_system()
    assert "You are Vinheim, a friendly guide." in prompt
    assert "You are Zak Code, a vendor-agnostic AI coding assistant" not in prompt


def test_self_md_only_workspace_surfaces_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The builder-construction fix: a workspace with ONLY self.md (no rules/skills) must still
    # surface the identity (the construction condition includes self.identity).
    _fake_home(monkeypatch, tmp_path / "home")
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "self.md").write_text("You are Vinheim.", encoding="utf-8")
    agent = _agent(ws, enable_rules=False, enable_skills=False)
    assert "You are Vinheim." in agent.loop._build_system()


def test_enable_identity_false_ignores_self_md(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _fake_home(monkeypatch, tmp_path / "home")
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "self.md").write_text("You are Vinheim.", encoding="utf-8")
    prompt = _agent(ws, enable_identity=False).loop._build_system()
    assert "You are Vinheim." not in prompt
    assert "You are Zak Code, a vendor-agnostic AI coding assistant" in prompt


def test_no_self_md_uses_default_identity(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_home(monkeypatch, tmp_path / "home")
    ws = tmp_path / "ws"
    ws.mkdir()
    prompt = _agent(ws).loop._build_system()
    assert "You are Zak Code, a vendor-agnostic AI coding assistant" in prompt


def test_explicit_identity_arg_beats_discovery(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _fake_home(monkeypatch, tmp_path / "home")
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "self.md").write_text("FROM FILE", encoding="utf-8")
    prompt = _agent(ws, identity="EXPLICIT IDENTITY").loop._build_system()
    assert "EXPLICIT IDENTITY" in prompt
    assert "FROM FILE" not in prompt
