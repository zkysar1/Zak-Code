"""ADR-0190: the model-visible tool names are Claude Code's; the old names are aliases.

A served Mind on a small model read ``Skill('aspirations') with args='loop'`` and answered
it in prose for hours, because its tool list showed ``use_skill`` — ``Skill`` was only an
alias, and a small model calls what it can SEE. These tests pin the rename at every seam:
the registry advertises Claude Code's names and resolves the old ones; the loop
canonicalizes a call (name AND Claude Code's argument keys) where it enters, so nothing
downstream ever matches an alias; the shape adapters for the tools that keep Zak Code's
own name (``task``, ``update_plan``); hook matchers, permission rules and operator config
in either spelling.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from zakcode.agent.loop import AgentLoop
from zakcode.agent.subagent import SubAgentResult
from zakcode.config import load_settings
from zakcode.evals.harness import ScriptedProvider, call_tool, reply
from zakcode.hooks import (
    CLAUDE_CODE_ARG_KEYS,
    HookEvent,
    HookManager,
    HookPayload,
    HookSpec,
    wire_payload,
)
from zakcode.messages import ToolResultBlock, ToolUseBlock
from zakcode.permissions import PermissionDecision, PermissionMode, PermissionPolicy
from zakcode.session.store import Session
from zakcode.tool_names import PRE_0190_TOOL_NAMES, canonical_tool_name
from zakcode.tools.base import (
    ConcurrencyClass,
    PermissionTier,
    SkillLoad,
    ToolContext,
    ToolRegistry,
    ToolSpec,
)
from zakcode.tools.builtins.default_registry import default_registry
from zakcode.tools.builtins.read_file import ReadFileTool
from zakcode.tools.builtins.task import TaskTool
from zakcode.tools.builtins.update_plan import UpdatePlanTool
from zakcode.tools.builtins.use_skill import UseSkillTool
from zakcode.usage import Usage

CLAUDE_CODE_NAMES = (
    "Read",
    "Write",
    "Edit",
    "LS",
    "Glob",
    "Grep",
    "Bash",
    "WebFetch",
    "WebSearch",
    "Skill",
    "ScheduleWakeup",
)


# ── (1) the registry: Claude Code's names are canonical; the old names resolve silently ──


def _agent_registry(tmp_path: Path) -> ToolRegistry:
    """The registry a real Agent builds: the default set plus Skill (skills on) and task
    (delegation on) — the two tools ``default_registry`` leaves to the Agent."""
    from zakcode import Agent
    from zakcode.config import Settings

    agent = Agent(
        settings=Settings(
            default_model="scripted/test", context_window=8192, workspace_root=tmp_path
        ),
        provider=ScriptedProvider([reply("x")]),
        enable_skills=True,
        enable_subagents=True,
    )
    return agent.registry


def test_default_registry_canonical_names_are_claude_codes(tmp_path: Path) -> None:
    registry = _agent_registry(tmp_path)
    names = set(registry.names())
    assert set(CLAUDE_CODE_NAMES) <= names
    # The pre-0190 spellings are not tools of their own — they resolve to the canonical one.
    for old, new in PRE_0190_TOOL_NAMES.items():
        assert old not in names, old
        assert registry.canonical(old) == new, (old, new)
        assert old in registry.aliases_of(new), (old, new)


def test_definitions_advertise_canonical_names_only(tmp_path: Path) -> None:
    advertised = [d["function"]["name"] for d in _agent_registry(tmp_path).definitions()]
    assert set(CLAUDE_CODE_NAMES) <= set(advertised)
    assert not set(PRE_0190_TOOL_NAMES) & set(advertised)
    assert len(advertised) == len(set(advertised))  # an alias never doubles a definition


def test_tools_that_keep_their_shape_take_claude_codes_names_as_aliases(tmp_path: Path) -> None:
    registry = _agent_registry(tmp_path)
    assert registry.canonical("Task") == "task"
    assert registry.canonical("Agent") == "task"
    assert registry.canonical("TodoWrite") == "update_plan"
    assert registry.canonical("TodoRead") == "plan_recall"


def test_canonical_tool_name_table_is_the_registry_fallback() -> None:
    assert canonical_tool_name("use_skill") == "Skill"
    assert canonical_tool_name("schedule_wakeup") == "ScheduleWakeup"
    assert canonical_tool_name("Bash") == "Bash"  # already canonical: identity
    assert canonical_tool_name("my_plugin_tool") == "my_plugin_tool"  # unknown: untouched


# ── (2) the loop canonicalizes every call where it enters ─────────────────────────


def _loop(
    provider: ScriptedProvider,
    tmp_path: Path,
    registry: ToolRegistry,
    *,
    hooks: HookManager | None = None,
    skill_resolver: Any = None,
) -> AgentLoop:
    return AgentLoop(
        provider,
        registry,
        Session(cwd=str(tmp_path), model="scripted/test"),
        settings=load_settings(workspace_root=tmp_path),
        workspace_root=tmp_path,
        permission_policy=PermissionPolicy(PermissionMode.ALLOW),
        hook_manager=hooks,
        max_iterations=4,
        skill_resolver=skill_resolver,
    )


async def _run(loop: AgentLoop, prompt: str, path: str) -> None:
    """One turn on the buffered path or the streamed one — the served path streams."""
    if path == "buffered":
        await loop.arun_turn(prompt)
        return
    async for _ in loop.astream_turn(prompt):
        pass


def _tool_uses(loop: AgentLoop) -> list[ToolUseBlock]:
    return [b for m in loop.session.messages for b in m.blocks if isinstance(b, ToolUseBlock)]


def _tool_results(loop: AgentLoop) -> list[ToolResultBlock]:
    return [b for m in loop.session.messages for b in m.blocks if isinstance(b, ToolResultBlock)]


@pytest.mark.parametrize("path", ["buffered", "streamed"])
async def test_a_pre_0190_call_lands_in_the_transcript_under_the_canonical_name(
    tmp_path: Path, path: str
) -> None:
    (tmp_path / "a.txt").write_text("alpha\n", encoding="utf-8")
    registry = ToolRegistry()
    registry.register(ReadFileTool(), aliases=["read_file"])
    provider = ScriptedProvider([call_tool("read_file", {"path": "a.txt"}), reply("done")])
    loop = _loop(provider, tmp_path, registry)
    await _run(loop, "read a.txt", path)
    (use,) = _tool_uses(loop)
    assert use.name == "Read"
    (res,) = _tool_results(loop)
    assert not res.is_error and "alpha" in res.output


async def test_a_bare_registry_still_resolves_the_old_spelling(tmp_path: Path) -> None:
    # A registry built WITHOUT the aliases (SDK embedders, older tests): the loop falls back
    # to the pre-0190 table, so the call is not an "unknown tool" divergence.
    (tmp_path / "a.txt").write_text("alpha\n", encoding="utf-8")
    registry = ToolRegistry()
    registry.register(ReadFileTool())
    provider = ScriptedProvider([call_tool("read_file", {"path": "a.txt"}), reply("done")])
    loop = _loop(provider, tmp_path, registry)
    await loop.arun_turn("read a.txt")
    (use,) = _tool_uses(loop)
    assert use.name == "Read"
    (res,) = _tool_results(loop)
    assert not res.is_error and "alpha" in res.output


@pytest.mark.parametrize("path", ["buffered", "streamed"])
async def test_claude_codes_argument_key_is_accepted_on_read(tmp_path: Path, path: str) -> None:
    # ``Read(file_path=...)`` is Claude Code's spelling; the schema says ``path``. Rewritten
    # where the call enters — the tool never sees ``file_path``. A streamed call enters
    # somewhere else than a buffered one, and until 2026-09-18 nothing rewrote it there: the
    # served path refused this call with "'path' is required".
    (tmp_path / "a.txt").write_text("alpha\n", encoding="utf-8")
    registry = ToolRegistry()
    registry.register(ReadFileTool())
    provider = ScriptedProvider([call_tool("Read", {"file_path": "a.txt"}), reply("done")])
    loop = _loop(provider, tmp_path, registry)
    await _run(loop, "read a.txt", path)
    (use,) = _tool_uses(loop)
    assert use.name == "Read" and use.input == {"path": "a.txt"}
    (res,) = _tool_results(loop)
    assert not res.is_error and "alpha" in res.output


def test_arg_key_map_is_the_inverse_of_the_hook_wire() -> None:
    # The hook wire renames ``path`` → ``file_path`` for Claude Code hooks; the loop applies
    # the inverse to model calls. Both directions come from one table.
    assert CLAUDE_CODE_ARG_KEYS["Read"] == {"file_path": "path"}
    assert CLAUDE_CODE_ARG_KEYS["Write"] == {"file_path": "path"}
    assert CLAUDE_CODE_ARG_KEYS["Edit"] == {"file_path": "path"}


# ── (3) Skill: ``skill`` is the parameter; ``name`` and ``use_skill`` still load ──


class _Resolver:
    def __init__(self, bodies: dict[str, str]) -> None:
        self._bodies = bodies
        self.loaded: list[tuple[str, str]] = []

    def names(self) -> list[str]:
        return list(self._bodies)

    def body(self, name: str) -> str | None:
        return self._bodies.get(name)

    async def load(self, name: str, *, query: str = "", args: str = "") -> SkillLoad:
        self.loaded.append((name, args))
        if name in self._bodies:
            return SkillLoad(found=True, name=name, body=self._bodies[name])
        return SkillLoad(found=False, name=name)


async def _run_skill_call(tmp_path: Path, tool: str, arguments: dict[str, Any]) -> _Resolver:
    registry = ToolRegistry()
    registry.register(UseSkillTool(), aliases=["use_skill"])
    resolver = _Resolver({"aspirations": "# Aspirations\n\n## Steps\n\n1. Loop.\n"})
    provider = ScriptedProvider([call_tool(tool, arguments), reply("done")])
    loop = _loop(provider, tmp_path, registry, skill_resolver=resolver)
    await loop.arun_turn("go")
    (use,) = _tool_uses(loop)
    assert use.name == "Skill"
    (res,) = _tool_results(loop)
    assert not res.is_error, res.output
    return resolver


async def test_skill_loads_under_claude_codes_call_shape(tmp_path: Path) -> None:
    resolver = await _run_skill_call(tmp_path, "Skill", {"skill": "aspirations", "args": "loop"})
    assert resolver.loaded == [("aspirations", "loop")]


async def test_skill_loads_under_the_pre_0190_call_shape(tmp_path: Path) -> None:
    resolver = await _run_skill_call(tmp_path, "use_skill", {"name": "aspirations", "args": "loop"})
    assert resolver.loaded == [("aspirations", "loop")]


def test_skill_schema_names_the_parameter_skill() -> None:
    spec = UseSkillTool().spec
    assert spec.name == "Skill"
    assert "skill" in spec.parameters["properties"]
    assert spec.parameters.get("required") == ["skill"]


# ── (4) shape adapters for the tools that keep Zak Code's own name ───────────────


async def test_todowrite_shape_builds_the_plan(tmp_path: Path) -> None:
    registry = ToolRegistry()
    registry.register(UpdatePlanTool(), aliases=["TodoWrite"])
    todos = [
        {"content": "Read the failing test", "status": "completed", "activeForm": "Reading"},
        {"content": "Fix the parser", "status": "in_progress", "activeForm": "Fixing"},
        {"content": "Run the suite", "status": "pending", "activeForm": "Running"},
    ]
    provider = ScriptedProvider([call_tool("TodoWrite", {"todos": todos}), reply("done")])
    loop = _loop(provider, tmp_path, registry)
    await loop.arun_turn("plan it")
    (use,) = _tool_uses(loop)
    assert use.name == "update_plan"
    (res,) = _tool_results(loop)
    assert not res.is_error, res.output
    assert res.data is not None and res.data["task_count"] == 3
    assert "Fix the parser" in res.output


class _FakeSpawner:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    async def spawn(self, *, type_name: str, prompt: str) -> SubAgentResult:
        self.calls.append((type_name, prompt))
        return SubAgentResult(name=type_name, summary=f"{prompt}!", usage=Usage(total_tokens=1))

    def available_types(self) -> list[str]:
        return ["general-purpose", "explore"]

    def default_type(self) -> str:
        return "general-purpose"


async def test_task_single_delegation_shape_is_one_subtask(tmp_path: Path) -> None:
    spawner = _FakeSpawner()
    result = await TaskTool().execute(
        {"description": "find it", "prompt": "find the parser", "subagent_type": "explore"},
        ToolContext(workspace_root=tmp_path, spawner=spawner),
    )
    assert not result.is_error, result.output
    assert spawner.calls == [("explore", "find the parser")]
    assert result.data is not None and result.data["count"] == 1


# ── (5) hooks: a matcher in any spelling gates the canonical tool ────────────────


def _hook_spec(matcher: str) -> HookSpec:
    return HookSpec(event=HookEvent.PRE_TOOL_USE, command=["x"], matcher=matcher)


def test_matcher_fires_in_every_spelling() -> None:
    assert _hook_spec("bash").matches("Bash", ("bash", "sh", "shell"))
    assert _hook_spec("bash").matches("Bash")  # the pre-0190 spelling needs no aliases
    assert _hook_spec("Bash").matches("Bash")
    assert _hook_spec("MultiEdit").matches("Edit")
    assert _hook_spec("TodoWrite").matches("update_plan")
    assert _hook_spec("use_skill").matches("Skill")
    assert _hook_spec("Skill").matches("Skill")
    assert not _hook_spec("Read").matches("Bash", ("bash",))


async def test_in_loop_hook_payload_carries_the_registry_aliases(tmp_path: Path) -> None:
    seen: list[HookPayload] = []

    def capture(payload: HookPayload) -> None:
        seen.append(payload)

    (tmp_path / "a.txt").write_text("alpha\n", encoding="utf-8")
    registry = ToolRegistry()
    registry.register(ReadFileTool(), aliases=["read_file", "cat"])
    hooks = HookManager(in_process={HookEvent.PRE_TOOL_USE: [capture]})
    provider = ScriptedProvider([call_tool("cat", {"path": "a.txt"}), reply("done")])
    loop = _loop(provider, tmp_path, registry, hooks=hooks)
    await loop.arun_turn("read")
    pre = [p for p in seen if p.event is HookEvent.PRE_TOOL_USE]
    assert pre and pre[0].tool_name == "Read"
    assert set(pre[0].tool_aliases) >= {"read_file", "cat"}
    # The in-process payload carries the aliases; the wire payload does not (Claude Code's
    # shape, nothing more) and spells the tool the way Claude Code does.
    wire = json.loads(wire_payload(pre[0]))
    assert wire["tool_name"] == "Read"
    assert "tool_aliases" not in wire
    # (the loop resolves the path against the workspace before the hook sees it)
    assert set(wire["tool_input"]) == {"file_path"}
    assert wire["tool_input"]["file_path"].endswith("a.txt")


def test_wire_payload_spells_the_shape_kept_tools_as_claude_code_does(tmp_path: Path) -> None:
    payload = HookPayload(
        event=HookEvent.PRE_TOOL_USE,
        tool_name="update_plan",
        arguments={"tasks": []},
        cwd=str(tmp_path),
        session_id="sid-1",
    )
    assert json.loads(wire_payload(payload))["tool_name"] == "TodoWrite"
    legacy = payload.model_copy(update={"tool_name": "use_skill", "arguments": {"skill": "x"}})
    assert json.loads(wire_payload(legacy))["tool_name"] == "Skill"


# ── (6) permissions and operator config resolve either spelling ─────────────────


def _spec(name: str, tier: PermissionTier) -> ToolSpec:
    return ToolSpec(
        name=name,
        description=name,
        parameters={"type": "object", "properties": {}},
        required_permission=tier,
        concurrency=ConcurrencyClass.NEVER_PARALLEL,
    )


BASH = _spec("Bash", PermissionTier.DANGER_FULL_ACCESS)
WEBFETCH = _spec("WebFetch", PermissionTier.READ_ONLY)


def test_trust_override_written_before_the_rename_still_applies() -> None:
    policy = PermissionPolicy(PermissionMode.ASK, tool_mode_overrides={"bash": "allow"})
    decision, _ = policy.decide(BASH, {"command": "ls"})
    assert decision is PermissionDecision.ALLOW


def test_denied_and_confirm_tools_resolve_either_spelling() -> None:
    denied = PermissionPolicy(PermissionMode.ALLOW, extra_denied_tools={"web_fetch"})
    assert denied.decide(WEBFETCH, {"url": "https://x"})[0] is PermissionDecision.DENY
    confirm = PermissionPolicy(PermissionMode.ALLOW, confirm_tools={"web_fetch"})
    assert confirm.decide(WEBFETCH, {"url": "https://x"})[0] is PermissionDecision.ASK


def test_a_session_grant_recorded_under_the_old_spelling_covers_the_tool() -> None:
    policy = PermissionPolicy(PermissionMode.ASK)
    policy._session_allow.add("bash")  # a grant restored from a pre-0190 session store
    assert policy.auto_allows(BASH, {"command": "ls"})


def test_exposure_filter_globs_match_aliases_too() -> None:
    registry = default_registry()
    registry.set_exposure_filter(deny=["web_*"])
    assert not registry.exposure_allows("WebFetch")
    assert not registry.exposure_allows("WebSearch")
    assert registry.exposure_allows("Read")
    advertised = {d["function"]["name"] for d in registry.definitions()}
    assert "WebFetch" not in advertised and "Read" in advertised
