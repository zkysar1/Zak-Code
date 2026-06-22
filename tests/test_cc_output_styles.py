"""Claude Code output-style support — the loader, prompt integration, and facade wiring.

These tests prove the Claude Code *output style* contract with GENERIC fixtures only (no
plug-in named): a workspace selects a style via ``outputStyle`` in ``.claude/settings.json``
(or ``settings.local.json``, which overrides it), the body lives in
``.claude/output-styles/<name>.md``, and a faithful host folds that body into the STABLE
(cacheable) system-prompt tier so it shapes generation — the same seam always-on rules use.

The load-bearing guarantee is the OFF default: with the flag off (or nothing configured),
the system prompt must be byte-identical to today. See docs/CLAUDE-CODE-HOST-ROADMAP.md.
"""

from __future__ import annotations

import json
from pathlib import Path

from zakcode.agent import DYNAMIC_BOUNDARY, SystemPromptBuilder
from zakcode.config import Settings, load_settings
from zakcode.output_styles import MAX_OUTPUT_STYLE_CHARS, load_active_output_style


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _settings(workspace: Path, name: str, *, local: bool = False) -> Path:
    filename = "settings.local.json" if local else "settings.json"
    return _write(workspace / ".claude" / filename, json.dumps({"outputStyle": name}))


def _style(workspace: Path, name: str, body: str) -> Path:
    return _write(workspace / ".claude" / "output-styles" / f"{name}.md", body)


# ── loader: load_active_output_style ─────────────────────────────────────────────


def test_selected_style_body_is_loaded_and_framed(tmp_path: Path) -> None:
    # The selected outputStyle's .md body is loaded and returned as a labelled block that
    # names the style (so the model knows which voice it is in).
    _settings(tmp_path, "terse")
    _style(tmp_path, "terse", "Answer in one sentence. No preamble.")
    block, reason = load_active_output_style(tmp_path)
    assert reason is None
    assert block is not None
    assert "Output style (terse)" in block
    assert "Answer in one sentence. No preamble." in block


def test_frontmatter_is_stripped_from_the_body(tmp_path: Path) -> None:
    # A style file may carry a `---` frontmatter block (CC writes name/description there);
    # the host strips it and injects only the instruction body.
    _settings(tmp_path, "house")
    _style(
        tmp_path,
        "house",
        "---\nname: House Voice\ndescription: the team's style\n---\nUSE_BULLET_POINTS_MARKER",
    )
    block, reason = load_active_output_style(tmp_path)
    assert reason is None
    assert block is not None
    assert "USE_BULLET_POINTS_MARKER" in block
    assert "description:" not in block  # frontmatter gone
    assert "the team's style" not in block


def test_unconfigured_workspace_injects_nothing(tmp_path: Path) -> None:
    # No outputStyle key anywhere → a clean no-op (None, reason), never a crash.
    block, reason = load_active_output_style(tmp_path)
    assert block is None
    assert reason is not None  # a human-readable "no outputStyle configured"


def test_missing_style_name_with_no_file_is_no_crash(tmp_path: Path) -> None:
    # A selection naming a style whose .md does not exist → (None, reason); defensive, no raise.
    _settings(tmp_path, "ghost")  # no .claude/output-styles/ghost.md
    block, reason = load_active_output_style(tmp_path)
    assert block is None
    assert reason is not None and "ghost" in reason


def test_empty_style_body_is_not_injected(tmp_path: Path) -> None:
    # A style file that is empty (or only frontmatter) yields nothing to inject.
    _settings(tmp_path, "blank")
    _style(tmp_path, "blank", "---\nname: blank\n---\n   \n")
    block, reason = load_active_output_style(tmp_path)
    assert block is None
    assert reason is not None and "empty" in reason


def test_malformed_settings_json_does_not_crash(tmp_path: Path) -> None:
    # A broken settings.json is data, not a crash: (None, reason) with a parse-error note.
    _write(tmp_path / ".claude" / "settings.json", "{ not json")
    block, reason = load_active_output_style(tmp_path)
    assert block is None
    assert reason is not None and "parse error" in reason


def test_settings_local_overrides_settings_json(tmp_path: Path) -> None:
    # CC layers settings.local.json OVER settings.json; the local selection wins.
    _settings(tmp_path, "verbose")  # settings.json selects "verbose"
    _settings(tmp_path, "terse", local=True)  # settings.local.json selects "terse"
    _style(tmp_path, "verbose", "VERBOSE_BODY_MARKER")
    _style(tmp_path, "terse", "TERSE_BODY_MARKER")
    block, reason = load_active_output_style(tmp_path)
    assert reason is None
    assert block is not None
    assert "TERSE_BODY_MARKER" in block  # local override wins
    assert "VERBOSE_BODY_MARKER" not in block


def test_local_selection_falls_back_when_local_lacks_the_key(tmp_path: Path) -> None:
    # settings.local.json present but WITHOUT an outputStyle key must not clobber the shared
    # selection (only a non-empty string is a selection).
    _settings(tmp_path, "verbose")
    _write(tmp_path / ".claude" / "settings.local.json", json.dumps({"model": "x"}))
    _style(tmp_path, "verbose", "VERBOSE_BODY_MARKER")
    block, reason = load_active_output_style(tmp_path)
    assert reason is None
    assert block is not None and "VERBOSE_BODY_MARKER" in block


def test_body_is_capped_to_the_char_budget(tmp_path: Path) -> None:
    # A huge style file is truncated to the per-style cap so the cached prefix stays bounded.
    _settings(tmp_path, "big")
    _style(tmp_path, "big", "z" * (MAX_OUTPUT_STYLE_CHARS * 2))
    block, reason = load_active_output_style(tmp_path)
    assert reason is None
    assert block is not None
    assert block.count("z") == MAX_OUTPUT_STYLE_CHARS  # body truncated to the cap


# ── prompt-builder integration ───────────────────────────────────────────────────


def test_output_style_lands_in_stable_tier(tmp_path: Path) -> None:
    # The style block is static, operator-selected guidance → it belongs in the cacheable
    # STABLE tier, above the dynamic boundary (alongside rules).
    settings = load_settings(workspace_root=tmp_path)
    style_block = "Output style (terse):\n\nSTYLE_MARKER_91"
    prompt = SystemPromptBuilder(output_style=style_block).build(settings)
    boundary_at = prompt.index(DYNAMIC_BOUNDARY)
    assert "STYLE_MARKER_91" in prompt[:boundary_at]


def test_no_output_style_is_byte_for_byte_default(tmp_path: Path) -> None:
    # Cache-stability guard: output_style=None must reproduce the default prompt EXACTLY — the
    # presence of the new parameter cannot perturb a prompt for a workspace that uses no style.
    settings = load_settings(workspace_root=tmp_path, default_model="openai/gpt-4o")
    assert SystemPromptBuilder(output_style=None).build(settings) == SystemPromptBuilder().build(
        settings
    )


def test_output_style_sits_after_rules_in_stable_tier(tmp_path: Path) -> None:
    # Ordering contract: rules first, then the output style, both above the boundary — so the
    # style refines behaviour established by the operator's standing rules.
    settings = load_settings(workspace_root=tmp_path)
    prompt = SystemPromptBuilder(
        rules="Project rules:\n\n## x\nRULE_MARKER",
        output_style="Output style (terse):\n\nSTYLE_MARKER",
    ).build(settings)
    stable = prompt[: prompt.index(DYNAMIC_BOUNDARY)]
    assert "RULE_MARKER" in stable and "STYLE_MARKER" in stable
    assert stable.index("RULE_MARKER") < stable.index("STYLE_MARKER")


# ── facade wiring (the real Agent path) ──────────────────────────────────────────


def _agent(workspace: Path, **kwargs: object) -> object:
    from zakcode import Agent
    from zakcode.evals.harness import ScriptedProvider, reply

    return Agent(
        provider=ScriptedProvider(script=[reply("ok")]),
        default_model="scripted/test",
        workspace_root=str(workspace),
        **kwargs,  # type: ignore[arg-type]
    )


def test_agent_enable_output_style_renders_into_prompt(tmp_path: Path) -> None:
    # The end-to-end path: enable_output_style=True + a configured style puts the body into the
    # built system prompt's stable tier.
    _settings(tmp_path, "terse")
    _style(tmp_path, "terse", "ENABLED_STYLE_MARKER")
    agent = _agent(tmp_path, enable_output_style=True)
    prompt = agent.loop.prompt_builder.build(agent.settings)  # type: ignore[attr-defined]
    boundary_at = prompt.index(DYNAMIC_BOUNDARY)
    assert "ENABLED_STYLE_MARKER" in prompt[:boundary_at]


def test_agent_off_by_default_omits_the_style(tmp_path: Path) -> None:
    # THE host promise: a workspace may carry another runtime's output-style config; without the
    # opt-in flag a faithful host injects NOTHING, and the prompt equals the no-config prompt.
    _settings(tmp_path, "terse")
    _style(tmp_path, "terse", "SHOULD_NOT_APPEAR_MARKER")
    agent = _agent(tmp_path)  # flag off (default)
    prompt = agent.loop.prompt_builder.build(agent.settings)  # type: ignore[attr-defined]
    assert "SHOULD_NOT_APPEAR_MARKER" not in prompt

    # And it is byte-identical to a fresh builder on the same settings (no perturbation at all).
    baseline = SystemPromptBuilder().build(agent.settings)  # type: ignore[attr-defined]
    assert prompt == baseline


def test_agent_settings_flag_enables_without_explicit_arg(tmp_path: Path) -> None:
    # The Settings.output_style flag (env ZAKCODE_OUTPUT_STYLE) enables it without an explicit
    # enable_output_style= arg — mirroring settings_hooks/settings_permissions.
    from zakcode import Agent
    from zakcode.evals.harness import ScriptedProvider, reply

    _settings(tmp_path, "terse")
    _style(tmp_path, "terse", "SETTINGS_FLAG_MARKER")
    agent = Agent(
        provider=ScriptedProvider(script=[reply("ok")]),
        settings=Settings(
            default_model="scripted/test", workspace_root=tmp_path, output_style=True
        ),
    )
    prompt = agent.loop.prompt_builder.build(agent.settings)
    assert "SETTINGS_FLAG_MARKER" in prompt[: prompt.index(DYNAMIC_BOUNDARY)]


def test_agent_missing_style_file_does_not_crash_construction(tmp_path: Path) -> None:
    # enable_output_style=True with a selection but NO body file must build cleanly (no inject,
    # an error recorded on the agent), never raise.
    _settings(tmp_path, "ghost")  # no matching .md
    agent = _agent(tmp_path, enable_output_style=True)
    prompt = agent.loop.prompt_builder.build(agent.settings)  # type: ignore[attr-defined]
    assert "ghost" not in prompt
    assert agent.output_style_error is not None  # type: ignore[attr-defined]


def test_injected_prompt_builder_is_populated_with_output_style(tmp_path: Path) -> None:
    # enable_output_style is not a silent no-op: an injected builder's empty output_style slot
    # gets filled (mirrors the rules slot).
    _settings(tmp_path, "terse")
    _style(tmp_path, "terse", "INJECTED_BUILDER_STYLE")
    pb = SystemPromptBuilder()
    agent = _agent(tmp_path, enable_output_style=True, prompt_builder=pb)
    assert agent.loop.prompt_builder is pb  # type: ignore[attr-defined]
    assert pb.output_style is not None and "INJECTED_BUILDER_STYLE" in pb.output_style
