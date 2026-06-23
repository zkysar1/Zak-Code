"""Output styles — Claude Code's active *output style* folded into the system prompt.

An *output style* is Claude Code's mechanism for reshaping how the assistant writes its
final answer (terse, explanatory, a custom house voice). The contract has two halves:

* a **selection** — ``outputStyle: "<name>"`` in ``<workspace>/.claude/settings.json``
  (and per-machine ``.claude/settings.local.json``, which overrides it), naming the style
  the workspace currently wants; and
* a **body** — the file ``<workspace>/.claude/output-styles/<name>.md``, a Markdown
  document (optionally with a ``---`` frontmatter block to strip) whose text is the
  style's standing instructions.

A faithful host honours that contract by injecting the selected style's body into the
**stable, cacheable tier** of the system prompt — the SAME seam :mod:`zakcode.rules`
uses for always-on guidance. That is the right tier: a style is constant for the session
(the selection does not change mid-turn), so a provider can cache the prefix, and it sits
beside the operator's rules as another piece of standing "how to behave" guidance.

This mirrors the rules model deliberately but is NOT the rules model: rules are a *flat
directory* of always-applied ``*.md`` files; an output style is a *single named file*
chosen by a settings key. We re-use the rules frontmatter splitter
(:func:`zakcode.rules._split_frontmatter`) so the two parse style files identically, but
discovery is its own thing.

Everything here is **opt-in and defensive**. The behaviour is gated behind
``Settings.output_style`` (env ``ZAKCODE_OUTPUT_STYLE``), off by default, so a workspace
carrying another runtime's output-style config does not silently reshape this engine's
voice. When off — or when nothing is configured, or the named file is missing/empty — the
loader injects nothing and the system prompt is byte-identical to today. A malformed
setting or unreadable file is reported as a reason string, never raised: a bad style is
data, not a crash.

Because the rendered block sits in the cached prefix it must be bounded: the style body
is capped at :data:`MAX_OUTPUT_STYLE_CHARS`, so a huge ``<name>.md`` can never blow the
context window (the same guarantee :mod:`zakcode.rules` makes for a large rules dir).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from zakcode.rules import _split_frontmatter

logger = logging.getLogger("zakcode.output_styles")

#: Largest slice of an output-style body folded into the prompt (~16 KB). A style is a
#: single document (not a directory of many files like rules), so it gets a more generous
#: cap than a single rule while still being firmly bounded for the cached prefix.
MAX_OUTPUT_STYLE_CHARS = 16_384

#: Settings files consulted for the ``outputStyle`` selection, in increasing precedence:
#: the shared ``settings.json`` first, then the per-machine ``settings.local.json`` (which
#: overrides it) — mirroring :func:`zakcode.hooks.settings_loader.load_settings_hooks`.
_SETTINGS_FILES = ("settings.json", "settings.local.json")


def _read_output_style_name(workspace_root: Path) -> tuple[str | None, str | None]:
    """Read the active ``outputStyle`` name from the workspace settings files.

    Returns ``(name, error)``. The shared ``.claude/settings.json`` is read first, then
    ``.claude/settings.local.json`` overrides it (local wins, exactly as Claude Code
    layers them). A missing file is simply skipped; an unparseable one is recorded in
    ``error`` but does not stop the other file from being read. ``name`` is ``None`` when
    no file declares a non-empty ``outputStyle``.
    """
    name: str | None = None
    error: str | None = None
    for filename in _SETTINGS_FILES:
        path = workspace_root / ".claude" / filename
        if not path.is_file():
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            # A bad settings file is data, not a crash: note it and keep going so a valid
            # later file (settings.local.json) can still supply the selection.
            error = f"{path.name}: parse error: {exc}"
            logger.warning("output style: %s", error)
            continue
        if not isinstance(data, dict):
            continue
        value = data.get("outputStyle")
        # Only a non-empty string is a selection; anything else (absent, null, a typo'd
        # non-string) leaves the running value untouched so local can still override.
        if isinstance(value, str) and value.strip():
            name = value.strip()
    return name, error


def load_active_output_style(workspace_root: str | Path) -> tuple[str | None, str | None]:
    """Load the workspace's active Claude Code output style as a prompt block.

    Resolves the ``outputStyle`` selection from ``.claude/settings.json`` then
    ``.claude/settings.local.json`` (local overrides), loads the body from
    ``.claude/output-styles/<name>.md``, strips an optional ``---`` frontmatter block, and
    returns a labelled, ready-to-inject block for the stable system-prompt tier.

    Returns ``(rendered_block, reason)``:

    * ``(block, None)`` — a style was selected and its non-empty body loaded; ``block`` is
      framed as ``Output style (<name>): …`` and capped at :data:`MAX_OUTPUT_STYLE_CHARS`.
    * ``(None, reason)`` — nothing to inject; ``reason`` says why (no selection, a missing
      or unreadable ``<name>.md``, an empty body, or a malformed settings file). The caller
      may log ``reason``; it is never raised.

    Defensive throughout: an unknown/missing style name or file yields ``(None, reason)``,
    never an exception, so enabling the feature can never destabilise a turn.
    """
    root = Path(workspace_root)
    name, settings_error = _read_output_style_name(root)
    if name is None:
        # No selection. Surface a parse error if that is *why* there is no name; otherwise
        # it is simply unconfigured (a no-op, the common case).
        return None, settings_error or "no outputStyle configured"

    style_path = root / ".claude" / "output-styles" / f"{name}.md"
    try:
        raw = style_path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        # UnicodeError (a ValueError, not an OSError) covers a non-UTF-8 style file; catch
        # it here so a stray encoding never crashes the prompt build.
        reason = f"output style {name!r}: cannot read {style_path}: {exc}"
        logger.warning("%s", reason)
        return None, reason

    _meta, body = _split_frontmatter(raw)
    content = body.strip()
    if not content:
        reason = f"output style {name!r}: empty body in {style_path}"
        logger.warning("%s", reason)
        return None, reason

    if len(content) > MAX_OUTPUT_STYLE_CHARS:
        content = content[:MAX_OUTPUT_STYLE_CHARS]

    # Frame it like the rules block: a labelled header (carrying the style name so the
    # model knows which voice it is in) followed by the body. Standing guidance on HOW to
    # respond — operator-selected, so injected un-fenced like rules, not as untrusted data.
    header = (
        f"Output style ({name}) — shape your responses to follow this style "
        "(operator-selected; always apply it):"
    )
    return f"{header}\n\n{content}", None


__all__ = [
    "MAX_OUTPUT_STYLE_CHARS",
    "load_active_output_style",
]
