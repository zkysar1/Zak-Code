"""Text-based tool-calling fallback for models without native function-calling.

Many capable *local* models (small Llama/Qwen/Mistral GGUFs served by Ollama,
llama.cpp, vLLM, ...) cannot emit the structured ``tool_calls`` field the OpenAI
wire format defines. When such a model is handed native ``tools`` it silently
ignores them and just chats — the agent loop sees ``has_tool_calls == False`` and
ends the turn, so the model can never actually *act*. That is the "local models
fall silent" gap.

:class:`TextToolCallingProvider` closes it as a **composable wrapper** around any
:class:`~zds_llm_provider.types.Provider`. It never imports a vendor SDK and never
touches the agent loop — it sits between them and, when needed, swaps native tool
wiring for a plain-text protocol the model *can* follow:

* **Outbound** — instead of passing ``tools=`` to the backend, it appends a
  human-readable *tool protocol* to the system prompt (each tool's name,
  description, and JSON-Schema parameters, plus the exact emit format) and
  rewrites any prior native tool-calls / tool-results in the history into the same
  ``<tool_call>`` / ``<tool_result>`` text shape (:func:`textify_messages`), so the
  *whole* multi-turn conversation stays legible to a model that has no tool role.
* **Inbound** — it parses ``<tool_call>{json}</tool_call>`` blocks (and a
  ```` ```tool_call ```` fenced alternate) back out of the assistant's text into
  canonical :class:`~zds_llm_provider.types.ToolCall` objects
  (:func:`parse_text_tool_calls`), strips them from the visible text, and returns a
  normal :class:`~zds_llm_provider.types.LLMResult`. The loop is none the wiser.

Three modes (``tool_calling_mode`` in :class:`~zakcode.config.Settings`):

* ``"native"`` — pure passthrough; never inject a protocol, never parse text.
* ``"text"``   — always use the text protocol (force it on, for any backend).
* ``"auto"``   — *(default)* use native when the wrapped provider reports
  ``capabilities().supports_tools``; otherwise fall back to text. In native mode it
  still *salvages* a text tool-call the model emitted on its own (a model that
  ignored its native tools but wrote a ``<tool_call>`` block anyway), so a
  capable-but-quirky model never falls silent either.

The markers are deliberately strict (an explicit tag/fence containing a JSON
object with a ``"name"`` key), so prose almost never trips the parser.
"""

from __future__ import annotations

import json
import re
from collections.abc import AsyncIterator, Collection, Iterator
from typing import Any

from zds_llm_provider.messages import (
    Message,
    TextBlock,
    ThinkingBlock,
    ToolResultBlock,
    ToolUseBlock,
)
from zds_llm_provider.types import (
    Capabilities,
    LLMResult,
    Provider,
    ProviderStreamEvent,
    StreamDone,
    StreamTextDelta,
    StreamToolCallDelta,
    StreamUsage,
    ToolCall,
)

#: Accepted tool-calling modes (anything else normalizes to ``"auto"``).
TOOL_CALLING_MODES = ("auto", "native", "text")

# A tool call is an explicit ``<tool_call>...</tool_call>`` tag (the canonical form,
# which Qwen/Hermes-style local models are already trained to emit) OR a
# ```` ```tool_call ... ``` ```` fenced block (a common Markdown alternate). The tag
# form is recovered by a balanced-brace scan (see :func:`_iter_tag_calls`), NOT this
# regex, so an argument value that itself contains a literal ``</tool_call>`` does not
# truncate the body the way a non-greedy ``...</tool_call>`` match would; ``_TAG_RE``
# remains only to recover a tag wrapped inside a fence (see :func:`_parse_fence_body`).
_TAG_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL | re.IGNORECASE)
_FENCE_RE = re.compile(r"```tool_call\s*(.*?)```", re.DOTALL | re.IGNORECASE)

# Opener / closer for the balanced-brace tag scan. Matching the opener alone (not a
# closed pair) lets :func:`_iter_tag_calls` locate the JSON object by brace-counting,
# which both survives an embedded ``</tool_call>`` and recovers a *truncated* trailing
# call whose object is otherwise complete (subsuming the old end-anchored regex).
_TAG_OPENER_RE = re.compile(r"<\s*tool_call\s*>", re.IGNORECASE)
_CLOSE_TAG_RE = re.compile(r"</\s*tool_call\s*>", re.IGNORECASE)

# End-anchored fallback for a *truncated* trailing fenced tool-call (the model hit its
# output budget mid-emit, so the closing fence never arrived). Requires a complete
# trailing JSON object so a half-written object is left in the text rather than
# mis-parsed; a properly-closed fence ends in `` ``` `` (not ``}``), so this never
# double-matches one ``_FENCE_RE`` already caught.
_FENCE_OPEN_RE = re.compile(r"```tool_call\s*(\{.*\})\s*$", re.DOTALL | re.IGNORECASE)

# Frame sentinels that untrusted tool output must not be able to forge when it is
# rendered into a ``<tool_result>`` block (see :func:`_defang_sentinels`). Matching
# is case-insensitive and covers the opening/closing tags and the fenced form.
_SENTINEL_RE = re.compile(r"</?\s*tool_(?:call|result)|```tool_call", re.IGNORECASE)

# Stop sequences injected on the TEXT-mode path so a weak local model cannot
# autoregress past one tool call into a fabricated ``<tool_result>`` or leak a
# chat-template token. These are protocol frame markers + chat-template sentinels
# common across open models (ChatML / Llama) — structural strings, never a
# vendor-name branch. litellm forwards ``stop`` to the backend; the text path only
# targets local backends (Ollama), which accept many stop strings.
_STOP_SENTINELS: tuple[str, ...] = (
    "<tool_result>",
    "<|im_start|>",
    "<|im_end|>",
    "<|eot_id|>",
    "<|eom_id|>",
)
#: Added to the stop list only in single-tool-per-turn mode: generation halts right
#: after the one tool call. KNOWN LIMITATION: this is a plain stop *string*, so a model
#: writing an argument value that itself contains the literal ``</tool_call>`` (e.g.
#: editing this protocol's own docs/tests) is truncated mid-emit by the backend and
#: that one call is lost. The inbound parser brace-balances and so survives an embedded
#: marker in a *fully received* block (multi-tool mode, where this stop is off), but it
#: cannot recover content the backend never sent. For that rare self-referential case,
#: run with ``single_tool_per_turn`` off or avoid emitting the literal marker.
_CLOSE_TAG = "</tool_call>"

#: Generic ``<|...|>`` chat-template tokens, defanged out of residual text so a leaked
#: sentinel never renders or re-enters history (defense-in-depth behind the stop
#: sequences). The ``<|...|>`` shape is a cross-model convention, not vendor-specific.
_TEMPLATE_TOKEN_RE = re.compile(r"<\|[^|>]{0,40}\|>")

#: A trailing ``<tool_call>`` opener with no matching close — a tool call the model began
#: but never finished (truncated/abandoned); stripped so it does not render as cruft.
_TRAILING_OPEN_CALL_RE = re.compile(
    r"<\s*tool_call\s*>(?:(?!</\s*tool_call\s*>).)*\Z", re.IGNORECASE | re.DOTALL
)


# ── protocol rendering (outbound) ────────────────────────────────────────────


def render_tool_protocol(tools: list[dict[str, Any]], *, single_tool: bool = False) -> str:
    """Render OpenAI-shaped tool definitions as a plain-text protocol block.

    The block tells the model exactly how to call a tool and lists every tool with
    its description and JSON-Schema parameters. Schemas are dumped with sorted keys
    so the rendered protocol is byte-stable across calls (cache-friendly and
    test-stable). Returns ``""`` for an empty tool list.
    """
    if not tools:
        return ""

    emit_rule = (
        "- Emit EXACTLY ONE `<tool_call>` block, then STOP — write nothing after it and "
        "wait for the result before doing anything else."
        if single_tool
        else "- Emit one `<tool_call>` block per tool call; you may emit several in a row."
    )

    lines: list[str] = [
        "# Tool calling",
        "",
        "You can call tools to accomplish the task. To call a tool, emit a "
        "`<tool_call>` block containing a single JSON object with a `name` and an "
        "`arguments` object, for example:",
        "",
        "<tool_call>",
        '{"name": "read_file", "arguments": {"path": "src/main.py"}}',
        "</tool_call>",
        "",
        "Rules:",
        emit_rule,
        "- `arguments` must be a JSON object matching that tool's parameters.",
        "- Put nothing but the JSON object inside a `<tool_call>` block; write any "
        "reasoning outside it.",
        "- NEVER write a `<tool_result>` block or invent a tool's output. You ONLY emit "
        "`<tool_call>`; the system runs the tool and gives you the real `<tool_result>` "
        "as your next input. Wait for it — do not imagine or continue the conversation.",
        "- NEVER emit special chat-template tokens (anything of the form `<|...|>`).",
        "- When the task is complete and you need no more tools, reply normally with "
        "no `<tool_call>` block.",
        "",
        "Worked example — you write ONLY the `<tool_call>`; the system then sends you the "
        "`<tool_result>` (you never write that yourself):",
        "",
        "<tool_call>",
        '{"name": "bash", "arguments": {"command": "pytest -q"}}',
        "</tool_call>",
        "",
        "the system then returns the real result, e.g.:",
        "",
        '<tool_result tool="bash">',
        "1 passed",
        "</tool_result>",
        "",
        "and only then do you continue, using that real result.",
        "",
        "Available tools:",
    ]

    for tool in tools:
        fn = tool.get("function", {}) if isinstance(tool, dict) else {}
        name = fn.get("name", "") if isinstance(fn, dict) else ""
        desc = (fn.get("description") or "").strip() if isinstance(fn, dict) else ""
        params = fn.get("parameters") if isinstance(fn, dict) else None
        if not isinstance(params, dict):
            params = {"type": "object", "properties": {}}
        lines.append("")
        lines.append(f"## {name}")
        if desc:
            lines.append(desc)
        lines.append("Parameters (JSON Schema):")
        lines.append(json.dumps(params, indent=2, sort_keys=True))

    return "\n".join(lines)


def _render_tool_call_text(name: str, arguments: dict[str, Any]) -> str:
    """Render one assistant tool-call as the ``<tool_call>`` text form."""
    payload = json.dumps({"name": name, "arguments": arguments}, default=str)
    return f"<tool_call>\n{payload}\n</tool_call>"


def _defang_sentinels(text: str) -> str:
    """Neutralize protocol frame markers in untrusted text.

    Tool output is attacker-influenceable (a read file, a shell echo, an MCP
    fetch). When it is rendered into a ``<tool_result>`` frame, a literal
    ``</tool_result>`` could close the frame early and let following bytes pose as
    harness text, and a literal ``<tool_call>`` block could try to look like an
    instruction. We insert a zero-width space after the leading ``<`` / `` ` `` of
    each sentinel so the marker no longer matches the frame (or the inbound
    parser, which never runs on this text anyway) while staying human-readable and
    leaving every other ``<``/``>`` untouched.
    """
    zwsp = "​"  # zero-width space, neutralizes the marker without hiding bytes
    return _SENTINEL_RE.sub(
        lambda m: m.group(0).replace("<", f"<{zwsp}", 1).replace("`", f"`{zwsp}", 1),
        text,
    )


def _strip_forged_frames(text: str) -> str:
    """Neutralize model-forged protocol frames + chat-template tokens in residual text.

    Defense-in-depth behind the stop sequences (:data:`_STOP_SENTINELS`): if a backend
    ignores ``stop``, a weak model may still emit a fabricated ``<tool_result>`` or leak
    a chat-template token (``<|im_start|>`` ...). Those are not recovered as tool calls,
    so without this they would render to the user *and* re-enter the next turn's context.
    We defang them in place with a zero-width space — never delete — so a legitimate
    answer that merely *quotes* such a token stays readable.
    """
    if not text:
        return text
    # Drop a trailing, unterminated <tool_call> opener — a tool call the model began but
    # never closed (truncated/abandoned). By the time this runs every COMPLETE call has
    # been parsed out, so a remaining opener is incomplete cruft, not a real call.
    text = _TRAILING_OPEN_CALL_RE.sub("", text).rstrip()
    return defang_untrusted(text)


def defang_untrusted(text: str) -> str:
    """Neutralize protocol frame markers AND chat-template tokens in untrusted text.

    Inserts a zero-width space after the leading ``<`` / `` ` `` of any ``<tool_call>`` /
    ``<tool_result>`` frame (or ```` ```tool_call ````) and any ``<|...|>`` chat-template
    token, so the bytes stay human-readable but can no longer forge a protocol frame or
    leak a template token. **Never deletes** content (unlike :func:`_strip_forged_frames`,
    which also drops a trailing unterminated opener) — so it is safe to run over content
    that must be preserved verbatim: file read-backs, shell output, skill bodies, or any
    untrusted bytes a *harness-authored* message embeds. This is the trust-boundary rule
    for the user role, mirroring :func:`_defang_sentinels` for the tool role.
    """
    if not text:
        return text
    zwsp = "​"  # zero-width space
    out = _defang_sentinels(text)  # <tool_call> / <tool_result> frames
    return _TEMPLATE_TOKEN_RE.sub(lambda m: f"<{zwsp}" + m.group(0)[1:], out)


def _sanitize_tool_name(name: str) -> str:
    """Strip characters that would break the ``tool="..."`` attribute (incl. MCP names)."""
    return re.sub(r'["<>]', "", name)


def _render_tool_result_text(name: str, output: str, is_error: bool) -> str:
    """Render one tool result as the ``<tool_result>`` text form fed back as input.

    The tool name and (untrusted) output are sanitized so neither can forge or
    escape the frame — see :func:`_defang_sentinels`.
    """
    safe_name = _sanitize_tool_name(name)
    attrs = f' tool="{safe_name}"' if safe_name else ""
    if is_error:
        attrs += ' error="true"'
    return f"<tool_result{attrs}>\n{_defang_sentinels(output)}\n</tool_result>"


def textify_messages(messages: list[Message]) -> list[Message]:
    """Rewrite native tool-calls/results in the history into text form.

    A text-protocol model has no ``assistant.tool_calls`` field and no ``tool``
    role, so replaying history that contains them would send the model a shape it
    never produced. This rewrites, without mutating the inputs:

    * each assistant :class:`~zds_llm_provider.messages.ToolUseBlock` -> a ``<tool_call>``
      text block (so the assistant "said" what it called), and
    * each ``tool``-role message -> a ``user`` message of ``<tool_result>`` blocks
      (so results arrive as ordinary input the model can read).

    ``ThinkingBlock``s are dropped (never replayed). User/system messages pass
    through untouched. Tool-result blocks are labelled with the tool name resolved
    from the preceding assistant turn's tool-call ids.
    """
    out: list[Message] = []
    id_to_name: dict[str, str] = {}

    for msg in messages:
        if msg.role == "assistant":
            parts: list[str] = []
            for block in msg.blocks:
                if isinstance(block, TextBlock):
                    if block.text:
                        parts.append(block.text)
                elif isinstance(block, ToolUseBlock):
                    id_to_name[block.id] = block.name
                    parts.append(_render_tool_call_text(block.name, block.input))
                elif isinstance(block, ThinkingBlock):
                    continue
            text = "\n\n".join(parts)
            out.append(Message(role="assistant", blocks=[TextBlock(text=text)] if text else []))
        elif msg.role == "tool":
            parts = []
            for block in msg.blocks:
                if isinstance(block, ToolResultBlock):
                    name = id_to_name.get(block.tool_use_id, "")
                    parts.append(_render_tool_result_text(name, block.output, block.is_error))
            text = "\n\n".join(parts)
            out.append(Message(role="user", blocks=[TextBlock(text=text)]))
        else:
            out.append(msg)

    return out


# ── parsing (inbound) ────────────────────────────────────────────────────────


def _scan_json_object(text: str, start: int) -> tuple[str, int] | None:
    """Extract one balanced JSON object from ``text`` starting at index ``start``.

    ``start`` must index the opening ``{``. Scans forward tracking brace depth while
    respecting JSON string literals (a ``{`` / ``}`` / ``"`` inside a string does not
    affect depth) and backslash escapes, returning ``(object_text, end)`` where
    ``end`` is just past the matching ``}``. Returns ``None`` when the object never
    closes (the model was truncated mid-emit). Because it brace-balances rather than
    trusting a delimiter, a literal ``</tool_call>`` inside a string value is handled
    correctly — the central fix for "argument content contains the close marker".
    """
    if start < 0 or start >= len(text) or text[start] != "{":
        return None
    depth = 0
    in_str = False
    escaped = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_str = False
        elif ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1], i + 1
    return None


def _iter_tag_calls(text: str) -> Iterator[tuple[int, int, tuple[str, dict[str, Any]]]]:
    """Yield ``(start, end, (name, arguments))`` for each ``<tool_call>`` tag block.

    For each ``<tool_call>`` opener the JSON object is extracted by brace-balancing
    (:func:`_scan_json_object`) rather than by matching a literal ``</tool_call>``
    delimiter, so an argument value that itself contains ``</tool_call>`` is parsed
    intact. Any prose between the object's closing ``}`` and the trailing
    ``</tool_call>`` is folded into the removed span (so it never leaks to residual),
    but only when no *new* opener intervenes — so we never swallow a following,
    separate tool call. A truncated opener whose object never closes is skipped (left
    for :func:`_strip_forged_frames`). This subsumes both the closed-tag form and the
    truncated-but-complete trailing-tag form the old end-anchored regex handled.
    """
    pos = 0
    while True:
        opener = _TAG_OPENER_RE.search(text, pos)
        if opener is None:
            return
        brace = text.find("{", opener.end())
        if brace == -1:
            return
        scanned = _scan_json_object(text, brace)
        if scanned is None:
            # No balanced object after this opener (truncated mid-emit); step past the
            # opener so a later well-formed call can still be recovered.
            pos = opener.end()
            continue
        obj_text, obj_end = scanned
        end = obj_end
        close = _CLOSE_TAG_RE.search(text, obj_end)
        if close is not None and _TAG_OPENER_RE.search(text, obj_end, close.start()) is None:
            end = close.end()
        parsed = _try_parse_call(obj_text)
        if parsed is not None:
            yield opener.start(), end, parsed
        pos = end


def _loads_leading_object(inner: str) -> Any:
    """Parse the leading JSON value of ``inner``, tolerating trailing prose.

    Tries a strict ``json.loads`` first; on failure falls back to
    ``JSONDecoder().raw_decode``, which consumes only the leading JSON object and
    ignores anything a weak model appended after it (``{...} done now``). Returns
    ``None`` when no JSON value starts the (stripped) string — so prose *before* the
    JSON is still rejected, leaving the block untouched rather than mis-parsed.
    """
    inner = inner.strip()
    if not inner:
        return None
    try:
        return json.loads(inner)
    except (json.JSONDecodeError, ValueError):
        pass
    try:
        obj, _end = json.JSONDecoder().raw_decode(inner)
    except (json.JSONDecodeError, ValueError):
        return None
    return obj


def _try_parse_call(inner: str) -> tuple[str, dict[str, Any]] | None:
    """Parse the JSON inside one tool-call block into ``(name, arguments)``.

    Returns ``None`` (so the block is left untouched in the text) when the content
    is not a JSON object carrying a non-empty string ``name``. ``arguments`` is
    coerced to a ``dict``: a JSON object is used as-is; a JSON *string* is decoded
    once more (some models double-encode); anything else degrades to ``{}`` or, for
    a non-dict non-empty value, ``{"_raw": <repr>}`` — never raising. Trailing prose
    after the JSON object is tolerated (see :func:`_loads_leading_object`).
    """
    obj = _loads_leading_object(inner)
    if not isinstance(obj, dict):
        return None
    name = obj.get("name")
    if not isinstance(name, str) or not name:
        return None

    raw_args = obj.get("arguments", {})
    args: dict[str, Any]
    if isinstance(raw_args, dict):
        args = raw_args
    elif isinstance(raw_args, str):
        try:
            parsed = json.loads(raw_args)
        except (json.JSONDecodeError, ValueError):
            args = {"_raw": raw_args}
        else:
            args = parsed if isinstance(parsed, dict) else {"_raw": raw_args}
    elif raw_args is None:
        args = {}
    else:
        args = {"_raw": str(raw_args)}
    return name, args


def _parse_fence_body(body: str) -> tuple[str, dict[str, Any]] | None:
    """Parse a ```` ```tool_call ```` fence body, recovering a wrapped tag if present.

    Models sometimes wrap the canonical ``<tool_call>`` tag inside a Markdown fence
    (``` ```tool_call\\n<tool_call>{...}</tool_call>\\n``` ```). Such a body is not
    valid JSON, so we fall back to extracting the inner tag — letting the *fence*
    span (which includes the ``` markers) be the removed region, so no orphaned
    fence scaffolding leaks into the residual text.
    """
    parsed = _try_parse_call(body)
    if parsed is not None:
        return parsed
    inner = _TAG_RE.search(body)
    if inner is not None:
        return _try_parse_call(inner.group(1).strip())
    return None


def parse_text_tool_calls(
    text: str,
    *,
    allowed_names: Collection[str] | None = None,
    max_calls: int | None = None,
) -> tuple[str, list[ToolCall]]:
    """Extract tool-call blocks from assistant text.

    Returns ``(residual_text, calls)``: ``residual_text`` is the input with every
    *recovered* tool-call block removed and stripped; ``calls`` are the recovered
    :class:`~zds_llm_provider.types.ToolCall`s in document order, with synthesized
    ids ``call_0``, ``call_1``, .... Unrecoverable blocks are left in the text. The id
    counter is per-call (each response restarts at 0); the loop only correlates ids
    within a single iteration's batch, so per-response uniqueness suffices.

    Recovers these shapes robustly: a ``<tool_call>`` tag (parsed by brace-balancing,
    so an argument value containing a literal ``</tool_call>`` does not truncate it,
    and trailing prose after the JSON object is ignored), a closed ```` ```tool_call ````
    fence (incl. one wrapping a tag), and a *truncated* trailing tag/fence whose JSON
    is otherwise complete (the model hit its output budget mid-emit). When
    ``allowed_names`` is given, only calls naming one of those tools are recovered —
    used on the salvage path so a model that merely *shows* a tool-call example is
    not made to execute it.
    """
    if not text:
        return text, []

    candidates: list[tuple[int, int, tuple[str, dict[str, Any]]]] = []
    candidates.extend(_iter_tag_calls(text))
    for rx in (_FENCE_RE, _FENCE_OPEN_RE):
        for m in rx.finditer(text):
            parsed = _parse_fence_body(m.group(1).strip())
            if parsed is not None:
                candidates.append((m.start(), m.end(), parsed))

    if allowed_names is not None:
        allowed = set(allowed_names)
        candidates = [c for c in candidates if c[2][0] in allowed]
    if not candidates:
        return text, []

    # Sort by position and drop overlaps (a fence may *contain* a tag, and the two
    # forms can otherwise coincide) by keeping only matches that start at/after the
    # previous kept match's end. This guarantees disjoint removed spans, so no
    # scaffolding from an enclosing block survives into the residual text.
    candidates.sort(key=lambda x: x[0])
    kept: list[tuple[int, int, tuple[str, dict[str, Any]]]] = []
    last_end = -1
    for start, end, parsed in candidates:
        if start < last_end:
            continue
        kept.append((start, end, parsed))
        last_end = end

    # ``max_calls`` (single-tool-per-turn) keeps only the first N calls AND drops any
    # text *after* the last kept call — that trailing text is either absent (a stop
    # sequence fired right after the call) or post-call hallucination.
    truncate_tail = max_calls is not None and bool(kept)
    if max_calls is not None and max_calls >= 0:
        kept = kept[:max_calls]

    calls = [
        ToolCall(id=f"call_{i}", name=parsed[0], arguments=parsed[1])
        for i, (_s, _e, parsed) in enumerate(kept)
    ]

    out_parts: list[str] = []
    cursor = 0
    for start, end, _parsed in kept:
        out_parts.append(text[cursor:start])
        cursor = end
    out_parts.append("" if truncate_tail else text[cursor:])
    residual = "".join(out_parts).strip()
    return residual, calls


# ── the wrapper provider ─────────────────────────────────────────────────────


class TextToolCallingProvider(Provider):
    """Wrap a :class:`Provider`, giving tool-less models tool-calling via text.

    ``inner`` is the real backend (e.g. the litellm-backed provider). ``mode`` is
    one of :data:`TOOL_CALLING_MODES`. The wrapper is transparent when
    no ``tools`` are requested and in ``"native"`` mode; otherwise it applies the
    text protocol as documented on this module.
    """

    def __init__(
        self, inner: Provider, *, mode: str = "auto", single_tool_per_turn: bool = False
    ) -> None:
        self.inner = inner
        self.mode = mode if mode in TOOL_CALLING_MODES else "auto"
        self.single_tool_per_turn = single_tool_per_turn
        self._cached_native: bool | None = None
        # Token cost of the most recently injected text-mode protocol. The loop calls
        # count_tokens() WITHOUT tools, so without this the compactor would under-count the
        # real text-mode request (raw messages + this multi-KB protocol) and risk silently
        # overrunning num_ctx. 0 on the native path and before the first text-mode call.
        self._text_overhead_tokens = 0

    # ── mode resolution ──────────────────────────────────────────────────────

    def _native_supported(self) -> bool:
        """Whether the wrapped model supports native tool-calling (cached)."""
        if self._cached_native is None:
            try:
                self._cached_native = bool(self.inner.capabilities().supports_tools)
            except Exception:  # noqa: BLE001 — a probe failure must not force text mode
                # Assume native on an unknown probe so capable models keep their
                # efficient path; the auto-mode salvage still rescues a stray text
                # call. A model that is genuinely tool-less but misreports should be
                # pinned with tool_calling_mode="text".
                self._cached_native = True
        return self._cached_native

    def _use_text_mode(self, tools: list[dict[str, Any]] | None) -> bool:
        """Whether *this* call should use the text protocol instead of native tools."""
        if not tools:
            return False
        if self.mode == "text":
            return True
        if self.mode == "native":
            return False
        return not self._native_supported()  # auto

    @staticmethod
    def _offered_names(tools: list[dict[str, Any]] | None) -> set[str]:
        """The set of tool names offered in ``tools`` (for salvage name-gating)."""
        names: set[str] = set()
        for tool in tools or []:
            fn = tool.get("function", {}) if isinstance(tool, dict) else {}
            name = fn.get("name") if isinstance(fn, dict) else None
            if isinstance(name, str) and name:
                names.add(name)
        return names

    def _merge_stop_sentinels(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        """Merge the text-mode stop sequences into any caller-provided ``stop``.

        Truncates generation at protocol/template sentinels so a weak model cannot
        fabricate a ``<tool_result>`` or leak chat-template tokens; in
        single-tool-per-turn mode it also stops right after the closing
        ``</tool_call>``. Sentinels are ordered HIGHEST-VALUE FIRST (the structural
        ``</tool_call>`` / ``<tool_result>`` before chat-template tokens) and the final
        list is capped to the backend's ``max_stop_sequences`` capability when it declares
        one (e.g. OpenAI rejects >4) — so a real ``openai/`` model in text mode does not
        hard-400 while the most important sentinels survive. Returns a new dict — never
        mutates the caller's kwargs.
        """
        merged = dict(kwargs)
        # Front-load the structurally-important sentinels so they survive any cap below.
        stops: list[str] = []
        if self.single_tool_per_turn:
            stops.append(_CLOSE_TAG)  # </tool_call>: bounds the turn to one tool call
        stops.extend(_STOP_SENTINELS)  # <tool_result> then chat-template tokens
        existing = merged.get("stop")
        if isinstance(existing, str):
            stops = [existing, *stops]
        elif isinstance(existing, list):
            stops = [*existing, *stops]
        seen: set[str] = set()
        deduped: list[str] = []
        for s in stops:
            if s not in seen:
                seen.add(s)
                deduped.append(s)
        limit = self._stop_limit()
        if limit is not None:
            deduped = deduped[:limit]
        merged["stop"] = deduped
        return merged

    def _stop_limit(self) -> int | None:
        """The backend's maximum allowed number of stop sequences, or None if unbounded."""
        try:
            limit = self.inner.capabilities().max_stop_sequences
        except Exception:  # noqa: BLE001 — a capability probe must never break a call
            return None
        return limit if isinstance(limit, int) and limit >= 0 else None

    # ── Provider interface ───────────────────────────────────────────────────

    async def acomplete(
        self,
        messages: list[Message],
        *,
        system: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        response_format: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> LLMResult:
        # Fold structured-output into kwargs ONLY when requested, so the wrapper stays a
        # transparent passthrough for inner providers whose signature predates response_format
        # — a None must never be injected into every inner call (it would break a minimal
        # Provider that accepts neither response_format nor **kwargs).
        if response_format is not None:
            kwargs["response_format"] = response_format
        if not tools:
            return await self.inner.acomplete(messages, system=system, tools=None, **kwargs)

        if self._use_text_mode(tools):
            protocol = render_tool_protocol(tools, single_tool=self.single_tool_per_turn)
            new_system = f"{system}\n\n{protocol}" if system else protocol
            self._record_text_overhead(protocol)
            call_kwargs = self._merge_stop_sentinels(kwargs)
            result = await self.inner.acomplete(
                textify_messages(messages), system=new_system, tools=None, **call_kwargs
            )
            max_calls = 1 if self.single_tool_per_turn else None
            residual, calls = parse_text_tool_calls(result.text, max_calls=max_calls)
            residual = _strip_forged_frames(residual)
            return result.model_copy(update={"text": residual, "tool_calls": calls})

        # Native path. In auto mode, salvage a text tool-call the model emitted even
        # though it had native tools (it ignored them but wrote a block anyway). The
        # salvage is name-gated to the offered tools so a model that merely *shows* a
        # tool-call example is not coerced into executing it.
        result = await self.inner.acomplete(messages, system=system, tools=tools, **kwargs)
        if self.mode == "auto" and not result.tool_calls and result.text:
            residual, calls = parse_text_tool_calls(
                result.text, allowed_names=self._offered_names(tools)
            )
            if calls:
                return result.model_copy(update={"text": residual, "tool_calls": calls})
        return result

    async def astream(
        self,
        messages: list[Message],
        *,
        system: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        response_format: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[ProviderStreamEvent]:
        """Stream a turn.

        In text mode the protocol response must be parsed as a whole, so this
        buffers via :meth:`acomplete` and re-emits the result as a stream (text
        delta, one tool-call delta per parsed call, usage, done) — exactly the
        base-provider shape, so the loop's accumulator reassembles the calls.

        On the native ``auto`` path it streams the inner provider's *true* token
        stream live, but also accumulates the text and, if the model finished
        without emitting any native tool-call, salvages a stray ``<tool_call>``
        block from the buffered text and emits synthetic tool-call deltas just
        before the terminal ``done`` — so a capable-but-quirky local model never
        "falls silent" on the streaming path either (it does on the buffered path
        via :meth:`acomplete`). In strict ``native`` mode it is a pure passthrough.
        """
        # See acomplete: forward structured-output only when set (transparent passthrough).
        if response_format is not None:
            kwargs["response_format"] = response_format
        if self._use_text_mode(tools):
            result = await self.acomplete(messages, system=system, tools=tools, **kwargs)
            if result.text:
                yield StreamTextDelta(text=result.text)
            for index, call in enumerate(result.tool_calls):
                yield StreamToolCallDelta(
                    index=index,
                    id=call.id,
                    name=call.name,
                    arguments_delta=json.dumps(call.arguments, default=str),
                )
            yield StreamUsage(usage=result.usage)
            yield StreamDone(finish_reason=result.finish_reason)
            return

        if self.mode == "auto" and tools:
            saw_tool_call = False
            text_parts: list[str] = []
            async for event in self.inner.astream(messages, system=system, tools=tools, **kwargs):
                if isinstance(event, StreamToolCallDelta):
                    saw_tool_call = True
                    yield event
                elif isinstance(event, StreamTextDelta):
                    text_parts.append(event.text)
                    yield event
                elif isinstance(event, StreamDone):
                    if not saw_tool_call:
                        _residual, calls = parse_text_tool_calls(
                            "".join(text_parts), allowed_names=self._offered_names(tools)
                        )
                        for index, call in enumerate(calls):
                            yield StreamToolCallDelta(
                                index=index,
                                id=call.id,
                                name=call.name,
                                arguments_delta=json.dumps(call.arguments, default=str),
                            )
                    yield event
                else:
                    yield event
            return

        async for event in self.inner.astream(messages, system=system, tools=tools, **kwargs):
            yield event

    def _record_text_overhead(self, protocol: str) -> None:
        """Remember the injected protocol's token cost (see ``_text_overhead_tokens``)."""
        try:
            self._text_overhead_tokens = self.inner.count_tokens([Message.user(protocol)])
        except Exception:  # noqa: BLE001 — overhead is an estimate; never break a real call
            self._text_overhead_tokens = 0

    def count_tokens(self, messages: list[Message], *, system: str | None = None) -> int:
        """Token count of the wrapped provider plus the injected text-mode protocol.

        In text mode the real request also carries the tool protocol (prepended to the
        system prompt) and textified history, which the raw ``messages`` the loop passes
        here do not reflect. Adding the most recent protocol's token cost keeps the
        compactor's estimate from under-counting and silently overrunning num_ctx. The
        overhead is 0 on the native path and before the first text-mode call, so this is a
        pure pass-through there (and the loop is always preceded by an ``acomplete``).
        """
        return self.inner.count_tokens(messages, system=system) + self._text_overhead_tokens

    def capabilities(self) -> Capabilities:
        """Report the wrapped model's capabilities unchanged.

        The loop reads only ``context_window`` from this; the wrapper does not
        rewrite ``supports_tools`` (it makes tools *work* via text but does not
        change what the underlying model natively is).
        """
        return self.inner.capabilities()


__all__ = [
    "TOOL_CALLING_MODES",
    "TextToolCallingProvider",
    "render_tool_protocol",
    "textify_messages",
    "parse_text_tool_calls",
    "defang_untrusted",
]
