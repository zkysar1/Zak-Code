"""Used-signal detection for the context gatherer -- closing the in-context-use label bias.

The reference proxy (does an offered ref appear in a tool-call arg or the assistant's text?) is
high-precision but STRUCTURALLY BLIND to the gatherer's designed-for case: it injects a file's
CONTENT, so the model can consume it in-context without ever re-fetching or naming the file. Those
turns get labelled ``used=False`` -- inverting the very signal a relevance ranker should learn.

``judge_used`` asks a cheap model the real question -- given what the assistant produced, which
offered items did its response actually DRAW ON (explicitly or in-context)? -- so in-context use is
labelled correctly. Fail-soft: any failure falls back to ``reference_used`` (the proxy), so a model
hiccup degrades to the cheap signal, never a broken turn. Mirrors the judge-primitive pattern
(json_object + local validation) used across the quality engine.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable, Sequence

from zakcode.messages import Message
from zakcode.providers.base import Provider
from zakcode.providers.structured import coerce_structured, make_response_format
from zakcode.usage import Usage

from .gatherer import Candidate

logger = logging.getLogger(__name__)

#: A used-signal detector: (task, offered candidates, turn evidence = tool-call args + final
#: assistant text) -> the refs the response actually used. Sync or async.
UsedDetector = Callable[[str, "Sequence[Candidate]", str], "set[str] | Awaitable[set[str]]"]


def _norm(text: str) -> str:
    """Normalise path separators so a Windows/abs tool arg matches a posix-relative ref."""
    return (text or "").replace("\\", "/")


def reference_used(task: str, offered: Sequence[Candidate], evidence: str) -> set[str]:
    """The cheap, no-model proxy: a ref is 'used' when it appears in the turn's evidence (tool-call
    args + final assistant text). High precision for explicit refs; blind to in-context use.
    """
    blob = _norm(evidence)
    return {c.ref for c in offered if c.ref and _norm(c.ref) in blob}


_SYSTEM = (
    "You label training data for a context ranker. The assistant was given background CONTEXT "
    "items (each with a `ref`) and produced a RESPONSE. Decide which items the response actually "
    "DREW ON -- whether quoted/named OR used implicitly. Items it did not need are NOT used. "
    'Reply with ONLY a JSON object: {"used": ["<ref>", ...]} of the refs the response drew on. '
    "No prose, no markdown fence."
)
_SCHEMA = {
    "type": "object",
    "properties": {"used": {"type": "array", "items": {"type": "string"}}},
    "required": ["used"],
}
_MAX_ITEMS = 40  # cap items put to the model so the prompt stays bounded
_MAX_SNIPPET = 400  # per-item content snippet chars
_MAX_TASK = 1000
_MAX_EVIDENCE = 2000


def _clip(text: str, n: int) -> str:
    text = text or ""
    return text if len(text) <= n else text[:n] + " …"


async def judge_used(
    provider: Provider,
    task: str,
    offered: Sequence[Candidate],
    evidence: str,
    *,
    temperature: float = 0.0,
) -> tuple[set[str], Usage]:
    """Ask a cheap model which offered items the response drew on -> ``(used_refs, usage)``.

    Catches IN-CONTEXT use the reference proxy misses. FAIL-SOFT: any error returns
    ``(set(), Usage())`` so the caller can fall back to :func:`reference_used`.
    """
    items = list(offered)[:_MAX_ITEMS]
    if not items:
        return set(), Usage()
    valid = {c.ref for c in items if c.ref}
    listing = "\n\n".join(
        f"[{i + 1}] ref={c.ref}\n{_clip(c.text, _MAX_SNIPPET)}" for i, c in enumerate(items)
    )
    payload = (
        f"<task>\n{_clip(task, _MAX_TASK)}\n</task>\n\n"
        f"<context>\n{listing}\n</context>\n\n"
        f"<response>\n{_clip(evidence, _MAX_EVIDENCE)}\n</response>"
    )
    try:
        result = await provider.acomplete(
            [Message.user(payload)],
            system=_SYSTEM,
            response_format=make_response_format(None),
            temperature=temperature,
        )
        data = coerce_structured(result.text, schema=_SCHEMA)
        used = {str(r) for r in (data.get("used", []) or []) if str(r) in valid}
        return used, result.usage
    except Exception:  # noqa: BLE001 - fail-soft: caller falls back to the reference proxy
        logger.debug("used-judge failed; caller will fall back to the proxy", exc_info=True)
        return set(), Usage()


class ModelUsedDetector:
    """A ``UsedDetector`` backed by one cheap model call per turn (see :func:`judge_used`).

    Labels in-context use correctly, then UNIONS the cheap reference proxy so an explicit ref is
    never lost -- and an empty/failed judge degrades to the proxy alone.

    ``on_usage`` (optional) accounts each call's :class:`~zakcode.usage.Usage` on the caller's
    session/budget; omitted, the small per-turn spend is simply not tracked.
    """

    def __init__(
        self,
        provider: Provider,
        *,
        temperature: float = 0.0,
        on_usage: Callable[[Usage], None] | None = None,
    ) -> None:
        self._provider = provider
        self._temperature = temperature
        self._on_usage = on_usage

    async def __call__(self, task: str, offered: Sequence[Candidate], evidence: str) -> set[str]:
        judged, usage = await judge_used(
            self._provider, task, offered, evidence, temperature=self._temperature
        )
        if self._on_usage is not None and usage.total_tokens:
            try:
                self._on_usage(usage)
            except Exception:  # noqa: BLE001 - accounting must never trap the turn
                logger.debug("on_usage callback failed", exc_info=True)
        # Union the proxy: the judge catches in-context use; the proxy guarantees explicit refs.
        return judged | reference_used(task, offered, evidence)
