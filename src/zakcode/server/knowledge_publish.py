"""Durable publish of the OKF knowledge bundle (PEARL §10.5).

The ``/knowledge/*`` browse routes read a bundle held on THIS box, so everything
they serve dies with the box. §10.5 wants an offline copy that survives a
restart. This module is the box side of that: it renders the same OKF transfer
bundle ``GET /knowledge/export`` returns and PUTs each file to the account/env
S3 prefix through the gateway's storage route.

**Nothing here is new infrastructure.** The bucket, its IAM grants, and the
CRUD route (``handleVinheimStorage``) were already provisioned; this is the
glue. Two properties of that route shape everything below and were read from
its source rather than assumed:

* The fence is ``vin_``-prefixed key -> ACTIVE key hash -> *the calling account
  owns this envId*. It is an OWNERSHIP fence, **not** a sharing-tier check —
  a 403 here means "not your env", never "this env is private".
* The S3 key is ``{accountId}/{envId}/{filePath}``. The account segment comes
  from the KEY, not from us, so a caller cannot write outside its own account
  no matter what path it sends.

Publishing is an ENHANCEMENT, never a precondition. An unconfigured box must
no-op, and a box whose publish fails must keep serving ``/knowledge/*``
normally — so no function here raises on a transport or HTTP error. Failures
are collected and returned for the caller to report.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

from zakcode.config import Settings

#: Bundle files land under this prefix inside the env, so a knowledge publish
#: can never collide with whatever else the env stores at its root.
KNOWLEDGE_PREFIX = "knowledge"

#: Bundle files are markdown. Sent explicitly because the route stores whatever
#: content-type it is given and hands it straight back on GET.
_CONTENT_TYPE = "text/markdown; charset=utf-8"

_TIMEOUT_SECONDS = 30.0


@dataclass
class PublishResult:
    """Outcome of one publish pass. Never an exception — always a report."""

    published: list[str] = field(default_factory=list)
    #: ``(bundle_path, reason)`` — reason is an HTTP status or an exception string.
    failed: list[tuple[str, str]] = field(default_factory=list)
    #: Set when the pass did not run at all (unconfigured box, empty bundle).
    skipped_reason: str | None = None

    @property
    def ok(self) -> bool:
        """A skipped pass is OK. A pass with any failure is not."""
        return not self.failed

    def summary(self) -> str:
        if self.skipped_reason:
            return f"skipped: {self.skipped_reason}"
        return f"published {len(self.published)}, failed {len(self.failed)}"


def publish_url(settings: Settings, bundle_path: str) -> str:
    """Absolute storage URL for one bundle file.

    ``bundle_path`` is the bundle-relative path (e.g. ``nodes/foo.md``); it is
    placed under :data:`KNOWLEDGE_PREFIX` within the env.
    """
    base = (settings.knowledge_publish_url or "").rstrip("/")
    env_id = (settings.knowledge_publish_env_id or "").strip()
    return f"{base}/v1/vinheim/storage/{env_id}/{KNOWLEDGE_PREFIX}/{bundle_path.lstrip('/')}"


def _headers(settings: Settings) -> dict[str, str]:
    # The key is a credential: it goes in the header and NOWHERE else — never
    # into a log line, an error message, or the URL (which lands in access logs).
    return {
        "authorization": f"Bearer {(settings.knowledge_publish_key or '').strip()}",
        "content-type": _CONTENT_TYPE,
    }


def publish_bundle(
    settings: Settings,
    bundle_files: dict[str, str],
    *,
    client: httpx.Client | None = None,
) -> PublishResult:
    """PUT every file of an already-rendered OKF bundle. Never raises.

    ``bundle_files`` is the ``files`` map from the OKF bundle: bundle-relative
    path -> markdown text. Pass ``client`` to drive this without a network
    (tests, or a caller that owns connection reuse).
    """
    if not settings.knowledge_publish_ready:
        # Partial configuration lands here too — see Settings.knowledge_publish_ready.
        return PublishResult(skipped_reason="publishing is not configured")
    if not bundle_files:
        return PublishResult(skipped_reason="bundle is empty")

    result = PublishResult()
    owned_client = client is None
    http = client or httpx.Client(timeout=_TIMEOUT_SECONDS)
    try:
        headers = _headers(settings)
        for bundle_path, content in sorted(bundle_files.items()):
            try:
                resp = http.put(
                    publish_url(settings, bundle_path),
                    content=content.encode("utf-8"),
                    headers=headers,
                )
            except Exception as exc:  # transport error — report, never propagate
                result.failed.append((bundle_path, type(exc).__name__))
                continue
            if 200 <= resp.status_code < 300:
                result.published.append(bundle_path)
            else:
                # Status only. The body can echo request detail, and this string
                # reaches logs.
                result.failed.append((bundle_path, f"HTTP {resp.status_code}"))
    finally:
        if owned_client:
            http.close()
    return result


def publish_workspace_bundle(
    settings: Settings,
    workspace_root: Path,
    *,
    client: httpx.Client | None = None,
) -> PublishResult:
    """Render this workspace's OKF bundle and publish it.

    Renders through the SAME two functions ``GET /knowledge/export`` uses, so
    the published bundle and the served bundle cannot drift into different
    shapes — one producer, two transports.
    """
    if not settings.knowledge_publish_ready:
        return PublishResult(skipped_reason="publishing is not configured")
    # Imported here, not at module scope: app.py imports fastapi, and the CLI
    # must not require the server extra just to parse its own commands.
    from zakcode.server.app import _okf_bundle, _read_knowledge_bundle

    bundle: dict[str, Any] = _okf_bundle(_read_knowledge_bundle(workspace_root))
    return publish_bundle(settings, bundle.get("files") or {}, client=client)
