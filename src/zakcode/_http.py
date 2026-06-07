"""Shared HTTP helpers for the web tools: an optional-``httpx`` loader and an SSRF URL guard.

Kept at the package root (like :mod:`zakcode._subprocess`) because both the search backends
(Tavily / SearXNG) and the ``web_fetch`` tool need them. ``httpx`` is an optional dependency
(the ``[web]`` extra), imported here under a guard so the package stays importable without it.

The SSRF guard is the security heart of ``web_fetch`` (the URL often comes from a model, so it
is untrusted). It does three things:

* **Classify the host** — refuse non-``http(s)`` schemes, ``localhost``, and any address that is
  loopback / private (RFC1918) / link-local (incl. the ``169.254.169.254`` cloud-metadata IP) /
  reserved / multicast — for IP literals, *obfuscated* IPv4 (decimal/octal/hex, which some
  resolvers accept), IPv4-mapped IPv6 (``::ffff:127.0.0.1``), AND resolved hostnames. Fail-closed
  on anything unparseable/unresolvable.
* **Pin the connection** — :func:`resolve_pinned_url` resolves the host ONCE and returns a URL
  that dials the *validated IP*, with the original host preserved in the ``Host`` header and TLS
  SNI. This closes the classic DNS-rebinding TOCTOU where the guard's lookup and httpx's
  connect-time lookup could disagree.
* **Re-check every redirect** — the caller re-pins each hop (httpx auto-redirect is disabled).
"""

from __future__ import annotations

import ipaddress
import json
import socket
from typing import Any
from urllib.parse import urlsplit, urlunsplit

#: Cap on a search backend's JSON response body. The endpoints are operator-configured (Tavily /
#: SearXNG), not controlled, but a misbehaving/compromised one must not OOM the process.
_MAX_JSON_BYTES = 5 * 1024 * 1024


class BlockedUrlError(ValueError):
    """A URL was rejected by the SSRF guard (bad scheme, or a private/loopback/metadata host)."""


def host_allowed(host: str, allowed: list[str]) -> bool:
    """Pure matcher: does ``host`` equal, or sit under, any domain in ``allowed``?

    Case-insensitive, trailing-dot-insensitive, subdomain-aware (``example.com`` matches
    ``docs.example.com`` but NOT ``notexample.com`` or ``example.com.evil.test``). An EMPTY
    ``allowed`` returns ``False`` (no match) — the caller decides what empty means (web_fetch
    treats no-allowlist as allow-all; the egress proxy treats it as deny-all).
    """
    if not host:
        return False
    h = host.lower().rstrip(".")
    for entry in allowed:
        d = entry.lower().strip().strip(".")
        if d and (h == d or h.endswith("." + d)):
            return True
    return False


def load_httpx() -> Any:
    """Return the ``httpx`` module, or raise ``ImportError`` with install guidance.

    Typed ``Any`` so callers use ``httpx.AsyncClient`` etc. without the module's stubs being
    required at type-check time (httpx is an optional dep).
    """
    try:
        import httpx
    except ImportError as exc:  # pragma: no cover - exercised via monkeypatch in tests
        raise ImportError(
            "httpx is required for web tools; install it with: pip install 'zakcode[web]'"
        ) from exc
    return httpx


def _resolve_host(host: str) -> list[str]:
    """Resolve ``host`` to its IP address strings. A module function so tests can monkeypatch it
    (the real path does a blocking DNS lookup; callers run it off the event loop)."""
    infos = socket.getaddrinfo(host, None, proto=socket.IPPROTO_TCP)
    return [str(info[4][0]) for info in infos]


def _as_ip_literal(host: str) -> str | None:
    """Canonicalize ``host`` to an IP string if it *is* an address, else ``None``.

    Covers normal IP literals AND obfuscated IPv4 (decimal ``2130706433``, hex ``0x7f000001``,
    octal, short ``127.1``) that ``ipaddress`` rejects but a libc resolver would happily dial —
    so the guard decides on the real target instead of leaning on per-platform resolver quirks.
    """
    try:
        return str(ipaddress.ip_address(host.split("%", 1)[0]))  # strip any IPv6 zone id
    except ValueError:
        pass
    try:
        packed = socket.inet_aton(host)  # accepts decimal/octal/hex/short IPv4 forms
    except OSError:
        return None
    return str(ipaddress.IPv4Address(packed))


def _ip_is_blocked(ip: str) -> bool:
    """True if ``ip`` is a non-public address we must never fetch (fail-closed on parse error)."""
    try:
        addr: ipaddress.IPv4Address | ipaddress.IPv6Address = ipaddress.ip_address(
            ip.split("%", 1)[0]
        )
    except ValueError:
        return True
    # An IPv4-mapped IPv6 address (``::ffff:127.0.0.1``) must be judged on its embedded IPv4,
    # which is_private/is_loopback do not reliably flag on the IPv6 form.
    if isinstance(addr, ipaddress.IPv6Address) and addr.ipv4_mapped is not None:
        addr = addr.ipv4_mapped
    return bool(
        addr.is_private
        or addr.is_loopback
        or addr.is_link_local  # incl. 169.254.169.254 cloud metadata
        or addr.is_reserved
        or addr.is_multicast
        or addr.is_unspecified
    )


def _validate_host(host: str | None) -> list[str]:
    """Return the validated IP(s) for ``host`` or raise :class:`BlockedUrlError`.

    Every resolved IP must pass (a multi-record / DNS-rebinding answer with even one private IP
    is rejected). Fail-closed on an empty/unresolvable host.
    """
    if not host:
        raise BlockedUrlError("URL has no host")
    low = host.lower()
    if low == "localhost" or low.endswith(".localhost"):
        raise BlockedUrlError(f"host {host!r} is a loopback name and is not allowed")
    literal = _as_ip_literal(host)
    if literal is not None:
        ips = [literal]
    else:
        try:
            ips = _resolve_host(host)
        except OSError as exc:
            raise BlockedUrlError(f"could not resolve host {host!r}: {exc}") from exc
    if not ips:
        raise BlockedUrlError(f"host {host!r} did not resolve to any address")
    for ip in ips:
        if _ip_is_blocked(ip):
            raise BlockedUrlError(f"host {host!r} resolves to a blocked address ({ip})")
    return ips


def ensure_url_allowed(url: str) -> str:
    """Validate ``url`` for SSRF safety; return it unchanged, or raise :class:`BlockedUrlError`.

    Validation only (no pinning). Used for cheap allow-checks and as the tested core; the actual
    fetch should use :func:`resolve_pinned_url` so the connection cannot be rebound.
    """
    parts = urlsplit(url)
    if parts.scheme not in ("http", "https"):
        raise BlockedUrlError(
            f"only http and https URLs are allowed (got scheme {parts.scheme or 'none'!r})"
        )
    _validate_host(parts.hostname)
    return url


def resolve_pinned_url(url: str) -> tuple[str, dict[str, str], dict[str, Any]]:
    """Validate ``url`` and PIN it to a checked IP, returning ``(connect_url, headers, ext)``.

    ``connect_url`` dials the validated IP literal (so httpx does not re-resolve the name — no
    DNS-rebinding window); ``headers`` carries the original ``Host`` so virtual-hosting and
    redirects still work; ``ext`` carries ``sni_hostname`` for HTTPS so TLS SNI + certificate
    verification still validate against the real hostname. Raises :class:`BlockedUrlError`.
    """
    parts = urlsplit(url)
    if parts.scheme not in ("http", "https"):
        raise BlockedUrlError(
            f"only http and https URLs are allowed (got scheme {parts.scheme or 'none'!r})"
        )
    host = parts.hostname
    ips = _validate_host(host)
    chosen = ips[0]
    ip_host = f"[{chosen}]" if ":" in chosen else chosen
    netloc = f"{ip_host}:{parts.port}" if parts.port else ip_host
    connect_url = urlunsplit((parts.scheme, netloc, parts.path or "/", parts.query, ""))
    host_header = f"{host}:{parts.port}" if parts.port else str(host)
    extensions: dict[str, Any] = {"sni_hostname": host} if parts.scheme == "https" else {}
    return connect_url, {"Host": host_header}, extensions


async def fetch_json_capped(
    httpx: Any,
    method: str,
    url: str,
    *,
    timeout: float,
    max_bytes: int = _MAX_JSON_BYTES,
    **kwargs: Any,
) -> Any:
    """Stream a request and parse JSON, refusing a body larger than ``max_bytes``.

    For the search backends' trusted-but-not-controlled endpoints: compression is disabled
    (``Accept-Encoding: identity``, so the on-wire size is the real size — no decompression
    bomb) and the body is read in bounded chunks, never buffered whole before the cap fires.
    Transport / HTTP-status / size / JSON errors propagate to the caller (which wraps them).
    """
    headers = {**kwargs.pop("headers", {}), "Accept-Encoding": "identity"}
    async with (
        httpx.AsyncClient(timeout=timeout) as client,
        client.stream(method, url, headers=headers, **kwargs) as resp,
    ):
        resp.raise_for_status()
        chunks: list[bytes] = []
        total = 0
        async for chunk in resp.aiter_bytes(chunk_size=64 * 1024):
            chunks.append(chunk)
            total += len(chunk)
            if total > max_bytes:
                raise ValueError(f"search response exceeded {max_bytes} bytes")
    return json.loads(b"".join(chunks))
