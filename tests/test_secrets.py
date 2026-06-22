"""Tests for the secret-redaction guard (zakcode.secrets)."""

from __future__ import annotations

from zakcode.secrets import redact_secrets, strip_url_credentials


def test_strip_url_credentials_masks_userinfo() -> None:
    # audit3 #7: an api_base with embedded user:pass@ must be masked for display/serialization,
    # preserving the host and path.
    masked = strip_url_credentials("https://user:s3cret-token@gateway.example.com:8443/v1")
    assert masked is not None
    assert "s3cret-token" not in masked and "user" not in masked
    assert "gateway.example.com:8443/v1" in masked
    # URLs without userinfo (and None) pass through unchanged.
    assert strip_url_credentials("http://127.0.0.1:11434/v1") == "http://127.0.0.1:11434/v1"
    assert strip_url_credentials(None) is None


def test_redacts_url_userinfo_in_freeform_text() -> None:
    # SRV-07: a provider error echoing a credentialed api_base must not leak the password.
    # redact_secrets is the chokepoint litellm_provider._map_error scrubs through.
    out, n = redact_secrets("RequestFailed: could not reach https://svc:Hunter2Pass@gw.example/v1")
    assert n >= 1
    assert "Hunter2Pass" not in out
    assert "://***@gw.example/v1" in out


def test_redacts_url_userinfo_with_only_a_token() -> None:
    # SRV-07 hardening (Phase 4): a USER-ONLY credential (no password) -- e.g. an API key carried as
    # the userinfo username -- must be masked, not leaked verbatim.
    out, n = redact_secrets("connect https://s3cr3t-token@gateway.example.com/v1 failed")
    assert n >= 1
    assert "s3cr3t-token" not in out
    assert "://***@gateway.example.com/v1" in out


def test_redacts_url_userinfo_with_at_in_password() -> None:
    # SRV-07 hardening (Phase 4): a password containing '@' must be FULLY masked (to the last @).
    out, n = redact_secrets("redis://default:p@ss@w0rd@cache.internal:6379 timed out")
    assert n >= 1
    assert "p@ss@w0rd" not in out and "ss@w0rd" not in out  # no password fragment survives
    assert "://***@cache.internal:6379" in out


def test_does_not_redact_url_without_userinfo() -> None:
    # SRV-07: a plain URL (no userinfo @) and a path '@' are NOT redacted (no false positive).
    out, n = redact_secrets("GET https://api.example.com/v1/users?at=@here returned 500")
    assert n == 0
    assert "https://api.example.com/v1/users" in out


def test_redacts_openai_style_key() -> None:
    out, n = redact_secrets("here it is sk-ABCDEFGHIJKLMNOP1234567890 ok")
    assert n == 1
    assert "sk-ABCDEFGHIJKLMNOP" not in out
    assert "[REDACTED]" in out


def test_redacts_aws_and_github_and_slack() -> None:
    out, n = redact_secrets(
        "AKIAIOSFODNN7EXAMPLE and ghp_0123456789012345678901234567890123 and "
        "xoxb-1234567890-abcdEFGH"
    )
    assert n == 3
    assert "AKIA" not in out
    assert "ghp_" not in out
    assert "xoxb-" not in out


def test_redacts_assignment_keeps_key_name() -> None:
    out, n = redact_secrets("password = hunter2hunter2")
    assert n == 1
    assert out.startswith("password")  # key name preserved
    assert "hunter2hunter2" not in out
    assert "[REDACTED]" in out


def test_redacts_pem_block() -> None:
    pem = "-----BEGIN RSA PRIVATE KEY-----\nMIIByyz...\n-----END RSA PRIVATE KEY-----"
    out, n = redact_secrets(f"key:\n{pem}\nend")
    assert n == 1
    assert "PRIVATE KEY" in out  # marker mentions it
    assert "MIIByyz" not in out


def test_ordinary_prose_is_untouched() -> None:
    text = "The API key lives in the environment, never in code. Run the tests."
    out, n = redact_secrets(text)
    assert n == 0  # no key=value, no token shape
    assert out == text


def test_empty_is_safe() -> None:
    assert redact_secrets("") == ("", 0)
