"""Tests for the secret-redaction guard (zakcode.secrets)."""

from __future__ import annotations

from zakcode.secrets import redact_secrets


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
