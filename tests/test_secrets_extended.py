"""Tests for the watch-surface redaction: gsk_/vin_ token patterns + redact_secrets_extended."""

from __future__ import annotations

from zakcode.secrets import redact_secrets, redact_secrets_extended


def test_base_redact_now_catches_groq_and_vinheim_keys() -> None:
    scrubbed, count = redact_secrets(
        "groq gsk_0123456789abcdefABCD and vin_0123456789abcdefABCD done"
    )
    assert "gsk_0123456789abcdefABCD" not in scrubbed
    assert "vin_0123456789abcdefABCD" not in scrubbed
    assert count >= 2


def test_extended_matches_exact_env_values() -> None:
    scrubbed, count = redact_secrets_extended(
        "token is s3cr3t-value-xyz here", secret_values=["s3cr3t-value-xyz"]
    )
    assert "s3cr3t-value-xyz" not in scrubbed
    assert count >= 1


def test_extended_skips_short_values() -> None:
    # a short value (< min_value_len) must survive — do not mask ordinary config.
    scrubbed, _ = redact_secrets_extended("mode is dev", secret_values=["dev"])
    assert "dev" in scrubbed


def test_extended_strips_workspace_paths() -> None:
    scrubbed, _ = redact_secrets_extended(
        "see /home/ec2-user/ws/research/journal.md", workspace_paths=["/home/ec2-user/ws"]
    )
    assert "/home/ec2-user/ws" not in scrubbed
    assert "[path]" in scrubbed


def test_extended_high_entropy_catch_all() -> None:
    token = "Zk9xQ2vB7nR4tW1pL5sJ8dF3gH6yK0mN2cX4bV7"  # 39 random chars, no known prefix
    scrubbed, count = redact_secrets_extended(f"leaked {token} end")
    assert token not in scrubbed
    assert count >= 1


def test_extended_keeps_low_entropy_digest() -> None:
    digest = "a" * 40  # entropy 0 → not a secret; must survive
    scrubbed, _ = redact_secrets_extended(f"commit {digest}")
    assert digest in scrubbed


def test_extended_empty_text() -> None:
    assert redact_secrets_extended("") == ("", 0)
