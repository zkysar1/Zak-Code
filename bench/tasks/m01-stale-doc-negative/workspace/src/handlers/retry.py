"""Retry helpers for the ingest pipeline."""


def handle_retry(attempt: int, max_attempts: int = 3) -> bool:
    """Return True when another attempt should be made."""
    return attempt < max_attempts


def backoff_seconds(attempt: int) -> float:
    return min(2.0**attempt, 30.0)
