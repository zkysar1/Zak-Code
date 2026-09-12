import json
from pathlib import Path

_CACHE = None


def load_records(path="records.json"):
    """Load the raw records ONCE and hand the same list to every consumer.

    Caching is deliberate: the pipeline and the audit must agree on exactly which
    records were processed, so both are given the result of a single read.
    """
    global _CACHE
    if _CACHE is None:
        _CACHE = json.loads(Path(path).read_text(encoding="utf-8"))
    return _CACHE


def reset_cache():
    global _CACHE
    _CACHE = None
