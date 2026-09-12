"""Existing coverage. Note these pass today even though `audit` reports problems."""
from pipeline import enrich, normalize, summarize


def test_normalize_coerces():
    out = normalize([{"id": 1, "amount": "3.50", "currency": " usd "}])
    assert out[0]["amount"] == 3.5
    assert out[0]["currency"] == "USD"


def test_enrich_converts():
    out = enrich([{"id": 1, "amount": 2.0, "currency": "EUR"}])
    assert out[0]["usd"] == 2.2


def test_summarize_totals():
    s = summarize([{"usd": 1.5}, {"usd": 2.5}])
    assert s == {"count": 2, "total_usd": 4.0}
