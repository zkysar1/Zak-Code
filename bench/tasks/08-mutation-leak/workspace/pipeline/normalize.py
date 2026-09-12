def normalize(records):
    """Coerce amounts to float and tidy the currency code."""
    out = []
    for r in records:
        r["amount"] = float(r["amount"])
        r["currency"] = r["currency"].strip().upper()
        out.append(r)
    return out
