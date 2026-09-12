RATES = {"USD": 1.0, "EUR": 1.1, "GBP": 1.27}


def enrich(records):
    out = []
    for r in records:
        e = dict(r)
        e["usd"] = round(r["amount"] * RATES[r["currency"]], 2)
        out.append(e)
    return out
