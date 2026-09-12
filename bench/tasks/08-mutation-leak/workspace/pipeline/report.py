def summarize(enriched):
    total = round(sum(e["usd"] for e in enriched), 2)
    return {"count": len(enriched), "total_usd": total}
