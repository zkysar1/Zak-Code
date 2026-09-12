from pipeline.load import load_records


def audit(summary):
    """Independent cross-check: recompute the record count straight from the raw
    records, and confirm every raw amount is still the untouched source value.

    The audit exists because the pipeline is allowed to derive new values but is
    NOT allowed to alter the source data it was handed.
    """
    raw = load_records()
    problems = []
    if summary["count"] != len(raw):
        problems.append(f"count mismatch: report {summary['count']} vs raw {len(raw)}")
    for r in raw:
        if not isinstance(r["amount"], str):
            problems.append(f"raw record {r['id']} amount is {type(r['amount']).__name__}, "
                            f"expected the untouched source string")
    return problems
