from pipeline.audit import audit
from pipeline.enrich import enrich
from pipeline.load import load_records
from pipeline.normalize import normalize
from pipeline.report import summarize

__all__ = ["run", "audit", "enrich", "load_records", "normalize", "summarize"]


def run(path="records.json"):
    raw = load_records(path)
    summary = summarize(enrich(normalize(raw)))
    return summary, audit(summary)
