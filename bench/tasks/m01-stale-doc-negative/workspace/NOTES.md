# Engineering notes

## 2026-08-14 — retry-logic cleanup
`handle_retry` was **removed** during the retry-logic cleanup. Nothing calls it
any more; `backoff_seconds` is the only surviving helper in that module.

## 2026-08-15 — symbol index regenerated
See `stale_index.txt` for the current exported symbols.
