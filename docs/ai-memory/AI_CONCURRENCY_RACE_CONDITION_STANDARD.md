# AI Concurrency and Race Condition Standard

Canonical source for shared-state and parallel-execution safety.

## Rules
- Define ownership of every mutable resource: row, key, file, queue message, partition, cache entry.
- Prefer single-writer patterns, transactional updates, or compare-and-set semantics over ad hoc locks.
- Make async consumers idempotent because duplicate delivery is normal.
- Treat ordering as explicit: if order matters, encode sequence, version, or fencing token.
- Bound parallelism per dependency and per tenant when contention risk exists.

## Backpressure and Queue SLOs (required)
Every async consumer / queue-backed path must define and enforce:

| Parameter | Required Default | Override Condition |
|---|---|---|
| Max queue lag (consumer behind) | ≤ 30s p95 | Document if higher; alert at 2× threshold |
| Max in-flight messages per consumer | ≤ 10 concurrent | Tune per measured throughput; never unbounded |
| Max in-flight per carrier / tenant | ≤ 5 concurrent | Prevents one tenant starving others |
| Circuit-open threshold | 5 consecutive failures or 50% error rate in 60s | |
| Circuit half-open probe interval | 30s | |
| Retry backoff | Exponential, base 1s, max 60s, jitter | |
| Dead-letter after N retries | 3–5 (configurable) | Document choice |

- These SLOs must be dashboarded; alerts must fire before the SLO is violated, not after.
- Per-carrier in-flight caps prevent a slow or failing carrier from blocking the global consumer pool.
- When circuit is open, fail fast and DLQ; do not block the worker thread.

## Required Checks
- Race test or reasoning for concurrent writes.
- Duplicate-delivery behavior documented.
- Timeout, retry, and cancellation semantics defined.
- Backpressure SLO table populated for every queue-backed path.

## Anti-Patterns
- Read-modify-write with no version check.
- Global mutex around a hot path.
- Assuming exactly-once delivery from the broker.
- Unbounded in-flight concurrency (no max_concurrent setting).
- No per-tenant isolation on shared consumers.

## Definition of Done
- Shared-state safety is explicit.
- Duplicate and out-of-order cases are handled.
- Backpressure SLO table exists and is monitored.
