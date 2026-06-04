# AI Concurrency and Race Condition Standard

Canonical source for shared-state and parallel-execution safety.

## Rules
- Define ownership of every mutable resource: row, key, file, queue message, partition, cache entry.
- Prefer single-writer patterns, transactional updates, or compare-and-set semantics over ad hoc locks.
- Make async consumers idempotent because duplicate delivery is normal.
- Treat ordering as explicit: if order matters, encode sequence, version, or fencing token.
- Bound parallelism per dependency and per tenant when contention risk exists.

## Required Checks
- Race test or reasoning for concurrent writes.
- Duplicate-delivery behavior documented.
- Timeout, retry, and cancellation semantics defined.

## Anti-Patterns
- Read-modify-write with no version check.
- Global mutex around a hot path.
- Assuming exactly-once delivery from the broker.

## Definition of Done
- Shared-state safety is explicit.
- Duplicate and out-of-order cases are handled.
