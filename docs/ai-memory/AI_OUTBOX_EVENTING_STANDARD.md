# AI Outbox Eventing Standard

Canonical source for reliable state change publication.

## Rules
- Write business state and outbox record in the same local transaction.
- Relay asynchronously from the outbox; do not publish directly from the request transaction and hope for the best.
- Consumers must be idempotent and replay-safe.
- Event schema/version, ordering expectations, and ownership must be explicit.
- Failed relay attempts must retry and then DLQ without silent drop.

## Anti-Patterns
- Dual write: DB commit plus broker publish in separate success paths.
- Publishing events before the transaction commits.
- Replay that re-applies side effects without dedup.

## Definition of Done
- State change and event publication are durable and recoverable.
