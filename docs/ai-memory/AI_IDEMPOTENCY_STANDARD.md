# AI Idempotency Standard

Canonical source for safe retries on write operations.

## Rules
- Require idempotency keys for externally retried create/update commands.
- Scope the key to actor plus operation intent, not just raw payload.
- Persist the first accepted result and return it on duplicate retries.
- Back idempotency with durable uniqueness where correctness depends on it.
- Consumers must deduplicate using event ID, command ID, or business key.

## Anti-Patterns
- Generating a fresh server-side key on every retry.
- Treating duplicate delivery as an edge case.
- Using non-durable cache-only dedup for money or irreversible writes.

## Definition of Done
- Retries cannot create duplicate side effects.
- Duplicate handling behavior is documented and testable.
