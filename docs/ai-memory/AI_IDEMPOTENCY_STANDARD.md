# AI Idempotency Standard

Canonical source for safe retries on write operations.

## Rules
- Require idempotency keys for externally retried create/update commands.
- Scope the key to actor plus operation intent, not just raw payload.
- Persist the first accepted result and return it on duplicate retries.
- Back idempotency with durable uniqueness where correctness depends on it.
- Consumers must deduplicate using event ID, command ID, or business key.

## Required Idempotency Key Schema Contract
Every write operation exposed to external callers or async consumers must define:

```
idempotency_key:
  scope:        tenant_id + endpoint_path + client_request_id
  hash_input:   SHA-256(tenant_id || endpoint || normalized_payload)
  ttl:          minimum 24 hours; 7 days for money/compliance paths
  storage:      durable (DB unique constraint or Redis with AOF persistence)
  replay_semantics:
    - return stored result on duplicate key; do not re-execute
    - return 200/201 (not 409) on duplicate with same key and identical payload
    - return 409 if same key but different payload (conflict, not duplicate)
  expiry_behavior:
    - after TTL, key is eligible for reuse; treat as new request
    - expired key reuse must not cause data corruption (design writes to be safe regardless)
```

- Idempotency key schema must be documented in the service's API spec.
- Non-durable cache-only dedup (e.g. Redis without AOF/RDB) is not permitted for money, irreversible, or compliance writes.

## Anti-Patterns
- Generating a fresh server-side key on every retry.
- Treating duplicate delivery as an edge case.
- Using non-durable cache-only dedup for money or irreversible writes.
- Idempotency key scoped only to payload hash (ignores tenant isolation).
- TTL shorter than the retry window of the calling client.

## Definition of Done
- Retries cannot create duplicate side effects.
- Idempotency key schema is documented and matches implementation.
- Duplicate handling behavior is documented and testable.
- TTL covers the realistic retry window for all callers.
