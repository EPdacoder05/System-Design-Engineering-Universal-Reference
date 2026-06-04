# AI Data Loss, DLQ, and Replay Standard

Canonical source for failed async work and loss prevention.

## Rules
- No silent drop. Every failed message is retried, DLQ'd, or rejected with explicit operator visibility.
- DLQ entries must include failure reason, attempt count, timestamps, and payload reference.
- Replay tooling must be safe by default: scoped, rate-limited, auditable, and idempotent.
- Poison messages require quarantine and root-cause analysis before bulk replay.
- Retention must outlast realistic detection and remediation windows.

## Required Ingest Outcome Matrix
Every ingest path must document its outcome decision for each failure class. No silent behavior allowed.

| Failure Class | Action | Alert Triggered | Notes |
|---|---|---|---|
| Transient network / timeout | **Retry** (max N, exponential backoff) | After max retries | Must be idempotent |
| Malformed / unparseable payload | **DLQ** | Immediately | Log parse error; never silently drop |
| Non-422 4xx (e.g. 409 conflict) | **DLQ** | Immediately | Do not retry client errors as if transient |
| 422 Unprocessable (schema validation) | **DLQ** | Immediately | Include field-level rejection reason |
| 5xx / server error (retryable) | **Retry** then **DLQ** | After max retries | |
| Business rule violation (known reject) | **Ack + log** | If above threshold | Explicit reject, not drop |
| Quota / rate limit exceeded | **Retry** with backoff | If sustained | Respect Retry-After header |
| Poison / repeated failure after replay | **Alert + quarantine** | Immediately | Block bulk replay until root cause resolved |

This table must exist in the service's runbook and match CI-tested behavior.

## Required Signals
- Retry count and age
- DLQ depth and oldest message age
- Replay success/failure counts
- Per-reason failure buckets

## Anti-Patterns
- Dropping malformed events without evidence.
- Infinite retry storms.
- Replay that bypasses normal validation or idempotency.
- A single "catch-all DLQ" with no per-reason classification.

## Definition of Done
- Failed work is recoverable.
- Outcome matrix is documented and tested.
- Operators can see, triage, and replay without data loss.
