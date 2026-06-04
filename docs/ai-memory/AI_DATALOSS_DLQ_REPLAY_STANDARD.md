# AI Data Loss, DLQ, and Replay Standard

Canonical source for failed async work and loss prevention.

## Rules
- No silent drop. Every failed message is retried, DLQ'd, or rejected with explicit operator visibility.
- DLQ entries must include failure reason, attempt count, timestamps, and payload reference.
- Replay tooling must be safe by default: scoped, rate-limited, auditable, and idempotent.
- Poison messages require quarantine and root-cause analysis before bulk replay.
- Retention must outlast realistic detection and remediation windows.

## Required Signals
- Retry count and age
- DLQ depth and oldest message age
- Replay success/failure counts
- Per-reason failure buckets

## Anti-Patterns
- Dropping malformed events without evidence.
- Infinite retry storms.
- Replay that bypasses normal validation or idempotency.

## Definition of Done
- Failed work is recoverable.
- Operators can see, triage, and replay without data loss.
