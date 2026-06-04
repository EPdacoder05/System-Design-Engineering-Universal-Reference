# AI Confirmation Protocol

Canonical source for when explicit human confirmation is required.

## Require Confirmation Before
- Deleting, truncating, backfilling, or repartitioning production data.
- Relaxing auth, security, CI, release, or container-hardening gates.
- Changing customer-visible API contracts or error semantics.
- Changing money movement, billing, or entitlement logic.
- Turning off DLQ/replay protections or reducing retention below policy.
- Choosing yearly partitioning where monthly would normally be the default.

## Confirmation Payload
- Change summary
- Why now
- Blast radius
- Rollback path
- Metrics and logs to watch
- Time window
- Owner/approver

## Anti-Patterns
- “Safe enough” with no rollback.
- Implicit approval from silence.
- Bundling low-risk and high-risk work into one confirmation.

## Definition of Done
- Explicit approval exists.
- Scope, risk, and rollback are written down.
- Approval expires when the change window closes.
