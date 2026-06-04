# AI Router Policy

Canonical source for routing work to the correct standard and approval level.

## Route Order
1. Identify change type: read-only, docs-only, code-path, data-path, infra-path, launch-path.
2. Check if the change touches auth, money, data deletion, schema, partitions, CI gates, container/runtime security, or customer-facing contracts.
3. Score the change with [AI_ROUTER_SCORING_MATRIX.md](./AI_ROUTER_SCORING_MATRIX.md).
4. If the score or category is high risk, require [AI_CONFIRMATION_PROTOCOL.md](./AI_CONFIRMATION_PROTOCOL.md).
5. Apply the narrowest canonical standard instead of writing a new local rule.

## Hard Routes
- Auth or identity changes -> [AI_CONFIRMATION_PROTOCOL.md](./AI_CONFIRMATION_PROTOCOL.md), [AI_SUPPLY_CHAIN_SECURITY_STANDARD.md](./AI_SUPPLY_CHAIN_SECURITY_STANDARD.md)
- Data write-path changes -> [AI_IDEMPOTENCY_STANDARD.md](./AI_IDEMPOTENCY_STANDARD.md), [AI_DATALOSS_DLQ_REPLAY_STANDARD.md](./AI_DATALOSS_DLQ_REPLAY_STANDARD.md), [AI_OUTBOX_EVENTING_STANDARD.md](./AI_OUTBOX_EVENTING_STANDARD.md)
- Query/model changes -> [AI_SQL_FIRST_ENGINEERING_STANDARD.md](./AI_SQL_FIRST_ENGINEERING_STANDARD.md), [AI_PARTITIONING_MONTHLY_VS_YEARLY_STANDARD.md](./AI_PARTITIONING_MONTHLY_VS_YEARLY_STANDARD.md)
- Delivery/runtime changes -> [AI_CI_CD_ENFORCEMENT_GATES.md](./AI_CI_CD_ENFORCEMENT_GATES.md), [AI_HEALTHCHECK_READINESS_LIVENESS_STANDARD.md](./AI_HEALTHCHECK_READINESS_LIVENESS_STANDARD.md), [AI_OBSERVABILITY_MINIMAL_LOGGING_STANDARD.md](./AI_OBSERVABILITY_MINIMAL_LOGGING_STANDARD.md)

## Non-Negotiable Policy Rules
- Do not put LLM calls in the v1 core synchronous path.
- Do not waive lint, test, security, or release gates without explicit approver, reason, expiry, and follow-up owner.
- Do not accept silent data loss, silent retries without limits, or silent fallback auth behavior.

## Definition of Done
- Every change maps to a standard.
- High-risk changes have a score and confirmation trace.
- New docs add links, not parallel policy text.
