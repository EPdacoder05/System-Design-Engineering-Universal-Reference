# AI Memory Universal Standard

Canonical source for the operating rules that apply across this standards pack.

## Core Rules
- Use one canonical standard per topic; link instead of copying.
- Prefer explicit ownership, reversibility, and observable failure modes.
- No LLM in the v1 core synchronous path: authn/authz, request admission, transaction commit, payment/money movement, or p95-critical request logic.
- Default to async decoupling for expensive, non-deterministic, or human-reviewable work.
- Default to monthly partitioning for high-write and event tables; yearly is an exception that must be justified.
- No silent drops. Failed async work must retry, DLQ, or fail closed with operator visibility.
- Auth hardening, CI gates, and container hardening are baseline standards, not optional add-ons.

## Decision Cascade
1. Classify the task with [AI_ROUTER_POLICY.md](./AI_ROUTER_POLICY.md).
2. Score risk with [AI_ROUTER_SCORING_MATRIX.md](./AI_ROUTER_SCORING_MATRIX.md).
3. Require confirmation when [AI_CONFIRMATION_PROTOCOL.md](./AI_CONFIRMATION_PROTOCOL.md) says so.
4. Apply the topic-specific canonical standard.
5. Record durable exceptions with [AI_DECISION_RECORD_TEMPLATE.md](./AI_DECISION_RECORD_TEMPLATE.md).

## Canonical Topic Map
- Risk control: [Blast radius](./AI_BLAST_RADIUS_STANDARD.md), [confirmation](./AI_CONFIRMATION_PROTOCOL.md)
- Data integrity: [Idempotency](./AI_IDEMPOTENCY_STANDARD.md), [DLQ/replay](./AI_DATALOSS_DLQ_REPLAY_STANDARD.md), [outbox](./AI_OUTBOX_EVENTING_STANDARD.md)
- Data design: [SQL-first](./AI_SQL_FIRST_ENGINEERING_STANDARD.md), [DB per service](./AI_DB_PER_MICROSERVICE_STANDARD.md), [partitioning](./AI_PARTITIONING_MONTHLY_VS_YEARLY_STANDARD.md)
- Runtime safety: [error contract](./AI_ERROR_HANDLING_CLIENT_CONTRACT.md), [observability](./AI_OBSERVABILITY_MINIMAL_LOGGING_STANDARD.md), [health checks](./AI_HEALTHCHECK_READINESS_LIVENESS_STANDARD.md)
- Delivery safety: [supply chain](./AI_SUPPLY_CHAIN_SECURITY_STANDARD.md), [dependency minimization](./AI_DEPENDENCY_MINIMIZATION_STANDARD.md), [SBOM/provenance/signing](./AI_SBOM_PROVENANCE_SIGNING_STANDARD.md), [CI/CD gates](./AI_CI_CD_ENFORCEMENT_GATES.md)

## Definition of Done
- Topic ownership is unambiguous.
- Cross-links replace duplicate prose.
- Exceptions are explicit, reviewable, and time-bounded.
