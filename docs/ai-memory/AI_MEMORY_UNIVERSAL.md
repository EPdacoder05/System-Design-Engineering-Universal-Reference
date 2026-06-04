# AI Memory Universal Standard

Canonical source for the operating rules that apply across this standards pack.

## Core Rules
- Use one canonical standard per topic; link instead of copying.
- Prefer the simplest design the owner can explain from memory; readability and explainability are safety properties.
- Prefer explicit ownership, reversibility, and observable failure modes.
- New wrappers, config layers, event hops, or abstractions must earn their keep with repeated need, not speculative flexibility.
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

## Agent Operating Conventions

**Token / output discipline**
- Low token density: answer directly, no filler ("Great question!", "Certainly!", "As an AI…"), no emoji.
- Scope responses to what changed; do not restate the full prior conversation.
- No placeholder prose in production code paths.

**File discipline**
- No new files for agent notes, context memory, or planning — update existing rubric/log files in place.
- Edit files in place; never recreate a file that already exists (data-loss risk).
- Encode lessons as table rows or checklist items in the relevant domain checklist.
- Log every rubric change in `rubrics/ROLLING_UPDATE_LOG.md` per the entry template.

**Engineering discipline**
- No over-engineering: new abstractions, wrappers, config layers, or event hops must be justified by repeated need, not speculative flexibility.
- Make the smallest change that fully solves the problem; do not fix unrelated issues.
- Use existing libraries; do not add or upgrade dependencies unless required.
- No LLM in the v1 core synchronous path (authn/authz, transaction commit, payment, p95-critical logic).

**CI gates (hard blockers)**
- ruff: F401 unused imports, F541 f-string without placeholders, F841 unused locals.
- mypy: annotate nested dicts (`Dict[str,Dict[str,str]]`), explicit return types, type all `.get()` locals.
- Gate order: lint → typecheck → test → build; each job declares `needs:` on the prior hard gate.
- Cross-repo CI failures → add avoid/correct pair to the relevant domain checklist; cite source repo + PR, no new files.

**Memory / learning**
- When a pattern causes a CI failure in any repo, add the avoid/correct pair to the relevant domain checklist here.
- Do not store secrets, credentials, or personal data in memory.

## Definition of Done
- Topic ownership is unambiguous.
- Cross-links replace duplicate prose.
- Exceptions are explicit, reviewable, and time-bounded.
