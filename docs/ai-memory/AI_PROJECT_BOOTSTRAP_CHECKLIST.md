# AI Project Bootstrap Checklist

Canonical source for starting a new backend/platform project.

## Checklist
- [ ] Owner, service boundary, and mission/money score are defined.
- [ ] SQL-first data model and DB-per-service boundary are chosen.
- [ ] High-write/event tables default to monthly partitions with precreate, retention, and health jobs.
- [ ] Write paths define idempotency, outbox, retry, and DLQ behavior.
- [ ] Auth hardening rules are documented and tested.
- [ ] Error contract, observability fields, and readiness/liveness endpoints are defined.
- [ ] CI gates block on lint, typecheck, test, build, and security.
- [ ] Container hardening, dependency minimization, SBOM, provenance, and signing are in scope.
- [ ] No LLM usage is placed in the v1 core synchronous path.
- [ ] Launch plan and rollback owner exist before production rollout.

## Definition of Done
- The service can ship without inventing foundational standards midstream.
