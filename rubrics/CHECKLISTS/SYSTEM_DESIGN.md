# System Design Checklist (Rolling Rubric)

Reference rubric version: **1.0.0**

## P0
- [ ] Idempotency boundaries identified for externally triggered workflows
- [ ] Outbox/inbox strategy defined for cross-service event reliability
- [ ] Failure-mode playbooks documented (timeouts, partial failure, dependency outage)
- [ ] Graceful degradation matrix documented per critical user journey
- [ ] Versioning lifecycle defined (introduce, support window, deprecate, remove)

## P1
- [ ] Capacity assumptions and scaling inflection points documented
- [ ] Data consistency and reconciliation strategy documented
- [ ] Observability requirements mapped to SLOs
