# AI Performance Failure Analysis Playbook

Canonical source for diagnosing production performance regressions.

## Triage Order
1. Confirm symptom: latency, throughput, error spike, queue lag, timeout, saturation.
2. Check recent changes, rollout scope, and feature flags.
3. Inspect dependency health, connection pools, lock waits, and partition pruning.
4. Inspect retries, DLQ growth, and consumer lag.
5. Compare before/after query plans and hot-path complexity.

## Required Evidence
- Time window and affected endpoints/jobs
- Resource saturation or wait type
- Query or downstream call deltas
- Rollback decision and operator owner

## Anti-Patterns
- Guessing from CPU alone.
- Treating retry traffic as normal load.
- Tuning symptoms before finding the constraint.

## Definition of Done
- Root constraint and mitigation path are explicit.
