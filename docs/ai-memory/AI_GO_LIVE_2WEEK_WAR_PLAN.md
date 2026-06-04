# AI Go-Live 2-Week War Plan

Canonical source for launch readiness in the final two weeks.

## T-14 to T-8
- Freeze high-risk scope churn.
- Reconfirm owners, rollback path, on-call coverage, dashboards, and alerts.
- Validate CI gates, container hardening, SBOM/provenance, and release artifacts.

## T-7 to T-2
- Run load test to breakpoint.
- Rehearse DLQ replay, rollback, and partition-health checks.
- Verify readiness/liveness behavior and customer-visible error contracts.

## T-1 to T+2
- Enable smallest safe rollout slice.
- Watch auth failures, error rate, p95/p99 latency, queue lag, DLQ depth, and partition anomalies.
- Stop or roll back on threshold breach; do not “monitor through” obvious failure.

## Definition of Done
- Launch owners, signals, and rollback criteria are explicit before go-live.
