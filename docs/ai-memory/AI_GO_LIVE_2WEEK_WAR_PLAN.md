# AI Go-Live 2-Week War Plan

Canonical source for launch readiness in the final two weeks.

## T-14 to T-8
- Freeze high-risk scope churn.
- Reconfirm owners, rollback path, on-call coverage, dashboards, and alerts.
- Validate CI gates, container hardening, SBOM/provenance, and release artifacts.
- Populate owner accountability table (see [AI_BLAST_RADIUS_STANDARD.md](./AI_BLAST_RADIUS_STANDARD.md)).

## T-7 to T-2
- Run load test to breakpoint.
- Rehearse DLQ replay, rollback, and partition-health checks.
- Verify readiness/liveness behavior and customer-visible error contracts.
- Confirm backpressure SLOs and circuit-breaker thresholds are dashboarded.

## T-1 to T+2: Hour-by-Hour Launch Gates
| Hour | Gate | Pass Criteria | Abort Trigger |
|---|---|---|---|
| H-2 | Pre-launch checklist | All items green | Any critical item red |
| H0 | Enable smallest slice | Canary traffic at 1–5% | Any 5xx spike > baseline |
| H+1 | RED metrics check | Error rate < 0.1%, p95 < SLO | Error rate > 1% or latency > 2× baseline |
| H+2 | DLQ / queue check | DLQ depth = 0, queue lag < SLO | DLQ growing, lag > threshold |
| H+4 | Expand to 25% | Stable metrics | Any threshold breach |
| H+8 | Expand to 50% | Stable metrics | Any threshold breach |
| H+24 | Full rollout | 24h stable | |

**Abort criteria (any one triggers immediate rollback):**
- Error rate > 1% sustained for 5 minutes
- p95 latency > 2× baseline for 5 minutes
- DLQ depth growing (not draining) for 10 minutes
- Auth failure rate > baseline + 0.5%
- Any data integrity anomaly detected

## Comms Matrix
| Event | Owner | Audience | Channel | SLA |
|---|---|---|---|---|
| Go/no-go decision | Launch lead | Eng + Ops | Slack #launch | T-1h |
| Rollback initiated | On-call | Eng + Ops + PM | Slack + PagerDuty | Immediate |
| All-clear | Launch lead | Eng + Ops + PM | Slack #launch | H+24 stable |
| Post-mortem (if any) | Incident lead | All stakeholders | Written doc | 48h after incident |

## Definition of Done
- Launch owners, signals, and rollback criteria are explicit before go-live.
- Hour-by-hour gates are documented and assigned.
- Abort criteria are agreed before traffic starts.
- Comms matrix is populated with named owners.
