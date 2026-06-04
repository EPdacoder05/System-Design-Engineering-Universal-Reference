# AI 30/60/90 Rollout Plan

Canonical source for staged adoption of this standards pack.

## Days 0-30
- Establish the canonical docs and link them from active work.
- Baseline CI gates, container hardening, auth-hardening checks, and DLQ visibility.
- Identify high-write tables that must move to monthly partition defaults.

## Days 31-60
- Enforce SQL-first, idempotency, outbox, and healthcheck standards on new work.
- Add precreate/retention/partition-health automation where missing.
- Require confirmation protocol for high-blast-radius changes.

## Days 61-90
- Make exception handling durable with decision records.
- Validate load-test breakpoint and perf-failure playbook usage on critical services.
- Audit duplicate docs and replace them with canonical links.

## Definition of Done
- The standards are used by default, not treated as side documentation.
