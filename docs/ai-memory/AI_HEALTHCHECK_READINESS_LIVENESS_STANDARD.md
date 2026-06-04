# AI Healthcheck, Readiness, and Liveness Standard

Canonical source for service health endpoints.

## Rules
- Liveness answers: should the process be restarted?
- Readiness answers: can this instance safely receive traffic now?
- Liveness must not depend on remote systems such as the database.
- Readiness must verify critical dependencies needed for safe serving.
- Startup checks may be separate when initialization is slow.
- `/health` (liveness): process-only check — no DB, no cache, no downstream calls.
- `/ready` (readiness): must query the local database connection and cache layer to confirm connectivity; must not execute business logic or heavy queries.
- Both endpoints must be unauthenticated and respond in < 200ms under normal load.
- Dependency timeout budgets must be short and bounded (e.g. 100ms for DB ping).

## Required Behavior
- Degraded dependency → fail readiness, not liveness.
- Health endpoints must return structured JSON: `{"status": "ok"|"degraded"|"down", "checks": {...}}`.
- Never run business logic, aggregations, or writes inside a health endpoint.

## Anti-Patterns
- One endpoint for everything.
- Health endpoint that always returns success.
- Expensive liveness query that causes restart loops.
- Healthcheck that calls downstream services — use readiness for direct deps only, not transitive chains.
- Unauthenticated health endpoint that leaks internal topology or version strings.

## Definition of Done
- Orchestrators can restart bad processes and stop routing to not-ready instances.
- Health check response time < 200ms p99.
- No business logic or sensitive data in health responses.
