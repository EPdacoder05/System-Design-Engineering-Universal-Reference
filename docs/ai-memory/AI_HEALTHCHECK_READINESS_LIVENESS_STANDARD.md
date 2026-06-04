# AI Healthcheck, Readiness, and Liveness Standard

Canonical source for service health endpoints.

## Rules
- Liveness answers: should the process be restarted?
- Readiness answers: can this instance safely receive traffic now?
- Liveness must not depend on remote systems such as the database.
- Readiness must verify critical dependencies needed for safe serving.
- Startup checks may be separate when initialization is slow.

## Required Behavior
- Degraded dependency -> fail readiness, not liveness.
- Dependency timeout budgets must be short and bounded.
- Health endpoints must be lightweight and unauthenticated only if exposure risk is acceptable.

## Anti-Patterns
- One endpoint for everything.
- Health endpoint that always returns success.
- Expensive liveness query that causes restart loops.

## Definition of Done
- Orchestrators can restart bad processes and stop routing to not-ready instances.
