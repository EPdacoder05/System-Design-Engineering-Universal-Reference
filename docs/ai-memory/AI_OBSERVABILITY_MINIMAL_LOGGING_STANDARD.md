# AI Observability Minimal Logging Standard

Canonical source for the minimum telemetry required in backend services.

## Required Structured Fields
- request_id or event_id
- trace_id/span_id where tracing exists
- tenant/account/user surrogate when applicable
- operation name
- result: success | failure | retry | dropped_to_dlq
- latency and attempt count

## RED Metrics (mandatory)
Emit exactly these three signals for every service boundary:
- **Rate**: requests per second
- **Errors**: failed requests (4xx that are server-actionable, all 5xx)
- **Duration**: latency histogram (p50, p95, p99)

All other metrics are optional. These three are the minimum viable production signal.

## Rules
- Log auth denials, permission denials, retries, circuit-open events, DLQ writes, and replay actions.
- Emit RED metrics for every service boundary and queue consumer.
- Only structured JSON logs are permitted in production paths — no `console.log`, `print()`, or unstructured string concatenation.
- CI must run an AST check to strip/reject bare `console.log` / `print` statements in production code (not test files).
- Never log secrets, raw tokens, or sensitive payload bodies by default.
- Correlate dashboards, traces, and logs with the same request/event/build identifiers.
- Backend logs are for operators; client responses are for callers — never conflate (see AI_ERROR_HANDLING_CLIENT_CONTRACT.md).

## Anti-Patterns
- Health checks with no dependency signal.
- Logging full request bodies for convenience.
- Missing correlation IDs on async hops.
- Green dashboards backed by swallowed exceptions or invisible retry storms.
- `console.log('here')` / `print(data)` left in production code paths.
- Novel-writing log messages — one structured line per event, no prose.

## Related Repo References
- [AI_ERROR_HANDLING_CLIENT_CONTRACT.md](./AI_ERROR_HANDLING_CLIENT_CONTRACT.md)
- [AI_HEALTHCHECK_READINESS_LIVENESS_STANDARD.md](./AI_HEALTHCHECK_READINESS_LIVENESS_STANDARD.md)

## Definition of Done
- A production incident can be triaged from logs, metrics, and traces without guesswork.
- RED metrics are dashboarded and alerted on for every service.
- No unstructured log statements reach production.
