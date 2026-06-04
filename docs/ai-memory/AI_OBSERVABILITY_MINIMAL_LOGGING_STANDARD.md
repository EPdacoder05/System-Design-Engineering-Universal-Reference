# AI Observability Minimal Logging Standard

Canonical source for the minimum telemetry required in backend services.

## Required Structured Fields
- request_id or event_id
- trace_id/span_id where tracing exists
- tenant/account/user surrogate when applicable
- operation name
- result: success | failure | retry | dropped_to_dlq
- latency and attempt count

## Rules
- Log auth denials, permission denials, retries, circuit-open events, DLQ writes, and replay actions.
- Emit metrics for error rate, latency, queue depth, retries, DLQ depth, and health status.
- Prometheus/OpenTelemetry/Grafana are only useful if catch/retry boundaries emit correlated error, attempt, and saturation signals.
- Correlate dashboards, traces, and logs with the same request/event/build identifiers so operators can pivot across systems without guesswork.
- Prefer structured logs over prose.
- Never log secrets, raw tokens, or sensitive payload bodies by default.

## Anti-Patterns
- Health checks with no dependency signal.
- Logging full request bodies for convenience.
- Missing correlation IDs on async hops.
- Green dashboards backed by swallowed exceptions or invisible retry storms.

## Definition of Done
- A production incident can be triaged from logs, metrics, and traces without guesswork.
