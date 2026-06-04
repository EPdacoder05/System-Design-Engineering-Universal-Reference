# AI Error Handling Client Contract

Canonical source for stable client-visible failure semantics.

## Standard Error Envelope
- Stable machine-readable code
- Human-readable message
- Retryability hint
- Correlation/request ID
- Optional field-level details for validation errors

## Rules
- Use 4xx for client/actionable faults and 5xx for server/operator faults.
- Do not leak secrets, stack traces, SQL, or internal topology.
- Keep error codes stable across versions unless a migration plan exists.
- Partial failure behavior must be explicit for batch APIs.

## Anti-Patterns
- Returning `200` with hidden failure payloads.
- Mapping all failures to `500`.
- Exposing raw downstream exception text.

## Definition of Done
- Clients can distinguish retry, fix-input, and escalate cases.
- Operators can correlate an error to logs and traces.
