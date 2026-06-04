# AI Error Handling Client Contract

Canonical source for stable client-visible failure semantics.

## Standard Error Envelope (RFC 7807 Problem Details)
All error responses must follow RFC 7807 JSON Problem Details:
```json
{
  "type": "https://api.yourdomain.com/errors/<slug>",
  "title": "Human-readable error title",
  "status": 429,
  "detail": "Actionable description. Retry after 30 seconds.",
  "instance": "/requests/<request-id>"
}
```
- `type`: stable URI identifying the error class (never changes across versions).
- `title`: human-readable, non-sensitive summary.
- `status`: mirrors HTTP status code.
- `detail`: actionable, non-sensitive description for the caller.
- `instance`: optional correlation/request ID.

## Rules
- Use 4xx for client/actionable faults and 5xx for server/operator faults.
- Never leak secrets, stack traces, SQL, internal module names, or topology in client responses.
- Keep error type URIs stable across versions unless a migration plan exists.
- Partial failure behavior must be explicit for batch APIs.
- Validation errors may include a `errors` array with field-level detail.

## Anti-Patterns
- Returning `200` with hidden failure payloads.
- Mapping all failures to `500`.
- Exposing raw downstream exception text, stack frames, or ORM query strings.
- Non-RFC-7807 ad-hoc error shapes that change per endpoint.

## Definition of Done
- Clients can distinguish retry, fix-input, and escalate cases.
- No internal detail surfaces in any client-visible field.
- Operators can correlate an error to logs and traces via `instance`/request ID.
