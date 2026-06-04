# AI Nested Loop and Complexity Guardrails

Canonical source for avoiding accidental quadratic or worse behavior.

## Guardrails
- No nested loop over unbounded collections in a hot path without proof it is safe.
- Push joins, filtering, grouping, and pagination into SQL before application loops.
- Pre-index data into maps/sets before repeated lookup.
- Batch remote calls; do not issue one query/request per item when a set-based path exists.
- State the expected complexity for new critical-path logic.

## Anti-Patterns
- O(n*m) diffing of request payloads against full tables.
- Per-item permission query inside a response serializer.
- Full partition scan when a bounded date/window predicate exists.

## Definition of Done
- Critical path complexity is explained.
- A bounded or set-based approach replaces naive nested loops.
