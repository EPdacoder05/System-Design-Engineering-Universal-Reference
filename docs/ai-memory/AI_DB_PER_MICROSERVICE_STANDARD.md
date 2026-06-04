# AI Database per Microservice Standard

Canonical source for service data ownership.

## Rules
- Each service owns its write schema and migration lifecycle.
- Cross-service reads should go through APIs, replicated read models, or events, not shared writes.
- Shared database servers may be acceptable; shared mutable tables are the exception and require explicit governance.
- Event and audit tables owned by a service should still follow the partitioning standard.

## Anti-Patterns
- Multiple services writing the same business table.
- Reporting jobs reaching across services with private table coupling.
- Hidden cross-service transactions.

## Definition of Done
- Ownership, schema changes, and rollback responsibility are clear per service.
