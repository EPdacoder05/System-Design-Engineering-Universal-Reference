# AI SQL-First Engineering Standard

Canonical source for data-intensive backend logic.

## Rules
- Prefer SQL for joins, filters, grouping, aggregation, deduplication, and pagination.
- Use the application layer for orchestration and policy, not bulk relational work.
- Review query shape and indexes for critical paths.
- Make migrations explicit, reversible where practical, and observable.
- ORMs are allowed, but they must not hide inefficient query behavior.

## Anti-Patterns
- Loading large result sets into memory for work SQL should do.
- N+1 query patterns hidden behind ORM accessors.
- Business-critical reports built from ad hoc application loops.

## Definition of Done
- Query logic is set-based and explainable.
- Performance-critical queries have an index and plan story.
