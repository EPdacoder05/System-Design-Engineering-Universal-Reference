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

## Strangler Fig / Legacy Migration Bridge Pattern
When migrating from a monolith or legacy platform to a new service:

### Anti-Corruption Layer (ACL)
- Insert a thin translation layer between legacy and new code; neither side knows the other's model directly.
- ACL translates legacy data shapes, error codes, and event formats into canonical domain types.
- Never let legacy field names or implicit nulls leak into new domain models.

### Phased Cutover Controls
1. **Parallel run**: new path runs alongside legacy; compare outputs, alert on divergence.
2. **Shadow traffic**: new path receives a copy of all requests but responses are not used.
3. **Canary cutover**: small % of real traffic routes to new path; monitor RED metrics.
4. **Full cutover**: legacy path demoted to read-only or retired; ACL kept until migration is complete.
5. **Decommission**: legacy tables/services removed only after full cutover is stable for at least one full retention window.

### Rules
- Never share a database between legacy and new service — data ownership must be explicit.
- Define a read-model sync or event bridge if both systems need the same data during transition.
- Keep feature flags per-phase; do not hard-code cutover logic in application code.
- Every rollback to legacy must be tested before go-live.
- ACL code is production code: it must be tested, linted, and monitored like any other path.

### Anti-Patterns
- Big-bang cutover with no parallel run.
- Legacy and new service writing to the same tables.
- ACL that accumulates business logic instead of pure translation.
- Feature flags that are never cleaned up post-cutover.

## Search / Indexing Architecture Patterns
For global or cross-entity search (e.g. Teams-scale, carrier search, policy search):

### Rules
- Treat the search index as a read-model, not the source of truth; source of truth is always the primary DB.
- Update the index via event/outbox pattern: write to primary DB first, publish change event, consumer updates index asynchronously.
- Accept bounded staleness: define and document the maximum acceptable index lag (e.g. < 5s p99).
- Never query the search index for transactional decisions (billing, auth, compliance); use primary DB.
- Index only the minimum fields required for search + display; never index full payload blobs.
- Design for replay: the entire index must be rebuildable from the event log or DB snapshot.

### Consistency / Freshness Tradeoffs
| Pattern | Consistency | Freshness | Use When |
|---|---|---|---|
| Synchronous dual-write | strong | immediate | small scale, single service |
| Outbox → consumer → index | eventual | seconds | recommended default |
| Full re-index from snapshot | strong (point-in-time) | minutes–hours | index schema migration, disaster recovery |
| CDC (change data capture) | eventual | sub-second | high-volume, low-latency requirement |

### Anti-Patterns
- Querying search index for authoritative data.
- Index updates in the same transaction as the primary write.
- No replay / rebuild path for the index.
- Indexing fields that require decryption at query time.

## Definition of Done
- Query logic is set-based and explainable.
- Performance-critical queries have an index and plan story.
- Migrations include a rollback path.
- Legacy bridge has ACL, phased cutover plan, and decommission criteria.
- Search index has defined staleness SLO and replay path.
