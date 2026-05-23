# SWE Checklist (Rolling Rubric)

Reference rubric version: **1.0.0**

## P0
- [ ] Pagination strategy documented (cursor/keyset/offset tradeoffs)
- [ ] Structured concurrency policy documented (task lifecycle ownership)
- [ ] Cancellation propagation behavior defined and tested
- [ ] API/schema compatibility policy defined (backward compatibility + deprecation)
- [ ] Shared mutable state minimized; immutable-by-default guidance documented

## P1
- [ ] Error taxonomy and retryability classification documented
- [ ] Idempotency strategy for externally visible operations
- [ ] Contract tests for critical API boundaries
- [ ] Performance budgets defined for critical endpoints
