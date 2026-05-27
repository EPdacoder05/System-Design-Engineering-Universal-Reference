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

## Python Linting (ruff — hard gate in CI)

Rules enforced; violations block merge:

| Rule | Pattern to avoid | Correct pattern |
|------|-----------------|-----------------|
| **F401** | `from datetime import datetime, timedelta` (timedelta unused) | Import only what is used; remove unused names |
| **F401** | `from unittest.mock import AsyncMock, MagicMock, Mock` (MagicMock unused) | Remove unused mock classes from import |
| **F541** | `print(f"static string")` — f-string with no `{}` placeholders | `print("static string")` — drop the `f` prefix |
| **F841** | `result = factory.create()` assigned but never referenced | Prefix with `_`: `_result = factory.create()` |

**Coding-agent rule:** before committing any `.py` file, mentally scan for (1) unused imports, (2) f-strings with no interpolation, (3) assigned-but-never-read locals. These are the three ruff rules that most frequently cause silent CI failures.
