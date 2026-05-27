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
| **I001** | Imports out of stdlib → third-party → local order | Use `isort`-compatible order; ruff auto-fixes with `ruff check --fix` |
| **E501** | Log statement or literal > 88 chars on one line | Wrap with `(` `)` across lines or use a variable |
| **F541** | `print(f"static string")` — f-string with no `{}` placeholders | `print("static string")` — drop the `f` prefix |
| **F841** | `result = factory.create()` assigned but never referenced | Prefix with `_`: `_result = factory.create()` |

**Coding-agent rule:** before committing any `.py` file, mentally scan for (1) unused imports, (2) import ordering, (3) line length > 88, (4) f-strings with no interpolation, (5) assigned-but-never-read locals.

## Python Type Annotations (mypy — hard gate in CI)

Violations that silently pass ruff but block mypy and downstream Docker builds:

| Pattern | Wrong | Correct |
|---------|-------|---------|
| Nested dict type | `mapping: Dict[str, str]` when values are dicts | `mapping: Dict[str, Dict[str, str]]` |
| Implicit Any return | `def cosine_sim(...):` with no return annotation | `def cosine_sim(...) -> float:` and return explicit `float` |
| Untyped local from `.get()` | `source_map = mapping.get(source, {})` | `source_map: Dict[str, str] = mapping.get(source, {})` |
| Missing annotation on public fn | `def transform(data):` | `def transform(data: dict) -> dict:` |

**Coding-agent rule:** `Build Docker Image` is downstream of `Type Check`; a mypy failure silently skips the build step. Always annotate return types on public functions and verify nested container shapes match the actual data.
