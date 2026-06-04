# AI Dependency Minimization Standard

Canonical source for adding and keeping dependencies under control.

## Rules
- Add a new dependency only when standard library or an approved existing library is insufficient.
- Prefer one well-supported package over multiple overlapping packages.
- Review transitive risk, maintenance quality, license, and removal cost before adoption.
- Remove unused dependencies quickly; stale packages are attack surface.
- Keep feature flags and adapters from forcing heavy dependencies into core paths.
- Import specific sub-modules, not entire packages: `from x import y` / `import { y } from 'x'` — not `import x` / `import 'x'` wholesale.
- CI must flag whole-package imports where a sub-module import is available; this is a hard gate, not advisory.

## Anti-Patterns
- Adding a library for a trivial helper.
- Pulling an SDK into the request path when a small HTTP client wrapper would do.
- Duplicating JSON, retry, or logging libraries.
- `import axios` / `import requests` when only one method is used — import the specific function.
- Importing a mega-package to use a single exported symbol.

## Definition of Done
- Every dependency has a clear owner and reason to exist.
- Unused or overlapping packages are pruned.
- All imports use the most granular form available.
