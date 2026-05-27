# Rolling Update Log

Use this template for each significant change.

## Entry Template

### Change ID
- ID: `YYYYMMDD-<short-name>`
- Date:
- Repository:
- Owner:
- Rubric Version: `1.0.0`

### Required Rolling Fields
- **What changed:**
- **Why:**
- **Impact:**
- **Next experiment:**
- **Deprecation notes:**

### Required Deltas
- **Rubric score delta:**
  - Correctness:
  - Reliability:
  - Security:
  - Scalability:
  - Operability:
  - Cost:
  - Maintainability:
  - Clarity:
  - Total: `before -> after`
- **Risk delta:**
- **Production readiness delta:**

### Validation
- Tests/lint/type checks run:
- Result summary:
- Known limitations:

### Source & Signal Notes (Optional)
- New source(s):
- Relevance to niche/industry/project:
- Quality/freshness checks:

---

### Change ID
- ID: `20260527-ruff-lint-rubric`
- Date: 2026-05-27
- Repository: EPdacoder05/System-Design-Engineering-Universal-Reference
- Owner: @copilot
- Rubric Version: `1.0.0`

### Required Rolling Fields
- **What changed:** Added Python linting section to `rubrics/CHECKLISTS/SWE.md` documenting ruff rules F401, F541, F841 with avoid/correct examples.
- **Why:** CI hard-gate (ruff) failed on PR #18 due to unused imports (F401), bare f-string (F541), and unused local variable (F841) in `testing/test_framework.py`. The root patterns were missing from the rubric, so agents repeated the mistakes.
- **Impact:** Future agents and contributors have an explicit, searchable reference to the three ruff rules most likely to silently break CI. Plug-and-play reuse of the testing scaffold will not hit the same lint errors.
- **Next experiment:** Add a pre-commit hook entry in the DevOps checklist that runs `ruff check --select F401,F541,F841` on staged `.py` files.
- **Deprecation notes:** None.

### Required Deltas
- **Rubric score delta:**
  - Maintainability: 3 → 4 (explicit lint rules in rubric)
  - Clarity: 3 → 4 (avoid/correct examples inline)
  - Total: `before +2`
- **Risk delta:** Reduced — documented patterns prevent recurring CI failures.
- **Production readiness delta:** CI hard-gate stays green; no new checklist items left open.

### Validation
- Tests/lint/type checks run: `ruff check testing/test_framework.py` + all changed `.py` files
- Result summary: All checks passed (exit 0)
- Known limitations: Rule list covers only the violations observed; expand as new patterns emerge.

---

### Change ID
- ID: `20260527-cross-repo-rubric-expansion`
- Date: 2026-05-27
- Repository: EPdacoder05/System-Design-Engineering-Universal-Reference
- Owner: @copilot
- Rubric Version: `1.0.0`
- Source PRs: security-data-fabric#12, ha-iot-stack#1–4, System-Design-Engineering-Universal-Reference#18

### Required Rolling Fields
- **What changed:**
  - `SWE.md`: extended Python Linting table (I001 import order, E501 line length); added mypy type annotation section (nested dict shapes, explicit return types, untyped `.get()` locals).
  - `DEVOPS.md`: added Docker Container Security table (privileged mode, pinned tags, healthchecks, read-only FS, explicit build context, missing requirements.txt).
  - `MASTER_RUBRIC.md`: added Section 7 Coding Agent Conventions (output quality, memory management, plug-and-play templates, cross-repo learning rule).
- **Why:** Three separate CI failures across two repos (security-data-fabric#12 mypy/lint; ha-iot-stack#2–3 Docker build; this repo#18 ruff) exposed patterns not yet in the rubric. Without codifying them, agents repeat the same breaks project to project.
- **Impact:** Any agent scaffolding a new Python service or Docker stack against this rubric will avoid the same regressions. Cross-repo learning loop is now self-documenting.
- **Next experiment:** Add pre-commit config template to DevOps checklist that wires `ruff`, `mypy`, and `docker compose config --quiet` as local gates.
- **Deprecation notes:** None.

### Required Deltas
- **Rubric score delta:**
  - Maintainability: 4 → 5
  - Clarity: 4 → 5
  - Operability: 3 → 4 (Docker security patterns explicit)
  - Total: `before +3`
- **Risk delta:** Reduced — lint/mypy/Docker break patterns now documented with avoid/correct pairs.
- **Production readiness delta:** CI hard-gates documented; Docker security baseline explicit.

### Validation
- Tests/lint/type checks run: doc-only change; no `.py` files modified
- Result summary: No CI gates applicable
- Known limitations: mypy nested-dict rule applies to Python ≥ 3.9 with `from __future__ import annotations` or `typing.Dict`; update when codebase migrates to built-in generics.
