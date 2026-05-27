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
