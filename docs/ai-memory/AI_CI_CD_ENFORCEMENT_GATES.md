# AI CI/CD Enforcement Gates

Canonical source for delivery gates and remediation principles.

## Canonical Gate Order (hard sequence — no skipping)
Each stage declares `needs:` on the prior. Any failure blocks all downstream stages.

1. **Lint** — ruff/eslint/stylelint; fails on F401, F541, F841 and equivalents
2. **Typecheck** — mypy/tsc strict; annotated dicts, explicit returns, typed locals
3. **Unit tests** — fast, no external I/O
4. **Contract tests** — API schema, event schema, DB migration compatibility
5. **Integration tests** — real dependencies (DB, queue, cache) in CI containers
6. **Security scans** — bandit, pip-audit/npm audit, secret scan, SAST; advisory unless critical
7. **Build / package** — Docker image, wheel, tarball; depends on all above
8. **Release / deploy approval** — human or policy gate; never auto-deploys to prod from a failed chain

Rationale: a type failure must not silently skip the build; a secret leak must not reach a release artifact.

## Hard Rules
- Downstream build and deploy steps must declare `needs:` on all upstream hard gates.
- CI exceptions need explicit approver, reason, expiry date, and compensating control — stored in `rubrics/ROLLING_UPDATE_LOG.md`.
- Failed gates must be visible; do not silently skip critical downstream jobs.
- Required checks must map to real blocking jobs; skipped or detached jobs do not count as enforcement.
- Advisory gates (mypy, bandit, pip-audit) are summarized by a ci-summary job; they do not silently pass.

## Anti-Overengineering Merge Gate
Before merging any new abstraction, wrapper, config layer, event hop, or interface:
- Declare "**repeated need evidence**": at least 2 existing callsites that will use it, or a documented future need within the current release cycle.
- Declare "**blast radius benefit**": how it reduces coupling, not how it adds flexibility.
- Reviewer must explicitly approve. PRs that add abstraction without evidence are rejected.

## Mission → Money → Ownership (MMMO) Required PR Section
Every non-trivial PR must include (enforced via PR template checklist):
- **Mission**: what user/operator problem this solves.
- **Money**: whether this touches billing, compliance, SLA, or revenue paths (yes/no + details).
- **Ownership**: named owner for the changed component + on-call rotation confirmation.
- **Rollback**: rollback path and estimated TTR if change causes production incident.

PRs missing this section are not eligible for merge.

## Cross-Repo De-duplication / Link-Check Gate
- Every canonical topic must have exactly one owning file in `docs/ai-memory/`; duplicate prose in other files is a lint failure.
- CI must run a link-check job against all `[...](.md)` references in `docs/ai-memory/` and `rubrics/`; broken links block merge.
- Periodic (weekly) prose-duplication scan: if the same policy paragraph appears in ≥ 2 files, one must be converted to a cross-link. Threshold: > 3 consecutive matching sentences.
- Cross-repo learning: when a CI failure pattern is fixed in any repo, the fix pair (avoid/correct + source repo + PR) must be added to the relevant domain checklist in this repo before that PR closes.

## Required Remediation Themes
- Auth hardening: fail closed, validate trust boundaries, keep privileged flows tested.
- CI gates: make critical failures blocking, not advisory by accident.
- Workflow integrity: verify runners, `needs`, permissions, and payload-driven conditionals before trusting green status.
- Container hardening: non-root, minimal image, pinned inputs, health checks.
- DLQ/no-silent-drop: async failure handling must be observable and enforceable.

## Definition of Done
- Gate order is explicit and enforced with `needs:` dependencies.
- MMMO section is present on every non-trivial PR.
- Anti-overengineering check is documented in the PR.
- Release path blocks unsafe changes.
- Gate bypasses are exceptional, documented, and time-bounded.
