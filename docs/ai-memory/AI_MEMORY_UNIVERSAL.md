# AI Memory Universal Standard

Canonical source for the operating rules that apply across this standards pack.

---

## First-Time-Right Protocol (FTRP)

Required preamble for every agent response to a non-trivial request.

### Response Eligibility Check (all must pass before producing output)
- [ ] Intent parsed — what the user/operator actually wants now
- [ ] Artifact count locked — exact number, not vague
- [ ] Depth locked — brief vs full implementation
- [ ] Constraints locked — time, quality, style, non-goals
- [ ] Failure mode check — what would make this output wrong or incomplete
- [ ] Acceptance criteria locked — explicit definition of "done right"

If any box is unchecked → emit **Clarification Lock Block** only (max 6 lines). No other text.

### Scope Ladder
- **S1**: single artifact
- **S2**: small set (2–5)
- **S3**: pack (6–15)
- **S4**: program (16+)

Default to user-stated scope. Never jump tiers without explicit `expand scope` command.

### QA Threshold
Score 0/1: intent accuracy, scope fidelity, technical depth, operational applicability, risk handling, testability, concision.
Minimum pass: 6/7. Regenerate if below threshold.

### Safety Mode (auto-trigger)
Enable when: production go-live window, security/compliance risk, data loss risk, large-scale/perf risk, or "must be right first time."
Behavior: no speculative claims, deterministic plan-first output, explicit rollback/failure paths, verification matrix.

---

## Core Rules
- Use one canonical standard per topic; link instead of copying.
- Prefer the simplest design the owner can explain from memory; readability and explainability are safety properties.
- Prefer explicit ownership, reversibility, and observable failure modes.
- New wrappers, config layers, event hops, or abstractions must earn their keep with repeated need, not speculative flexibility.
- No LLM in the v1 core synchronous path: authn/authz, request admission, transaction commit, payment/money movement, or p95-critical request logic.
- Default to async decoupling for expensive, non-deterministic, or human-reviewable work.
- Default to monthly partitioning for high-write and event tables; yearly is an exception that must be justified.
- No silent drops. Failed async work must retry, DLQ, or fail closed with operator visibility.
- Auth hardening, CI gates, and container hardening are baseline standards, not optional add-ons.

## Decision Cascade
1. Classify the task with [AI_ROUTER_POLICY.md](./AI_ROUTER_POLICY.md).
2. Score risk with [AI_ROUTER_SCORING_MATRIX.md](./AI_ROUTER_SCORING_MATRIX.md).
3. Require confirmation when [AI_CONFIRMATION_PROTOCOL.md](./AI_CONFIRMATION_PROTOCOL.md) says so.
4. Apply the topic-specific canonical standard.
5. Record durable exceptions with [AI_DECISION_RECORD_TEMPLATE.md](./AI_DECISION_RECORD_TEMPLATE.md).

## Canonical Topic Map
- Risk control: [Blast radius + GitNexus impact trace](./AI_BLAST_RADIUS_STANDARD.md), [confirmation](./AI_CONFIRMATION_PROTOCOL.md)
- Data integrity: [Idempotency + key schema](./AI_IDEMPOTENCY_STANDARD.md), [DLQ/replay + outcome matrix](./AI_DATALOSS_DLQ_REPLAY_STANDARD.md), [outbox](./AI_OUTBOX_EVENTING_STANDARD.md)
- Data design: [SQL-first + strangler migration + search/indexing](./AI_SQL_FIRST_ENGINEERING_STANDARD.md), [DB per service](./AI_DB_PER_MICROSERVICE_STANDARD.md), [partitioning](./AI_PARTITIONING_MONTHLY_VS_YEARLY_STANDARD.md)
- Runtime safety: [error contract (RFC 7807)](./AI_ERROR_HANDLING_CLIENT_CONTRACT.md), [observability + RED metrics](./AI_OBSERVABILITY_MINIMAL_LOGGING_STANDARD.md), [health checks](./AI_HEALTHCHECK_READINESS_LIVENESS_STANDARD.md), [concurrency + backpressure SLOs](./AI_CONCURRENCY_RACE_CONDITION_STANDARD.md)
- Delivery safety: [supply chain](./AI_SUPPLY_CHAIN_SECURITY_STANDARD.md), [dependency minimization + minimal imports](./AI_DEPENDENCY_MINIMIZATION_STANDARD.md), [SBOM/provenance/signing](./AI_SBOM_PROVENANCE_SIGNING_STANDARD.md), [CI/CD gates + MMMO + anti-overengineering](./AI_CI_CD_ENFORCEMENT_GATES.md)
- Launch: [2-week war plan + abort criteria](./AI_GO_LIVE_2WEEK_WAR_PLAN.md)

---

## AuthZ Precedence Model (canonical policy)
Evaluation order is **strict**; first matching rule wins:

1. **Explicit Deny** — any explicit deny on the principal, resource, or action wins unconditionally.
2. **User Allow** — explicit allow granted directly to the authenticated user.
3. **Role / Group Allow** — allow inherited through role or group membership.
4. **Default Deny** — if no rule matches, access is denied.

Rules:
- Every auth enforcement point must implement this exact precedence order.
- Fail-closed: on auth system error or timeout, deny access; never default-allow.
- JWT validation: verify signature, `iss`, `aud`, `exp`, and `sub` on every request; do not trust claims without verification.
- Auth decisions must be logged (not the token itself) with result, principal, resource, and timestamp.

## Identity / Config Drift Detection
- Any identity provider mapping (e.g. Okta group → local role) must be reconciled on a schedule: at minimum daily.
- Drift detection job: compare IdP state vs local DB state; alert on any divergence > 0 within SLA.
- Remediation SLA: **critical drift** (e.g. orphaned admin access) → remediate within 1 hour; **non-critical drift** → remediate within 24 hours.
- Drift events must be immutably logged and reviewable by security owners.
- Config drift (env vars, feature flags, infrastructure state) follows the same reconcile → alert → remediate loop.

## Break-Glass / PIM Operating Controls
When a privileged access request bypasses normal access flow:

- **Approval**: requires named approver (not self-approve); approval is logged with reason and requestor.
- **TTL**: maximum session TTL for break-glass access is 4 hours; shorter TTLs preferred.
- **Auto-revoke**: access must be automatically revoked at TTL expiry; no manual extension without re-approval.
- **Immutable audit log**: all actions taken during break-glass session must be logged to an append-only, tamper-evident store.
- **Post-access review**: within 24 hours of session expiry, the access event and actions taken must be reviewed by a security owner.
- **Justification**: every break-glass activation requires a linked incident or change ticket.

---

## Agent Operating Conventions

**Token / output discipline**
- Low token density: answer directly, no filler ("Great question!", "Certainly!", "As an AI…"), no emoji.
- Scope responses to what changed; do not restate the full prior conversation.
- No placeholder prose in production code paths.

**File discipline**
- No new files for agent notes, context memory, or planning — update existing rubric/log files in place.
- Edit files in place; never recreate a file that already exists (data-loss risk).
- Temporary helper files must live under `/tmp`, never in repo.
- Encode lessons as table rows or checklist items in the relevant domain checklist.
- Log every rubric change in `rubrics/ROLLING_UPDATE_LOG.md` per the entry template.

**Exception: when creating a new file is allowed**
A new file in `docs/ai-memory/` is allowed only when ALL of the following are true:
1. The topic has no existing canonical owner in the topic map above.
2. The content cannot be a section in an existing file without making that file incoherent.
3. The file will be immediately linked in the Canonical Topic Map in this document.
4. The creating agent logs the addition in `rubrics/ROLLING_UPDATE_LOG.md`.

**Engineering discipline**
- No over-engineering: new abstractions, wrappers, config layers, or event hops must be justified by repeated need, not speculative flexibility.
- Make the smallest change that fully solves the problem; do not fix unrelated issues.
- Use existing libraries; do not add or upgrade dependencies unless required.
- No LLM in the v1 core synchronous path (authn/authz, transaction commit, payment, p95-critical logic).
- Run existing lint/test gates before and after every code change.

**Cross-repo learning loop**
When a CI failure or production incident reveals a pattern:
1. Fix the immediate issue in the source repo.
2. Add the avoid/correct pair to the relevant domain checklist in `rubrics/CHECKLISTS/`.
3. Log the entry in `rubrics/ROLLING_UPDATE_LOG.md` with source repo + PR/incident number.
4. If the pattern affects a canonical standard, update the relevant `docs/ai-memory/` file.
All four steps must happen in the same PR or a follow-up PR tagged as "cross-repo learning."

**Supply chain (hard gates)**
- Cross-reference every import against the committed lockfile on every PR; reject if new transitive packages appear without explicit approval.
- Restrict lifecycle scripts in CI: `npm ci --ignore-scripts` / `pip install --no-build-isolation` or equivalent.
- Import specific sub-modules, not whole packages; CI bot must flag whole-package imports.
- Rotate all secrets immediately if any affected package version was installed during a known compromise window.
- Valid SLSA provenance is necessary but not sufficient — compromised CI can produce valid-looking provenance.
- See [AI_SUPPLY_CHAIN_SECURITY_STANDARD.md](./AI_SUPPLY_CHAIN_SECURITY_STANDARD.md) and [AI_DEPENDENCY_MINIMIZATION_STANDARD.md](./AI_DEPENDENCY_MINIMIZATION_STANDARD.md).

**Error contract (hard gates)**
- All error responses must follow RFC 7807 Problem Details JSON: `type`, `title`, `status`, `detail`, `instance`.
- Never surface stack traces, SQL, internal module names, or topology in client-visible responses.
- 4xx = client/actionable fault; 5xx = server/operator fault. Never `200` with hidden error payload.
- See [AI_ERROR_HANDLING_CLIENT_CONTRACT.md](./AI_ERROR_HANDLING_CLIENT_CONTRACT.md).

**Logging / observability (hard gates)**
- Only structured JSON logs in production paths — no `console.log`, `print()`, or unstructured strings.
- CI must run an AST check rejecting bare `console.log` / `print` in production code.
- Emit RED metrics (Rate, Errors, Duration) at every service boundary; no other metrics are required at minimum.
- Backend logs are for operators; client error responses are for callers — never conflate.
- See [AI_OBSERVABILITY_MINIMAL_LOGGING_STANDARD.md](./AI_OBSERVABILITY_MINIMAL_LOGGING_STANDARD.md).

**Health checks**
- `/health` (liveness): process-only, no DB/cache calls, responds < 200ms.
- `/ready` (readiness): queries local DB connection and cache to confirm connectivity; no business logic.
- Both return structured JSON `{"status": "ok"|"degraded"|"down", "checks": {...}}`.
- See [AI_HEALTHCHECK_READINESS_LIVENESS_STANDARD.md](./AI_HEALTHCHECK_READINESS_LIVENESS_STANDARD.md).

**CI gates (hard blockers)**
- ruff: F401 unused imports, F541 f-string without placeholders, F841 unused locals.
- mypy: annotate nested dicts (`Dict[str,Dict[str,str]]`), explicit return types, type all `.get()` locals.
- Full gate order: lint → typecheck → unit → contract → integration → security → build. Each declares `needs:` on prior.
- Cross-repo CI failures → add avoid/correct pair to the relevant domain checklist; cite source repo + PR, no new files.

**Memory / learning**
- When a pattern causes a CI failure in any repo, add the avoid/correct pair to the relevant domain checklist here.
- Do not store secrets, credentials, or personal data in memory.

---

## Definition of Done
- Topic ownership is unambiguous.
- Cross-links replace duplicate prose.
- Exceptions are explicit, reviewable, and time-bounded.
- FTRP eligibility check passes before any substantive agent output.
