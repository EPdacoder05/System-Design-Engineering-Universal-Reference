# AI Memory Standards Pack

This directory is the canonical backend/platform engineering standards pack for this repository.

## Reading Order
1. [AI_MEMORY_UNIVERSAL.md](./AI_MEMORY_UNIVERSAL.md)
2. [AI_ROUTER_POLICY.md](./AI_ROUTER_POLICY.md)
3. [AI_ROUTER_SCORING_MATRIX.md](./AI_ROUTER_SCORING_MATRIX.md)
4. [AI_CONFIRMATION_PROTOCOL.md](./AI_CONFIRMATION_PROTOCOL.md)
5. [AI_BLAST_RADIUS_STANDARD.md](./AI_BLAST_RADIUS_STANDARD.md)
6. Implementation standards:
   - [AI_SQL_FIRST_ENGINEERING_STANDARD.md](./AI_SQL_FIRST_ENGINEERING_STANDARD.md)
   - [AI_DB_PER_MICROSERVICE_STANDARD.md](./AI_DB_PER_MICROSERVICE_STANDARD.md)
   - [AI_OUTBOX_EVENTING_STANDARD.md](./AI_OUTBOX_EVENTING_STANDARD.md)
   - [AI_PARTITIONING_MONTHLY_VS_YEARLY_STANDARD.md](./AI_PARTITIONING_MONTHLY_VS_YEARLY_STANDARD.md)
   - [AI_IDEMPOTENCY_STANDARD.md](./AI_IDEMPOTENCY_STANDARD.md)
   - [AI_DATALOSS_DLQ_REPLAY_STANDARD.md](./AI_DATALOSS_DLQ_REPLAY_STANDARD.md)
   - [AI_CONCURRENCY_RACE_CONDITION_STANDARD.md](./AI_CONCURRENCY_RACE_CONDITION_STANDARD.md)
   - [AI_NESTED_LOOP_AND_COMPLEXITY_GUARDRAILS.md](./AI_NESTED_LOOP_AND_COMPLEXITY_GUARDRAILS.md)
7. Runtime and delivery standards:
   - [AI_ERROR_HANDLING_CLIENT_CONTRACT.md](./AI_ERROR_HANDLING_CLIENT_CONTRACT.md)
   - [AI_OBSERVABILITY_MINIMAL_LOGGING_STANDARD.md](./AI_OBSERVABILITY_MINIMAL_LOGGING_STANDARD.md)
   - [AI_HEALTHCHECK_READINESS_LIVENESS_STANDARD.md](./AI_HEALTHCHECK_READINESS_LIVENESS_STANDARD.md)
   - [AI_SUPPLY_CHAIN_SECURITY_STANDARD.md](./AI_SUPPLY_CHAIN_SECURITY_STANDARD.md)
   - [AI_DEPENDENCY_MINIMIZATION_STANDARD.md](./AI_DEPENDENCY_MINIMIZATION_STANDARD.md)
   - [AI_SBOM_PROVENANCE_SIGNING_STANDARD.md](./AI_SBOM_PROVENANCE_SIGNING_STANDARD.md)
   - [AI_CI_CD_ENFORCEMENT_GATES.md](./AI_CI_CD_ENFORCEMENT_GATES.md)
8. Review and ops protocols:
   - [AI_CODE_REVIEW_ANTI_SLOP_CHECKLIST.md](./AI_CODE_REVIEW_ANTI_SLOP_CHECKLIST.md)
   - [AI_LOAD_TEST_BREAKPOINT_PROTOCOL.md](./AI_LOAD_TEST_BREAKPOINT_PROTOCOL.md)
   - [AI_PERF_FAILURE_ANALYSIS_PLAYBOOK.md](./AI_PERF_FAILURE_ANALYSIS_PLAYBOOK.md)
   - [AI_MISSION_MONEY_OWNERSHIP_SCORECARD.md](./AI_MISSION_MONEY_OWNERSHIP_SCORECARD.md)
   - [AI_30_60_90_ROLLOUT_PLAN.md](./AI_30_60_90_ROLLOUT_PLAN.md)
   - [AI_GO_LIVE_2WEEK_WAR_PLAN.md](./AI_GO_LIVE_2WEEK_WAR_PLAN.md)
9. Reusable formats and adapters:
   - [AI_DECISION_RECORD_TEMPLATE.md](./AI_DECISION_RECORD_TEMPLATE.md)
   - [AI_KNOWLEDGE_ARTICLE_TEMPLATE.md](./AI_KNOWLEDGE_ARTICLE_TEMPLATE.md)
   - [AI_PLATFORM_ADAPTERS_CURSOR_VSCODE_COPILOT_PYCHARM.md](./AI_PLATFORM_ADAPTERS_CURSOR_VSCODE_COPILOT_PYCHARM.md)
   - [AI_PROJECT_BOOTSTRAP_CHECKLIST.md](./AI_PROJECT_BOOTSTRAP_CHECKLIST.md)

## Canonical-Source and De-Dup Rules
- One topic, one canonical file in this directory.
- Other docs may mention a rule in one sentence, then link here; they must not restate the full standard.
- If overlap is discovered, move the full rule into the canonical file and replace duplicates with links.
- Templates are canonical for format only; they must not become policy documents.
- Broader repository docs stay broad references. Example: container details stay in `docker/DOCKER_SECURITY.md`; this pack points to them when the topic needs deeper implementation detail.

## Contribution and Edit Rules
- Edit the canonical file first, then update inbound links only if scope changed.
- Keep language short: checklist, guardrail, anti-pattern, definition of done.
- Generalize lessons into standards; do not copy repo-specific incident prose into this pack.
- Prefer updating an existing canonical file over adding another memory/context file for nearby guidance.
- Optimize for maintainer comprehension: prefer the smallest rule set and shortest checklist that still preserves safety.
- Preserve the default: monthly partitioning for high-write and event tables, with precreate, retention, and partition-health jobs.
- Preserve the v1 rule: no LLM calls in the core synchronous path for auth, request admission, transaction commit, or money movement.
- Use [AI_CONFIRMATION_PROTOCOL.md](./AI_CONFIRMATION_PROTOCOL.md) for destructive or high-blast-radius changes.

## External Reference Links Inside This Repo
- `rubrics/MASTER_RUBRIC.md`
- `rubrics/CHECKLISTS/DEVOPS.md`
- `CYBERSEC_GUARDRAILS.md`
- `docker/DOCKER_SECURITY.md`
- `.github/workflows/ci.yml`
