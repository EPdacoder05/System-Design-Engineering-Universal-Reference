# Rolling Master Rubric

**Version:** 1.0.0  
**Status:** Active  
**Update Mode:** Rolling (continuous improvement)

This rubric is the canonical scoring and governance template for all related repositories.

## 1) Fixed Scoring Dimensions

Score each dimension from **1 (weak)** to **5 (strong)**.

| Dimension | What to evaluate |
|---|---|
| Correctness | Meets requirements, handles edge cases, validated outputs |
| Reliability | Failure tolerance, retries, timeouts, graceful degradation |
| Security | Authentication, authorization, secret hygiene, vuln controls |
| Scalability | Throughput/latency behavior under growth and concurrency |
| Operability | Deployability, observability, runbooks, incident readiness |
| Cost | Resource efficiency, measurable cost/benefit, waste controls |
| Maintainability | Simplicity, modularity, readability, testability |
| Clarity | Clear docs, assumptions, tradeoffs, handoff readiness |

**Overall Score:** sum / 40

---

## 2) Required Rolling Fields (Per Change)

Every meaningful update must include:

1. **What changed**
2. **Why**
3. **Impact** (users/systems/risk/cost)
4. **Next experiment**
5. **Deprecation notes** (if any)

Use: [`ROLLING_UPDATE_LOG.md`](./ROLLING_UPDATE_LOG.md)

---

## 3) Required Deltas (Per Update)

Every update must record:

- **Rubric score delta** (before → after)
- **Risk delta** (increased / neutral / reduced + reason)
- **Production readiness delta** (checklist movement)

---

## 4) Domain Checklists

- SWE: [`CHECKLISTS/SWE.md`](./CHECKLISTS/SWE.md)
- DevOps: [`CHECKLISTS/DEVOPS.md`](./CHECKLISTS/DEVOPS.md)
- CS Fundamentals: [`CHECKLISTS/CS_FUNDAMENTALS.md`](./CHECKLISTS/CS_FUNDAMENTALS.md)
- System Design: [`CHECKLISTS/SYSTEM_DESIGN.md`](./CHECKLISTS/SYSTEM_DESIGN.md)

---

## 5) Security Canonical Clarification (P0)

Use this quick reference for identity and authorization decisions:

- [`SECURITY/OIDC_OAUTH2_QUICKREF.md`](./SECURITY/OIDC_OAUTH2_QUICKREF.md)

---

## 6) Rolling Governance Cadence

- **Per change:** update rolling log + deltas
- **Monthly:** review rubric drift and scoring consistency
- **Quarterly:** promote proven patterns, retire weak patterns

Default posture: improvements ship on a **rolling update basis**, with versioned traceability.

---

## 7) Coding Agent Conventions

Rules for any AI/LLM agent writing code or docs in these repos. Violations degrade signal quality and increase token waste across sessions.

**Output quality**
- No emoji spam; omit entirely or use once where semantically meaningful
- No filler phrases ("Great question!", "Certainly!", "As an AI...") — start with the answer
- No placeholder prose ("TODO: add logic here") in production code paths
- Responses scoped to what changed; no restating the full prior conversation

**Memory / context management**
- Do not create new files solely for agent notes or context memory — update existing rubric/log files
- Encode lessons as table rows or checklist items in the relevant domain checklist
- Log every rubric change in `ROLLING_UPDATE_LOG.md` per the entry template

**Plug-and-play templates**
- New project scaffolds must pass `ruff check` + `mypy --strict` (or equivalent) before first commit
- CI pipeline gate order: lint → typecheck → test → build; downstream jobs must declare `needs:` on upstream hard gates
- `Build Docker Image` (or equivalent build step) must declare `needs: [lint, typecheck, test]` so a type failure does not silently skip the build

**Cross-repo learning**
- When a pattern causes a CI failure in any repo, add the avoid/correct pair to the relevant domain checklist here
- Source repo + PR number as a comment is sufficient citation; no new files needed
