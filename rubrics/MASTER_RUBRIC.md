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
