## Mission → Money → Ownership (MMMO)

**Mission**: What user/operator problem does this solve?

**Money**: Does this touch billing, compliance, SLA, or revenue paths?
- [ ] No
- [ ] Yes → details:

**Ownership**: Named owner of changed component:
On-call rotation confirmed: [ ] Yes

**Rollback**: Rollback path and estimated TTR if this causes a production incident:

---

## Impact Trace (required for non-trivial changes)

- **Changed**: 
- **Upstream deps** (what calls/imports this):
- **Downstream deps** (what this calls/imports):
- **Contracts broken** (API, event schema, DB columns, env vars):
- **Affected tests**:
- **Required follow-on changes**:

---

## Blast Radius

Score (0–18 per [AI_ROUTER_SCORING_MATRIX.md](../docs/ai-memory/AI_ROUTER_SCORING_MATRIX.md)):

Isolation strategy (flag / partition / canary / region):

---

## Checklist

- [ ] MMMO section complete
- [ ] Impact trace complete (or N/A for trivial change)
- [ ] Blast radius scored
- [ ] Rollback path documented
- [ ] CI gates passing (lint → typecheck → unit → contract → integration → security → build)
- [ ] New abstraction justified with "repeated need evidence" (or N/A)
- [ ] Lockfile diff reviewed — no unexpected new transitive packages
- [ ] Structured logs only — no `console.log` / `print` in production paths
- [ ] Error responses follow RFC 7807 (or N/A)
