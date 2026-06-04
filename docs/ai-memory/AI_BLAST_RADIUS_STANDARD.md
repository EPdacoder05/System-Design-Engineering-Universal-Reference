# AI Blast Radius Standard

Canonical source for assessing impact before a change ships.

## Minimum Assessment
- Who is affected: operator, tenant, customer segment, all customers.
- What is affected: data correctness, auth, latency, availability, money, compliance.
- How recovery works: rollback, replay, restore, or manual remediation.
- Whether the change is isolated by flag, partition, queue, region, tenant, or environment.

## GitNexus Impact-Trace (required merge gate)
Before merge, every non-trivial change must produce a coupling evidence block:

```
Impact Trace:
  Changed:        <file or module>
  Upstream deps:  <what calls / imports this>
  Downstream deps:<what this calls / imports>
  Contracts broken: <API signatures, event schemas, DB columns, env vars>
  Affected tests: <test files that directly exercise changed paths>
  Required follow-on changes: <list every file that must also change>
```

- Reviewers must reject PRs where required follow-on changes are undeclared.
- For systems with ≥ 5 coupled files, run a dependency diff tool (e.g. GitNexus, `pydeps`, `madge`, `dependency-cruiser`) and attach the 1D impact plane output to the PR.
- "Change one, break nine" anti-pattern: if a change touches > 3 contracts (API, schema, event, config), require explicit sign-off from each contract owner.

## Owner Accountability Table (required for high-blast-radius changes)
| Component | Owner | SLO | Rollback Authority | On-call Rotation |
|---|---|---|---|---|
| (fill per change) | | | | |

Every high-blast-radius PR (score ≥ 9 per [AI_ROUTER_SCORING_MATRIX.md](./AI_ROUTER_SCORING_MATRIX.md)) must include a populated table before merge.

## Guardrails
- Start with the smallest release slice that proves the change.
- High-blast-radius changes require confirmation, rollback, and monitoring owners.
- If rollback is not fast, reduce the release slice or redesign first.

## Anti-Patterns
- Global changes with no canary.
- Shared mutable state with no containment boundary.
- Customer-wide migrations with no pause point.
- PRs that silently break downstream contracts without listing them.

## Definition of Done
- Blast radius is scored.
- Impact trace is declared and reviewed.
- Owner accountability table is populated for high-risk changes.
- Isolation strategy exists.
- Rollback path is practiced or at least credible.
