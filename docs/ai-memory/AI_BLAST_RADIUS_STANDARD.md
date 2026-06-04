# AI Blast Radius Standard

Canonical source for assessing impact before a change ships.

## Minimum Assessment
- Who is affected: operator, tenant, customer segment, all customers.
- What is affected: data correctness, auth, latency, availability, money, compliance.
- How recovery works: rollback, replay, restore, or manual remediation.
- Whether the change is isolated by flag, partition, queue, region, tenant, or environment.

## Guardrails
- Start with the smallest release slice that proves the change.
- High-blast-radius changes require confirmation, rollback, and monitoring owners.
- If rollback is not fast, reduce the release slice or redesign first.

## Anti-Patterns
- Global changes with no canary.
- Shared mutable state with no containment boundary.
- Customer-wide migrations with no pause point.

## Definition of Done
- Blast radius is scored.
- Isolation strategy exists.
- Rollback path is practiced or at least credible.
