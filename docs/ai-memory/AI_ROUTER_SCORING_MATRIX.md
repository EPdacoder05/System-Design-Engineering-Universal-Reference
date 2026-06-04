# AI Router Scoring Matrix

Canonical source for deciding whether a change can proceed automatically, needs review, or needs explicit approval.

## Score Each Dimension (0-3)
| Dimension | 0 | 1 | 2 | 3 |
|---|---|---|---|---|
| Blast radius | isolated | one service | multi-service | org/customer wide |
| Data impact | none | read-only | write/update | delete/corrupt/irreversible |
| Security impact | none | internal only | authz/config | authn/secret/trust boundary |
| Runtime criticality | async/non-critical | backoffice | customer request path | money/mission/safety critical |
| Reversibility | instant rollback | same day | migration needed | hard/impossible rollback |
| Operational novelty | known pattern | small variant | new dependency/path | new operating model |

## Routing Thresholds
- 0-4: proceed with normal review.
- 5-8: require topic-standard check and reviewer acknowledgment.
- 9-12: require confirmation before execution.
- 13-18: require confirmation, blast-radius plan, rollback plan, and post-change monitoring owner.
- Any score 3 in security, delete/corruption, or money path -> escalate regardless of total.

## Required Attachments for Escalated Work
- Relevant standard link
- Rollback statement
- Metrics/logs to watch
- Owner and expiry for any exception

## Definition of Done
- Scoring is recorded before execution.
- Escalation rule is followed, not bypassed.
