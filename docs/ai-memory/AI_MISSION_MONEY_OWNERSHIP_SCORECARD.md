# AI Mission/Money Ownership Scorecard

Canonical source for deciding ownership rigor.

## Score 0-2 Per Area
| Area | 0 | 1 | 2 |
|---|---|---|---|
| Mission criticality | convenience | important | hard outage if broken |
| Money movement | none | indirect | direct charge/payout |
| Data integrity | low impact | recoverable | legal/financial/audit critical |
| Auth/trust | internal only | user-facing | privileged or compliance boundary |
| Operational burden | simple | moderate | 24x7/high-severity |

## Interpretation
- 0-3: standard ownership
- 4-6: named service owner and runbook required
- 7-10: senior approval, explicit rollback owner, launch war-plan required

## Definition of Done
- Ownership level matches business and operational risk.
