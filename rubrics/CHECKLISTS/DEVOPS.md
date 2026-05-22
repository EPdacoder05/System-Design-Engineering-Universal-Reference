# DevOps Checklist (Rolling Rubric)

Reference rubric version: **1.0.0**

## P0
- [ ] Rolling deployment is default strategy
- [ ] Progressive delivery path defined (canary and/or blue-green)
- [ ] Rollback triggers tied to SLO thresholds
- [ ] Retry budgets and load-shedding policy defined
- [ ] DLQ and poison-message handling defined for async pipelines

## P1
- [ ] Runbook links attached to critical alerts
- [ ] Release gates include security + reliability checks
- [ ] Disaster recovery exercises scheduled and tracked
- [ ] Cost controls included in deploy review (right-sizing/budgets)
