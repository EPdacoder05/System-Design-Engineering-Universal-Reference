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

## Docker Container Security (hard gate — never ship privileged containers)

| Pattern to avoid | Correct pattern |
|-----------------|-----------------|
| `privileged: true` on any service | `cap_drop: [ALL]` + surgical `cap_add` only for needed caps (e.g., `NET_ADMIN`) |
| `image: service:latest` | Pin to explicit version tag: `image: service:2024.2.0` |
| No healthcheck | `healthcheck: test/interval/timeout/retries/start_period` on every service |
| No security context | `security_opt: [no-new-privileges:true]` on every service |
| Writable root filesystem | `read_only: true` + `tmpfs: [/tmp:size=100M,mode=1777]` where possible |
| Implicit build context | `build: {context: ./service, dockerfile: Dockerfile}` — always explicit |
| Missing `requirements.txt` referenced in CI | Create file before CI references it; missing file causes `[Errno 2]` and skips all downstream steps |

**Coding-agent rule:** `Build Docker Image` is downstream of `Lint` and `Type Check`. If either hard gate fails, Docker build is silently skipped — fix lint/mypy first, then verify the build step actually ran.
