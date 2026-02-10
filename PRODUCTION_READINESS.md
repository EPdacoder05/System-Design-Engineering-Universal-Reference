# ✅ Production Readiness Checklist

Universal checklist applicable to ALL projects before production deployment.

**Use this checklist for**: APIs, microservices, ML systems, data pipelines, web applications

---

## 📦 Code & Dependencies

### Source Control
- [ ] Code committed to version control (Git)
- [ ] `.gitignore` configured (exclude build artifacts, secrets, IDE files)
- [ ] `README.md` with setup instructions
- [ ] `CHANGELOG.md` maintained
- [ ] Feature branches merged to `main`/`master`
- [ ] No merge conflicts
- [ ] Git tags for releases (semantic versioning: v1.0.0)

### Dependencies
- [ ] `requirements.txt` / `package.json` / `go.mod` committed
- [ ] `poetry.lock` / `package-lock.json` / `go.sum` committed (if applicable)
- [ ] All dependencies pinned to specific versions (no `*` or `^`)
- [ ] No unused dependencies (audit with `pip-audit`, `npm audit`)
- [ ] No vulnerable dependencies (run security scanners)
- [ ] License compatibility verified (no GPL in commercial projects)
- [ ] Dependency update strategy documented (Dependabot, Renovate)

### Code Quality
- [ ] Linting passes (`ruff`, `flake8`, `eslint`)
- [ ] Type checking passes (`mypy`, `TypeScript`)
- [ ] Code formatting consistent (`black`, `prettier`)
- [ ] No commented-out code blocks
- [ ] No `TODO` or `FIXME` for critical functionality
- [ ] Docstrings/JSDoc for public APIs
- [ ] Complexity metrics acceptable (McCabe < 10)

---

## 🔐 Security

### Secrets Management
- [ ] **No secrets in code** (passwords, API keys, tokens)
- [ ] No secrets in `.env` files (use environment variables)
- [ ] No secrets in Git history (use `git-secrets`, `gitleaks`)
- [ ] Secrets stored in vault (AWS Secrets Manager, Azure Key Vault)
- [ ] Secrets rotation schedule documented (90-day recommended)
- [ ] Service accounts use least privilege
- [ ] API keys have expiration dates

### Authentication & Authorization
- [ ] Authentication implemented (JWT, OAuth, API keys)
- [ ] Authorization enforced (RBAC, ABAC)
- [ ] MFA enabled for admin accounts
- [ ] Session timeouts configured (30 min recommended)
- [ ] Refresh token rotation implemented (single-use)
- [ ] Password policy enforced (12+ chars, complexity)
- [ ] Rate limiting enabled (prevent brute force)

### Input Validation
- [ ] **All 32 attack patterns mitigated** (SQL injection, XSS, etc.)
- [ ] Input length limits enforced
- [ ] File upload restrictions (extension, MIME, size)
- [ ] Regex timeout protection (prevent ReDoS)
- [ ] Content-Type validation
- [ ] Sanitization before storage
- [ ] Parameterized queries (no string concatenation)
- [ ] CSRF tokens for state-changing operations

### Network Security
- [ ] HTTPS enforced (TLS 1.2+)
- [ ] HSTS headers configured (Strict-Transport-Security)
- [ ] Security headers (CSP, X-Frame-Options, X-Content-Type-Options)
- [ ] CORS configured (not `*` in production)
- [ ] Firewall rules (restrict to necessary ports)
- [ ] DDoS protection enabled (CloudFlare, AWS Shield)
- [ ] VPC/network isolation (private subnets)

### Encryption
- [ ] Data at rest encrypted (AES-256)
- [ ] Data in transit encrypted (TLS)
- [ ] Database encryption enabled
- [ ] Backup encryption enabled
- [ ] Key management documented
- [ ] Certificate rotation scheduled
- [ ] Password hashing (PBKDF2, bcrypt, Argon2)

### Compliance
- [ ] SOC2/ISO27001 controls documented (if applicable)
- [ ] GDPR compliance (if handling EU data)
- [ ] PII handling documented
- [ ] Data retention policy defined
- [ ] Audit logging enabled (7-year retention)
- [ ] Quarterly access reviews scheduled
- [ ] Security incident response plan documented

---

## 🧪 Testing

### Test Coverage
- [ ] **95%+ code coverage** (line coverage)
- [ ] Unit tests written (fast, isolated)
- [ ] Integration tests written (database, APIs)
- [ ] End-to-end tests written (critical user flows)
- [ ] Performance tests written (load, stress)
- [ ] Security tests written (OWASP Top 10)
- [ ] Smoke tests for health checks

### Test Types
- [ ] Property-based testing (Hypothesis, fast-check) for business logic
- [ ] Mutation testing (verify test quality)
- [ ] Chaos engineering (failure injection)
- [ ] Accessibility tests (WCAG 2.1 AA)
- [ ] Browser/device compatibility tests
- [ ] API contract tests (Pact, OpenAPI validation)
- [ ] Database migration tests (rollback scenarios)

### Test Environment
- [ ] Test data fixtures available
- [ ] Test database with sample data
- [ ] Mock services for external dependencies
- [ ] CI runs tests on every commit
- [ ] Tests run in isolated environments
- [ ] Test execution time <10 minutes (CI)
- [ ] Parallel test execution enabled

---

## 🔄 CI/CD

### Continuous Integration
- [ ] CI pipeline configured (GitHub Actions, GitLab CI, Jenkins)
- [ ] Automated testing on every commit
- [ ] Linting in CI pipeline
- [ ] Type checking in CI pipeline
- [ ] Security scanning in CI (CodeQL, Snyk)
- [ ] Dependency vulnerability scanning (pip-audit, npm audit)
- [ ] Container scanning (Trivy, Clair)
- [ ] SBOM generation (Software Bill of Materials)
- [ ] Code coverage reported
- [ ] Build artifacts stored

### Continuous Deployment
- [ ] Deployment pipeline automated
- [ ] Blue-green deployment configured
- [ ] Canary deployment available (gradual rollout)
- [ ] Feature flags for toggles
- [ ] Rollback strategy documented
- [ ] **Automatic rollback on health check failure**
- [ ] Database migration strategy (forward-compatible)
- [ ] Zero-downtime deployment validated
- [ ] Smoke tests run post-deployment
- [ ] Deployment approval gate for production

### Infrastructure as Code
- [ ] Infrastructure defined as code (Terraform, Pulumi, CloudFormation)
- [ ] IaC in version control
- [ ] State file managed securely (S3 backend)
- [ ] Terraform plan reviewed before apply
- [ ] Resource tags standardized
- [ ] Cost estimation automated (Infracost)
- [ ] Disaster recovery plan documented

---

## 📊 Observability

### Logging
- [ ] Structured logging (JSON format)
- [ ] Log levels configured (DEBUG, INFO, WARN, ERROR)
- [ ] Correlation IDs for request tracing
- [ ] PII not logged
- [ ] Log rotation configured
- [ ] Centralized log aggregation (ELK, CloudWatch, DataDog)
- [ ] Log retention policy defined (7 years for audit logs)
- [ ] Log analysis alerts configured

### Metrics
- [ ] **Golden Signals monitored** (latency, traffic, errors, saturation)
- [ ] Business metrics tracked (signups, transactions, revenue)
- [ ] Custom metrics defined (cache hit rate, queue depth)
- [ ] Prometheus/Grafana dashboards configured
- [ ] Metrics exported in standard format (Prometheus, StatsD)
- [ ] Metrics retention policy defined
- [ ] Metric-based auto-scaling configured

### Alerting
- [ ] Alerts configured for critical issues
- [ ] Alert thresholds tuned (minimize false positives)
- [ ] On-call rotation defined
- [ ] PagerDuty/OpsGenie integration
- [ ] Alert escalation policy documented
- [ ] Alert fatigue prevention (80%+ confidence threshold)
- [ ] Runbooks linked to alerts
- [ ] Alert acknowledgment tracked

### Tracing
- [ ] Distributed tracing enabled (Jaeger, Zipkin, DataDog APM)
- [ ] Request spans captured
- [ ] External API calls traced
- [ ] Database queries traced
- [ ] Trace sampling configured (1-10%)
- [ ] Trace storage retention defined

### SLA Tracking
- [ ] SLO defined (99.9% uptime, P95 latency < 100ms)
- [ ] **Error budget tracked** (0.1% failure budget)
- [ ] Error budget policy documented (freeze on depletion)
- [ ] SLA reports automated (weekly)
- [ ] Uptime monitoring (Pingdom, UptimeRobot)
- [ ] Incident postmortems documented

---

## 🗄️ Database

### Configuration
- [ ] Connection pooling enabled (20-50 connections)
- [ ] Query timeout configured (30s default)
- [ ] Slow query logging enabled (>1s)
- [ ] Indexes optimized (B-tree, GiST, HNSW)
- [ ] Query planner statistics updated
- [ ] Vacuum/analyze scheduled (PostgreSQL)
- [ ] Read replicas configured (if applicable)

### Backup & Recovery
- [ ] Automated daily backups
- [ ] Backup retention policy (30 days minimum)
- [ ] Backup encryption enabled
- [ ] Point-in-time recovery (PITR) enabled
- [ ] Backup restoration tested (quarterly)
- [ ] RTO/RPO defined (e.g., RTO=1hr, RPO=15min)
- [ ] Disaster recovery plan documented

### Migrations
- [ ] Migration tool configured (Alembic, Flyway, Liquibase)
- [ ] Migrations version controlled
- [ ] Migrations tested in staging
- [ ] Rollback migrations available
- [ ] Data backfill strategies documented
- [ ] Large migrations planned for low-traffic windows

---

## 🚀 Performance

### Optimization
- [ ] Caching implemented (L1/L2/L3 strategy)
- [ ] Database queries optimized (EXPLAIN ANALYZE)
- [ ] N+1 query problems resolved
- [ ] API response time P95 < 100ms
- [ ] Static assets served from CDN
- [ ] Image optimization (WebP, compression)
- [ ] Lazy loading for frontend assets
- [ ] Batch processing for bulk operations

### Load Testing
- [ ] Load tests conducted (Apache Bench, wrk, Locust)
- [ ] Stress tests conducted (beyond expected capacity)
- [ ] Sustained load testing (6+ hours)
- [ ] Spike testing (sudden traffic increases)
- [ ] Performance benchmarks documented
- [ ] Bottlenecks identified and resolved
- [ ] Auto-scaling thresholds validated

### Resource Limits
- [ ] Request timeout configured (30s)
- [ ] Max request size limit (10MB recommended)
- [ ] Rate limiting per user/IP
- [ ] Memory limits per process
- [ ] Disk space monitoring (alert at 80%)
- [ ] CPU throttling thresholds
- [ ] Connection limits enforced

---

## 📖 Documentation

### Code Documentation
- [ ] Architecture diagram (system design)
- [ ] API documentation (OpenAPI/Swagger)
- [ ] Database schema diagram (ER diagram)
- [ ] Sequence diagrams for complex flows
- [ ] Inline comments for complex logic
- [ ] README with setup instructions
- [ ] CONTRIBUTING.md for contributors

### Operational Documentation
- [ ] **Runbooks for common incidents**
- [ ] Deployment guide
- [ ] Rollback procedure
- [ ] Disaster recovery procedure
- [ ] On-call escalation guide
- [ ] Incident response plan
- [ ] Security incident response plan
- [ ] Capacity planning guide

### User Documentation
- [ ] User guide / manual
- [ ] API reference documentation
- [ ] SDK documentation (if applicable)
- [ ] Troubleshooting guide
- [ ] FAQ section
- [ ] Release notes / changelog

---

## 🐳 Containerization & Orchestration

### Docker
- [ ] Dockerfile optimized (multi-stage build)
- [ ] Non-root user configured
- [ ] Health check defined
- [ ] Image tagged properly (semantic versioning)
- [ ] Image scanned for vulnerabilities (Trivy)
- [ ] Image size optimized (<500MB)
- [ ] Base image pinned to specific version
- [ ] `.dockerignore` configured

### Kubernetes (if applicable)
- [ ] Resource requests/limits defined
- [ ] Liveness probe configured
- [ ] Readiness probe configured
- [ ] HPA (Horizontal Pod Autoscaler) configured
- [ ] Pod disruption budget defined
- [ ] Network policies configured
- [ ] Secrets stored in Kubernetes Secrets (not ConfigMaps)
- [ ] Ingress/service mesh configured

---

## 💰 Cost Management

### Optimization
- [ ] Right-sized instances (not over-provisioned)
- [ ] Reserved instances for predictable workloads
- [ ] Spot instances for batch workloads
- [ ] Auto-scaling configured (scale down during off-hours)
- [ ] S3 lifecycle policies (transition to Glacier)
- [ ] Database connection pooling
- [ ] Caching strategy to reduce API costs
- [ ] CloudWatch cost alarms configured

### Tracking
- [ ] Cost dashboard configured (AWS Cost Explorer)
- [ ] Cost allocation tags applied
- [ ] Budget alerts configured (20% over threshold)
- [ ] Monthly cost review scheduled
- [ ] FinOps analysis documented
- [ ] Unit economics tracked (cost per request/user)

---

## 🛠️ Operational Excellence

### Monitoring
- [ ] Uptime monitoring (external service)
- [ ] Performance monitoring (APM)
- [ ] Error tracking (Sentry, Rollbar)
- [ ] User analytics (if applicable)
- [ ] Business metrics dashboard
- [ ] Weekly metrics review meeting

### Incident Management
- [ ] Incident response plan documented
- [ ] On-call rotation schedule
- [ ] Postmortem template available
- [ ] Blameless postmortem culture
- [ ] Incident severity levels defined
- [ ] Mean time to recovery (MTTR) tracked
- [ ] Incident communication plan (status page)

### Continuous Improvement
- [ ] Quarterly security audits
- [ ] Quarterly dependency updates
- [ ] Quarterly performance reviews
- [ ] Quarterly access reviews
- [ ] Postmortem action items tracked
- [ ] Tech debt backlog maintained
- [ ] Innovation time allocated (20%)

---

## 🎓 Team Readiness

### Knowledge Transfer
- [ ] Onboarding documentation
- [ ] Architecture decision records (ADRs)
- [ ] Code walkthrough sessions
- [ ] Disaster recovery drills (quarterly)
- [ ] Security training completed
- [ ] OWASP Top 10 awareness
- [ ] Multiple team members can deploy

### Process
- [ ] Code review process defined (2+ reviewers)
- [ ] Definition of done documented
- [ ] Incident escalation process
- [ ] Change management process
- [ ] Release versioning strategy
- [ ] Hotfix process defined
- [ ] Communication channels defined (Slack, email)

---

## 📋 Pre-Launch Checklist Summary

### 🔴 **CRITICAL** (Must have before launch)
- [ ] No secrets in code
- [ ] All 32 attack patterns mitigated
- [ ] HTTPS enforced
- [ ] Automated backups configured
- [ ] Health checks working
- [ ] Rollback strategy tested
- [ ] Monitoring & alerting configured
- [ ] 95%+ test coverage
- [ ] Security scanning in CI/CD
- [ ] Runbooks documented

### 🟡 **HIGH PRIORITY** (Should have before launch)
- [ ] Blue-green deployment
- [ ] Error budget tracked
- [ ] Auto-scaling configured
- [ ] Load testing completed
- [ ] Disaster recovery plan
- [ ] API documentation
- [ ] Dependency scanning
- [ ] SOC2/ISO27001 controls

### 🟢 **NICE TO HAVE** (Can add post-launch)
- [ ] Chaos engineering
- [ ] A/B testing framework
- [ ] Feature flags
- [ ] Distributed tracing
- [ ] User analytics
- [ ] Performance benchmarks published

---

## ✅ Sign-off

| Role | Name | Date | Signature |
|------|------|------|-----------|
| Engineering Lead | | | |
| Security Officer | | | |
| DevOps Lead | | | |
| Product Manager | | | |
| QA Lead | | | |

---

**Checklist Version**: 1.0  
**Last Updated**: February 2026  
**Next Review**: Quarterly
