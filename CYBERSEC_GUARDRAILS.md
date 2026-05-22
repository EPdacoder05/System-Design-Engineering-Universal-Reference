# Cybersecurity Guardrails (Defense-First)

This guide defines a security posture for high-risk environments using **defense in depth** and **continuous hardening**.

## Core Doctrine
- Build guardrails inside the system and strong perimeter controls outside it.
- Assume breach, detect early, contain fast, recover safely.
- Improve controls on a rolling basis after every incident or near miss.

## Non-Negotiable Rules
- **No doxxing** or personal targeting.
- **No unauthorized access** or exploitation.
- **No offensive misuse** of recon or testing tools.
- Use lawful, authorized, and policy-compliant security workflows only.

## Layered Defense Architecture

### 1) Identity and Access
- Enforce MFA for privileged access.
- Use least-privilege RBAC/ABAC.
- Rotate credentials and short-lived tokens.
- Prefer OIDC + OAuth2 patterns for modern authn/authz.

### 2) Application Guardrails
- Input validation and output encoding.
- Parameterized queries and secure deserialization.
- Rate limiting, circuit breakers, and strict timeout policies.
- Security headers, CSRF defenses, and dependency pinning.

### 3) Infrastructure Hardening
- Patch cadence and immutable image strategy.
- Network segmentation and zero-trust boundaries.
- Non-root containers, minimal base images, signed artifacts.
- Secrets in vaults, never hardcoded.

### 4) Detection and Telemetry
- Centralized structured logs and correlation IDs.
- Alerting on auth anomalies, privilege escalation, and data exfil indicators.
- WAF/IDS/EDR signal integration where applicable.
- Retention and audit trail policies aligned to compliance requirements.

### 5) Incident Response (Rolling)
- Prepare: playbooks, ownership, and escalation tree.
- Detect: triage, classify severity, preserve evidence.
- Contain: isolate affected components and revoke compromised credentials.
- Eradicate: remove persistence, patch root cause.
- Recover: phased restore with enhanced monitoring.
- Learn: postmortem, control updates, and rubric delta logging.

## Threat Intelligence and Recon (Defensive Use)
Use open-source intelligence and external signal collection only for:
- Asset inventory and exposed surface monitoring
- Brand/domain impersonation detection
- Credential leak monitoring
- Vulnerability prioritization

Always require:
- Legal authorization
- Scope boundaries
- Audit logging of collection and access
- Data minimization and retention controls

## High-Value Operational Checklist
- [ ] Privileged accounts protected by MFA and break-glass controls
- [ ] OIDC/OAuth2 validation checks enforced (`iss`, `aud`, signature, rotation)
- [ ] Security logging with tamper-evident retention
- [ ] IR runbook tested via tabletop or simulation
- [ ] Dependency and container scanning on every release path
- [ ] Recovery objectives (RTO/RPO) defined and rehearsed

## Integration With This Repository
- Security patterns: `SECURITY_PATTERNS.md`
- Production checklist: `PRODUCTION_READINESS.md`
- OIDC/OAuth2 quick reference: `rubrics/SECURITY/OIDC_OAUTH2_QUICKREF.md`
- Docker hardening: `docker/DOCKER_SECURITY.md`

## Rolling Improvement Requirement
Every significant security change should update:
- What changed
- Why
- Impact
- Risk delta
- Production readiness delta

Use: `rubrics/ROLLING_UPDATE_LOG.md`
