# System Design & Engineering Universal Reference — Project Summary

## Status: Active, living architecture reference

**Repository**: EPdacoder05/System-Design-Engineering-Universal-Reference  
**Role**: Canonical reusable reference for backend + AI systems architecture  
**Scope**: API patterns, security, database/vector primitives, CI/test templates, decision rubrics, multi-agent RAG workflow map  
**Not in scope**: Product-specific executable logic — that belongs in the owning product repo  

---

## What Lives Here

### Reusable reference patterns (stay here)
- `api/` — FastAPI service template, GraphQL, gRPC, WebSocket, Cloudflare platform, rate limiter, idempotency
- `database/` — async SQLAlchemy, pgvector/RAG primitives, part-type catalog
- `security/` — auth framework, input validation (32+ patterns), encryption, AI-era security, IaC hardening
- `cicd/` — CI pipeline templates, security scan workflows, Dockerfile, Terraform module template
- `testing/` — pytest fixtures, factory patterns, async test helpers
- `patterns/` — medallion architecture, service patterns (circuit breaker, saga, retry), advanced distributed patterns
- `performance/` — multi-tier caching, async patterns, Big-O cheatsheet
- `monitoring/` — observability toolkit (structured logging, metrics, SLA)
- `rubrics/` — master rubric, rolling update log, domain checklists (SWE, DevOps, CS, system design), OIDC/OAuth2 quick-ref
- `tools/opsmemory/` — memory/retrieval primitive (ingest, redact, embed, retrieve, MCP server)

### Product code moved out (not here)
- `tools/jarvis/` — extracted to `EPdacoder05/Jarvis-AI-Assistant`; see `tools/jarvis/MIGRATION.md`
- ECC and other repo-specific systems — referenced as architecture notes, not copied in

---

## Backend + AI Systems Learning Path

End-to-end production architecture covered by this repo:

| Layer | Reference file(s) | Production role |
|-------|------------------|-----------------|
| API gateway / edge | `api/cloudflare_platform.py`, `security/iac/cloudflare_terraform.tf` | Entry point, geo-IP rate limiting, WAF, zero-trust |
| Auth | `security/auth_framework.py`, `rubrics/SECURITY/OIDC_OAUTH2_QUICKREF.md` | JWT, RBAC, MFA, PKCE, token rotation |
| Service layer | `api/service_template.py`, `api/grpc_reference.py`, `api/graphql_reference.py` | REST, gRPC, GraphQL, idempotency |
| Async / queuing | `performance/async_patterns.py`, `patterns/service_patterns.py` | Fan-out, saga, retry with backoff |
| Storage / ORM | `database/connection.py`, `database/model_patterns.py` | Async SQLAlchemy, UUID PKs, soft-delete, audit trails |
| Vector retrieval | `database/vector_search.py`, `tools/opsmemory/` | pgvector embeddings, RAG pipeline |
| Observability | `monitoring/observability.py` | Structured logging, metrics, SLA tracking |
| CI / release gates | `.github/workflows/ci.yml`, `cicd/test-pipeline.yml`, `cicd/security-scan.yml` | Parallel lint→type→unit→integration→e2e→security gate |
| Decision rubrics | `rubrics/MASTER_RUBRIC.md`, `TRADEOFFS.md` | Architecture decisions, 50+ tradeoff analyses |

---

## Ownership Rules

| This repo owns | Product repos own |
|---------------|------------------|
| Reusable patterns and templates | Executable product logic |
| Decision rubrics and checklists | Product-specific MCP servers and vaults |
| Architecture maps and workflow diagrams | Deployment wiring and dashboards |
| CI/CD template files | Product-specific CI configurations |
| When in doubt — keep a reference here | When in doubt — move the implementation there |

---

## Key principles

- Every module is standalone; no internal cross-module imports
- No hardcoded secrets — all configuration from environment variables
- No PII, no sector-specific data — patterns only
- Validation: ruff (F401/F541/F841), mypy, pytest run in CI on every PR
