# Architecture Summary: Backend + AI Systems Reference

## What this repo is

A canonical, lightweight reference for backend and AI systems architecture.
It is not a product, not a finished deployment, and not a one-time deliverable.
Every file here is either a reusable pattern, a decision rubric, an architecture diagram, or a CI/test template.

---

## Architecture layers and their reference files

### 1. Edge / API gateway
- `api/cloudflare_platform.py` — CORS builder, security headers, geo-IP rate limiter, Workers/Durable Objects templates, Turnstile, D1/KV/R2/Queues feature matrix
- `security/iac/cloudflare_terraform.tf` — full Terraform config: TLS, WAF, OWASP CRS, geo-IP block rules, HSTS, rate limits, R2, Access, Tunnel, DNS, DMARC

**Production role:** Entry point. All traffic hits the edge before reaching the origin. WAF and bot filtering run here. Auth enforcement starts here via Cloudflare Access or JWT header inspection.

### 2. Auth
- `security/auth_framework.py` — JWT (access + refresh + rotation), RBAC, MFA/TOTP, API keys with rate limiting
- `rubrics/SECURITY/OIDC_OAUTH2_QUICKREF.md` — canonical OIDC vs OAuth2 decision rule, PKCE, audience/issuer validation, least-privilege scopes

**Production role:** Every inbound request must carry a verifiable identity. Auth framework handles token issuance; OIDC quick-ref covers federated identity and machine-to-machine flows.

### 3. Service layer
- `api/service_template.py` — FastAPI template: request ID middleware, CORS, security headers, health/ready endpoints, structured error responses, rate limiting, API versioning, OpenAPI
- `api/graphql_reference.py` — Strawberry + FastAPI: DataLoader (N+1 elimination), depth/complexity limits, JWT context, Relay pagination, subscriptions, Apollo Federation, APQ
- `api/grpc_reference.py` — all four RPC patterns, JWT interceptor, mTLS, retry policy, deadline propagation, health checking, reflection service
- `api/websocket_reference.py` — WebSocket room manager, JWT on upgrade, per-connection rate limiting, Redis Pub/Sub bridge, SSE, HMAC webhook verification
- `api/idempotency.py` — idempotency key pattern for safe retries
- `api/rate_limiter.py` — token bucket + sliding window (both O(1)), RateLimiterRegistry per carrier/tenant

**Production role:** Choose the protocol based on consumer: REST for public/third-party, gRPC for internal high-throughput, GraphQL for flexible BFF, WebSocket/SSE for real-time push.

### 4. Async jobs / event-driven
- `patterns/service_patterns.py` — circuit breaker (closed/open/half-open), retry with exponential backoff + jitter, fan-out/fan-in, saga (orchestration-based), all async with httpx
- `performance/async_patterns.py` — semaphore-based concurrency limiting, batch processing, async context managers, `asyncio.gather()` with error handling

**Production role:** Async jobs decouple write path from read path. Circuit breakers prevent cascading failures. Saga handles distributed transactions without two-phase commit.

### 5. Storage / ORM
- `database/connection.py` — async SQLAlchemy engine, connection pooling, session factory, health checks
- `database/model_patterns.py` — UUID PKs, audit mixin (created/updated/created_by), soft delete, composite indexes, relationship patterns
- `database/catalog.py` — part-type catalog backed by DB (PartTypeCatalog), soft validation via warn-not-reject

**Production role:** Async SQLAlchemy + UUID PKs + audit trails are the standard ORM baseline. Soft delete avoids data loss. Composite indexes carry query plans.

### 6. Vector retrieval / RAG
- `database/vector_search.py` — pgvector: embedding storage, cosine similarity search, embedding cache, batch insertion, semantic search with filtering
- `tools/opsmemory/` — full RAG pipeline: ingest → redact → embed → retrieve → MCP server, connectors (GitHub), providers (LiteLLM embeddings + LLM), storage (SQLAlchemy + pgvector)

**Production role:** pgvector turns Postgres into a vector store — no separate service required for most workloads. opsmemory provides the complete memory/retrieval primitive that agents call via MCP.

### 7. Observability
- `monitoring/observability.py` — structured JSON logging with correlation IDs, metrics (counters/gauges/histograms), SLA tracking (uptime, latency percentiles), alert thresholds

**Production role:** Correlation IDs link logs across service calls. SLA tracking feeds error budget calculations. Prometheus scrape targets emit the Golden Signals (latency, traffic, errors, saturation).

### 8. CI / release gates
- `.github/workflows/ci.yml` — parallel gate: lint (ruff), typecheck (mypy), unit tests (pytest), coverage
- `cicd/ci-python.yml` — reusable Python CI template
- `cicd/test-pipeline.yml` — matrix testing (Python 3.10/3.11/3.12), coverage artifact upload
- `cicd/security-scan.yml` — CodeQL, Trivy, pip-audit, safety, bandit, gitleaks, SBOM, weekly schedule

**Production role:** Every PR must pass lint + typecheck + unit before merge. Security scan runs weekly and on demand. Matrix testing catches version regressions early.

---

## Multi-agent RAG workflow

Defined as a repo-level architecture. `tools/opsmemory/` is the memory/retrieval primitive.

```
ingest → redact → embed → retrieve → answer → evaluate → refactor → retest
```

| Stage | opsmemory component | Agent role |
|-------|-------------------|------------|
| ingest | `mcp/tools/ingest.py`, `connectors/` | Builder: pull raw data from GitHub, repos, manual input |
| redact | `agent/redactor.py` | Builder: strip PII and secrets before embedding |
| embed | `providers/embeddings/` (LiteLLM) | Builder: generate vectors, batch insert via pgvector |
| retrieve | `mcp/tools/query.py`, `database/vector_search.py` | Reviewer: semantic search, return top-k with scores |
| answer | `providers/llm/` (LiteLLM) | Reviewer/Refactorer: LLM call over retrieved context |
| evaluate | `rubrics/MASTER_RUBRIC.md`, test suite | Reviewer: rubric score delta, test coverage delta |
| refactor | service layer + patterns | Refactorer: apply patterns from `api/`, `patterns/` |
| retest | CI gate | Break-fixer + release gate: all runners must pass |

### Agent roles and validation stages

| Agent | Responsibility | Validation stage |
|-------|---------------|-----------------|
| Builder | Implement from spec, ingest context | Unit tests pass |
| Reviewer/Refactorer | Code review, rubric scoring, refactor | Unit + integration pass |
| Break-fixer | Diagnose CI failure, patch without regression | Unit + integration + e2e pass |
| Test runner | Execute all stages, report coverage | Unit + integration + e2e + security |
| Release gate | Block merge on any failing gate | Unit + integration + e2e + security + regression |

---

## CI/testing architecture (full engineering loop)

The CI gate is not "run pytest." It is:

```
lint (ruff F401/F541/F841)
  → typecheck (mypy: annotate nested dicts, explicit return types)
    → unit tests (pytest, fast, no I/O)
      → integration tests (pytest, real DB, mock external)
        → e2e tests (pytest, full stack or staging env)
          → security scan (bandit, pip-audit, gitleaks)
            → [PASS] merge allowed / [FAIL] → break-fixer agent
```

Post-failure troubleshooting loop:

```
CI failure
  → break-fixer reads failure logs
    → patches root cause (never suppresses lint/type errors)
      → re-runs from lint stage
        → regression check (existing tests still pass)
          → rubric delta logged in rubrics/ROLLING_UPDATE_LOG.md
```

See `.github/workflows/ci.yml` for the base parallel gate implementation.

---

## ECC (external corpus / cross-cutting) architecture reference

ECC contributes the following generalizable patterns for backend/AI design:

- **Carrier identity normalization** — `api/carrier_identity.py`: data-driven catalog, fuzzy matching with confidence tiers (≥0.90 accept / 0.72–0.89 review / <0.72 reject), per-tenant isolation, SHA-256 hash-chained audit trail
- **Part-type catalog** — `database/catalog.py`: soft validation (warn, not reject), DB-backed type registry, type_code restricted to `[a-z0-9_]`, new types via DB insert not code deploy
- **Medallion architecture** — `patterns/medallion_architecture.py`: Bronze→Silver→Gold ETL, schema enforcement at each tier, metadata tracking

Only patterns that generalize to any backend/AI system are kept here. Product-specific ECC implementation details live in the ECC product repo.

---

## Jarvis migration record

`tools/jarvis/` — homelab intelligence MCP server — has been extracted to `EPdacoder05/Jarvis-AI-Assistant`.

See `tools/jarvis/MIGRATION.md` for the inventory and pointer.

The reusable MCP integration pattern (how opsmemory connects to an external consumer via MCP) remains in `tools/opsmemory/mcp/` and `tools/opsmemory/integrations/jarvis/` as a reference integration example.

