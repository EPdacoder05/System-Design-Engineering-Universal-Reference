# 🎯 System Design & Engineering Universal Reference Library

![Python](https://img.shields.io/badge/Python-3.11%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Production--Ready-brightgreen)
![Platform](https://img.shields.io/badge/Platform-Agnostic-orange)

**A complete, modular, copy-paste-ready engineering reference library** across all disciplines. This is a portable toolkit that can be cloned and used at ANY company, ANY project, ANY role — from analyst to engineer to pentester. **No PII, no sector-specific info. Pure engineering patterns.**

## 🚀 Quick Start

Need authentication? → `cp security/auth_framework.py your-project/`  
Need caching? → `cp performance/caching.py your-project/`  
Need CI/CD? → `cp cicd/test-pipeline.yml .github/workflows/`

**Every file works standalone. Take what you need. Ship it.**

## 🗺️ Portfolio Ecosystem

This repository is the **foundation** of a portable engineering portfolio:

```
EPdacoder05/
├── System-Design-Engineering-Universal-Reference  ← You are here (reference library)
├── TF2S3-migration                                ← IaC automation (uses cicd/ templates)
├── finops-cost-control-as-code                    ← FinOps (uses ml/anomaly_detector.py patterns)
├── NullPointVector                                ← Security platform (uses security/ patterns)
├── Sportsbook-aggregation                         ← Real-time analytics (uses patterns/ + api/)
├── incident-replay-tool                           ← ML prediction (uses ml/ + monitoring/)
├── ha-iot-stack                                   ← IoT infrastructure (uses cicd/ + Dockerfile)
└── ha-ble-mqtt-bridge                             ← IoT bridge (uses cicd/ templates)
```

Patterns from this library power every project in the ecosystem.

## 📚 Table of Contents

- [Architecture Patterns](#-architecture-patterns)
- [Security](#-security)
- [API Development](#-api-development)
- [Performance](#-performance)
- [Database](#-database)
- [CI/CD](#-cicd)
- [Machine Learning](#-machine-learning)
- [Monitoring](#-monitoring)
- [Configuration](#-configuration)
- [Testing](#-testing)
- [Engineering Tradeoffs](#-engineering-tradeoffs)
- [Rolling Rubrics](#-rolling-rubrics)
- [Performance & Operations](#-performance--operations)
- [Pentester — Ethical Hacking Reference](#-pentester--ethical-hacking-reference)

---

## 🏗️ Architecture Patterns

Production-grade design patterns for scalable systems.

### [`patterns/medallion_architecture.py`](patterns/medallion_architecture.py)
**Bronze → Silver → Gold** ETL pipeline implementation
- ✅ Data validation between tiers
- ✅ Schema enforcement with Pydantic
- ✅ Metadata tracking (ingestion time, source, quality score)
- ✅ Example: raw JSON → cleaned DataFrame → aggregated Gold table

**Apply to:** Data lakes, analytics pipelines, data warehousing

### [`patterns/service_patterns.py`](patterns/service_patterns.py)
Essential distributed system patterns
- ✅ **Circuit Breaker** (closed, open, half-open states)
- ✅ **Retry with exponential backoff** + jitter
- ✅ **Fan-out/Fan-in** parallel execution
- ✅ **Saga pattern** (orchestration-based)
- ✅ All async with httpx

**Apply to:** Microservices, service mesh, resilient APIs

### [`patterns/advanced_architecture.py`](patterns/advanced_architecture.py)
**Advanced patterns from security-data-fabric and Sportsbook-aggregation**
- ✅ **Redis distributed cache** with AES-256 encryption
- ✅ **MFA integration** (TOTP, QR codes, backup codes)
- ✅ **Service-to-service JWT** auth with scope-based authorization
- ✅ **Audit logging** with 7-year retention (SOC2/ISO27001)
- ✅ **Refresh token rotation** (single-use)
- ✅ **Autonomous engine pattern** (self-healing, scheduled tasks)
- ✅ **Prometheus/Grafana monitoring** (Golden Signals)
- ✅ Real-time data aggregation

**Apply to:** Distributed systems, secure data platforms, real-time processing

---

## 🔐 Security

Zero-trust security patterns. No hardcoded secrets.

### [`security/auth_framework.py`](security/auth_framework.py)
Complete authentication & authorization framework
- ✅ **JWT** token creation, validation, refresh
- ✅ **API key** generation and validation with rate limiting
- ✅ **RBAC** with hierarchical permission checking
- ✅ **Token rotation** mechanism
- ✅ **MFA** TOTP generation and verification
- ✅ All secrets from environment variables

**Apply to:** APIs, web apps, internal tools, admin panels

### [`security/input_validator.py`](security/input_validator.py)
**32+ attack pattern detection** with severity mapping
- ✅ SQL Injection (26 parameterized patterns)
- ✅ XSS (10 patterns + bleach sanitization)
- ✅ Path Traversal / Directory traversal
- ✅ Command Injection (shell, OS)
- ✅ LDAP Injection (RFC 4515 escaping)
- ✅ XML/XXE Injection
- ✅ SSRF patterns
- ✅ Header Injection / CRLF
- ✅ Template Injection (Jinja2, EL)
- ✅ Log Injection (structured JSON logging)
- ✅ Email Header Injection
- ✅ Unicode attacks
- ✅ Null byte injection
- ✅ Deserialization attacks
- ✅ Buffer overflow indicators
- ✅ Integer overflow/underflow
- ✅ Race conditions (TOCTOU)
- ✅ CSRF patterns
- ✅ Open redirect
- ✅ IDOR (Insecure Direct Object Reference)
- ✅ Mass assignment
- ✅ **Supply Chain Attack** (hash pinning, SCA)
- ✅ **Side-Channel Attack** (metadata stripping)
- ✅ **Business Logic Vulnerabilities** (property-based testing)
- ✅ **Build System Hijack** (ephemeral runners, signed commits, SLSA)

**Apply to:** User input validation, API endpoints, form processing, zero-trust security

### [`security/zero_day_shield.py`](security/zero_day_shield.py)
Defense-in-depth zero-day protection utilities
- ✅ **SecureDeserializer** - Whitelist-based deserialization
- ✅ **SecureHasher** - Timing attack protection (HMAC)
- ✅ **SecureTokenGenerator** - Cryptographic tokens
- ✅ **SecureValidator** - Enhanced ReDoS prevention with thread-based timeout
- ✅ **MetadataSanitizer** - Side-channel leak prevention
- ✅ **DefenseInDepthValidator** - Multi-layer validation

**Apply to:** Zero-trust architectures, defense-in-depth security, production systems

### [`security/ai_era_security.py`](security/ai_era_security.py) 🆕
**AI-Era Security Patterns (2026)** for AI-augmented applications
- ✅ **Pattern 28: Prompt Injection Detection** - Direct/indirect injection, system override, jailbreak detection
- ✅ **Pattern 29: AI Package Hallucination Protection** - Verify packages exist, detect typosquatting
- ✅ **Pattern 30: AI Agent Identity & Access** - Agent-specific OIDC, human-in-the-loop for high-regret actions
- ✅ **Enhanced ReDoS Protection** - Thread-based timeout that actually stops catastrophic backtracking
- ✅ **Instruction Hierarchy** - Separate system instructions from user input
- ✅ **Semantic Filtering** - Risk scoring for prompt injection attempts
- ✅ **Package Whitelist** - Validate against approved dependencies
- ✅ **Agent Rate Limiting** - Per-agent action limits
- ✅ **Audit Trail** - Complete logging of agent actions

**Apply to:** LLM applications, AI agents, autonomous systems, vibe coding environments

### [`security/circuit_breaker.py`](security/circuit_breaker.py)
Circuit breaker pattern for resilience
- ✅ **3 states** - Closed, Open, Half-Open
- ✅ **Automatic failure detection** and recovery
- ✅ **Configurable thresholds** - Failure rate, timeout
- ✅ **Thread-safe** implementation
- ✅ **Metrics tracking** - Success rate, failure rate
- ✅ **Circuit Breaker Registry** - Multi-service management
- ✅ **Decorator pattern** - Easy integration

**Apply to:** Microservices, external APIs, preventing cascading failures

### [`security/encryption.py`](security/encryption.py)
Cryptographic toolkit
- ✅ **AES-256-GCM** encryption/decryption
- ✅ **SHA-256/SHA-512** hashing
- ✅ **PBKDF2** password hashing with salt
- ✅ Secure random token generation
- ✅ Key derivation functions

**Apply to:** Data at rest encryption, password storage, token generation

---

## 📐 API Development

Production FastAPI templates with security, observability, and best practices.

### [`api/service_template.py`](api/service_template.py)
**Battle-tested FastAPI template**
- ✅ Request ID middleware (UUID per request)
- ✅ CORS configuration
- ✅ Security headers middleware (CSP, HSTS, X-Frame-Options)
- ✅ Health check endpoints (`/health`, `/ready`)
- ✅ Structured JSON error responses
- ✅ Request/response logging middleware
- ✅ Rate limiting middleware
- ✅ API versioning pattern
- ✅ OpenAPI/Swagger auto-documentation

**Apply to:** REST APIs, microservices, internal services, public APIs

---

### [`api/graphql_reference.py`](api/graphql_reference.py)
**Complete GraphQL reference — Strawberry + FastAPI**
- ✅ Schema design: types, inputs, unions, enums
- ✅ **DataLoader** — eliminates N+1 queries (one DB round-trip per batch)
- ✅ **Depth limiting** (MAX_DEPTH=7) + **complexity limiting** (MAX_COMPLEXITY=50)
- ✅ JWT authentication via context injection + field-level permission classes
- ✅ **Relay cursor-based pagination** (spec-compliant, prevents offset injection)
- ✅ **Subscriptions** over WebSocket (graphql-ws protocol)
- ✅ **Apollo Federation v2** subgraph pattern
- ✅ **Automatic Persisted Queries** (APQ) cache for bandwidth reduction + allow-listing
- ✅ Typed mutation result errors (never leak stack traces)
- ✅ Disable introspection in production

**Apply to:** BFF (Backend for Frontend), flexible data APIs, federated micro-service graphs

---

### [`api/grpc_reference.py`](api/grpc_reference.py)
**Complete gRPC reference — all four RPC patterns**
- ✅ **Unary, server-streaming, client-streaming, bidirectional streaming**
- ✅ Protobuf schema reference (embedded)
- ✅ **JWT interceptor** — validates every non-exempt method
- ✅ **mTLS channel factory** — mutual certificate authentication
- ✅ **Retry policy** with exponential backoff (UNAVAILABLE / RESOURCE_EXHAUSTED)
- ✅ **Deadline propagation** and keepalive pings
- ✅ Health checking (`grpc.health.v1`) — Kubernetes liveness/readiness compatible
- ✅ Reflection service (dev tooling — grpcurl, Postman)
- ✅ ThreadPool size cap (prevents thread-exhaustion DoS)

**Apply to:** High-throughput internal microservice communication, streaming pipelines, polyglot service meshes

---

### [`api/websocket_reference.py`](api/websocket_reference.py)
**WebSocket + Server-Sent Events (SSE) + Webhooks**
- ✅ **WebSocket room manager** — fan-out to all subscribers, async broadcast
- ✅ **JWT authentication on upgrade** (before `ws.accept()`)
- ✅ **Per-connection rate limiting** (token bucket, 30 req burst / 10 req/s)
- ✅ **Ping/pong keepalive** — detects stale connections on load balancers
- ✅ **Redis Pub/Sub bridge** — horizontal scaling across multiple server nodes
- ✅ **SSE endpoint** — unidirectional server push (notifications, feeds)
- ✅ **HMAC webhook verification** — compatible with GitHub, Stripe, Shopify
- ✅ **AsyncAPI 2.x schema** reference (embedded)
- ✅ Protocol choice guide: REST vs GraphQL vs gRPC vs WebSocket vs SSE

**Apply to:** Chat, collaborative editing, live dashboards, real-time notifications, webhooks

---

### [`api/cloudflare_platform.py`](api/cloudflare_platform.py)
**Cloudflare S-Tier Platform Reference**
- ✅ **CORS policy builder** — strict, public API, internal modes with `Vary: Origin`
- ✅ **Full security header suite** — CSP, HSTS (preload), Permissions-Policy, COEP/COOP
- ✅ **Geo-IP rate limiter** — per-country risk tiers (blocked/elevated/normal/trusted)
- ✅ **Workers templates** (JS) — CORS + security headers + geo-IP blocking + KV rate limiting
- ✅ **Durable Objects** template — strongly consistent per-IP rate limiting actor
- ✅ **Turnstile** bot-protection integration
- ✅ **Platform feature matrix** — Workers, D1, KV, R2, Queues, Durable Objects, Access
- ✅ **Cache-Control recipes** — immutable assets, API CDN cache, private session, no-cache
- ✅ Concurrency model guide (isolates vs. DO actors)

**Apply to:** API gateways, CDN-accelerated apps, edge auth middleware, zero-trust access

---

## ⚡ Performance

Optimization patterns with cost models and complexity analysis.

### [`performance/caching.py`](performance/caching.py)
**L1 → L2 → L3** multi-tier caching strategy
- ✅ **L1:** In-memory cache (TTL-based dict, sub-ms latency)
- ✅ **L2:** Distributed Redis cache (1-5ms latency)
- ✅ **L3:** Database fallback (10-50ms latency)
- ✅ Cache invalidation strategies
- ✅ Consistent hashing for key generation
- ✅ **Cost model:** `$X/month` savings estimates with commentary

**Apply to:** High-traffic APIs, read-heavy workloads, cost optimization

### [`performance/complexity_cheatsheet.py`](performance/complexity_cheatsheet.py)
**Big-O reference** for everything
- ✅ Data structures (array, linked list, hash map, BST, heap, trie)
- ✅ Sorting algorithms (quick, merge, heap, radix, tim)
- ✅ Searching algorithms (binary, linear, BFS, DFS)
- ✅ Database operations (SELECT, JOIN, INDEX scan, full table scan)
- ✅ ML algorithms (training vs inference complexity)
- ✅ Space complexity included
- ✅ "When to use" notes for each

**Apply to:** Algorithm selection, performance interviews, capacity planning

### [`performance/async_patterns.py`](performance/async_patterns.py)
**Async/await** best practices
- ✅ Semaphore-based concurrency limiting
- ✅ Batch processing with configurable batch sizes
- ✅ Exponential backoff with jitter
- ✅ Async context managers
- ✅ `asyncio.gather()` with error handling
- ✅ Rate-limited async execution

**Apply to:** I/O-bound operations, API clients, data pipelines

---

## 🗄️ Database

Async SQLAlchemy patterns with indexing strategies and semantic search.

### [`database/connection.py`](database/connection.py)
**Production database connection management**
- ✅ Async SQLAlchemy engine with connection pooling
- ✅ Session factory with context manager
- ✅ Transaction management (commit/rollback)
- ✅ Connection health checks
- ✅ Pool size configuration from env vars

**Apply to:** APIs, background workers, data pipelines

### [`database/model_patterns.py`](database/model_patterns.py)
**SQLAlchemy best practices**
- ✅ UUID primary keys (not auto-increment)
- ✅ Audit mixin (created_at, updated_at, created_by)
- ✅ Soft delete mixin
- ✅ Composite indexing examples
- ✅ Relationship patterns (one-to-many, many-to-many)
- ✅ Example models: User, Role, AuditLog

**Apply to:** ORM design, data modeling, audit trails

### [`database/vector_search.py`](database/vector_search.py)
**pgvector semantic search**
- ✅ Embedding storage and retrieval
- ✅ Cosine similarity search
- ✅ Embedding cache layer (avoid re-computing)
- ✅ Batch embedding insertion
- ✅ Semantic search function with filtering

**Apply to:** RAG systems, semantic search, recommendation engines

---

## 🔄 CI/CD

GitHub Actions workflows and infrastructure-as-code templates.

### [`cicd/test-pipeline.yml`](cicd/test-pipeline.yml)
**GitHub Actions CI template**
- ✅ Matrix testing (Python 3.10, 3.11, 3.12)
- ✅ Linting (ruff/flake8)
- ✅ Type checking (mypy)
- ✅ Test execution (pytest with coverage)
- ✅ Coverage reporting
- ✅ Artifact upload

### [`cicd/security-scan.yml`](cicd/security-scan.yml)
**Weekly security scanning**
- ✅ CodeQL static analysis
- ✅ Trivy container scanning
- ✅ Dependency vulnerability check (pip-audit/safety)
- ✅ SARIF upload to GitHub Security tab
- ✅ Manual trigger option

### [`.github/workflows/security-scan-universal.yml`](.github/workflows/security-scan-universal.yml)
**🔐 Plug-and-play universal security scanning** (NEW)
- ✅ **CodeQL** - Multi-language static analysis
- ✅ **Trivy** - Container vulnerability scanning
- ✅ **pip-audit** - Python CVE scanning
- ✅ **Safety** - Known vulnerabilities
- ✅ **Bandit** - Python security issues
- ✅ **Gitleaks** - Secrets detection
- ✅ **SBOM generation** - Software Bill of Materials
- ✅ **Node.js audit** - npm/yarn vulnerabilities (if applicable)
- ✅ **Works with**: Python, Node.js, Go, Java, Ruby projects
- ✅ **Copy-paste ready** - No configuration needed

**Apply to:** Any project requiring comprehensive security scanning

### [`cicd/dependabot.yml`](cicd/dependabot.yml)
**Automated dependency updates**
- ✅ pip ecosystem updates (weekly)
- ✅ GitHub Actions updates (weekly)
- ✅ Docker updates (monthly)
- ✅ Auto-label PRs
- ✅ Commit message prefix configuration

### [`cicd/Dockerfile`](cicd/Dockerfile)
**Multi-stage production Docker build**
- ✅ Python 3.11-slim base
- ✅ Non-root user
- ✅ Health check
- ✅ Proper layer caching
- ✅ Security best practices (no cache, minimal image)

### Docker Security Standards

All projects in the ecosystem follow these Docker security principles:
- ✅ **Minimal base images** — `python:3.x-slim-bookworm` (not `python:3.x`)
- ✅ **Multi-stage builds** — Builder stage for deps, runtime stage for execution
- ✅ **Non-root user** — `appuser` with no shell, minimal permissions
- ✅ **HEALTHCHECK** — Every Dockerfile includes a health check instruction
- ✅ **No privileged mode** — Use specific `cap_add` and device mappings instead
- ✅ **Pinned versions** — Specific image tags, not `:latest`
- ✅ **`.dockerignore`** — Prevents secrets and unnecessary files from entering images

See `cicd/Dockerfile` for the reference implementation.

### [`cicd/terraform_module_template.tf`](cicd/terraform_module_template.tf)
**Universal IaC template**
- ✅ Variable definitions with validation
- ✅ Provider configuration
- ✅ Resource group / project setup
- ✅ Output definitions
- ✅ Tags/labels pattern
- ✅ Comments explaining customization points

**Apply to:** CI/CD pipelines, security automation, infrastructure provisioning

---

### [`security/iac/cloudflare_terraform.tf`](security/iac/cloudflare_terraform.tf)
**Cloudflare IaaC — Terraform (S-Tier configuration)**
- ✅ **Zone settings** — TLS 1.2+, TLS 1.3+0-RTT, HSTS, brotli, HTTP/2+3, Early Hints
- ✅ **WAF Managed Rules** — Cloudflare Managed Ruleset + OWASP CRS (block mode)
- ✅ **WAF Custom Rules** — geo-IP block, threat score, bot filtering, scanner UA, path traversal
- ✅ **Geo-IP Rate Limiting** — OFAC countries blocked, elevated-risk countries 30 req/min, auth endpoints 10 req/min
- ✅ **Security Headers Transform Rule** — CSP, HSTS, Permissions-Policy, COEP/COOP injected on every response
- ✅ **Cache Rules** — immutable assets, CDN short-cache for APIs, no-store for auth
- ✅ **Workers Route** binding
- ✅ **R2 Bucket** — zero-egress object storage with lifecycle protection
- ✅ **Cloudflare Access** — Zero Trust SSO + MFA for admin panel
- ✅ **Cloudflare Tunnel** — origin never exposed to public internet
- ✅ **DNS records** — A, CNAME, SPF, DMARC (p=reject)

**Apply to:** Any Cloudflare-fronted service — APIs, web apps, SaaS, internal tools

---

## 🧠 Machine Learning

Anomaly detection and time-series forecasting for capacity planning.

### [`ml/anomaly_detector.py`](ml/anomaly_detector.py)
**Z-score + Isolation Forest** anomaly detection with incident prediction
- ✅ Z-score baseline analysis
- ✅ Isolation Forest (scikit-learn)
- ✅ **Configurable thresholds** (1.5σ, 3.0σ, 4.5σ)
- ✅ **Trajectory prediction** with confidence scoring
- ✅ **Alert fatigue prevention** (80%+ confidence threshold)
- ✅ Anomaly scoring with 4-level classification
- ✅ Batch and streaming detection modes
- ✅ **SDF Gold layer integration bridge**
- ✅ Human-readable explanations

**Apply to:** Fraud detection, incident prediction, system monitoring, outlier detection

**Production example:** [finops-cost-control-as-code](https://github.com/EPdacoder05/finops-cost-control-as-code) — deployed AWS system using anomaly detection patterns from this module

### [`ml/forecaster.py`](ml/forecaster.py)
**Random Forest time-series forecasting**
- ✅ Feature engineering (lag features, rolling stats)
- ✅ Train/predict pipeline
- ✅ Confidence intervals
- ✅ Model persistence (joblib)
- ✅ Capacity planning: "When will resource X hit limit?"

**Apply to:** Capacity planning, demand forecasting, resource scaling

---

## 📊 Monitoring

Structured logging, metrics, and SLA tracking.

### [`monitoring/observability.py`](monitoring/observability.py)
**Production observability toolkit**
- ✅ Structured JSON logging with correlation IDs
- ✅ Metrics collection (counters, gauges, histograms)
- ✅ SLA tracking (uptime, latency percentiles)
- ✅ Log levels configuration
- ✅ Request tracing context
- ✅ Alert threshold definitions

**Apply to:** Production debugging, incident response, SLA monitoring

---

## 🔧 Configuration

Environment-aware configuration management with secret generation.

### [`config/settings.py`](config/settings.py)
**Pydantic BaseSettings** with env file support
- ✅ Environment-aware (dev/staging/prod)
- ✅ Secret generation utilities
- ✅ Database URL construction
- ✅ Redis URL construction
- ✅ API key validation
- ✅ **Never hardcode secrets** — all from environment

**Apply to:** 12-factor apps, config management, secret rotation

---

## 🧪 Testing

Pytest fixtures, factories, and async testing patterns.

### [`testing/test_framework.py`](testing/test_framework.py)
**Pytest best practices**
- ✅ Fixture patterns (session, function, module scope)
- ✅ Factory pattern for test data generation
- ✅ Async test helpers
- ✅ Mock/patch patterns for external services
- ✅ Database test fixtures (transaction rollback)
- ✅ API client test fixtures
- ✅ Coverage configuration example

**Apply to:** Unit tests, integration tests, API tests

---

## ⚖️ Engineering Tradeoffs

### [`TRADEOFFS.md`](TRADEOFFS.md)
**The Crown Jewel:** 50+ real-world engineering tradeoffs
- **Architecture:** Monolith vs Microservices, REST vs GraphQL vs gRPC, Sync vs Async, SQL vs NoSQL, Event Sourcing vs CRUD, Serverless vs Containers
- **Security:** JWT vs Session tokens, API keys vs OAuth, Encryption at rest vs in transit, WAF vs Application-level validation
- **Performance:** Cache vs Fresh data, Horizontal vs Vertical scaling, CDN vs Origin, Connection pooling sizes
- **Database:** Normalization vs Denormalization, Read replicas vs Sharding, Indexes vs Write speed, ACID vs BASE
- **DevOps:** Blue-green vs Canary vs Rolling deploys, Terraform vs Pulumi, GitHub Actions vs Jenkins
- **ML:** Real-time vs Batch inference, Accuracy vs Latency, Simple models vs Deep learning

Each tradeoff includes: Description, When to choose A, When to choose B, Real-world example, Cost implications

---

## 📈 Rolling Rubrics

### [`rubrics/MASTER_RUBRIC.md`](rubrics/MASTER_RUBRIC.md)
**Universal rolling scoring template**
- ✅ Fixed dimensions: correctness, reliability, security, scalability, operability, cost, maintainability, clarity
- ✅ Required rolling fields: what changed, why, impact, next experiment, deprecation notes
- ✅ Required deltas: rubric score delta, risk delta, production readiness delta
- ✅ Versioned governance for compounding improvements

### [`rubrics/ROLLING_UPDATE_LOG.md`](rubrics/ROLLING_UPDATE_LOG.md)
**Standardized rolling update log**
- ✅ Change entry template with required rubric and risk deltas
- ✅ Validation and known limitation capture
- ✅ Source/signal notes for dynamic updates

### [`rubrics/SECURITY/OIDC_OAUTH2_QUICKREF.md`](rubrics/SECURITY/OIDC_OAUTH2_QUICKREF.md)
**P0 identity/auth quick reference**
- ✅ Canonical OIDC vs OAuth2 clarification
- ✅ Decision rule for OAuth2-only vs OIDC+OAuth2
- ✅ Minimum checks: PKCE, `iss`, `aud`, token/key rotation, least-privilege scopes, signature verification

### [`rubrics/CHECKLISTS/`](rubrics/CHECKLISTS/)
**Domain checklists wired to master rubric version**
- ✅ SWE checklist
- ✅ DevOps checklist
- ✅ CS fundamentals checklist
- ✅ System design checklist

---

## 📊 Performance & Operations

### [`PERFORMANCE_BENCHMARKS.md`](PERFORMANCE_BENCHMARKS.md)
**Consolidated performance metrics across all projects**
- **security-data-fabric**: Cache latency (<1ms), vector search (<100ms), ML forecast (<500ms)
- **incident-predictor-ml**: Prediction cycle (<2s), anomaly detection (<200ms)
- **NullPointVector**: Input validation (<10ms), circuit breaker (<1ms)
- **Sportsbook-aggregation**: Scraper cycle (<30s), real-time aggregation (<200ms)
- **Load testing results**: 1,247 req/s, P95 latency 18ms
- **Cost savings**: $2,650/month from caching (88% reduction)
- **SLA compliance**: 99.94% uptime, P95 <100ms

### [`COST_ANALYSIS.md`](COST_ANALYSIS.md)
**FinOps template for cost optimization**
- **Baseline cost calculation**: Infrastructure, external APIs, hidden costs
- **Optimization phases**: Quick wins, architectural changes, advanced optimization
- **ROI methodology**: Payback period, decision matrix
- **Cost breakdown**: By service, team, category
- **Budget planning**: Quarterly projections, annual forecasts
- **Example savings**: $499/month (28% reduction)
- **Unit economics**: Cost per request, per user, per prediction

### [`PRODUCTION_READINESS.md`](PRODUCTION_READINESS.md)
**Universal production deployment checklist**
- **Code & Dependencies**: Version control, dependency pinning, code quality
- **Security**: 32 attack patterns mitigated, secrets management, compliance
- **Testing**: 95%+ coverage, property-based testing, chaos engineering
- **CI/CD**: Blue-green deployment, automatic rollback, zero-downtime
- **Observability**: Golden Signals, error budget tracking, runbooks
- **Database**: Backups, PITR, migration strategy
- **Performance**: Load testing, caching, auto-scaling
- **Documentation**: Architecture diagrams, runbooks, API docs
- **Cost Management**: Right-sizing, reserved instances, budget alerts
- **Team Readiness**: On-call rotation, incident response, knowledge transfer

### [`INTEGRATION_MAP.md`](INTEGRATION_MAP.md)
**Cross-project integration architecture**
- **Visual ecosystem map**: Data flow between all projects
- **Integration patterns**: REST API, Kafka, JWT auth
- **Authentication chain**: Service-to-service JWT with scopes
- **Monitoring integration**: Prometheus/Grafana, alert routing
- **Disaster recovery**: RTO 1 hour, RPO 15 minutes
- **Real-world examples**: End-to-end latency 11.5s
- **Best practices**: Circuit breakers, retry logic, input validation

---

## 🎯 How to Use This Repository

### Copy-Paste Workflow
```bash
# Clone the repo
git clone https://github.com/EPdacoder05/System-Design-Engineering-Universal-Reference.git
cd System-Design-Engineering-Universal-Reference

# Copy what you need to your project
cp security/auth_framework.py ../my-project/
cp api/service_template.py ../my-project/
cp cicd/test-pipeline.yml ../my-project/.github/workflows/

# Install dependencies for modules you're using
pip install fastapi pydantic python-jose
```

### Modular Design
- **Every file works standalone** — no internal dependencies
- Take only what you need — no bloat
- Customize for your use case — clear comments explain where to edit

### Best Practices
- **No PII** — zero personal data, company names, sector-specific info
- **Production-grade** — patterns used in scaled systems, not toy examples
- **Well-documented** — docstrings explain "Apply to: [use case]"
- **Platform-agnostic** — works with AWS, Azure, GCP, any cloud

---

## 📦 Installation

```bash
# Install all dependencies (optional)
pip install -r requirements.txt

# Or install selectively based on what you're using
pip install fastapi sqlalchemy redis scikit-learn
```

## 🔒 Security

### Cybersecurity Guardrails
For a defense-first cybersecurity playbook, see [`CYBERSEC_GUARDRAILS.md`](CYBERSEC_GUARDRAILS.md)
- Includes an authorized-learning reference to `LuanMattos/ethical-hacking` with strict legal-use boundaries

### Docker Security
For production-ready Docker hardening patterns, see [`docker/DOCKER_SECURITY.md`](docker/DOCKER_SECURITY.md)

- All secrets managed via environment variables
- **32+ attack pattern detection** included (SQL injection, XSS, supply chain, build system hijack, etc.)
- Zero-day shield utilities (secure deserialization, timing attack protection)
- Circuit breaker for resilience
- Universal security scanning workflow (CodeQL, Trivy, Gitleaks, SBOM)
- Regular dependency updates via Dependabot
- Production readiness checklist with SOC2/ISO27001 controls

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

## 🤝 Contributing

This is a reference library — customize it for your needs. No contributions needed, but feel free to fork and adapt.

## 🌟 Key Principles

1. **Portable** — Works anywhere, any company, any project
2. **Modular** — Take only what you need
3. **Production-Ready** — Battle-tested patterns
4. **Copy-Paste Friendly** — No internal dependencies
5. **Well-Documented** — Clear use cases and examples

## 🛣️ Roadmap
- [ ] Terraform Compliance Scanner (standalone repo, shares TF parsing from cicd/)
- [ ] Secrets Rotation Engine (extends security/ patterns)  
- [ ] Cost Anomaly Detector (production deployment of ml/anomaly_detector.py)
- [ ] Docker security template (hardened Dockerfile patterns for all projects)

---

**Built for engineers, by engineers. Ship faster. Ship better.**

---

## 🔐 Pentester — Ethical Hacking Reference

> ⚠️ **LEGAL NOTICE**: All offensive-security tools are for **authorised use in
> isolated lab environments only** (VMs you own, test networks, CTF challenges).
> See [`pentester/README.md`](pentester/README.md) for full legal notice.

A reference toolkit for cybersecurity professionals, students, and red/blue/purple
teamers — covering the offensive mindset alongside the defensive patterns in
`security/`.

### [`pentester/`](pentester/)
**Red-team / ethical hacking reference — lab use only**
- ✅ **Advanced Port Scanner** — `advancedscanner.py` (TCP/SYN/ICMP/UDP/FIN, presets, Rich UI)
- ✅ **Backdoor & Listener** — reverse shell pattern for detection-rule building
- ✅ **ARP Spoofing** — MITM via gratuitous ARP (scapy, lab VMs)
- ✅ **Hasher** — MD5, SHA-1, SHA-224, SHA-256, SHA-512 in one tool
- ✅ **CVE Notes** — vsftpd 2.3.4 (CVE-2011-2523) analysis + Metasploit walkthrough
- ✅ **Wi-Fi Notes** — WPA handshake capture guide (aircrack-ng)

### [`security/advanced_scanner.py`](security/advanced_scanner.py)
**Multi-method port scanner — 5 scan methods, 11 presets, Rich UI**

```bash
python advancedscanner.py -H 192.168.1.1 -x web          # web preset
python advancedscanner.py -H 192.168.1.1 -p 22,80,443 -M syn  # SYN stealth
python advancedscanner.py -i                               # interactive mode
python advancedscanner.py --list                           # show all presets
```

| Method | Flag | Privilege | Notes |
|--------|------|-----------|-------|
| TCP Connect | `-M tcp` | None | Default; full 3-way handshake |
| SYN Stealth | `-M syn` | Admin/Root | Half-open; auto-fallback to TCP |
| ICMP Ping | `-M icmp` | Admin/Root | Host discovery |
| UDP Scan | `-M udp` | None | Connectionless |
| FIN/NULL/Xmas | `-M fin` | Admin/Root | Stealth flags; auto-fallback |

**Presets:** `web` · `ssh` · `database` · `mail` · `dns` · `directory` ·
`monitoring` · `ntp` · `vpn` · `common` · `all`

**Apply to:** Authorised network audits, CTF challenges, service discovery,
lab infrastructure assessment, building detection signatures for blue teams.
