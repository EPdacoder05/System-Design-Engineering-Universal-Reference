# ⚖️ Engineering Tradeoffs: The Crown Jewel

**50+ real-world engineering tradeoffs** you'll face in production systems. Each decision has consequences. This guide helps you choose wisely.

## Table of Contents
- [Architecture](#architecture)
- [Security](#security)
- [Performance](#performance)
- [Database](#database)
- [DevOps](#devops)
- [Machine Learning](#machine-learning)

---

## Architecture

### 1. Monolith vs Microservices

**Description:** Single codebase vs distributed services.

**Choose Monolith when:**
- Team <10 engineers
- MVP or early-stage product
- Domain not well understood yet
- Deployment simplicity matters
- Debugging needs to be straightforward

**Choose Microservices when:**
- Team >20 engineers with clear ownership boundaries
- Need independent scaling (e.g., payments service needs 10x more capacity than user service)
- Different tech stacks required per service
- Deploy frequency >10x/day
- Failure isolation critical (one service down ≠ entire system down)

**Real-world example:** Netflix started as monolith, moved to microservices at scale. Resulted in 1000+ services but enabled global scaling and 4-5 deployments per day.

**Cost implications:**
- Monolith: $5K-20K/month infrastructure, 2-3 DevOps engineers
- Microservices: $50K-500K/month infrastructure, 5-10 DevOps/SRE engineers, service mesh costs

---

### 2. REST vs GraphQL vs gRPC

**Description:** API communication protocol.

**Choose REST when:**
- Public API (widest client compatibility)
- CRUD operations dominate
- Caching via HTTP is valuable
- Team familiar with HTTP/JSON

**Choose GraphQL when:**
- Frontend needs flexible data fetching (avoid over-fetching)
- Multiple clients with different data needs (mobile, web, desktop)
- Rapid iteration on data requirements
- N+1 query problem solved via DataLoader

**Choose gRPC when:**
- Internal microservice communication
- Performance critical (10x faster than REST in some cases)
- Bidirectional streaming needed
- Strong typing via protobuf required

**Real-world example:** GitHub uses REST for public API, GraphQL for github.com, gRPC for internal services.

**Cost implications:**
- REST: Standard, no extra costs
- GraphQL: +20% backend complexity, requires caching layer ($500-5K/month)
- gRPC: +30% tooling complexity, service mesh recommended ($2K-10K/month)

---

### 3. Synchronous vs Asynchronous Processing

**Description:** Block and wait vs fire-and-forget.

**Choose Synchronous when:**
- User needs immediate response
- Operation <100ms (database reads, cache hits)
- Simpler error handling needed
- Consistency critical (e.g., payment authorization)

**Choose Asynchronous when:**
- Operation >1 second (email sending, video processing, report generation)
- Can retry on failure
- Peak load smoothing needed (queue absorbs spikes)
- User doesn't need to wait

**Real-world example:** Stripe payment processing is synchronous (authorization), but webhook delivery is asynchronous (eventual consistency acceptable).

**Cost implications:**
- Synchronous: Simple, but ties up resources during wait
- Asynchronous: +$500-5K/month for message queue (SQS, RabbitMQ, Kafka), +40% code complexity

---

### 4. SQL vs NoSQL

**Description:** Relational vs document/key-value storage.

**Choose SQL when:**
- Complex queries with JOINs needed
- ACID transactions required
- Data highly relational (users, orders, products)
- Schema changes infrequent
- Ad-hoc analytics queries

**Choose NoSQL when:**
- Massive scale (>10TB, >100K writes/sec)
- Schema evolves rapidly
- Document-oriented data (JSON blobs)
- Horizontal scaling critical
- Key-value access pattern dominates

**Real-world example:** Uber uses PostgreSQL for trip records (transactions), Cassandra for time-series data (GPS coordinates every 4 seconds).

**Cost implications:**
- SQL: $100-5K/month (RDS/Cloud SQL), read replicas add 2x cost
- NoSQL: $500-50K/month (DynamoDB, Cosmos DB), can scale infinitely but pay per throughput

---

### 5. Event Sourcing vs CRUD

**Description:** Store events vs store current state.

**Choose CRUD when:**
- Simple business logic
- History not critical
- Team new to event sourcing
- Read-heavy workload

**Choose Event Sourcing when:**
- Audit trail legally required
- Temporal queries needed ("what was inventory on March 15?")
- Undo/replay functionality required
- Event-driven architecture already in place

**Real-world example:** Banking systems use event sourcing (every transaction recorded), e-commerce carts use CRUD (only current state matters).

**Cost implications:**
- CRUD: Simple, standard database costs
- Event Sourcing: +50% storage costs (store all events), +60% complexity, requires event store ($1K-10K/month)

---

### 6. Serverless vs Containers

**Description:** FaaS (AWS Lambda) vs orchestrated containers (Kubernetes).

**Choose Serverless when:**
- Sporadic traffic (idle time >70%)
- Startup time <10 seconds acceptable
- Stateless workloads
- No long-running processes
- Want zero infrastructure management

**Choose Containers when:**
- Predictable traffic (always >1 instance needed)
- Cold start time unacceptable
- Stateful workloads or persistent connections
- Custom runtime/dependencies
- Vendor lock-in concerns

**Real-world example:** News sites use serverless for breaking news spikes (idle 80% of time), streaming services use containers (always serving traffic).

**Cost implications:**
- Serverless: $0 at idle, $50-500/month for 1M requests, $5K+ at high scale
- Containers: $200-2K/month minimum (always-on), scales linearly, Kubernetes overhead $5K-20K/month

---

### 7. Monorepo vs Polyrepo

**Description:** One repository vs multiple repositories.

**Choose Monorepo when:**
- Shared code libraries common
- Atomic cross-project changes needed
- Unified CI/CD pipeline
- Team <100 engineers

**Choose Polyrepo when:**
- Independent release cycles critical
- Different access controls per project
- Clear service boundaries
- Team >100 engineers with separate orgs

**Real-world example:** Google uses monorepo (2B+ lines of code), Amazon uses polyrepo (thousands of independent repos).

**Cost implications:**
- Monorepo: +20% CI/CD costs (test all on every commit), tooling $5K-20K/year (Bazel, Nx)
- Polyrepo: +30% overhead for dependency management, code duplication increases

---

## Security

### 8. JWT vs Session Tokens

**Description:** Stateless vs stateful authentication.

**Choose JWT when:**
- Microservices architecture (no shared session store)
- Mobile apps (offline validation)
- Third-party integrations (OAuth flows)
- Horizontal scaling without sticky sessions

**Choose Session Tokens when:**
- Immediate revocation required (logout, ban user)
- Token size matters (JWTs can be 1KB+, sessions 16 bytes)
- Single monolithic application
- No cross-domain access needed

**Real-world example:** Auth0 uses JWT for APIs, session cookies for web apps.

**Cost implications:**
- JWT: No session store needed, but can't revoke easily
- Session Tokens: Redis/Memcached session store $50-500/month, adds latency

---

### 9. API Keys vs OAuth 2.0

**Description:** Simple keys vs delegated authorization.

**Choose API Keys when:**
- Server-to-server communication
- No user context needed
- Simple authentication sufficient
- Internal APIs or trusted partners

**Choose OAuth 2.0 when:**
- User authorization required ("Act on my behalf")
- Third-party integrations
- Granular scopes needed (read-only, write, admin)
- Token refresh needed

**Real-world example:** Stripe uses API keys for merchant API, OAuth for Connect (third-party platforms).

**Cost implications:**
- API Keys: Simple, no extra infrastructure
- OAuth: Identity provider $500-5K/month (Auth0, Okta), +50% complexity

---

### 10. Encryption at Rest vs In Transit

**Description:** Protect stored data vs data in motion.

**Choose Encryption at Rest when:**
- Compliance requires it (HIPAA, PCI-DSS)
- Data breach risk high (database theft)
- Physical security concerns (stolen drives)

**Choose Encryption in Transit when:**
- Network eavesdropping risk (public internet)
- Man-in-the-middle attacks possible
- Always use HTTPS/TLS for production

**Choose BOTH when:** Handling sensitive data (PII, financial, health)

**Real-world example:** Healthcare apps encrypt at rest (HIPAA) and in transit (TLS 1.3).

**Cost implications:**
- At Rest: 5-10% performance overhead, KMS costs $1-5/month per key
- In Transit: 2-5% CPU overhead for TLS, certificate management $0-100/year

---

### 11. WAF vs Application-Level Validation

**Description:** Network-level firewall vs code-level validation.

**Choose WAF when:**
- OWASP Top 10 protection needed
- DDoS protection critical
- Zero-day exploit mitigation
- Compliance requires it

**Choose Application-Level Validation when:**
- Business logic validation (e.g., "age must be 18+")
- Complex input rules
- Custom attack patterns specific to your app

**Choose BOTH for:** Defense in depth (WAF blocks 99%, app validation catches edge cases)

**Real-world example:** Banks use WAF (Cloudflare, AWS WAF) + input validation in code.

**Cost implications:**
- WAF: $20-200/month (basic), $2K-20K/month (advanced with rate limiting)
- App Validation: Built into code, no extra cost but +30% dev time

---

## Performance

### 12. Cache vs Fresh Data

**Description:** Serve stale data fast vs always fetch latest.

**Choose Cache when:**
- Data changes infrequently (<1x/minute)
- Read:Write ratio >10:1
- Latency critical (<50ms)
- Database load high

**Choose Fresh Data when:**
- Real-time requirements (stock prices, live sports)
- Consistency critical (inventory counts, payment status)
- Data changes frequently (>1x/second)

**Real-world example:** E-commerce product pages cached (15-60 second TTL), checkout inventory is fresh (real-time).

**Cost implications:**
- Cache: Redis/Memcached $50-500/month, 10x database cost reduction, 10x latency improvement
- Fresh Data: Higher database costs, slower response times, but always accurate

---

### 13. Horizontal vs Vertical Scaling

**Description:** Add more machines vs bigger machines.

**Choose Horizontal Scaling when:**
- Stateless services (web servers, APIs)
- Linear cost model acceptable
- High availability critical (N+1 redundancy)
- Cloud-native architecture

**Choose Vertical Scaling when:**
- Stateful services (databases, caches)
- Software licensing per-instance
- Coordination overhead expensive
- Quick fix needed (double CPU)

**Real-world example:** Web tier: horizontal (50x 2-core instances), Database: vertical (1x 64-core instance + read replicas).

**Cost implications:**
- Horizontal: $100/month x N instances, load balancer $20-50/month
- Vertical: $200-2K/month per instance, diminishing returns at high end (128 cores = 10x cost, 5x performance)

---

### 14. CDN vs Origin Server

**Description:** Edge caching vs serve from central location.

**Choose CDN when:**
- Static assets (images, CSS, JS)
- Global audience
- Latency <100ms critical
- DDoS protection needed

**Choose Origin Server when:**
- Dynamic content (personalized pages)
- Small geographic region
- Cache invalidation complex
- Debugging simplicity valued

**Real-world example:** Netflix uses CDN for video streaming (90% of traffic), origin for API calls (10%).

**Cost implications:**
- CDN: $0.02-0.08 per GB transfer, 5-10x faster, $100-10K/month at scale
- Origin: Standard egress $0.08-0.12 per GB, slower but simpler

---

### 15. Connection Pooling Size

**Description:** How many database connections to maintain.

**Choose Small Pool (5-10) when:**
- Low traffic (<100 req/sec)
- Database cost-sensitive
- Simple queries (<10ms)

**Choose Large Pool (50-100) when:**
- High traffic (>1000 req/sec)
- Queries slow (>100ms)
- Many concurrent users

**Formula:** `pool_size = (core_count * 2) + effective_spindle_count`

**Real-world example:** PgBouncer with pool of 20 can serve 1000 clients.

**Cost implications:**
- Small Pool: Fewer connections = lower DB costs, but risk of exhaustion
- Large Pool: More connections = higher DB license costs ($100-500/month per 10 connections)

---

### 16. Eager Loading vs Lazy Loading

**Description:** Load all data upfront vs on-demand.

**Choose Eager Loading when:**
- Data needed 100% of the time
- Avoid N+1 query problem
- Predictable access patterns

**Choose Lazy Loading when:**
- Data rarely needed (<20%)
- Memory constrained
- Large objects (images, videos)

**Real-world example:** Blog post with comments: eager load (always shown), author profile image: lazy load (not always needed).

**Cost implications:**
- Eager: +50% query time upfront, avoids 10-100x queries later
- Lazy: Slower on demand, but saves bandwidth if unused

---

## Database

### 17. Normalization vs Denormalization

**Description:** Split data vs duplicate data.

**Choose Normalization (3NF) when:**
- Write consistency critical
- Data changes frequently
- Storage costs high
- Strong ACID guarantees needed

**Choose Denormalization when:**
- Read performance critical (JOIN-free queries)
- Data rarely changes
- Analytics/reporting workload
- NoSQL database

**Real-world example:** User profiles normalized (avoid update anomalies), analytics tables denormalized (fast aggregation).

**Cost implications:**
- Normalization: Slower reads (JOIN overhead 10-100ms), smaller storage (50% less)
- Denormalization: 10x faster reads, 2-3x storage costs, eventual consistency risk

---

### 18. Read Replicas vs Sharding

**Description:** Copy entire database vs split data across databases.

**Choose Read Replicas when:**
- Read:Write ratio >10:1
- Data fits in single database (<1TB)
- Eventual consistency acceptable (replica lag 100-500ms)

**Choose Sharding when:**
- Data >10TB
- Write throughput needs to scale
- Single database max capacity reached
- Geographic partitioning (EU data stays in EU)

**Real-world example:** Social media: read replicas for feeds (read-heavy), sharding for user data (billions of users).

**Cost implications:**
- Read Replicas: 2-5x database cost (1 primary + 1-4 replicas), simple setup
- Sharding: 10x complexity, application-level routing, but infinite scale

---

### 19. Indexes vs Write Speed

**Description:** Fast reads vs fast writes.

**Add Index when:**
- Query scans >10% of table
- Slow query log shows sequential scans
- Column used in WHERE/JOIN/ORDER BY
- Read:Write ratio >10:1

**Skip Index when:**
- Write-heavy table (logs, time-series)
- Small table (<10K rows)
- Column has low cardinality (gender, boolean)

**Real-world example:** User table: index on email (login query), no index on "last_login" (only admin reports use it).

**Cost implications:**
- Index: +10-50ms per INSERT/UPDATE/DELETE, +20% storage, 10-1000x faster reads
- No Index: Fast writes, slow reads (full table scan 100ms-10s on large tables)

---

### 20. ACID vs BASE

**Description:** Strong consistency vs eventual consistency.

**Choose ACID when:**
- Financial transactions
- Inventory management
- Multi-step workflows (order + payment)

**Choose BASE when:**
- Social media feeds (eventual consistency fine)
- Analytics data (eventually accurate)
- High availability > consistency
- NoSQL databases (Cassandra, DynamoDB)

**Real-world example:** Bank transfers: ACID (must be atomic), Twitter timeline: BASE (eventual consistency acceptable).

**Cost implications:**
- ACID: 2-phase commit overhead, single point of failure, vertical scaling expensive
- BASE: High availability, but application must handle inconsistency (+30% code complexity)

---

### 21. Optimistic Locking vs Pessimistic Locking

**Description:** Assume no conflicts vs lock preemptively.

**Choose Optimistic Locking when:**
- Low contention (<1% conflict rate)
- Read:Write ratio >10:1
- Distributed systems (no shared lock manager)

**Choose Pessimistic Locking when:**
- High contention (>10% conflict rate)
- Critical sections (inventory reservation)
- Short lock duration (<100ms)

**Real-world example:** Shopping cart: optimistic (rarely conflicting edits), seat reservation: pessimistic (must prevent double-booking).

**Cost implications:**
- Optimistic: No lock overhead, but retry logic needed (5-10% performance hit on conflict)
- Pessimistic: Lock overhead 10-50ms, deadlock risk, but guaranteed consistency

---

## DevOps

### 22. Blue-Green vs Canary vs Rolling Deployment

**Description:** Deployment strategies for zero-downtime releases.

**Choose Blue-Green when:**
- Instant rollback critical (<1 minute)
- Database schema unchanged
- Sufficient capacity for 2x environments

**Choose Canary when:**
- Gradual rollout needed (1% → 10% → 100%)
- Monitor metrics before full rollout
- A/B testing deployment

**Choose Rolling when:**
- Resource-constrained (can't double environment)
- Stateless services
- Gradual rollout acceptable (10% at a time)

**Real-world example:** Netflix uses canary (test on 1% traffic for 1 hour, then 100%).

**Cost implications:**
- Blue-Green: 2x infrastructure for switchover (30 min - 2 hours), then tear down
- Canary: +20% overhead for traffic splitting, requires feature flags
- Rolling: No extra cost, but slower rollout (30 min - 2 hours)

---

### 23. Terraform vs Pulumi vs CloudFormation

**Description:** Infrastructure as Code tool selection.

**Choose Terraform when:**
- Multi-cloud (AWS + Azure + GCP)
- Large ecosystem of providers
- HCL acceptable (declarative language)
- Community support critical

**Choose Pulumi when:**
- Prefer general-purpose language (Python, TypeScript, Go)
- Complex logic in IaC (loops, conditionals)
- Testing IaC code
- Team already knows Python/TypeScript

**Choose CloudFormation when:**
- AWS-only shop
- Deep AWS integration needed
- No external tooling allowed

**Real-world example:** Uber uses Terraform (multi-cloud), Pulumi users include Mercedes-Benz (complex logic).

**Cost implications:**
- Terraform: Free (OSS), Terraform Cloud $20-70/user/month
- Pulumi: Free (OSS), Pulumi Cloud $50-100/user/month
- CloudFormation: Free, but AWS-locked

---

### 24. GitHub Actions vs Jenkins vs CircleCI

**Description:** CI/CD platform selection.

**Choose GitHub Actions when:**
- GitHub-native workflow
- Simple pipelines (<30 min)
- Open-source project (free minutes)

**Choose Jenkins when:**
- Self-hosted requirement (security, compliance)
- Complex pipelines (>1 hour)
- Extensive plugins needed
- On-premise infrastructure

**Choose CircleCI when:**
- Fast feedback (<5 min builds)
- Docker-native workflows
- Paid support needed

**Real-world example:** Open-source projects use GitHub Actions (free), enterprises use Jenkins (self-hosted).

**Cost implications:**
- GitHub Actions: 2000 free minutes/month, then $0.008/min (Linux), $0.08/min (macOS)
- Jenkins: Self-hosted $500-5K/month (infrastructure + maintenance)
- CircleCI: $30-200/month for small teams, $2K-10K/month at scale

---

### 25. Docker vs VM vs Bare Metal

**Description:** Container vs virtual machine vs physical server.

**Choose Docker when:**
- Microservices architecture
- Fast startup needed (<1 second)
- Lightweight isolation sufficient
- DevOps culture strong

**Choose VM when:**
- Strong isolation required (multi-tenant)
- Run different OS (Windows on Linux host)
- Legacy applications

**Choose Bare Metal when:**
- Maximum performance (databases, ML training)
- Predictable performance (no noisy neighbor)
- Security isolation critical

**Real-world example:** Web apps: Docker, Database: VM or bare metal.

**Cost implications:**
- Docker: 10-20% overhead, $50-500/month for orchestration (ECS, Kubernetes)
- VM: 20-30% overhead, standard cloud VM pricing
- Bare Metal: 0% overhead but 2-5x cost, $200-5K/month per server

---

## Machine Learning

### 26. Real-Time vs Batch Inference

**Description:** Predict on-demand vs precompute predictions.

**Choose Real-Time Inference when:**
- Personalized results (user input needed)
- Low latency critical (<100ms)
- Feature values change frequently
- Inference cost <$0.01 per request

**Choose Batch Inference when:**
- Predictions for all users (recommendations)
- Latency acceptable (24-hour delay)
- Expensive models (>1 second per prediction)
- Scheduled updates (nightly ETL)

**Real-world example:** Search ranking: real-time (user query), email spam: batch (scan all emails nightly).

**Cost implications:**
- Real-Time: $500-10K/month for API (SageMaker, Vertex AI), <100ms latency
- Batch: $100-2K/month for compute, 24-hour latency acceptable

---

### 27. Accuracy vs Latency

**Description:** Better predictions vs faster predictions.

**Choose Accuracy when:**
- High-stakes decisions (medical, financial)
- Latency >1 second acceptable
- Complex models (deep learning)

**Choose Latency when:**
- User-facing applications (<100ms)
- Good-enough accuracy (80% vs 85%)
- Simple models (linear, tree-based)

**Real-world example:** Fraud detection: accuracy (deep learning, 500ms), autocomplete: latency (trie, <10ms).

**Cost implications:**
- Accuracy: 10-100x more compute, $5K-50K/month GPU costs
- Latency: Simpler models, $500-5K/month CPU costs

---

### 28. Simple Models vs Deep Learning

**Description:** Linear regression vs neural networks.

**Choose Simple Models (Linear, Trees, XGBoost) when:**
- Tabular data
- <100K training samples
- Interpretability required
- Training time <1 hour
- Inference <10ms

**Choose Deep Learning when:**
- Unstructured data (images, text, audio)
- >1M training samples
- State-of-art accuracy needed
- GPU available

**Real-world example:** Credit scoring: XGBoost (interpretable, regulated), image recognition: deep learning (accuracy critical).

**Cost implications:**
- Simple Models: $100-1K/month CPU, train in minutes-hours
- Deep Learning: $5K-50K/month GPU, train in days-weeks

---

### 29. Feature Engineering vs Feature Learning

**Description:** Manual features vs learned representations.

**Choose Feature Engineering when:**
- Domain expertise available
- Tabular data
- Small datasets (<10K samples)
- Interpretability required

**Choose Feature Learning when:**
- High-dimensional data (images, text)
- Large datasets (>1M samples)
- Deep learning models
- Transfer learning (pretrained embeddings)

**Real-world example:** House price prediction: feature engineering (square feet, location), image classification: feature learning (CNN).

**Cost implications:**
- Feature Engineering: High human cost (data scientists), low compute
- Feature Learning: Low human cost, high compute cost (GPUs)

---

## Additional Tradeoffs

### 30. Pull vs Push (Data Pipeline)

**Choose Pull:** Consumer controls rate (backpressure), simple architecture  
**Choose Push:** Real-time updates, producer controls rate

---

### 31. Strong Types vs Dynamic Types

**Choose Strong:** Compile-time safety (TypeScript, Rust), +20% dev time upfront  
**Choose Dynamic:** Rapid prototyping (Python, Ruby), +30% runtime errors

---

### 32. Code Generation vs Hand-Written

**Choose Generated:** APIs from OpenAPI, ORMs from schema, consistency  
**Choose Hand-Written:** Custom logic, performance-critical paths

---

### 33. Feature Flags vs Branch Deploys

**Choose Feature Flags:** Production testing, gradual rollout, A/B testing  
**Choose Branches:** Simpler, but all-or-nothing deployment

---

### 34. Managed Service vs Self-Hosted

**Choose Managed:** Less operational overhead, +50-200% cost premium  
**Choose Self-Hosted:** Full control, lower cost, +2-5 DevOps engineers needed

---

### 35. Polling vs Webhooks

**Choose Polling:** Simple, but inefficient (wastes API calls)  
**Choose Webhooks:** Real-time, but requires public endpoint + retry logic

---

### 36. JWT in Cookie vs LocalStorage

**Choose Cookie:** XSS protection (httpOnly), CSRF risk (use SameSite)  
**Choose LocalStorage:** No CSRF, but XSS steals token easily

---

### 37. API Gateway vs Direct Service

**Choose API Gateway:** Centralized auth, rate limiting, logging, +10-20ms latency  
**Choose Direct:** Lower latency, but auth logic duplicated per service

---

### 38. E2E Encryption vs Server-Side Encryption

**Choose E2E:** Zero-trust (server never sees plaintext), complex key management  
**Choose Server-Side:** Simpler, but server compromise exposes all data

---

### 39. Saga vs 2PC (Distributed Transactions)

**Choose Saga:** High availability, eventual consistency, complex compensation logic  
**Choose 2PC:** Strong consistency, but coordinator single point of failure

---

### 40. Rate Limiting: Fixed Window vs Sliding Window

**Choose Fixed:** Simple, but burst at window boundary  
**Choose Sliding:** Smooth rate limiting, +complexity (token bucket)

---

### 41. GraphQL N+1 Problem: DataLoader vs Join

**Choose DataLoader:** Batching per request, good for APIs  
**Choose Join:** Single query, but less flexible

---

### 42. Hot vs Cold Storage

**Choose Hot (SSD):** Frequent access (<1ms), $0.10-0.20/GB/month  
**Choose Cold (S3 Glacier):** Infrequent access, $0.004/GB/month, retrieval time hours

---

### 43. Monolithic Frontend vs Micro-Frontends

**Choose Monolithic:** Simpler, but one team owns entire UI  
**Choose Micro-Frontends:** Team autonomy, but coordination overhead

---

### 44. Shared Database vs Database per Service

**Choose Shared:** Easier JOINs, but tight coupling  
**Choose Per-Service:** Loose coupling, but distributed queries complex

---

### 45. Long Polling vs WebSocket vs SSE

**Choose Long Polling:** Simple, works through proxies, inefficient  
**Choose WebSocket:** Bidirectional, full-duplex, best for real-time  
**Choose SSE:** Server-to-client only, simpler than WebSocket

---

### 46. Circuit Breaker: Open vs Half-Open Threshold

**Open after:** 5 failures in 10 seconds (fail fast)  
**Half-Open after:** 30 seconds (test if recovered)  
**Close after:** 3 successes in half-open (fully recovered)

---

### 47. Retry: Immediate vs Exponential Backoff

**Choose Immediate:** Transient network glitch (<1% failure rate)  
**Choose Exponential:** Service overload (avoid thundering herd), +jitter

---

### 48. Password Hashing: bcrypt vs Argon2

**Choose bcrypt:** Widely supported, 10 rounds = 100ms, battle-tested  
**Choose Argon2:** Winner of 2015 competition, memory-hard (GPU-resistant), newer

---

### 49. API Versioning: URL vs Header

**Choose URL (`/v1/users`):** Explicit, cacheable, simpler  
**Choose Header (`Accept: application/vnd.api+json;version=1`):** REST purist, but harder to use

---

### 50. Container Orchestration: Docker Swarm vs Kubernetes vs ECS

**Choose Swarm:** Simple, small teams (<5 engineers), <50 containers  
**Choose Kubernetes:** Complex, large scale, ecosystem rich, steep learning curve  
**Choose ECS:** AWS-native, middle complexity, vendor lock-in

---

### 51. Monitoring: Pull (Prometheus) vs Push (Graphite)

**Choose Pull (Prometheus):** Service discovery, short-lived jobs harder  
**Choose Push (Graphite):** Works everywhere, but no service discovery

---

### 52. Secrets: Environment Variables vs Secret Manager

**Choose Env Vars:** Simple, but leak risk (logs, error messages)  
**Choose Secret Manager:** Encrypted, audit logs, rotation, +complexity

---

### 53. Idempotency: Client ID vs Server State

**Choose Client ID:** Client generates UUID, server deduplicates  
**Choose Server State:** Server tracks operations, client retries safely

---

### 54. Time-Series DB: InfluxDB vs Prometheus vs TimescaleDB

**Choose InfluxDB:** General-purpose time-series, SQL-like, writes >1M/sec  
**Choose Prometheus:** Metrics + alerting, pull-based, PromQL powerful  
**Choose TimescaleDB:** PostgreSQL extension, SQL, relational joins work

---

### 55. Message Queue: Kafka vs RabbitMQ vs SQS

**Choose Kafka:** Event streaming, high throughput (>1M msg/sec), replay  
**Choose RabbitMQ:** Traditional queue, routing complex, lower throughput  
**Choose SQS:** AWS-managed, simpler, lower scale (<10K msg/sec)

---

## Summary

**Key Insight:** There are no silver bullets. Every decision is a tradeoff between:
- **Complexity** vs Simplicity
- **Cost** vs Performance  
- **Consistency** vs Availability
- **Flexibility** vs Speed to Market

**Best Practice:** Choose boring technology for non-core systems. Innovate on your competitive advantage.

**Remember:** The best architecture is the one that ships. Don't over-engineer for scale you don't have yet.

---

**Use this document to:** Justify technical decisions, onboard new engineers, debate architecture, prep for interviews.
