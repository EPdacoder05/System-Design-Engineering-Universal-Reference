# 🔗 Cross-Project Integration Map

Visual guide showing how all repositories in the ecosystem connect and interact.

---

## 🎯 Ecosystem Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                      PROJECT ECOSYSTEM                               │
└─────────────────────────────────────────────────────────────────────┘

┌──────────────────────┐
│ Sportsbook-          │  Data Collection Layer
│ aggregation          │  - Multi-source scraping
│                      │  - Real-time data processing
│ (Data Collection)    │  - Rate limiting
└──────────┬───────────┘
           │ Scraped data (JSON)
           │ HTTP/REST
           ↓
┌──────────────────────┐
│ security-data-       │  Data Platform Layer
│ fabric               │  - Redis cache (AES-256)
│                      │  - PostgreSQL + pgvector
│ (Data Platform)      │  - MFA + JWT auth
│                      │  - Audit logging
└──────────┬───────────┘
           │ Gold layer events
           │ Kafka/Event stream
           ↓
┌──────────────────────┐
│ incident-predictor-  │  ML Analytics Layer
│ ml                   │  - Z-score anomaly detection
│                      │  - Isolation Forest
│ (Prediction Engine)  │  - Trajectory prediction
│                      │  - Alert fatigue prevention
└──────────┬───────────┘
           │ Alerts + predictions
           │ REST API / webhooks
           ↓
┌──────────────────────┐
│ NullPointVector      │  Security Validation Layer
│                      │  - 32 attack patterns
│ (Security Shield)    │  - Input validation
│                      │  - Zero-day shield
│                      │  - Circuit breaker
└──────────┬───────────┘
           │ Validated/sanitized data
           │
           ↓
┌──────────────────────────────────────────────────────────────────┐
│ System-Design-Engineering-Universal-Reference                     │
│                                                                    │
│ (Pattern Library - "The Bible")                                   │
│ - All patterns, templates, best practices                         │
│ - Copy-paste ready implementations                                │
└────────────────────────────────────────────────────────────────────┘
           ↑
           │ Patterns extracted from all projects
           │ Continuous updates
           └────────────────────┘
```

---

## 🏗️ Detailed Integration Architecture

### 1. Sportsbook-aggregation → security-data-fabric

**Integration Type**: REST API + Message Queue

**Data Flow**:
```
Scrapers (Sportsbook) → Normalization → API Gateway → security-data-fabric
                                              ↓
                                       Bronze Layer (Raw)
                                              ↓
                                       Silver Layer (Cleaned)
                                              ↓
                                       Gold Layer (Aggregated)
```

**Endpoints**:
- `POST /api/v1/ingest/bronze` - Raw data ingestion
- `POST /api/v1/events/batch` - Batch event submission
- `GET /api/v1/health` - Health check

**Authentication**: 
- Service-to-service JWT with scope `data:write`
- API key with rate limit: 1000 req/min

**Data Format**:
```json
{
  "source": "sportsbook_scraper",
  "timestamp": "2026-02-10T02:30:00Z",
  "event_type": "odds_update",
  "data": {
    "sport": "basketball",
    "teams": ["Team A", "Team B"],
    "odds": {"Team A": 1.85, "Team B": 2.10}
  },
  "metadata": {
    "scraper_version": "2.1.0",
    "confidence": 0.95
  }
}
```

**Error Handling**:
- Retry with exponential backoff (3 attempts)
- Circuit breaker: open after 5 failures in 60s
- Dead letter queue for failed events

---

### 2. security-data-fabric → incident-predictor-ml

**Integration Type**: Event Stream (Kafka) + REST API

**Data Flow**:
```
Gold Layer Events → Kafka Topic → incident-predictor-ml → Predictions
                                           ↓
                                    Anomaly Detector
                                           ↓
                                  Trajectory Predictor
                                           ↓
                                    Alert Generator
```

**Kafka Topics**:
- `gold.events` - All Gold layer events
- `gold.anomalies` - Detected anomalies only
- `predictions.alerts` - High-confidence alerts

**Event Schema**:
```json
{
  "event_id": "evt_123456",
  "timestamp": "2026-02-10T02:30:00Z",
  "event_type": "metric_update",
  "severity": "NORMAL",
  "value": 42.5,
  "metadata": {
    "source": "security-data-fabric",
    "layer": "gold",
    "quality_score": 0.98
  }
}
```

**ML Pipeline**:
1. **Consume events** from Kafka (batch of 100)
2. **Feature extraction** (lag features, rolling stats)
3. **Anomaly detection** (Z-score + Isolation Forest)
4. **Trajectory prediction** (5-step ahead forecast)
5. **Alert evaluation** (confidence threshold: 80%)
6. **Publish alerts** back to Kafka

**API Endpoints**:
- `POST /api/v1/predict` - On-demand prediction
- `GET /api/v1/alerts` - Retrieve recent alerts
- `GET /api/v1/metrics` - Model performance metrics

---

### 3. incident-predictor-ml → NullPointVector

**Integration Type**: REST API with validation

**Data Flow**:
```
Predictions → Input Validation → Zero-Day Shield → Sanitized Output
                    ↓                   ↓
              Attack Detection    Secure Hashing
                    ↓                   ↓
              Blocked/Allowed     HMAC Verification
```

**Validation Pipeline**:
1. **Length check** (max 10KB per request)
2. **Attack pattern detection** (32 patterns checked)
3. **Deserialization** (JSON whitelist validation)
4. **Business logic validation** (no negative values, reasonable ranges)
5. **Sanitization** (strip metadata, escape special chars)

**Example Integration**:
```python
from security.input_validator import detect_attack_patterns
from security.zero_day_shield import DefenseInDepthValidator

# Validate prediction data before processing
validator = DefenseInDepthValidator()
prediction_data = '{"value": 42.5, "confidence": 0.85}'

validated = validator.validate_multi_layer(
    data=prediction_data,
    expected_type='json',
    max_length=10_000
)

if validated:
    # Process prediction
    pass
else:
    # Reject malicious input
    raise SecurityException("Validation failed")
```

**Circuit Breaker Integration**:
```python
from security.circuit_breaker import get_circuit_breaker

# Protect external API calls
prediction_cb = get_circuit_breaker(
    name="prediction_api",
    config=CircuitBreakerConfig(
        failure_threshold=5,
        recovery_timeout=30
    )
)

result = prediction_cb.call(
    make_prediction_api_call,
    data=validated_data
)
```

---

### 4. All Projects → System-Design-Engineering-Universal-Reference

**Integration Type**: Pattern extraction + documentation

**Pattern Flow**:
```
Project Implementations → Code Review → Pattern Extraction → Documentation
                                              ↓
                                      Universal Reference
                                              ↓
                                  Copy-paste ready modules
                                              ↓
                                  Future projects reuse
```

**Pattern Categories Extracted**:

| Project | Patterns Contributed |
|---------|---------------------|
| **security-data-fabric** | Redis caching, MFA, JWT auth, Audit logging, Azure Key Vault, Prometheus monitoring, Error budget, SOC2 controls |
| **NullPointVector** | 32 attack patterns, Input validation, Circuit breaker, Zero-day shield, Defense-in-depth |
| **incident-predictor-ml** | Z-score anomaly detection, Isolation Forest, Trajectory prediction, Alert fatigue prevention, SDF bridge |
| **Sportsbook-aggregation** | Autonomous engine, Multi-source scraping, Rate limiting, Real-time aggregation, Health checks |

**Usage Pattern**:
```bash
# Copy patterns to new project
cp security/input_validator.py new-project/security/
cp security/circuit_breaker.py new-project/security/
cp .github/workflows/security-scan-universal.yml new-project/.github/workflows/

# Customize for your use case
vim new-project/security/input_validator.py
```

---

## 🔄 Data Flow Examples

### End-to-End: Data Collection → Alert Generation

```
Step 1: Sportsbook Scraper collects odds data
   ↓ HTTP POST
Step 2: security-data-fabric ingests to Bronze layer
   ↓ Validation + transformation
Step 3: Data moves to Silver layer (cleaned)
   ↓ Aggregation
Step 4: Data moves to Gold layer (ready for analytics)
   ↓ Kafka event stream
Step 5: incident-predictor-ml consumes Gold events
   ↓ ML pipeline
Step 6: Anomaly detected (confidence: 87%)
   ↓ Alert evaluation (>80% threshold)
Step 7: Alert sent to monitoring system
   ↓ PagerDuty notification
Step 8: On-call engineer investigates
```

**Timing**:
- Scraper → Bronze: ~2s
- Bronze → Silver: ~5s
- Silver → Gold: ~3s
- Gold → ML prediction: ~1.5s
- **Total latency**: ~11.5s (end-to-end)

---

## 🔐 Security Integration Points

### Authentication Chain

```
User/Service → API Gateway → JWT Validation → RBAC Check → Resource Access
                    ↓              ↓              ↓
               Rate Limit    Refresh Token   Permission
                Check         Rotation         Scope
```

**Token Flow**:
1. **Login** → Access token (15 min) + Refresh token (7 days)
2. **API call** → Access token in `Authorization: Bearer <token>`
3. **Token expired** → Use refresh token to get new access token
4. **Refresh token used** → Rotated (single-use)
5. **Logout** → Revoke refresh token

**Service-to-Service Auth**:
```python
from security.auth_framework import create_service_token

# security-data-fabric issues token to incident-predictor-ml
token = create_service_token(
    service_name="incident-predictor-ml",
    scopes=["data:read", "events:subscribe"]
)

# incident-predictor-ml uses token to access data
headers = {"Authorization": f"Bearer {token}"}
response = requests.get(
    "https://sdf.example.com/api/v1/events",
    headers=headers
)
```

---

## 📊 Monitoring Integration

### Metrics Collection Flow

```
Application → Prometheus Client → Prometheus Server → Grafana
                                          ↓
                                    AlertManager
                                          ↓
                                     PagerDuty
```

**Shared Metrics**:
- `http_request_duration_seconds` - API latency
- `http_requests_total` - Request count
- `cache_hit_rate` - Cache efficiency
- `circuit_breaker_state` - Circuit breaker status
- `anomaly_detection_count` - Anomalies detected
- `prediction_confidence_score` - ML confidence

**Grafana Dashboard**:
- **Row 1**: Golden Signals (latency, traffic, errors, saturation)
- **Row 2**: Cache metrics (hit rate, memory usage)
- **Row 3**: ML metrics (prediction count, confidence distribution)
- **Row 4**: Security metrics (blocked requests, attack patterns)

---

## 🚨 Alert Routing

### Alert Flow

```
Detection → Evaluation → Routing → Notification
                ↓           ↓          ↓
           Confidence   Severity   Channel
            (>80%)    (CRITICAL)  (PagerDuty)
```

**Alert Routing Table**:

| Severity | Confidence | Route To | Response Time |
|----------|-----------|----------|---------------|
| EXTREME | >80% | PagerDuty + Slack | Immediate |
| CRITICAL | >80% | PagerDuty | <15 min |
| WARNING | >80% | Slack | <1 hour |
| NORMAL | Any | Logs only | N/A |

**Alert Fatigue Prevention**:
- Confidence threshold: **80%** (suppress low-confidence alerts)
- Rate limiting: Max 10 alerts/hour per type
- Aggregation: Group similar alerts (5-minute window)
- Snooze: Temporarily silence known issues

---

## 💾 Data Retention Policy

| Layer | Retention | Storage | Purpose |
|-------|-----------|---------|---------|
| **Bronze** (Raw) | 30 days | S3 Standard | Debugging, reprocessing |
| **Silver** (Cleaned) | 90 days | S3 Standard | Analytics |
| **Gold** (Aggregated) | 1 year | S3 IA | Reporting, ML training |
| **Audit Logs** | 7 years | S3 Glacier | Compliance |
| **Predictions** | 6 months | PostgreSQL | Model evaluation |
| **Alerts** | 1 year | PostgreSQL | Incident analysis |

---

## 🔄 Disaster Recovery

### Recovery Strategy

**RTO (Recovery Time Objective)**: 1 hour  
**RPO (Recovery Point Objective)**: 15 minutes

**Failure Scenarios**:

| Scenario | Impact | Recovery Procedure |
|----------|--------|-------------------|
| **Database failure** | No reads/writes | Failover to replica (5 min) |
| **Redis failure** | Cache miss (slower) | Traffic to database (degraded) |
| **Kafka failure** | No events | Buffer in memory (max 1 hour) |
| **ML service down** | No predictions | Use last known predictions |
| **Region failure** | Full outage | Failover to DR region (1 hour) |

**Recovery Steps**:
1. **Detect failure** (automated monitoring)
2. **Trigger runbook** (on-call engineer)
3. **Failover to backup** (automated or manual)
4. **Verify health** (smoke tests)
5. **Resume traffic** (gradual ramp-up)
6. **Postmortem** (blameless, within 48 hours)

---

## 🎓 Integration Best Practices

### Do's ✅
- Use circuit breakers for external calls
- Implement retry with exponential backoff
- Validate all inputs (32 attack patterns)
- Use structured logging (JSON format)
- Monitor Golden Signals (latency, traffic, errors, saturation)
- Set up alerts with confidence thresholds (80%+)
- Document all integration points
- Test failure scenarios (chaos engineering)

### Don'ts ❌
- Don't share secrets in code or environment files
- Don't skip input validation
- Don't ignore monitoring gaps
- Don't use blocking I/O in async code
- Don't hardcode URLs (use service discovery)
- Don't skip authentication between services
- Don't forget rate limiting
- Don't skip load testing

---

## 📖 Quick Reference

### Service URLs (Example)

| Service | URL | Port |
|---------|-----|------|
| security-data-fabric | `https://sdf.example.com` | 443 |
| incident-predictor-ml | `https://predictor.example.com` | 443 |
| NullPointVector | N/A (library) | N/A |
| Sportsbook-aggregation | `https://sportsbook.example.com` | 443 |
| Prometheus | `https://prometheus.example.com` | 9090 |
| Grafana | `https://grafana.example.com` | 3000 |

### Health Check Endpoints

All services expose:
- `GET /health` - Liveness probe
- `GET /ready` - Readiness probe
- `GET /metrics` - Prometheus metrics

---

**Document Version**: 1.0  
**Last Updated**: February 2026  
**Next Review**: Quarterly
