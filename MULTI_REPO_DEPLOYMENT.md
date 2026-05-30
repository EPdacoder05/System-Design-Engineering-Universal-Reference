# Multi-Repo Ownership and Migration Map

**Purpose:** Define what lives where, who owns it, and how migrations work.  
This is an architecture ownership document, not a copy-paste deployment guide.

---

## Repository roles

| Repository | Role | Owns |
|------------|------|------|
| `EPdacoder05/System-Design-Engineering-Universal-Reference` | **Canonical reference** | Reusable patterns, templates, rubrics, architecture maps |
| `EPdacoder05/Jarvis-AI-Assistant` | **Jarvis product** | MCP server, vault, dashboards, homelab event pipelines |
| `EPdacoder05/security-data-fabric` | **Security data product** | ECC carrier identity, audit pipelines, security platform |
| `EPdacoder05/NullPointVector` | **Security testing platform** | Offensive/defensive security tooling, pentest workflows |
| `EPdacoder05/finops-cost-control-as-code` | **FinOps** | Cost anomaly detection deployment, AWS resource management |
| `EPdacoder05/incident-replay-tool` | **Incident ML** | Incident prediction, replay tooling |
| `EPdacoder05/ha-iot-stack` | **IoT** | Home automation stack, HA + MQTT deployment |
| `EPdacoder05/ha-ble-mqtt-bridge` | **IoT bridge** | BLE-to-MQTT bridge firmware and deployment |
| `EPdacoder05/TF2S3-migration` | **IaC automation** | Terraform-to-S3 migration tooling |

---

## What each product repo imports from this repo

Each product repo should **reference** these patterns, not copy them wholesale. Use `git submodule`, a shared package, or direct copy of the relevant module — the key is one canonical source.

| Pattern file | Which product repos use it |
|-------------|---------------------------|
| `security/auth_framework.py` | security-data-fabric, NullPointVector, incident-replay-tool, finops-cost-control-as-code |
| `security/input_validator.py` | All services with user/device input |
| `security/ai_era_security.py` | security-data-fabric (prompt injection, agent access control) |
| `security/iac/cloudflare_terraform.tf` | Any Cloudflare-fronted service |
| `api/service_template.py` | Any new FastAPI service |
| `api/cloudflare_platform.py` | Any edge-deployed service |
| `database/vector_search.py` | Any RAG system (Jarvis, opsmemory consumers) |
| `cicd/ci-python.yml` | All Python repos |
| `cicd/security-scan.yml` | All repos requiring weekly security gate |
| `tools/opsmemory/` | Jarvis-AI-Assistant (as primary consumer), any agent needing persistent memory |

---

## Jarvis extraction (Phase 3)

### Migration inventory

| Item | Location in this repo | Destination | Status |
|------|----------------------|-------------|--------|
| MCP server | `tools/jarvis/mcp/` | `EPdacoder05/Jarvis-AI-Assistant/mcp/` | To migrate |
| Vault (raw + wiki) | `tools/jarvis/vault/` | `EPdacoder05/Jarvis-AI-Assistant/vault/` | To migrate |
| Slash commands | `tools/jarvis/.claude/commands/` | `EPdacoder05/Jarvis-AI-Assistant/.claude/commands/` | To migrate |
| Agent prompt | `tools/jarvis/CLAUDE.md` | `EPdacoder05/Jarvis-AI-Assistant/CLAUDE.md` | To migrate |
| Tests | `tools/jarvis/tests/` | `EPdacoder05/Jarvis-AI-Assistant/tests/` | To migrate |
| opsmemory Jarvis integration | `tools/opsmemory/integrations/jarvis/` | Stays here as reusable pattern reference | Keep |
| opsmemory Jarvis test | `tools/opsmemory/tests/test_jarvis_client.py` | Stays here as integration test for the pattern | Keep |

### Migration steps (execute from Jarvis-AI-Assistant repo)

```bash
# In EPdacoder05/Jarvis-AI-Assistant
git remote add ref https://github.com/EPdacoder05/System-Design-Engineering-Universal-Reference.git
git fetch ref

# Copy Jarvis product files
git checkout ref/main -- tools/jarvis/mcp
git checkout ref/main -- tools/jarvis/vault
git checkout ref/main -- tools/jarvis/.claude
git checkout ref/main -- tools/jarvis/tests
git checkout ref/main -- tools/jarvis/CLAUDE.md

# Move to repo root structure
git mv tools/jarvis/mcp mcp
git mv tools/jarvis/vault vault
git mv tools/jarvis/.claude .claude
git mv tools/jarvis/tests tests
git mv tools/jarvis/CLAUDE.md CLAUDE.md

git commit -m "chore: import Jarvis product code from universal reference"
```

After migration, delete product code from this repo and leave only `tools/jarvis/MIGRATION.md`.

---

## ECC architecture reference

ECC (external corpus / cross-cutting) patterns that generalize to any backend/AI system:

### Patterns extracted to this repo

| Pattern | File here | Original domain | Generalizes to |
|---------|-----------|----------------|----------------|
| Carrier identity normalization | `api/carrier_identity.py` | Insurance carrier lookup | Any entity-identity resolution with fuzzy matching and confidence tiers |
| Part-type catalog | `database/catalog.py` | Parts inventory | Any DB-backed type registry with soft validation |
| Medallion ETL | `patterns/medallion_architecture.py` | Data lake ingestion | Any Bronze→Silver→Gold data pipeline |

### Patterns that stay in ECC product repo

- Carrier-specific business rules and scoring logic
- Insurer ID tenant mapping tables
- ECC-specific schema migrations
- Product-specific audit trail consumers

---

## Ownership rules

### This repo owns

- Reusable pattern files (standalone, no product logic)
- Architecture decision rubrics (`rubrics/`)
- CI/CD template files (`cicd/`)
- Security pattern references (`security/`)
- API protocol references (`api/`)
- Multi-agent RAG workflow map (`tools/opsmemory/`, `IMPLEMENTATION_SUMMARY.md`)
- Engineering tradeoff analysis (`TRADEOFFS.md`)

### Product repos own

- Executable product logic (services, workers, pipelines)
- Product-specific MCP servers and vault structures
- Dashboards and deployment wiring
- Product-specific CI configurations that extend the templates here
- Secrets management and environment-specific config

### Decision rule

> If a file can be copy-pasted into any backend project without modification, it belongs here.  
> If a file only makes sense in one specific product context, it belongs in that product repo.

---

## Adding a new pattern to this repo

1. Confirm it generalizes — usable in at least two different projects without modification
2. Strip any product-specific terminology, credentials, or business logic
3. Add an `Apply to:` docstring section explaining the use case
4. Add a row to the ownership table above
5. Update `README.md` TOC and section
6. CI must pass: ruff + mypy + pytest

---

## Security distribution model

Each product repo pulls security patterns from this repo. The canonical security baseline is:

```
security/input_validator.py     — input validation gate (32+ patterns)
security/auth_framework.py      — identity and access
security/ai_era_security.py     — AI-era attack surface (patterns 28-30)
security/circuit_breaker.py     — resilience
cicd/security-scan.yml          — weekly automated scan
```

Product repos add product-specific security config on top. They do not fork the base patterns — they import them.

**Last updated:** 2026-05-30  
**Status:** Active — update when ownership changes
