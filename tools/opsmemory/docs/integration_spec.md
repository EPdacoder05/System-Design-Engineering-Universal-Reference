# OpsMemory — Integration Spec & Platform Plan

> **Version:** 1.0  
> **Status:** Active  
> **Public-repo safe:** Yes — no personal identifiers, private endpoints, or
> internal workflow details.

---

## 1. Purpose

This document specifies the platform integration architecture for OpsMemory,
covering:

1. **Jarvis MCP client integration** — how an assistant/orchestrator connects
   to OpsMemory tools.
2. **Media Processing evidence ingestion** — how a media pipeline pushes
   normalised evidence into OpsMemory.
3. **Local-first auth model** — API key governance without a cloud IdP.
4. **Service topology** — Docker Compose / deployment layout and trust
   boundaries.
5. **Future gateway mode** — optional centralised LiteLLM proxy path.

---

## 2. Repository Layout

```
System-Design-Engineering-Universal-Reference/  ← platform/pattern home (this repo)
├── tools/opsmemory/
│   ├── api/app.py                  ← FastAPI service
│   ├── mcp/server.py               ← FastMCP server
│   ├── auth.py                     ← local-first auth module
│   ├── integrations/
│   │   ├── jarvis/                 ← Jarvis MCP client scaffolding
│   │   │   ├── config.py
│   │   │   ├── mcp_client.py
│   │   │   └── examples.py
│   │   └── media_pipeline/         ← Media pipeline ingestion scaffolding
│   │       ├── models.py
│   │       ├── client.py
│   │       └── examples.py
│   ├── docker/
│   │   ├── docker-compose.yml      ← development topology
│   │   └── docker-compose.secure.yml ← hardened local deployment
│   └── docs/
│       └── integration_spec.md     ← this document

Jarvis-AI-Assistant/                ← assistant/orchestrator (separate repo)
  (imports tools/opsmemory/integrations/jarvis/ as a dependency or copies it)

Media-Processing-Pipeline/          ← evidence producer (separate repo)
  (imports tools/opsmemory/integrations/media_pipeline/ as a dependency)
```

---

## 3. Jarvis MCP Integration Design

### 3.1 Overview

A Jarvis-style assistant orchestrator integrates with OpsMemory to:

- **Retrieve context** before generating answers (semantic search over evidence
  and memories).
- **Persist outcomes** after task completion (evidence for future queries).
- **Introspect** store counts and registered sources (observability).

### 3.2 Transport Options

| Transport | When to use |
|-----------|-------------|
| REST API (default) | Always available; recommended for service-to-service |
| MCP SSE | When a native MCP client SDK is available |
| MCP stdio | For subprocess embedding (Claude Desktop, Cursor, IDE plugins) |

The scaffolding in `integrations/jarvis/mcp_client.py` uses REST by default.
Swap to SSE/stdio by implementing the `OpsMemoryMCPTransportClient` stub.

### 3.3 Canonical Interaction Pattern

```
Jarvis Assistant
      │
      ├─► query_memory(question)          ← GET /query?q=...
      │       returns: citations + memories
      │
      ├─► [generate answer using context]
      │
      └─► ingest_session_outcome(result)  ← POST /ingest
              persists: session outcome as evidence
```

### 3.4 Configuration

All settings via environment variables:

| Variable | Default | Description |
|---|---|---|
| `OPSMEMORY_API_URL` | `http://localhost:8000` | OpsMemory API base URL |
| `OPSMEMORY_API_KEY` | _(none)_ | Bearer token for auth |
| `OPSMEMORY_MCP_TOKEN` | _(falls back to API_KEY)_ | MCP-specific token |
| `OPSMEMORY_MCP_URL` | `http://localhost:8100` | MCP SSE server URL |
| `JARVIS_MEMORY_QUERY_LIMIT` | `5` | Evidence items per query |
| `JARVIS_SESSION_SOURCE_TYPE` | `jarvis_session` | Source type label |
| `OPSMEMORY_REQUEST_TIMEOUT` | `30` | HTTP timeout (seconds) |

### 3.5 Downstream Adoption

A downstream Jarvis repository can adopt this scaffolding by:

1. Copying `tools/opsmemory/integrations/jarvis/` into its source tree, **or**
2. Importing directly if this repository is included as a dependency package.

No changes to OpsMemory's server-side code are required.

---

## 4. Media Pipeline Evidence Ingestion Design

### 4.1 Overview

A media processing pipeline ingests its outputs (transcripts, OCR extractions,
metadata, enrichments) into OpsMemory as structured evidence items.

### 4.2 Ingestion Flow

```
Media Pipeline Stage
      │
      ├─► build MediaEvidenceBase subclass
      │       (TranscriptResult / OcrExtractionResult / etc.)
      │
      ├─► MediaIngestionClient.ingest(result)
      │       POST /ingest with normalised payload
      │       ← retry on 429/5xx (exponential back-off)
      │
      └─► OpsMemory stores evidence
              ← auto-redaction
              ← embedding generation
              ← periodic consolidation into memories
```

### 4.3 Evidence Payload Schema

```json
{
  "text":         "Human-readable extracted content",
  "source_type":  "media_transcript",
  "source_ref":   "media://recordings/session-001.wav",
  "author":       "transcription-service",
  "repo":         "org/media-pipeline",
  "native_id":    "session-001",
  "occurred_at":  "2026-03-15T14:00:00Z",
  "metadata": {
    "language":          "en",
    "confidence":        0.97,
    "duration_seconds":  180.0
  }
}
```

### 4.4 Source Types

| `source_type` | Model | Description |
|---|---|---|
| `media_transcript` | `TranscriptResult` | Speech-to-text output |
| `media_ocr` | `OcrExtractionResult` | OCR from images/PDFs |
| `media_metadata` | `MediaMetadataResult` | Technical asset properties |
| `media_enrichment` | `EnrichmentResult` | NLP / annotation stage output |

Custom source types can be added by subclassing `MediaEvidenceBase`.

### 4.5 Deduplication

OpsMemory deduplicates evidence using a SHA-256 hash of
`source_type:repo:native_id:occurred_at`.  Ensure `native_id` and
`occurred_at` are stable and reproducible for a given asset to avoid
duplicate ingestion.

### 4.6 Configuration

| Variable | Default | Description |
|---|---|---|
| `OPSMEMORY_API_URL` | `http://localhost:8000` | OpsMemory API base URL |
| `OPSMEMORY_API_KEY` | _(none)_ | Bearer token for auth |
| `MEDIA_INGESTION_MAX_RETRIES` | `3` | Max retry attempts |
| `MEDIA_INGESTION_RETRY_BACKOFF` | `2` | Back-off base (seconds) |
| `OPSMEMORY_REQUEST_TIMEOUT` | `30` | HTTP timeout (seconds) |

---

## 5. Local Auth Model

### 5.1 Design Principles

- **No cloud IdP dependency** — no Okta, Auth0, Cognito, or Google Identity.
- **Local-first** — a single shared API key is sufficient for self-hosted
  deployments.
- **Secure-by-default for exposed deployments** — `OPSMEMORY_REQUIRE_API_KEY`
  defaults to `false` for local-only developer convenience, but the
  `docker-compose.secure.yml` profile enables it explicitly.
- **Exempt paths** — `/health` and `/ready` are always accessible (for
  container orchestrator health probes).

### 5.2 Environment Variables

| Variable | Default | Description |
|---|---|---|
| `OPSMEMORY_REQUIRE_API_KEY` | `false` | Set `true` to enable auth |
| `OPSMEMORY_API_KEY` | _(none)_ | Bearer token for API access |
| `OPSMEMORY_MCP_TOKEN` | _(falls back to API_KEY)_ | Optional separate MCP token |

### 5.3 Generating a Key

```bash
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

Store the result in `.env` or a secret manager.  Never commit it.

### 5.4 Request Format

Authenticated requests must include:

```
Authorization: Bearer <OPSMEMORY_API_KEY>
```

### 5.5 Implementation

The auth logic is in `tools/opsmemory/auth.py` (`apply_auth_middleware`,
`require_api_key`).  Constant-time string comparison (`secrets.compare_digest`)
prevents timing attacks.

---

## 6. Service Topology

### 6.1 Development Topology (`docker-compose.yml`)

```
Host machine
│
├── 127.0.0.1:5432  ← postgres (pgvector:pg16)
└── 127.0.0.1:8000  ← opsmemory (FastAPI)
```

Auth disabled by default.  Suitable for local development only.

### 6.2 Secure Local Topology (`docker-compose.secure.yml`)

```
Host machine
│
├── 127.0.0.1:8000  ← opsmemory (FastAPI) — auth required
├── 127.0.0.1:8100  ← opsmemory-mcp (FastMCP SSE) — auth required
└── [litellm-proxy]  ← 127.0.0.1:4000 (disabled by default; profile=litellm)

Internal Docker network (opsmemory_net — bridge, no outbound)
├── postgres:5432    (no host port)
├── opsmemory:8000
├── opsmemory-mcp:8100
└── litellm-proxy:4000 (when enabled)
```

### 6.3 Recommended Exposure Model

| Service | Recommended exposure |
|---------|---------------------|
| `postgres` | Internal only — never expose publicly |
| `opsmemory` | `127.0.0.1` binding + TLS reverse proxy for remote access |
| `opsmemory-mcp` | `127.0.0.1` binding; stdio mode preferred for local MCP clients |
| `litellm-proxy` | `127.0.0.1` binding only |

### 6.4 Trust Boundaries

```
┌─────────────────────────────────────────────────────┐
│  Host / localhost trust zone                         │
│                                                      │
│  ┌──────────────┐    ┌────────────────────────────┐  │
│  │  MCP clients  │    │  Jarvis / media producers  │  │
│  │  (stdio/SSE)  │    │  (HTTP REST)               │  │
│  └──────┬───────┘    └─────────┬──────────────────┘  │
│         │ MCP protocol          │ HTTPS (if remote)   │
│         │                       │ or HTTP (localhost)  │
│  ┌──────▼───────────────────────▼──────────────────┐  │
│  │           OpsMemory API + MCP Server             │  │
│  │           (Auth: Bearer token via env var)       │  │
│  └──────────────────────┬───────────────────────────┘  │
│                          │                              │
│  ┌───────────────────────▼──────────────────────────┐  │
│  │  Internal Docker network (opsmemory_net)          │  │
│  │  PostgreSQL + pgvector (no external exposure)     │  │
│  └──────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
```

---

## 7. Secret Handling

| Secret | Recommended storage |
|--------|---------------------|
| `OPSMEMORY_API_KEY` | `.env` file (not committed) or secret manager |
| `OPSMEMORY_MCP_TOKEN` | Same as above |
| `GITHUB_TOKEN` | Repository Action secret / `.env` |
| Provider API keys (`ANTHROPIC_API_KEY`, etc.) | Secret manager / `.env` |
| `POSTGRES_PASSWORD` | `.env` or Docker secret |

**Rules:**
- Never hardcode secrets in source code or Docker images.
- Never commit `.env` files with real values.
- Rotate keys whenever team membership changes.
- Use `secrets.token_urlsafe(32)` to generate strong random keys.

---

## 8. Environment Variable Reference

### OpsMemory Core

| Variable | Default | Description |
|---|---|---|
| `DATABASE_URL` | `postgresql+asyncpg://...` | PostgreSQL DSN |
| `DB_POOL_SIZE` | `10` | Connection pool size |
| `DB_MAX_OVERFLOW` | `20` | Extra connections above pool |
| `OPSMEMORY_REQUIRE_API_KEY` | `false` | Enable API key auth |
| `OPSMEMORY_API_KEY` | _(none)_ | API bearer token |
| `OPSMEMORY_MCP_TOKEN` | _(falls back to API_KEY)_ | MCP bearer token |
| `OPSMEMORY_API_URL` | `http://localhost:8000` | API base URL (for MCP + connectors) |
| `OPSMEMORY_LLM_PROVIDER` | `mock` | LLM provider |
| `OPSMEMORY_LLM_MODEL` | `anthropic/claude-3-haiku-20240307` | LLM model |
| `OPSMEMORY_EMBEDDING_PROVIDER` | `mock` | Embedding provider |
| `OPSMEMORY_EMBEDDING_MODEL` | `openai/text-embedding-3-small` | Embedding model |
| `OPSMEMORY_LITELLM_BASE_URL` | _(none)_ | LiteLLM proxy/compatible endpoint |
| `OPSMEMORY_LITELLM_API_KEY` | _(none)_ | LiteLLM API key override |
| `OPSMEMORY_REQUEST_TIMEOUT` | `30` | HTTP request timeout (seconds) |

### GitHub Connector

| Variable | Default | Description |
|---|---|---|
| `GITHUB_TOKEN` | _(none)_ | GitHub PAT |
| `GITHUB_OWNER` | _(resolved from token)_ | GitHub user or org |
| `GITHUB_INCLUDE_REPOS` | _(all)_ | Allowlist of repo names |
| `GITHUB_EXCLUDE_REPOS` | _(none)_ | Denylist of repo names |
| `GITHUB_POLL_INTERVAL` | `3600` | Sweep interval (seconds) |
| `OPSMEMORY_INGEST_URL` | `http://localhost:8000/ingest` | Ingest target URL |

### Jarvis Integration

| Variable | Default | Description |
|---|---|---|
| `OPSMEMORY_MCP_URL` | `http://localhost:8100` | MCP SSE server URL |
| `JARVIS_MEMORY_QUERY_LIMIT` | `5` | Max evidence items per query |
| `JARVIS_SESSION_SOURCE_TYPE` | `jarvis_session` | Source type for session outcomes |

### Media Pipeline Integration

| Variable | Default | Description |
|---|---|---|
| `MEDIA_INGESTION_MAX_RETRIES` | `3` | Max retry attempts |
| `MEDIA_INGESTION_RETRY_BACKOFF` | `2` | Back-off base (seconds) |

---

## 9. Future: Centralised LiteLLM Proxy Mode

When multiple services (Jarvis, media pipeline, other tools) need LLM / embedding
access, a centralised LiteLLM proxy eliminates the need for each service to hold
its own provider API keys.

### Architecture

```
┌────────────────────────────────────────────────────┐
│  LiteLLM Proxy (127.0.0.1:4000)                    │
│  Holds all provider API keys                        │
│  Presents a single OpenAI-compatible API            │
└──────────────────────────┬─────────────────────────┘
                            │
          ┌─────────────────┼──────────────────┐
          │                 │                  │
   OpsMemory API      Jarvis assistant   Media pipeline
   (no raw keys)      (no raw keys)      (no raw keys)
```

### Activation

```bash
# Enable in docker-compose.secure.yml
docker compose -f tools/opsmemory/docker/docker-compose.secure.yml \
  --profile litellm up -d

# Point OpsMemory at the proxy
export OPSMEMORY_LITELLM_BASE_URL=http://litellm-proxy:4000
export OPSMEMORY_LLM_PROVIDER=litellm
export OPSMEMORY_EMBEDDING_PROVIDER=litellm
```

Only `LITELLM_MASTER_KEY` and provider keys are needed on the proxy service.
All downstream services use `OPSMEMORY_LITELLM_BASE_URL` with no provider keys.

---

## 10. Public-Repo Safety Checklist

Before merging any integration code or docs into a public repository:

- [ ] No personal usernames in examples (use generic placeholders like `your-org`).
- [ ] No private or internal endpoints in examples.
- [ ] No real API keys, tokens, or passwords.
- [ ] No private workflow details (personal task names, internal project names).
- [ ] All secrets referenced via `${ENV_VAR}` substitution only.
- [ ] `.env` files with real values are listed in `.gitignore`.
- [ ] Docker Compose examples use environment variable substitution, not hardcoded values.
- [ ] MCP config examples use `${ENV_VAR}` for all secret fields.
