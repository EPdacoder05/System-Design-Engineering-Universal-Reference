# OpsMemory — AI Memory Platform Primitive

> Provider-agnostic, persistent, searchable memory backed by PostgreSQL + pgvector.
> Compatible with Claude / Anthropic, OpenAI, AWS Bedrock, and local backends via LiteLLM.
> MCP-accessible via FastMCP.

---

## Overview

OpsMemory is a reusable AI memory platform primitive that:

1. **Ingests** evidence from any source (GitHub commits, PRs, manual text, files) via HTTP, an inbox directory, or MCP tools.
2. **Redacts** secrets automatically before anything reaches the database.
3. **Embeds** each evidence item using a configurable embedding provider (mock by default; LiteLLM-backed for production).
4. **Consolidates** batches of evidence into higher-level Memory records on a configurable timer, optionally using an LLM for summarisation.
5. **Answers** natural-language queries using semantic search over the embedding store.
6. **Exposes** all core operations as MCP tools via FastMCP — accessible from Claude Desktop, Cursor, and other MCP clients.

OpsMemory is designed to be:
- **Provider-agnostic** — swap LLM and embedding providers via environment config, not code changes.
- **Cloud-modular** — compatible with Anthropic, OpenAI, AWS Bedrock, and OpenAI-compatible local backends.
- **MCP-native** — all operations are available as structured MCP tools.
- **PostgreSQL-backed** — pgvector is the only persistence layer; no SQLite.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                FastMCP Server (port 8100 / stdio)            │
│  memory_ingest_text  memory_query  memory_status             │
│  memory_consolidate  memory_list_sources                     │
│  memory_sync_github_owner  memory_delete_*                   │
└────────────────────┬─────────────────────────────────────────┘
                     │ HTTP → OPSMEMORY_API_URL
┌────────────────────▼─────────────────────────────────────────┐
│                     FastAPI (port 8000)                      │
│  /ingest  /query  /consolidate  /memories  /evidence         │
│  /sources  /status  DELETE /evidence/{id}  DELETE            │
│  /memories/{id}  POST /clear                                 │
└────────────────────┬─────────────────────────────────────────┘
                     │ async
          ┌──────────▼──────────┐
          │    Agent Runtime     │
          │  ┌───────────────┐  │
          │  │  Ingestor      │  │◄── inbox/ (*.txt / *.json)
          │  │  (inbox watch) │  │
          │  └──────┬────────┘  │
          │         │ redact    │
          │  ┌──────▼────────┐  │
          │  │  Consolidator  │  │  (every 30 min)
          │  └──────┬────────┘  │
          └─────────┼───────────┘
                    │
          ┌─────────▼───────────┐    ┌──────────────────────────┐
          │   PostgreSQL + pgvector │  │   Provider Layer          │
          │  evidence_items         │  │  providers/llm/           │
          │  evidence_embeddings    │  │    base.py (abstract)     │
          │  memories               │  │    litellm.py (SDK/proxy) │
          │  consolidation_runs     │  │  providers/embeddings/    │
          │  sources                │  │    base.py (abstract)     │
          └─────────────────────────┘  │    litellm.py (SDK/proxy) │
                                       │  providers/__init__.py    │
                                       │    (factory / loader)     │
                                       │  providers/model_registry │
                                       │    .yaml                  │
                                       └──────────────────────────┘

          ┌──────────────────────────┐    ┌───────────────────────────┐
          │   GitHub Connector       │    │   Streamlit Dashboard     │
          │  (commits + PRs → /ingest│    │  dashboard.py (port 8501) │
          └──────────────────────────┘    └───────────────────────────┘
```

---

## Prerequisites

| Requirement | Version |
|---|---|
| Docker + Docker Compose | 20.10+ |
| Python | 3.11+ |
| PostgreSQL | 16 with pgvector extension |

---

## Quick Start

```bash
# 1. Clone / enter the repo
cd tools/opsmemory/docker

# 2. Copy and configure environment variables
cp tools/opsmemory/.env.example .env
# Edit .env — set GITHUB_TOKEN, provider keys, etc.

# 3. Start Postgres + OpsMemory API
docker compose up -d

# 4. Verify the API is healthy
curl http://localhost:8000/health

# 5. Ingest some text
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{"text": "Deployed v2.3.1 to production at 14:00 UTC", "source_type": "manual"}'

# 6. Query
curl "http://localhost:8000/query?q=What+was+deployed%3F"

# 7. Trigger consolidation
curl -X POST http://localhost:8000/consolidate

# 8. Dashboard (optional — requires streamlit)
cd ..
streamlit run dashboard.py
# Opens at http://localhost:8501
```

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `DATABASE_URL` | `postgresql+asyncpg://opsmemory:opsmemory@localhost:5432/opsmemory` | SQLAlchemy async DSN |
| `DB_POOL_SIZE` | `10` | Connection pool size |
| `DB_MAX_OVERFLOW` | `20` | Max extra connections above pool size |
| `GITHUB_TOKEN` | _(none)_ | GitHub personal access token (for private repos / higher rate limits) |
| `GITHUB_OWNER` | _(resolved from token)_ | GitHub username or organisation to enumerate repositories for. If unset, the connector calls `GET /user` with the supplied token to resolve the authenticated identity automatically. |
| `GITHUB_INCLUDE_REPOS` | _(all)_ | Comma-separated allowlist of repo names |
| `GITHUB_EXCLUDE_REPOS` | _(none)_ | Comma-separated denylist of repo names |
| `GITHUB_POLL_INTERVAL` | `3600` | Seconds between GitHub sweeps |
| `OPSMEMORY_INGEST_URL` | `http://localhost:8000/ingest` | Internal URL the GitHub connector POSTs to |
| `OPSMEMORY_LLM_PROVIDER` | `mock` | LLM provider: `litellm` or `mock` |
| `OPSMEMORY_LLM_MODEL` | `anthropic/claude-3-haiku-20240307` | LiteLLM-format LLM model string |
| `OPSMEMORY_EMBEDDING_PROVIDER` | `mock` | Embedding provider: `litellm` or `mock` |
| `OPSMEMORY_EMBEDDING_MODEL` | `openai/text-embedding-3-small` | LiteLLM-format embedding model string |
| `OPSMEMORY_LITELLM_BASE_URL` | _(none)_ | Optional proxy/compatible endpoint base URL (enables proxy mode) |
| `OPSMEMORY_LITELLM_API_KEY` | _(none)_ | Optional API key override forwarded to LiteLLM |
| `OPSMEMORY_API_URL` | `http://localhost:8000` | Base URL used by the MCP server to reach the FastAPI service |
| `ANTHROPIC_API_KEY` | _(none)_ | Anthropic API key (for Anthropic-direct models) |
| `OPENAI_API_KEY` | _(none)_ | OpenAI API key |
| `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` | _(none)_ | AWS credentials for Bedrock (prefer IAM roles in production) |

See `.env.example` for a complete annotated template.

> **Security note:** Supply all secrets via environment variables, a `.env` file (not committed to source control), or a secure secret store such as [GitHub Actions secrets](https://docs.github.com/en/actions/security-guides/using-secrets-in-github-actions).  Never commit tokens or credentials to version control.

---

## API Reference

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | API liveness check |
| `/ready` | GET | Returns 200 when the database is reachable |
| `/ingest` | POST | Ingest new text evidence |
| `/query` | GET | Semantic search over evidence + memories |
| `/consolidate` | POST | Trigger a consolidation cycle |
| `/status` | GET | Evidence and memory counts |
| `/memories` | GET | List consolidated memory records |
| `/memories/{id}` | DELETE | Delete a single memory record |
| `/evidence` | GET | List raw evidence items |
| `/evidence/{id}` | DELETE | Delete a single evidence item |
| `/sources` | GET | List registered data sources |
| `/clear` | POST | Delete all evidence and memories (full reset) |

### `GET /health`
```bash
curl http://localhost:8000/health
# {"status":"ok","timestamp":"2024-01-15T12:00:00+00:00"}
```

### `GET /ready`
Returns `200 {"status":"ready"}` when the database is reachable, `503` otherwise.

### `POST /ingest`
```bash
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Merged PR #42: Add rate limiting",
    "source_type": "github_pr",
    "source_ref": "https://github.com/your-org/myrepo/pull/42",
    "author": "alice"
  }'
# {"evidence_id":"<uuid>","redacted":false,"correlation_id":"<uuid>"}
```

### `POST /consolidate`
```bash
curl -X POST http://localhost:8000/consolidate
# {"run_id":"<uuid>","memories_created":3,"evidence_consolidated":27}
```

### `GET /status`
```bash
curl http://localhost:8000/status
# {"evidence_total":42,"evidence_unconsolidated":15,"memories":5}
```

### `GET /query`
```bash
curl "http://localhost:8000/query?q=recent+deployments&limit=5"
# {
#   "query": "recent deployments",
#   "answer": "Based on 3 evidence items and 2 memories: ...",
#   "citations": [...],
#   "memories": [...]
# }
```

### `GET /memories`
```bash
curl http://localhost:8000/memories
```

### `DELETE /memories/{id}`
```bash
curl -X DELETE http://localhost:8000/memories/<uuid>
# {"deleted":"<uuid>"}
```

### `GET /evidence`
```bash
curl http://localhost:8000/evidence
```

### `DELETE /evidence/{id}`
```bash
curl -X DELETE http://localhost:8000/evidence/<uuid>
# {"deleted":"<uuid>"}
```

### `GET /sources`
```bash
curl http://localhost:8000/sources
```

### `POST /clear`
```bash
curl -X POST http://localhost:8000/clear
# {"evidence_deleted":42,"memories_deleted":5}
```

---

## Streamlit Dashboard

`dashboard.py` provides a point-and-click interface that connects to the running OpsMemory API.

```bash
# With the API already running on port 8000:
streamlit run tools/opsmemory/dashboard.py
# Opens at http://localhost:8501
```

The dashboard provides:

| Feature | Description |
|---|---|
| 📊 Live stats | Evidence total / unconsolidated / memory count, refreshed each page load |
| 📥 Ingest text | Paste any text and pick a source type |
| 📎 Upload file | Upload `.txt` or `.json` files directly |
| 🔍 Query | Natural-language search with expandable citation cards |
| 🧠 Memories | Browse consolidated memory records; delete individual items |
| 📋 Evidence | Browse raw evidence items; delete individual items |
| 🔄 Consolidate | One-click consolidation trigger from the sidebar |
| 🗑️ Clear all | Full reset from the sidebar (two-click confirmation) |

---

## Provider Layer

OpsMemory uses a pluggable provider abstraction for LLM generation and embeddings.
Providers are selected via environment variables — no code changes are needed to
switch between Anthropic, OpenAI, Bedrock, or local backends.

### SDK mode (default)

LiteLLM is used in-process as the SDK.  Set the provider and model:

```bash
# Use Claude via Anthropic directly
export OPSMEMORY_LLM_PROVIDER=litellm
export OPSMEMORY_LLM_MODEL=anthropic/claude-3-haiku-20240307
export ANTHROPIC_API_KEY=sk-ant-...

# Use OpenAI embeddings
export OPSMEMORY_EMBEDDING_PROVIDER=litellm
export OPSMEMORY_EMBEDDING_MODEL=openai/text-embedding-3-small
export OPENAI_API_KEY=sk-...
```

### Proxy mode (future / advanced)

To route all LLM and embedding calls through a LiteLLM proxy or any OpenAI-compatible
endpoint, set `OPSMEMORY_LITELLM_BASE_URL`:

```bash
export OPSMEMORY_LITELLM_BASE_URL=http://litellm-proxy:4000
# No provider key needed — the proxy handles authentication.
```

Business logic and OpsMemory internals are unchanged in proxy mode.

### Mock provider (dev / test)

The default provider (`mock`) generates deterministic random embeddings and canned
LLM responses with no external API calls.  Tests always use the mock provider.

```bash
# Explicit (also the default)
export OPSMEMORY_EMBEDDING_PROVIDER=mock
export OPSMEMORY_LLM_PROVIDER=mock
```

### Adding a new provider

Create a subclass of `BaseLLMProvider` or `BaseEmbeddingProvider` in
`providers/llm/` or `providers/embeddings/` respectively, then register it in
`providers/__init__.py`.  No other files need to change.

---

## FastMCP Server

OpsMemory exposes its full tool surface as MCP tools via FastMCP.

### Available tools

| Tool | Description |
|---|---|
| `memory_ingest_text` | Ingest free-form text evidence |
| `memory_query` | Semantic search over evidence + memories |
| `memory_status` | Return current store counts |
| `memory_consolidate` | Trigger a consolidation cycle |
| `memory_list_sources` | List registered data sources |
| `memory_sync_github_owner` | Run a GitHub ingestion sweep |
| `memory_delete_memory` | Delete a consolidated memory record by UUID |
| `memory_delete_evidence` | Delete a raw evidence item by UUID |

### Running the MCP server

**stdio transport** (for Claude Desktop / Cursor / IDE integrations):

```bash
# Start the OpsMemory API first
uvicorn tools.opsmemory.api.app:app --port 8000

# Start the MCP server (stdio)
python -m tools.opsmemory.mcp.server
# or
python tools/opsmemory/mcp/server.py
```

**SSE transport** (for debugging / HTTP-based clients):

```bash
python -m tools.opsmemory.mcp.server --transport sse --port 8100
```

### Claude Desktop configuration

Add to `~/Library/Application Support/Claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "opsmemory": {
      "command": "python",
      "args": ["-m", "tools.opsmemory.mcp.server"],
      "env": {
        "OPSMEMORY_API_URL": "http://localhost:8000"
      }
    }
  }
}
```

---

## Model Registry

`providers/model_registry.yaml` tracks approved and experimental models with their
capabilities (generation, embeddings, structured output, MCP-safe, production-approved).

Validate the registry:

```bash
python tools/opsmemory/scripts/validate_model_registry.py
```

---

## GitHub Connector

The GitHub connector (`connectors/github_connector.py`) runs a polling loop that:

1. Resolves the target owner (from `GITHUB_OWNER` or the authenticated token identity).
2. Detects whether the owner is a personal account or organisation and picks the correct API endpoint.
3. Lists repositories for the resolved owner.
4. Fetches recent commits and PRs for each repo.
5. Normalises each item into structured text.
6. POSTs to `OPSMEMORY_INGEST_URL`.

It uses exponential back-off retry (tenacity) on 429 / 5xx responses.

### Running standalone

```bash
python - <<'EOF'
import asyncio
from tools.opsmemory.connectors.github_connector import GitHubConnector, GitHubConnectorConfig

async def main():
    connector = GitHubConnector()
    stats = await connector.run_once()
    print(stats)

asyncio.run(main())
EOF
```

---

## Development Setup (without Docker)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Copy and configure environment
cp tools/opsmemory/.env.example .env
# Edit .env with your credentials

# 3. Start Postgres with pgvector (example via Docker)
docker run -d \
  -e POSTGRES_USER=opsmemory \
  -e POSTGRES_PASSWORD=opsmemory \
  -e POSTGRES_DB=opsmemory \
  -p 5432:5432 \
  pgvector/pgvector:pg16

# 4. Run migrations
python tools/opsmemory/scripts/migrate.py

# 5. Start the API
uvicorn tools.opsmemory.api.app:app --reload --port 8000

# 6. (Optional) Start the MCP server
python -m tools.opsmemory.mcp.server

# 7. Run tests
pytest tools/opsmemory/tests/ -v
```

---

## Production Notes

- **Embedding model**: Set `OPSMEMORY_EMBEDDING_PROVIDER=litellm` and `OPSMEMORY_EMBEDDING_MODEL` to your chosen model.  Update `embedding_dim` in `ConsolidationConfig` if the model uses a non-1536 dimension.
- **ivfflat index**: The migration script creates an ivfflat index on `evidence_embeddings.embedding`. ivfflat requires data to already exist before the index can select centroids. For empty-table deployments, use `hnsw` instead, or run a separate re-index step after loading representative data.
- **Secret redaction**: Patterns in `agent/redactor.py` cover common cases. Review and extend for your environment.
- **Consolidation interval**: Default is 30 minutes (`ConsolidationConfig.interval_seconds=1800`). Tune via environment or code.
- **Database pool**: Increase `DB_POOL_SIZE` and `DB_MAX_OVERFLOW` for high-throughput deployments.
- **Secret management**: Use [GitHub Actions secrets](https://docs.github.com/en/actions/security-guides/using-secrets-in-github-actions), AWS Secrets Manager, or HashiCorp Vault for production credentials.  Never commit `.env` files or API keys to version control.

---

## Repos Included by Default

All public repositories for the resolved owner are included by default.
- If `GITHUB_OWNER` is set, that username or organisation is used directly.
- If `GITHUB_OWNER` is unset, the connector resolves the owner from the authenticated `GITHUB_TOKEN` via `GET /user`.
- Organisation owners are supported — the connector detects org accounts automatically and uses the `/orgs/{owner}/repos` endpoint.
- Use `GITHUB_INCLUDE_REPOS` or `GITHUB_EXCLUDE_REPOS` to restrict the set.

> **Security note:** Supply `GITHUB_TOKEN` and any other secrets via environment variables, a `.env` file (not committed to source control), or a secure secret store such as [GitHub Actions secrets](https://docs.github.com/en/actions/security-guides/using-secrets-in-github-actions). Never commit tokens or credentials to version control.
