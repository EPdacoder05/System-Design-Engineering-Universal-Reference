# OpsMemory — Always-On Memory Agent

> Persistent, searchable memory for your GitHub activity and operational events, backed by PostgreSQL + pgvector.

---

## Overview

OpsMemory is an always-on background agent that:

1. **Ingests** evidence from any source (GitHub commits, PRs, manual text, files) via HTTP or an inbox directory.
2. **Redacts** secrets automatically before anything reaches the database.
3. **Embeds** each evidence item using a vector model (mock by default; plug in OpenAI or any other provider).
4. **Consolidates** batches of evidence into higher-level Memory records on a configurable timer.
5. **Answers** natural-language queries using semantic search over the embedding store.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
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
          ┌─────────▼───────────┐
          │   PostgreSQL + pgvector │
          │  evidence_items         │
          │  evidence_embeddings    │
          │  memories               │
          │  consolidation_runs     │
          │  sources                │
          └─────────────────────────┘

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

# 2. Start Postgres + OpsMemory API
docker compose up -d

# 3. Verify the API is healthy
curl http://localhost:8000/health

# 4. Ingest some text
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{"text": "Deployed v2.3.1 to production at 14:00 UTC", "source_type": "manual"}'

# 5. Query
curl "http://localhost:8000/query?q=What+was+deployed%3F"

# 6. Trigger consolidation
curl -X POST http://localhost:8000/consolidate

# 7. Dashboard (optional — requires streamlit)
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

# 2. Start Postgres with pgvector (example via Docker)
docker run -d \
  -e POSTGRES_USER=opsmemory \
  -e POSTGRES_PASSWORD=opsmemory \
  -e POSTGRES_DB=opsmemory \
  -p 5432:5432 \
  pgvector/pgvector:pg16

# 3. Run migrations
python tools/opsmemory/scripts/migrate.py

# 4. Start the API
uvicorn tools.opsmemory.api.app:app --reload --port 8000

# 5. Run tests
pytest tools/opsmemory/tests/ -v
```

---

## Production Notes

- **Embedding model**: Replace `generate_embedding` in `agent/consolidator.py` with a real API call (e.g. `openai.embeddings.create`). Update `embedding_dim` in `ConsolidationConfig` to match.
- **ivfflat index**: The migration script creates an ivfflat index on `evidence_embeddings.embedding`. ivfflat requires data to already exist before the index can select centroids. For empty-table deployments, use `hnsw` instead, or run a separate re-index step after loading representative data.
- **Secret redaction**: Patterns in `agent/redactor.py` cover common cases. Review and extend for your environment.
- **Consolidation interval**: Default is 30 minutes (`ConsolidationConfig.interval_seconds=1800`). Tune via environment or code.
- **Database pool**: Increase `DB_POOL_SIZE` and `DB_MAX_OVERFLOW` for high-throughput deployments.

---

## Repos Included by Default

All public repositories for the resolved owner are included by default.
- If `GITHUB_OWNER` is set, that username or organisation is used directly.
- If `GITHUB_OWNER` is unset, the connector resolves the owner from the authenticated `GITHUB_TOKEN` via `GET /user`.
- Organisation owners are supported — the connector detects org accounts automatically and uses the `/orgs/{owner}/repos` endpoint.
- Use `GITHUB_INCLUDE_REPOS` or `GITHUB_EXCLUDE_REPOS` to restrict the set.

> **Security note:** Supply `GITHUB_TOKEN` and any other secrets via environment variables, a `.env` file (not committed to source control), or a secure secret store such as [GitHub Actions secrets](https://docs.github.com/en/actions/security-guides/using-secrets-in-github-actions). Never commit tokens or credentials to version control.
