# Jarvis MCP Client Integration

Reusable scaffolding for connecting a Jarvis-style assistant / orchestrator
to OpsMemory.

---

## Overview

This directory provides a generic, public-safe MCP client adapter that enables
any assistant or orchestrator to:

- **Query memory** before generating a response (provides historical context).
- **Ingest session outcomes** after task completion (persists results for future
  retrieval).
- **Check status and list sources** (introspection / health visibility).

The implementation uses the OpsMemory **REST API** as the transport layer (always
available) with a clearly marked TODO stub for native MCP protocol wiring if you
prefer the `fastmcp` / official `mcp` SDK transport.

---

## Files

| File | Purpose |
|------|---------|
| `config.py` | `JarvisClientConfig` — reads all settings from environment variables |
| `mcp_client.py` | `OpsMemoryClient` — async HTTP wrapper for OpsMemory tools |
| `examples.py` | Three canonical usage examples (query, ingest, status) |

---

## Quick Start

### 1. Configure environment

Copy the example env file and fill in your values:

```bash
cp tools/opsmemory/.env.jarvis.example .env
# Edit .env — set OPSMEMORY_API_URL, OPSMEMORY_API_KEY, etc.
```

### 2. Use the client

```python
import asyncio
from tools.opsmemory.integrations.jarvis.mcp_client import OpsMemoryClient

async def handle_user_request(question: str) -> str:
    client = OpsMemoryClient()          # reads config from env

    # Step 1 — retrieve relevant context from memory
    context = await client.query_memory(question, limit=5)
    citations = context.get("citations", [])
    memories = context.get("memories", [])

    # Step 2 — generate answer using citations + memories as context
    # ... your LLM call here ...
    answer = f"Based on {len(citations)} evidence items: ..."

    # Step 3 — ingest the outcome for future retrieval
    await client.ingest_session_outcome(
        text=f"Q: {question}\nA: {answer}",
        session_id="session-001",
        author="jarvis-assistant",
    )

    return answer

asyncio.run(handle_user_request("What services were deployed recently?"))
```

### 3. Run examples

```bash
# Start OpsMemory first (see tools/opsmemory/docker/)
uvicorn tools.opsmemory.api.app:app --port 8000

# Run examples (requires a running OpsMemory instance)
python -m tools.opsmemory.integrations.jarvis.examples
```

---

## Configuration Reference

All settings are read from environment variables.  See `.env.jarvis.example`
for a full annotated template.

| Variable | Default | Description |
|---|---|---|
| `OPSMEMORY_API_URL` | `http://localhost:8000` | OpsMemory REST API base URL |
| `OPSMEMORY_API_KEY` | _(none)_ | Bearer token (required when auth is enabled) |
| `OPSMEMORY_MCP_TOKEN` | _(falls back to API_KEY)_ | Optional separate MCP token |
| `OPSMEMORY_MCP_URL` | `http://localhost:8100` | OpsMemory MCP SSE server URL |
| `JARVIS_MEMORY_QUERY_LIMIT` | `5` | Max evidence items per query |
| `JARVIS_SESSION_SOURCE_TYPE` | `jarvis_session` | Source type for ingested outcomes |
| `OPSMEMORY_REQUEST_TIMEOUT` | `30` | HTTP request timeout (seconds) |

---

## Auth

When the OpsMemory API is deployed with `OPSMEMORY_REQUIRE_API_KEY=true`,
all requests must include:

```
Authorization: Bearer <OPSMEMORY_API_KEY>
```

The `OpsMemoryClient` adds this header automatically when `OPSMEMORY_API_KEY`
is set in the environment.  For local development with auth disabled, leave
`OPSMEMORY_API_KEY` empty.

---

## MCP Transport (Advanced)

The current implementation uses the REST API.  For native MCP protocol transport
(stdio or SSE), see the `TODO` block at the bottom of `mcp_client.py`.

**SSE transport setup:**

```bash
# Start the OpsMemory MCP server in SSE mode
python -m tools.opsmemory.mcp.server --transport sse --port 8100

# Point your MCP client at http://localhost:8100
export OPSMEMORY_MCP_URL=http://localhost:8100
```

**stdio transport:**

Add to your MCP client configuration (e.g. Claude Desktop
`claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "opsmemory": {
      "command": "python",
      "args": ["-m", "tools.opsmemory.mcp.server"],
      "env": {
        "OPSMEMORY_API_URL": "http://localhost:8000",
        "OPSMEMORY_API_KEY": "${OPSMEMORY_API_KEY}"
      }
    }
  }
}
```

---

## Public-Repo Safety

- No personal usernames, private endpoints, or internal workflow details.
- All config via environment variables or `.env` files (never committed).
- Example flows use generic, placeholder-based data only.
