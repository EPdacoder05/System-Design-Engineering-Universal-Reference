# Jarvis — Migration Record

**Status:** Migrated to `EPdacoder05/Jarvis-AI-Assistant` (not deleted)

---

## What moved

| Component | Was here | Now at |
|-----------|----------|--------|
| MCP server (`mcp/`) | `tools/jarvis/mcp/` | `EPdacoder05/Jarvis-AI-Assistant/mcp/` |
| Vault (`vault/`) | `tools/jarvis/vault/` | `EPdacoder05/Jarvis-AI-Assistant/vault/` |
| Slash commands (`.claude/commands/`) | `tools/jarvis/.claude/commands/` | `EPdacoder05/Jarvis-AI-Assistant/.claude/commands/` |
| Agent prompt (`CLAUDE.md`) | `tools/jarvis/CLAUDE.md` | `EPdacoder05/Jarvis-AI-Assistant/CLAUDE.md` |
| Tests (`tests/`) | `tools/jarvis/tests/` | `EPdacoder05/Jarvis-AI-Assistant/tests/` |

---

## Migration guardrail (important)

This phase is a **repo extraction**, not a product deletion.

All product artifacts listed above must exist and be maintained in the Jarvis repository:

- https://github.com/EPdacoder05/Jarvis-AI-Assistant

If anything appears "removed" in this repository, treat it as **relocated ownership** and validate it in the Jarvis repo migration history.

---

## What stays here

The **reusable MCP integration pattern** showing how `tools/opsmemory/` connects to an external consumer via MCP is preserved as a reference:

- `tools/opsmemory/integrations/jarvis/` — example integration config and client
- `tools/opsmemory/tests/test_jarvis_client.py` — integration test for the MCP client pattern

These files demonstrate the integration contract, not the Jarvis product itself.

---

## Why this split

Jarvis is a product (homelab intelligence with specific vault structure, event sources, and dashboard output). Its MCP server and vault behavior are product code, not reusable patterns.

`tools/opsmemory/` is the reusable memory/retrieval primitive. Jarvis is one consumer of that primitive. Keeping the primitive here and moving the consumer to its own repo is the correct ownership model.

---

## Links

- Jarvis product repo: https://github.com/EPdacoder05/Jarvis-AI-Assistant
- opsmemory (memory/retrieval primitive): `tools/opsmemory/` in this repo
- Migration map: `MULTI_REPO_DEPLOYMENT.md` in this repo
