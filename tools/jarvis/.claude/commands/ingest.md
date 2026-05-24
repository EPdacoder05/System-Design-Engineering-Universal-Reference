# /ingest — Compile raw events into wiki dashboards

Scan `vault/raw/` for new files that have not yet been ingested.  For each new file:

1. Parse the raw payload (JSON, CSV, or plain text).
2. Extract the key fields relevant to the domain (see `CLAUDE.md` Domain Context table).
3. Append a clean Markdown summary entry to the correct `wiki/` dashboard.
4. If any errors are present in the payload, also append an entry to `wiki/errors.md` with a wikilink back to the source dashboard.
5. Mark the file as processed by recording its `native_id` (filename + mtime) so re-runs are idempotent.

**Output format** for each wiki entry:

```markdown
### <ISO-8601 timestamp> — <brief title>

| Field | Value |
|-------|-------|
| source | `<relative path to raw file>` |
| status | success / partial / failed |
| <domain-specific fields> | … |
```

Do not write to `vault/raw/` under any circumstances.
