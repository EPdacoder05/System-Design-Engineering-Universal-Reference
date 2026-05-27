# /lint — Health-check wiki state against live system reality

Cross-check the compiled wiki state against the actual running system.  Report any drift.

Checks to perform:

1. **Docker containers** — for every `\`service-name\`` mentioned in the wiki, verify the container is actually running via `docker ps`.
2. **Cron jobs** — for every scheduled job referenced in the wiki, verify it exists in `crontab -l` or the system cron directory.
3. **File paths** — for every `\`/path/to/file\`` referenced, verify the file or directory exists.
4. **Raw backlog** — count files in `vault/raw/` that have not yet been ingested and report the count.

**Output format:**

```
## Lint Report — <ISO-8601 timestamp>

### ✅ Passing
- <item>: OK

### ⚠️ Drift Detected
- <item>: expected <X>, found <Y>

### 📥 Raw Backlog
- <N> files pending ingestion in vault/raw/
```

Do not modify any wiki files.  Lint is read-only.
