# /log — Quick-capture a note into the journal

Append a timestamped entry to `vault/wiki/journal.md`.

Usage: `/log <your note text>`

The entry is formatted as:

```markdown
### <ISO-8601 UTC timestamp>

<your note text>

---
```

This command is intentionally lightweight — no parsing, no cross-linking.  
Use it for dev thoughts, architecture ideas, temporary override notes, or anything you want to capture quickly without a full ingest cycle.

After appending, confirm: "Logged to [[journal]] at <timestamp>."
