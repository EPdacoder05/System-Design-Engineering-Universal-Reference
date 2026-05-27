# /query — Answer questions from compiled wiki state

Given a natural-language question, search `vault/wiki/` Markdown files for the answer.

Rules:
1. **Never read `vault/raw/`** — query compiled state only.
2. Extract the answer from the wiki pages and cite the exact file and section.
3. If the answer is not found in the wiki, respond: "No record found in wiki state. Run `/ingest` to compile the latest raw data."
4. Return a structured response:

```
Answer: <concise answer>
Source: [[<wiki page>]] — "<section heading>"
Confidence: high / medium / low
```

Use `low` confidence when the relevant wiki entry is marked `status: needs_review`.
