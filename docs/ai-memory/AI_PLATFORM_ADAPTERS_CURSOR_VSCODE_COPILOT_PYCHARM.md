# AI Platform Adapters: Cursor, VS Code, Copilot, PyCharm

Canonical source for applying this pack consistently across AI-enabled tooling.

## Adapter Rules
- Keep the standards in-repo and linkable; do not depend on hidden tool memory for critical policy.
- Route requests through the same order: universal -> router policy -> scoring -> confirmation -> topic standard.
- Tool choice may differ; policy does not.
- Generated suggestions must respect the v1 no-LLM-in-core-sync-path rule.

## Minimum Expectations Per Tool
- Cursor/VS Code/Copilot/PyCharm should surface links to canonical docs during planning and review.
- High-risk changes require visible confirmation and rollback notes regardless of editor.
- Local prompts may summarize a rule, but the canonical file remains the source of truth.

## Definition of Done
- Different editor agents converge on the same standards and approvals.
