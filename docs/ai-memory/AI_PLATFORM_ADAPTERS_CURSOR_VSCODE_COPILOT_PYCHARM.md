# AI Platform Adapters: Cursor, VS Code, Copilot, PyCharm

Canonical source for applying this pack consistently across AI-enabled tooling.

## Adapter Rules
- Keep the standards in-repo and linkable; do not depend on hidden tool memory for critical policy.
- Use the smallest capable agent/tool for the task; escalate to stronger models only for genuinely cross-cutting or high-risk complexity.
- Route requests through the same order: universal -> router policy -> scoring -> confirmation -> topic standard.
- Tool choice may differ; policy does not.
- Generated suggestions must respect the v1 no-LLM-in-core-sync-path rule.
- Treat generated output as a first draft that must be understood, simplified, and owned before merge.

## Minimum Expectations Per Tool
- Cursor/VS Code/Copilot/PyCharm should surface links to canonical docs during planning and review.
- High-risk changes require visible confirmation and rollback notes regardless of editor.
- Local prompts may summarize a rule, but the canonical file remains the source of truth.
- Keep prompts and summaries concise; low-token guidance should still preserve the canonical links and decision trail.

## Definition of Done
- Different editor agents converge on the same standards and approvals.
