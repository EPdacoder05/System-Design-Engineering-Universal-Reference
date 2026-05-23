"""Example flows for the Jarvis → OpsMemory integration.

These examples demonstrate the three canonical usage patterns:
1. Query memory before answering a user question.
2. Ingest session / task outcomes after completion.
3. Check status and list sources.

All examples use environment variables for configuration.  No secrets or
personal workflow details are included.

Run any example directly::

    python -m tools.opsmemory.integrations.jarvis.examples
"""

from __future__ import annotations

import asyncio
import logging
import os

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helper: skip examples that require a live server
# ---------------------------------------------------------------------------

def _should_run_live_examples() -> bool:
    """Return True unless ``OPSMEMORY_SKIP_LIVE=true`` is set.

    Set ``OPSMEMORY_SKIP_LIVE=true`` in CI or unit-test runs to skip examples
    that require a running OpsMemory instance.
    """
    return os.environ.get("OPSMEMORY_SKIP_LIVE", "false").lower() not in (
        "1", "true", "yes"
    )


# ---------------------------------------------------------------------------
# Example 1: Query memory before answering
# ---------------------------------------------------------------------------


async def example_query_before_answering() -> None:
    """Retrieve relevant context from OpsMemory before generating a response.

    A Jarvis assistant should call this pattern at the start of each task so
    it has access to relevant historical evidence.
    """
    from tools.opsmemory.integrations.jarvis.mcp_client import OpsMemoryClient

    client = OpsMemoryClient()

    user_question = "What services were deployed recently?"
    log.info("Querying OpsMemory for context: %s", user_question)

    result = await client.query_memory(user_question, limit=5)

    log.info("Memory context retrieved:")
    log.info("  Answer hint: %s", result.get("answer", ""))
    log.info("  Citations  : %d items", len(result.get("citations", [])))
    log.info("  Memories   : %d records", len(result.get("memories", [])))

    # The assistant would now incorporate `result["citations"]` and
    # `result["memories"]` into its context window before answering.
    return result  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Example 2: Ingest session outcome after task completion
# ---------------------------------------------------------------------------


async def example_ingest_after_task() -> None:
    """Persist a task / session outcome into OpsMemory for future retrieval.

    Call this pattern after each task or conversation turn completes so the
    result is stored as searchable evidence.
    """
    from tools.opsmemory.integrations.jarvis.mcp_client import OpsMemoryClient

    client = OpsMemoryClient()

    # Hypothetical task outcome — replace with real data in your integration.
    outcome_text = (
        "Completed: deployed service-api v2.1.0 to staging environment. "
        "All health checks passed. Rollout took 4 minutes. "
        "Triggered by automated pipeline on branch 'release/v2.1'."
    )

    log.info("Ingesting session outcome into OpsMemory...")
    response = await client.ingest_session_outcome(
        text=outcome_text,
        session_id="task-deploy-2026-03-15-001",
        author="jarvis-assistant",
        occurred_at="2026-03-15T14:30:00Z",
    )

    log.info("Ingested. evidence_id=%s", response.get("evidence_id"))
    return response  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Example 3: Request status and sources
# ---------------------------------------------------------------------------


async def example_request_status() -> None:
    """Retrieve current OpsMemory store counts and registered sources."""
    from tools.opsmemory.integrations.jarvis.mcp_client import OpsMemoryClient

    client = OpsMemoryClient()

    log.info("Fetching OpsMemory status...")
    status = await client.get_status()
    log.info(
        "Status — evidence_total=%s unconsolidated=%s memories=%s",
        status.get("evidence_total"),
        status.get("evidence_unconsolidated"),
        status.get("memories"),
    )

    log.info("Fetching registered sources...")
    sources = await client.list_sources()
    log.info("Sources: %d registered", len(sources))
    for src in sources[:5]:
        log.info("  %s / %s (%s)", src.get("owner"), src.get("repo"), src.get("source_type"))

    return status, sources  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


async def main() -> None:
    if not _should_run_live_examples():
        log.info("OPSMEMORY_SKIP_LIVE=true — skipping live server examples.")
        log.info(
            "Start OpsMemory (docker compose up) and unset OPSMEMORY_SKIP_LIVE "
            "to run these examples against a real server."
        )
        return

    log.info("=== Example 1: Query memory before answering ===")
    try:
        await example_query_before_answering()
    except Exception as exc:
        log.error("Example 1 failed: %s", exc)

    log.info("=== Example 2: Ingest session outcome ===")
    try:
        await example_ingest_after_task()
    except Exception as exc:
        log.error("Example 2 failed: %s", exc)

    log.info("=== Example 3: Status and sources ===")
    try:
        await example_request_status()
    except Exception as exc:
        log.error("Example 3 failed: %s", exc)


if __name__ == "__main__":
    asyncio.run(main())
