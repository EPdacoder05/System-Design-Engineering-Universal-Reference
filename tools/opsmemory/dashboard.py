"""Streamlit dashboard for OpsMemory — Always-On Memory Agent.

Connects to the OpsMemory FastAPI backend and provides a visual interface for:
  - Ingesting text and uploading files
  - Querying memory with natural language
  - Browsing and deleting evidence items
  - Browsing and deleting consolidated memories
  - Triggering consolidation on demand
  - Viewing status / statistics

Usage::

    streamlit run tools/opsmemory/dashboard.py

Set the API URL in the sidebar (defaults to http://localhost:8000).
"""

from __future__ import annotations

import json
from typing import Any, Dict, Optional

import httpx
import streamlit as st

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="OpsMemory",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# API helper functions
# ---------------------------------------------------------------------------


def _get(path: str, params: Optional[Dict[str, Any]] = None) -> Any:
    """GET request to the OpsMemory API."""
    resp = httpx.get(f"{st.session_state['api_url']}{path}", params=params, timeout=15)
    resp.raise_for_status()
    return resp.json()


def _post(path: str, json_body: Optional[Dict[str, Any]] = None) -> Any:
    """POST request to the OpsMemory API."""
    resp = httpx.post(
        f"{st.session_state['api_url']}{path}",
        json=json_body,
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


def _delete(path: str) -> Any:
    """DELETE request to the OpsMemory API."""
    resp = httpx.delete(f"{st.session_state['api_url']}{path}", timeout=15)
    resp.raise_for_status()
    return resp.json()


# ---------------------------------------------------------------------------
# Sidebar — configuration & live stats
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("🧠 OpsMemory")
    st.caption("Always-On Memory Agent")
    st.divider()

    api_url = st.text_input(
        "API URL",
        value=st.session_state.get("api_url", "http://localhost:8000"),
        help="Base URL of the running OpsMemory FastAPI server.",
    )
    st.session_state["api_url"] = api_url.rstrip("/")

    st.divider()
    st.subheader("Stats")

    try:
        stat = _get("/status")
        col1, col2, col3 = st.columns(3)
        col1.metric("Evidence", stat.get("evidence_total", 0))
        col2.metric("Unconsolidated", stat.get("evidence_unconsolidated", 0))
        col3.metric("Memories", stat.get("memories", 0))
        st.caption("✅ API reachable")
    except Exception as exc:
        st.error(f"API unreachable: {exc}")

    st.divider()
    if st.button("🔄 Consolidate now", use_container_width=True):
        try:
            result = _post("/consolidate")
            st.success(
                f"Created {result['memories_created']} memories "
                f"from {result['evidence_consolidated']} items."
            )
            st.rerun()
        except Exception as exc:
            st.error(f"Consolidation failed: {exc}")

    if st.button("🗑️ Clear all (reset)", use_container_width=True, type="secondary"):
        if st.session_state.get("confirm_clear"):
            try:
                result = _post("/clear")
                st.success(
                    f"Deleted {result['evidence_deleted']} evidence items "
                    f"and {result['memories_deleted']} memories."
                )
                st.session_state["confirm_clear"] = False
                st.rerun()
            except Exception as exc:
                st.error(f"Clear failed: {exc}")
        else:
            st.session_state["confirm_clear"] = True
            st.warning("Click again to confirm full reset.")

# ---------------------------------------------------------------------------
# Main content — tabs
# ---------------------------------------------------------------------------

tab_ingest, tab_query, tab_memories, tab_evidence = st.tabs(
    ["📥 Ingest", "🔍 Query", "🧠 Memories", "📋 Evidence"]
)

# ── Tab 1: Ingest ──────────────────────────────────────────────────────────
with tab_ingest:
    st.header("Ingest")
    st.write("Add new evidence to the memory store.")

    ingest_mode = st.radio("Input mode", ["Text", "File upload"], horizontal=True)

    if ingest_mode == "Text":
        with st.form("ingest_text_form"):
            text_input = st.text_area("Content", height=150, placeholder="Paste any text…")
            source_type = st.selectbox(
                "Source type",
                ["manual", "note", "meeting", "article", "github_commit", "github_pr", "other"],
            )
            source_ref = st.text_input("Source reference (URL, filename, etc.)", value="")
            author = st.text_input("Author (optional)", value="")
            submitted = st.form_submit_button("Ingest", type="primary")

        if submitted:
            if not text_input.strip():
                st.error("Content is required.")
            else:
                try:
                    payload: Dict[str, Any] = {
                        "text": text_input,
                        "source_type": source_type,
                    }
                    if source_ref:
                        payload["source_ref"] = source_ref
                    if author:
                        payload["author"] = author
                    result = _post("/ingest", payload)
                    st.success(f"Ingested. Evidence ID: `{result['evidence_id']}`")
                    if result.get("redacted"):
                        st.warning("⚠️ Secrets were automatically redacted from the text.")
                except Exception as exc:
                    st.error(f"Ingest failed: {exc}")

    else:
        uploaded = st.file_uploader(
            "Upload a text file (.txt or .json IngestPayload)",
            type=["txt", "json"],
            accept_multiple_files=False,
        )
        if uploaded is not None:
            raw = uploaded.read().decode("utf-8", errors="replace")
            st.text_area("Preview", value=raw[:1000] + ("…" if len(raw) > 1000 else ""), height=120)
            if st.button("Ingest file", type="primary"):
                try:
                    if uploaded.name.endswith(".json"):
                        data = json.loads(raw)
                        result = _post("/ingest", data)
                    else:
                        result = _post(
                            "/ingest",
                            {
                                "text": raw,
                                "source_type": "file",
                                "source_ref": uploaded.name,
                            },
                        )
                    st.success(f"Ingested. Evidence ID: `{result['evidence_id']}`")
                    if result.get("redacted"):
                        st.warning("⚠️ Secrets were automatically redacted from the text.")
                except Exception as exc:
                    st.error(f"Ingest failed: {exc}")

# ── Tab 2: Query ───────────────────────────────────────────────────────────
with tab_query:
    st.header("Query")
    st.write("Ask any question — the agent searches over evidence and memories.")

    with st.form("query_form"):
        query_text = st.text_input("Question", placeholder="What should I focus on?")
        limit = st.slider("Max citations", min_value=1, max_value=20, value=5)
        search = st.form_submit_button("Search", type="primary")

    if search:
        if not query_text.strip():
            st.error("Please enter a question.")
        else:
            try:
                result = _get("/query", params={"q": query_text, "limit": limit})
                st.info(f"**Answer:** {result.get('answer', '—')}")

                citations = result.get("citations", [])
                if citations:
                    st.subheader(f"Citations ({len(citations)})")
                    for c in citations:
                        with st.expander(
                            f"[{c.get('source_type', '?')}] "
                            + (c.get("source_ref") or c.get("evidence_id", ""))[:80]
                        ):
                            st.write("**Evidence ID:**", c.get("evidence_id"))
                            if c.get("repo"):
                                st.write("**Repo:**", c["repo"])
                            if c.get("occurred_at"):
                                st.write("**Occurred at:**", c["occurred_at"])
                            excerpt = c.get("excerpt") or ""
                            if excerpt:
                                st.text(excerpt[:500])

                memories = result.get("memories", [])
                if memories:
                    st.subheader(f"Memories ({len(memories)})")
                    for m in memories:
                        st.markdown(f"- **{m.get('id', '')}**: {m.get('summary', '')[:200]}")
            except Exception as exc:
                st.error(f"Query failed: {exc}")

# ── Tab 3: Memories ────────────────────────────────────────────────────────
with tab_memories:
    st.header("Memories")
    st.write("Consolidated memory records synthesized from evidence batches.")

    if st.button("Refresh", key="refresh_memories"):
        st.rerun()

    try:
        memories = _get("/memories")
        if not memories:
            st.info("No memories yet. Ingest some evidence then trigger consolidation.")
        else:
            for mem in memories:
                with st.expander(
                    f"🧠 {mem.get('id', '')[:8]}… — {(mem.get('summary') or '')[:100]}"
                ):
                    st.write("**ID:**", mem.get("id"))
                    st.write("**Created:**", mem.get("created_at", "—"))
                    st.text_area(
                        "Content",
                        value=mem.get("content", ""),
                        height=120,
                        key=f"mem_content_{mem.get('id')}",
                        disabled=True,
                    )
                    if st.button(
                        "🗑️ Delete this memory",
                        key=f"del_mem_{mem.get('id')}",
                    ):
                        try:
                            _delete(f"/memories/{mem['id']}")
                            st.success("Deleted.")
                            st.rerun()
                        except Exception as exc:
                            st.error(f"Delete failed: {exc}")
    except Exception as exc:
        st.error(f"Failed to load memories: {exc}")

# ── Tab 4: Evidence ────────────────────────────────────────────────────────
with tab_evidence:
    st.header("Evidence")
    st.write("Raw ingested evidence items (before or after consolidation).")

    if st.button("Refresh", key="refresh_evidence"):
        st.rerun()

    try:
        items = _get("/evidence")
        if not items:
            st.info("No evidence yet. Use the Ingest tab to add some.")
        else:
            for item in items:
                label = (
                    f"[{item.get('source_type', '?')}] "
                    + (item.get("source_ref") or item.get("evidence_id", ""))[:60]
                    + (" ✅" if item.get("consolidated") else " ⏳")
                )
                with st.expander(label):
                    st.write("**Evidence ID:**", item.get("evidence_id") or item.get("id"))
                    st.write("**Source type:**", item.get("source_type"))
                    if item.get("repo"):
                        st.write("**Repo:**", item["repo"])
                    if item.get("author"):
                        st.write("**Author:**", item["author"])
                    if item.get("occurred_at"):
                        st.write("**Occurred at:**", item["occurred_at"])
                    st.write("**Ingested at:**", item.get("ingested_at", "—"))
                    st.write("**Consolidated:**", "Yes" if item.get("consolidated") else "No")
                    excerpt = item.get("excerpt") or ""
                    if excerpt:
                        st.text(excerpt[:500])
                    if st.button(
                        "🗑️ Delete this item",
                        key=f"del_ev_{item.get('id')}",
                    ):
                        try:
                            _delete(f"/evidence/{item['id']}")
                            st.success("Deleted.")
                            st.rerun()
                        except Exception as exc:
                            st.error(f"Delete failed: {exc}")
    except Exception as exc:
        st.error(f"Failed to load evidence: {exc}")
