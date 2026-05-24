"""Jarvis /query tool.

Searches compiled wiki state (vault/wiki/) for answers to natural-language
questions.  Never reads vault/raw/.

Strategy
--------
1. Load all Markdown files from ``vault/wiki/``.
2. Split each file into section chunks (split on ``###`` headings).
3. Find sections whose text contains any of the query tokens (case-insensitive).
4. Return the best-matching section with citation metadata.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional


def _vault_path() -> Path:
    env = os.environ.get("JARVIS_VAULT_PATH")
    if env:
        return Path(env)
    return Path(__file__).parent.parent.parent / "vault"


def _load_wiki_files(wiki_root: Path) -> Dict[str, str]:
    """Return {filename: full_text} for every .md file in wiki_root."""
    docs: Dict[str, str] = {}
    if not wiki_root.exists():
        return docs
    for md_file in sorted(wiki_root.glob("*.md")):
        try:
            docs[md_file.name] = md_file.read_text(encoding="utf-8")
        except OSError:
            pass
    return docs


def _split_sections(text: str) -> List[Dict[str, str]]:
    """Split a Markdown document into sections by ### headings."""
    sections: List[Dict[str, str]] = []
    current_heading = "(preamble)"
    current_lines: List[str] = []

    for line in text.splitlines():
        if line.startswith("### "):
            if current_lines:
                sections.append({"heading": current_heading, "body": "\n".join(current_lines)})
            current_heading = line[4:].strip()
            current_lines = []
        else:
            current_lines.append(line)

    if current_lines:
        sections.append({"heading": current_heading, "body": "\n".join(current_lines)})

    return sections


def _score_section(section: Dict[str, str], tokens: List[str]) -> int:
    """Return count of query tokens found in the section heading + body."""
    text = (section["heading"] + " " + section["body"]).lower()
    return sum(1 for t in tokens if t in text)


async def jarvis_query(
    question: str,
    vault_path: str = "",
    limit: int = 3,
) -> Dict[str, Any]:
    """Answer a question from compiled wiki state.

    Parameters
    ----------
    question:
        Natural-language question (e.g. ``"Did the SSD transfer finish last night?"``).
    vault_path:
        Absolute path to the vault root.  Defaults to ``JARVIS_VAULT_PATH`` env var
        or the default ``tools/jarvis/vault/`` directory.
    limit:
        Maximum number of matching sections to return (clamped to 1–20).

    Returns
    -------
    dict with ``question``, ``answer`` (best match text), ``citations`` list, and
    ``confidence`` (``"high"`` / ``"medium"`` / ``"low"`` / ``"none"``).
    """
    limit = max(1, min(20, limit))
    vault = Path(vault_path) if vault_path else _vault_path()
    wiki_root = vault / "wiki"

    docs = _load_wiki_files(wiki_root)
    if not docs:
        return {
            "question": question,
            "answer": "No wiki state found. Run jarvis_ingest to compile raw data.",
            "citations": [],
            "confidence": "none",
        }

    # Tokenise the question (words ≥ 3 chars, lower-cased)
    tokens = [t.lower() for t in re.findall(r"\w+", question) if len(t) >= 3]

    results: List[Dict[str, Any]] = []
    for filename, text in docs.items():
        for section in _split_sections(text):
            score = _score_section(section, tokens)
            if score > 0:
                results.append(
                    {
                        "score": score,
                        "file": filename,
                        "heading": section["heading"],
                        "body": section["body"].strip(),
                    }
                )

    if not results:
        return {
            "question": question,
            "answer": "No record found in wiki state. Run jarvis_ingest to compile the latest raw data.",
            "citations": [],
            "confidence": "none",
        }

    results.sort(key=lambda r: r["score"], reverse=True)
    top = results[:limit]

    best = top[0]
    confidence = "high" if best["score"] >= 3 else "medium" if best["score"] >= 2 else "low"

    # Check if best result is marked needs_review
    if "needs_review" in best["body"].lower():
        confidence = "low"

    wiki_stem = best["file"].replace(".md", "")
    answer = (
        f"[[{wiki_stem}]] — \"{best['heading']}\"\n\n"
        + best["body"][:800]
        + ("…" if len(best["body"]) > 800 else "")
    )

    citations = [
        {"file": r["file"], "section": r["heading"], "score": r["score"]}
        for r in top
    ]

    return {
        "question": question,
        "answer": answer,
        "citations": citations,
        "confidence": confidence,
    }
