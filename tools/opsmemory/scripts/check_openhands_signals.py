"""
Ecosystem Signal Collector for OpsMemory

Queries GitHub APIs to gather signals about:
  - BerriAI/litellm  — upstream dependency source of truth
  - OpenHands/litellm — downstream ecosystem observer (NOT a dependency source)

Policy
------
  BerriAI/litellm  →  dependency source, version pin, security patches
  OpenHands/litellm → ecosystem signal only; integration ideas, drift tracking

Usage
-----
    python check_openhands_signals.py [--output PATH] [--verbose]

Environment
-----------
    GITHUB_TOKEN   Optional. Raises API rate limit from 60 to 5 000 req/h.

Output
------
    Writes ecosystem_signals.json to tools/opsmemory/providers/ (or --output).
    Prints a human-readable summary to stdout.
    Exits with code 1 if notable signals warrant human review (used by workflow).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

# ── paths ────────────────────────────────────────────────────────────────────
_SCRIPT_DIR = Path(__file__).parent
_DEFAULT_OUTPUT = _SCRIPT_DIR.parent / "providers" / "ecosystem_signals.json"

# ── files to watch for integration-pattern changes in OpenHands fork ─────────
WATCHED_FILES = [
    "litellm/__init__.py",
    "litellm/main.py",
    "litellm/router.py",
    "litellm/utils.py",
    "model_prices_and_context_window.json",
    "litellm/integrations/",
]

# ── notable-change thresholds ─────────────────────────────────────────────────
DRIFT_ALERT_THRESHOLD = 200   # open issue if divergence grew by this many commits


# ── GitHub API helper ─────────────────────────────────────────────────────────

def _github_api(path: str, token: Optional[str] = None) -> Any:
    """Call the GitHub REST API and return parsed JSON.

    Raises ``urllib.error.URLError`` on network errors and
    ``RuntimeError`` on non-200 responses.
    """
    url = f"https://api.github.com/{path.lstrip('/')}"
    req = urllib.request.Request(url)
    req.add_header("Accept", "application/vnd.github.v3+json")
    req.add_header("User-Agent", "opsmemory-ecosystem-watch/1.0")
    if token:
        req.add_header("Authorization", f"token {token}")
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"GitHub API {path!r} returned HTTP {exc.code}") from exc


# ── signal collectors ─────────────────────────────────────────────────────────

def _collect_berri(token: Optional[str]) -> Dict[str, Any]:
    """Collect signals for BerriAI/litellm (upstream source of truth)."""
    out: Dict[str, Any] = {}
    try:
        release = _github_api("repos/BerriAI/litellm/releases/latest", token)
        branch = _github_api("repos/BerriAI/litellm/branches/main", token)
        out = {
            "latest_release_tag": release.get("tag_name"),
            "latest_release_date": release.get("published_at"),
            "release_url": release.get("html_url"),
            "main_branch_sha": branch["commit"]["sha"],
            "main_branch_updated": branch["commit"]["commit"]["committer"]["date"],
        }
    except Exception as exc:
        out["error"] = str(exc)
    return out


def _collect_openhands(token: Optional[str]) -> Dict[str, Any]:
    """Collect signals for OpenHands/litellm (ecosystem observer)."""
    out: Dict[str, Any] = {}
    try:
        repo = _github_api("repos/OpenHands/litellm", token)
        branch = _github_api("repos/OpenHands/litellm/branches/main", token)
        out = {
            "head_sha": branch["commit"]["sha"],
            "last_updated": repo.get("updated_at"),
            "pushed_at": repo.get("pushed_at"),
            "open_issues_count": repo.get("open_issues_count"),
            "fork_of": "BerriAI/litellm",
            "role": "ecosystem_signal_only",
        }
    except Exception as exc:
        out["error"] = str(exc)
        return out

    # Try commit-divergence via GitHub compare API
    try:
        berri_data = _collect_berri(token)
        berri_sha = berri_data.get("main_branch_sha", "main")
        oh_sha = out.get("head_sha", "main")
        compare = _github_api(
            f"repos/BerriAI/litellm/compare/{oh_sha}...{berri_sha}", token
        )
        out["commits_behind"] = compare.get("behind_by")
        out["commits_ahead"] = compare.get("ahead_by")
    except Exception as exc:
        out["divergence_note"] = f"Could not compute divergence: {exc}"

    return out


def _collect_watched_files(token: Optional[str]) -> Dict[str, Any]:
    """Check recent activity in OpenHands-tracked files of interest."""
    results: Dict[str, Any] = {}
    for fpath in WATCHED_FILES:
        try:
            commits = _github_api(
                f"repos/OpenHands/litellm/commits?path={fpath}&per_page=1", token
            )
            if commits:
                c = commits[0]
                results[fpath] = {
                    "sha": c["sha"][:8],
                    "date": c["commit"]["committer"]["date"],
                    "message": c["commit"]["message"][:120].replace("\n", " "),
                }
            else:
                results[fpath] = {"status": "no_commits"}
        except Exception as exc:
            results[fpath] = {"error": str(exc)}
    return results


# ── notable-change detection ──────────────────────────────────────────────────

def _detect_notable_changes(
    current: Dict[str, Any], previous: Optional[Dict[str, Any]]
) -> list[str]:
    """Return a list of human-readable notable-change reasons (empty = nothing actionable)."""
    reasons: list[str] = []

    # First run — no prior baseline to compare
    if previous is None:
        return reasons

    oh_cur = current.get("openhands_fork", {})
    oh_prev = previous.get("openhands_fork", {})

    # Drift grew significantly
    cur_behind = oh_cur.get("commits_behind")
    prev_behind = oh_prev.get("commits_behind")
    if (
        isinstance(cur_behind, int)
        and isinstance(prev_behind, int)
        and (cur_behind - prev_behind) >= DRIFT_ALERT_THRESHOLD
    ):
        reasons.append(
            f"OpenHands fork divergence grew by "
            f"{cur_behind - prev_behind} commits "
            f"({prev_behind} → {cur_behind} behind BerriAI main)"
        )

    # OpenHands pushed recently (active development = may have useful patterns)
    pushed = oh_cur.get("pushed_at", "")
    prev_pushed = oh_prev.get("pushed_at", "")
    if pushed and pushed != prev_pushed:
        reasons.append(
            f"OpenHands fork was updated since last check "
            f"(pushed_at changed: {prev_pushed!r} → {pushed!r}). "
            "Review for new integration patterns."
        )

    # Watched file activity changed
    watched_cur = current.get("openhands_watched_files", {})
    watched_prev = previous.get("openhands_watched_files", {})
    for fpath, info in watched_cur.items():
        prev_sha = watched_prev.get(fpath, {}).get("sha")
        cur_sha = info.get("sha")
        if cur_sha and prev_sha and cur_sha != prev_sha:
            reasons.append(
                f"OpenHands changed {fpath!r} "
                f"(commit {prev_sha} → {cur_sha}): "
                f"{info.get('message', '')[:80]}"
            )

    return reasons


# ── main ──────────────────────────────────────────────────────────────────────

def collect(
    output_path: Path = _DEFAULT_OUTPUT,
    verbose: bool = False,
    token: Optional[str] = None,
) -> Dict[str, Any]:
    """Collect all ecosystem signals and write to *output_path*.

    Returns the signals dict.  Exits with code 1 when notable changes are
    detected (consumed by the GitHub Actions workflow to open an issue).
    """
    if token is None:
        token = os.getenv("GITHUB_TOKEN")

    # Load previous snapshot for change-detection
    previous: Optional[Dict[str, Any]] = None
    if output_path.exists():
        try:
            with open(output_path) as fh:
                previous = json.load(fh)
        except (json.JSONDecodeError, OSError):
            pass

    if verbose:
        print("Collecting signals from GitHub API …")

    signals: Dict[str, Any] = {
        "schema_version": "1.1",
        "last_checked_at": datetime.now(timezone.utc).isoformat(),
        "policy": {
            "berri_role": "upstream dependency source of truth — PyPI package source",
            "openhands_role": (
                "ecosystem signal / integration pattern observer "
                "— NOT a runtime dependency source"
            ),
        },
        "litellm_upstream": _collect_berri(token),
        "openhands_fork": _collect_openhands(token),
        "openhands_watched_files": _collect_watched_files(token),
    }

    # Write snapshot
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as fh:
        json.dump(signals, fh, indent=2)

    # Print summary
    up = signals["litellm_upstream"]
    oh = signals["openhands_fork"]
    behind = oh.get("commits_behind", "unknown")

    print(f"✓ BerriAI latest release : {up.get('latest_release_tag', 'unknown')}")
    print(f"✓ BerriAI main SHA       : {up.get('main_branch_sha', 'unknown')[:12]}…")
    print(f"✓ OpenHands head SHA     : {oh.get('head_sha', 'unknown')[:12]}…")
    print(f"✓ OpenHands commits behind BerriAI main: {behind}")
    print(f"✓ Signals written to {output_path}")

    # Detect notable changes
    notable = _detect_notable_changes(signals, previous)
    if notable:
        print("\n⚠  NOTABLE CHANGES DETECTED — recommend human review:")
        for reason in notable:
            print(f"   • {reason}")
        # Output for workflow to consume
        print(f"\n::set-output name=notable::true")
        print(f"::set-output name=summary::{'; '.join(notable)[:500]}")
        sys.exit(1)  # Non-zero exit → workflow opens an issue
    else:
        print("\n✓ No notable changes since last check.")
        print("::set-output name=notable::false")

    if verbose:
        print("\n--- Full signals JSON ---")
        print(json.dumps(signals, indent=2))

    return signals


def main() -> None:
    parser = argparse.ArgumentParser(
        description="OpsMemory Ecosystem Signal Collector",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=_DEFAULT_OUTPUT,
        help=f"Output JSON path (default: {_DEFAULT_OUTPUT})",
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()
    collect(output_path=args.output, verbose=args.verbose)


if __name__ == "__main__":
    main()
