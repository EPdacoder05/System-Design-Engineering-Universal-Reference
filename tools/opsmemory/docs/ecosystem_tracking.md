# Ecosystem Tracking Policy

> OpsMemory uses the official **BerriAI/litellm** package from PyPI as the
> dependency source of truth. Forks such as **OpenHands/litellm** are tracked
> for ecosystem signals, integration ideas, and compatibility patterns **only**.
> They are **not** used as runtime dependency sources.

---

## Table of Contents

- [Upstream vs Observer](#upstream-vs-observer)
- [What We Track From OpenHands](#what-we-track-from-openhands)
- [Automation Layer](#automation-layer)
- [Alert Policy](#alert-policy)
- [How to Interpret Signals](#how-to-interpret-signals)
- [Manual Review Checklist](#manual-review-checklist)

---

## Upstream vs Observer

| Repo | Role | Automation |
|------|------|-----------|
| **[BerriAI/litellm](https://github.com/BerriAI/litellm)** | Upstream dependency source of truth | Auto-PR on new PyPI release (`litellm-upstream-sync.yml`) |
| **[OpenHands/litellm](https://github.com/OpenHands/litellm)** | Ecosystem signal / integration observer | Weekly read-only signal collection (`openhands-ecosystem-watch.yml`) |

**Never** install packages from `OpenHands/litellm` directly.  
**Never** mirror or auto-apply OpenHands patches into this codebase.  
**Do** borrow integration *ideas* from OpenHands through manual review.

---

## What We Track From OpenHands

### A. Fork Drift
How many commits behind BerriAI main is OpenHands? If the gap is still large
(it was 5 199 commits when this tracking was introduced), that reinforces the
policy of ignoring the fork as a package source. If the gap *shrinks*, it may
indicate OpenHands has rebased — review their changelog for ideas.

### B. Integration Patterns
We watch key files in the OpenHands fork for changes:

| File | Why |
|------|-----|
| `litellm/__init__.py` | Entry-point changes, new default model support |
| `litellm/main.py` | Core routing / completion logic changes |
| `litellm/router.py` | Fallback routing, load-balancing patterns |
| `litellm/utils.py` | Utility helpers, encoding fixes |
| `model_prices_and_context_window.json` | Model capability metadata |
| `litellm/integrations/` | New provider integrations |

Changes in these files may carry useful ideas even when the fork is stale.

### C. Model Ecosystem Metadata
OpenHands' LLM support work can inform updates to our own
`tools/opsmemory/providers/model_registry.yaml`:
- New model families they adopt before we do
- New `production_approved` or `mcp_safe` patterns
- New provider capability flags

### D. Breakage Signals
If OpenHands patches a LiteLLM or provider bug, that can be an early warning
of an issue we may encounter too.

---

## Automation Layer

### `litellm-upstream-sync.yml` — BerriAI auto-update (daily)
- Polls PyPI JSON API for newest `litellm` version
- If newer than the `requirements.txt` pin → opens a PR to bump it
- **This is the only automation that modifies runtime dependencies**

### `openhands-ecosystem-watch.yml` — Observer (weekly)
- Calls `tools/opsmemory/scripts/check_openhands_signals.py`
- Writes/updates `tools/opsmemory/providers/ecosystem_signals.json`
- Compares to previous snapshot for notable changes
- **Only opens an issue** (never modifies `requirements.txt` or source code)

### `dependabot.yml` — Broad dependency hygiene (weekly)
- Covers all `pip` deps including `litellm`
- Covers `github-actions` versions
- Major-version bumps excluded from auto-open (require human review)

---

## Alert Policy

| Trigger | Automated Action |
|---------|-----------------|
| New BerriAI/litellm PyPI release | Open PR bumping `requirements.txt` |
| OpenHands fork divergence grew ≥ 200 commits | Open GitHub Issue for awareness |
| OpenHands pushed (active development detected) | Open GitHub Issue for review |
| Watched file changed in OpenHands fork | Open GitHub Issue listing changes |
| Breaking BerriAI release (semver major) | Dependabot PR + human required |

**Issues opened by the watcher** are labelled `ecosystem-signal` and are
**low priority** — they exist to surface information, not demand action.

---

## How to Interpret Signals

The machine-readable snapshot lives at:

```
tools/opsmemory/providers/ecosystem_signals.json
```

Key fields to review:

```json
{
  "litellm_upstream": {
    "latest_release_tag": "v1.xx.x",      ← compare with requirements.txt pin
    "main_branch_sha":    "abc123..."      ← BerriAI HEAD
  },
  "openhands_fork": {
    "commits_behind": 5199,               ← drift from BerriAI main
    "pushed_at":      "2026-...",         ← last OpenHands activity
    "head_sha":       "def456..."         ← OpenHands HEAD
  },
  "openhands_watched_files": {
    "litellm/router.py": {
      "sha": "aaa111",
      "message": "Add fallback logic for..."
    }
  }
}
```

### Decision guide

| Signal | Interpretation | Action |
|--------|---------------|--------|
| `commits_behind` still large (>1000) | Fork is stale; policy intact | No action |
| `commits_behind` shrinking rapidly | Fork may have rebased; check changelog | Review manually |
| Watched file changed with interesting message | Possible integration pattern | Read the diff, borrow idea if useful |
| OpenHands `pushed_at` changed | Fork is actively maintained | Review their recent commits for ideas |

---

## Manual Review Checklist

When the ecosystem watcher opens an issue, use this checklist:

- [ ] Check `ecosystem_signals.json` drift numbers — is the gap growing or shrinking?
- [ ] Read OpenHands commit messages for watched files — any useful ideas?
- [ ] Cross-reference with `providers/model_registry.yaml` — any new model families?
- [ ] If an integration idea is worth borrowing: open a separate issue/PR for it
- [ ] Close the ecosystem-signal issue once reviewed
- [ ] If no action needed, leave a comment explaining why and close

---

*Last updated: 2026-03-16 — maintained by the OpsMemory engineering team.*
