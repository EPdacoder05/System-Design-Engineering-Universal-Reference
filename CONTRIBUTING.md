# Contributing

Thanks for contributing.

## Scope

This repository is a portable engineering reference library. Keep contributions:
- modular
- public-safe (no PII, no private endpoints, no secrets)
- reusable across projects

## Getting Started

1. Fork and create a feature branch.
2. Make focused changes with clear rationale.
3. Run local validation:

```bash
python -m ruff check .
python -m mypy .
python -m pytest -q
```

4. Open a PR using the pull request template.

## Contribution Types

- Bug fixes in reference modules or docs
- Security hardening patterns
- CI/CD template improvements
- Additional reusable architecture examples
- OpsMemory integration improvements

## Quality Gates

- Keep files standalone where possible
- Avoid breaking existing examples
- Maintain clear docs and usage notes
- Follow rubric checks in `rubrics/MASTER_RUBRIC.md`

## Good First Issues

Look for issues labeled `good first issue` and `documentation`/`tests`/`templates`.

## Security

Do not open public issues for sensitive vulnerabilities. Use `SECURITY.md`.
