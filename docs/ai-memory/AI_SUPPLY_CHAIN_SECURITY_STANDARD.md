# AI Supply Chain Security Standard

Canonical source for dependency, artifact, and build trust controls.

## Rules
- Pin dependency versions and container image digests for releasable builds.
- Build as non-root, run as non-root, and keep images minimal.
- Scan code, dependencies, containers, and secrets on the release path.
- Prefer trusted registries and verified publishers.
- Treat CI configuration as production security code.
- Cross-reference every import in a PR against the committed lockfile; reject PRs that introduce packages not already present in the lockfile.
- Restrict lifecycle scripts (preinstall/postinstall) in CI; never run untrusted install-time scripts against unreviewed packages.
- Import specific sub-modules, not entire packages where granular imports exist — minimizes attack surface from compromised transitive code.
- Rotate all secrets (GitHub PATs, npm/PyPI tokens, cloud IAM, SSH keys) immediately if any affected package version was installed during a known compromise window.

## Required Controls
- Dependency audit (pip-audit / npm audit)
- Lockfile diff gate on every PR: block if new transitive packages appear without explicit approval
- Preinstall/postinstall script allow-list in CI (e.g. `npm ci --ignore-scripts`)
- Minimal-import CI check: flag whole-package imports where sub-module imports are available
- Static analysis / security lint
- Secret scanning
- Container scanning
- SLSA provenance verification for high-risk packages; valid provenance alone does not guarantee safety (see TanStack/AntV incidents)
- Artifact integrity checks

## Anti-Patterns
- Trusting `npm install` / `pip install` without `--ignore-scripts` or equivalent
- Accepting PRs that silently expand the transitive dependency graph
- Importing a full SDK or mega-package when only one exported function is needed
- Treating valid SLSA provenance as sufficient — compromised CI can produce valid-looking provenance

## Related Repo References
- [docker/DOCKER_SECURITY.md](../../docker/DOCKER_SECURITY.md)
- [CYBERSEC_GUARDRAILS.md](../../CYBERSEC_GUARDRAILS.md)
- [.github/workflows/ci.yml](../../.github/workflows/ci.yml)
- [AI_DEPENDENCY_MINIMIZATION_STANDARD.md](./AI_DEPENDENCY_MINIMIZATION_STANDARD.md)

## Definition of Done
- Build inputs are attributable.
- Every PR lockfile diff is reviewed and approved.
- Runtime artifacts are hardened and scanned before release.
