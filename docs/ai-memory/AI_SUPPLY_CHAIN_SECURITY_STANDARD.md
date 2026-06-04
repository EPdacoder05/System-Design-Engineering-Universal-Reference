# AI Supply Chain Security Standard

Canonical source for dependency, artifact, and build trust controls.

## Rules
- Pin dependency versions and container image digests for releasable builds.
- Build as non-root, run as non-root, and keep images minimal.
- Scan code, dependencies, containers, and secrets on the release path.
- Prefer trusted registries and verified publishers.
- Treat CI configuration as production security code.

## Required Controls
- Dependency audit
- Static analysis/security lint
- Secret scanning
- Container scanning
- Artifact integrity checks

## Related Repo References
- [docker/DOCKER_SECURITY.md](../../docker/DOCKER_SECURITY.md)
- [CYBERSEC_GUARDRAILS.md](../../CYBERSEC_GUARDRAILS.md)
- [.github/workflows/ci.yml](../../.github/workflows/ci.yml)

## Definition of Done
- Build inputs are attributable.
- Runtime artifacts are hardened and scanned before release.
