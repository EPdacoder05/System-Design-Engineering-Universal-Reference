# AI SBOM, Provenance, and Signing Standard

Canonical source for release attestations.

## Rules
- Generate an SBOM for every releasable artifact.
- Produce provenance that ties artifact -> source revision -> build workflow.
- Sign container images and release artifacts.
- Verify signatures and provenance before deploy where platform support exists.
- Store attestations with enough retention for incident response and audit.

## Minimum Artifact Set
- SBOM
- Build provenance/attestation
- Signature
- Scan results reference

## Anti-Patterns
- Unsigned mutable release artifacts.
- SBOM generated once but not kept with releases.
- Provenance that cannot identify the workflow or commit.

## Definition of Done
- Released artifacts can be traced, verified, and audited.
