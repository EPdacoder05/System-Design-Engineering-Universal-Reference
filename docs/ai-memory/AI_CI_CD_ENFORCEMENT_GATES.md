# AI CI/CD Enforcement Gates

Canonical source for delivery gates and remediation principles.

## Gate Order
1. Lint
2. Typecheck/static validation
3. Tests
4. Build/package
5. Security scans
6. Release/deploy approval

## Hard Rules
- Downstream build and deploy steps must depend on upstream gates.
- Verify workflow triggers, branch/path filters, runner selection, permissions, and event payload assumptions as part of the gate design.
- Auth hardening, data-loss prevention, and container hardening changes require tests plus security validation.
- CI exceptions need explicit approver, reason, expiry, and compensating control.
- Failed gates must be visible; do not silently skip critical downstream jobs.
- Required checks must map to real blocking jobs; skipped or detached jobs do not count as enforcement.

## Required Remediation Themes
- Auth hardening: fail closed, validate trust boundaries, keep privileged flows tested.
- CI gates: make critical failures blocking, not advisory by accident.
- Workflow integrity: verify runners, `needs`, permissions, and payload-driven conditionals before trusting green status.
- Container hardening: non-root, minimal image, pinned inputs, health checks.
- DLQ/no-silent-drop: async failure handling must be observable and enforceable.

## Definition of Done
- The release path blocks unsafe changes.
- Gate bypasses are exceptional, documented, and time-bounded.
