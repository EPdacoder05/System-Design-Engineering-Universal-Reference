# AI Code Review Anti-Slop Checklist

Canonical review checklist for rejecting low-signal or dangerous changes.

## Checklist
- [ ] Uses the canonical standard instead of inventing a duplicate rule.
- [ ] The owner can explain the change from memory without rereading generated scaffolding.
- [ ] No auth bypass, weak default, or fail-open behavior.
- [ ] No silent catch, silent retry loop, or silent message drop.
- [ ] Retries are bounded and observable, with backoff/jitter where repetition is expected.
- [ ] No unbounded nested loop or N+1 query in hot paths.
- [ ] No unbounded queue consumer concurrency.
- [ ] No speculative wrapper, config system, or event hop where a direct call or constant would do.
- [ ] No new dependency without clear need and transitive-risk check.
- [ ] No mutable `latest` image or unsigned release artifact.
- [ ] No fake health check that always returns healthy.
- [ ] No LLM call inserted into the v1 core synchronous path.
- [ ] Definition of done and operator signals are explicit.

## Anti-Pattern Callouts
- Copy-pasted docs that drift from the canonical file.
- Boilerplate abstractions that make a small service look like a 50-team platform.
- “Temporary” security exceptions with no expiry.
- Replacing deterministic logic with model output on critical paths.

## Definition of Done
- Review comments focus on correctness, safety, and operability.
- Known anti-patterns are either fixed or explicitly rejected.
