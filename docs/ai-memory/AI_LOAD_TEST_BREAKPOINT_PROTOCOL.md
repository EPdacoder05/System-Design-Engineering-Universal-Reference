# AI Load Test Breakpoint Protocol

Canonical source for finding the first real scaling limit.

## Protocol
- Define workload model, SLOs, and stop conditions before the test.
- Ramp gradually until the first sustained breakpoint appears.
- Record the limiting resource: CPU, lock, queue, DB, downstream, partition, or network.
- Repeat after the main fix to prove the bottleneck moved.

## Required Outputs
- Breakpoint throughput and concurrency
- p50/p95/p99 latency at and before failure
- Error/retry/DLQ behavior
- Primary bottleneck hypothesis and evidence

## Anti-Patterns
- One giant spike test with no ramp.
- Declaring success from average latency only.
- Ignoring queue lag, retries, or partition skew.

## Definition of Done
- The first bottleneck is identified with evidence.
- The next action is clear: tune, redesign, or cap load.
