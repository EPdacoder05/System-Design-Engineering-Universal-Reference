# AI Partitioning: Monthly vs Yearly Standard

Canonical source for time partitioning defaults and exceptions.

## Default Rule
- Use monthly partitions by default for high-write, append-heavy, and event/audit tables.
- Yearly partitioning is allowed only when write volume is low, pruning remains effective, and operational evidence shows monthly partitions add unnecessary overhead.

## Monthly Default Triggers
- Event, audit, log, ledger, outbox, inbox, or telemetry tables.
- Sustained write-heavy tables.
- Tables with retention, archive, replay, or backfill workflows.
- Tables where incident isolation by time window matters.

## Longevity Guardrails
- Precreate future partitions on a scheduled job; keep at least the next 2-3 months ready.
- Run a partition-health job at least daily to detect missing future partitions, skew, failed pruning, retention lag, and unexpected writes to default partitions.
- Enforce retention/archive policy with explicit drop/archive windows and success/failure alerting.
- Keep partition key choice stable and query predicates aligned to it.
- If live partition counts or maintenance cost grow, solve with archive/tiering/retention first; do not collapse to yearly without evidence.

## Anti-Patterns
- Single giant yearly partition for a hot event stream.
- Creating partitions manually during an incident.
- Using partitioning without pruning-friendly query predicates.

## Definition of Done
- Partition interval is justified.
- Precreate, retention, and health automation exist.
- Operators can detect missing or unhealthy partitions before write failures occur.
