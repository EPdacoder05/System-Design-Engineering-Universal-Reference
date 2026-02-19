# Docker Security Knowledge Base

Production-ready Docker hardening patterns for all EPdacoder05 repositories.

**Reference implementations in this repo:**
- [`cicd/Dockerfile`](../cicd/Dockerfile) — hardened multi-stage template
- [`cicd/.dockerignore`](../cicd/.dockerignore) — comprehensive ignore template
- [`.github/workflows/security-scan-universal.yml`](../.github/workflows/security-scan-universal.yml) — Trivy scanning workflow

---

## Table of Contents

1. [Multi-Stage Builds](#1-multi-stage-builds)
2. [Non-Root Users](#2-non-root-users)
3. [Base Image Security](#3-base-image-security)
4. [Linux Capabilities](#4-linux-capabilities)
5. [Docker Daemon Socket Protection](#5-docker-daemon-socket-protection)
6. [.dockerignore](#6-dockerignore)
7. [Healthchecks](#7-healthchecks)
8. [Automated Vulnerability Scanning](#8-automated-vulnerability-scanning)
9. [Read-Only Root Filesystem](#9-read-only-root-filesystem)
10. [Security Scorecard Matrix](#10-security-scorecard-matrix)

---

## 1. Multi-Stage Builds

### Why They Matter
- **Attack surface reduction** — build tools (gcc, pip, make) never reach production
- **Image size** — separating build and runtime stages routinely cuts 80–90% of image size
- **Secret isolation** — build-time secrets (tokens, SSH keys) never enter the final layer

### Template

```dockerfile
# ── Stage 1: Builder ────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

# ── Stage 2: Runtime ────────────────────────────────────────────────────────
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH=/home/appuser/.local/bin:$PATH

# Only runtime system packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy only the installed packages from builder
COPY --from=builder --chown=appuser:appuser /root/.local /home/appuser/.local
```

### Layer Caching Best Practices
- Copy `requirements.txt` (or `package.json`) **before** copying application code so dependency layers are cached independently.
- Sort `apt-get install` packages alphabetically — makes diffs easier and avoids accidental duplicates.
- Combine `RUN` commands with `&&` and clean up in the **same layer**: `rm -rf /var/lib/apt/lists/*`.

---

## 2. Non-Root Users

### Security Rationale
Running as root inside a container means a container-escape vulnerability gives an attacker **root on the host**. Non-root users limit the blast radius.

### How to Create and Switch

```dockerfile
# Create non-root user with explicit UID/GID
RUN groupadd --gid 1001 appgroup && \
    useradd --uid 1001 --gid 1001 --no-create-home --shell /sbin/nologin appuser

WORKDIR /app
COPY --chown=appuser:appgroup . .

USER appuser
```

### UID/GID Best Practices
- Use **UID ≥ 1000** (UIDs below 1000 are conventionally reserved for system accounts).
- The **1001 convention** avoids collision with the default `ubuntu`/`debian` non-root user (UID 1000).
- Use `--no-create-home` for service accounts that do not need a home directory.
- Use `--shell /sbin/nologin` to prevent interactive login.

### File Permissions

```dockerfile
# Application files — readable by appuser only
COPY --chown=appuser:appgroup --chmod=550 entrypoint.sh .

# Config files — readable, not writable
COPY --chown=root:appgroup --chmod=440 config/ ./config/
```

---

## 3. Base Image Security

### Why Pin Digests
Using `:latest` or even a mutable tag like `python:3.11-slim` means your build can silently pull a different image tomorrow. Pinning to a **digest** (SHA-256 hash) gives you:
- **Immutability** — the exact bytes you tested are the exact bytes that ship
- **Supply chain security** — a compromised upstream registry cannot silently substitute a malicious image

### How to Find and Pin Digests

```bash
# Pull and inspect
docker pull python:3.11-slim
docker inspect python:3.11-slim --format='{{index .RepoDigests 0}}'
# → python@sha256:abc123...

# Or use crane / skopeo without pulling
crane digest python:3.11-slim
```

Use in Dockerfile:

```dockerfile
FROM python@sha256:abc123def456...  # python:3.11-slim
```

### Minimal Base Images

| Image | Typical Size | Shell | Package Manager | Use When |
|-------|-------------|-------|-----------------|----------|
| `python:3.x-slim` | ~130 MB | bash | apt | General Python apps |
| `python:3.x-alpine` | ~50 MB | sh | apk | Size-critical; note musl libc differences |
| `gcr.io/distroless/python3` | ~50 MB | none | none | Maximum hardening; no shell at all |
| `scratch` | 0 MB | none | none | Statically compiled Go/Rust binaries |

### Update Strategy
- **Dependabot** — add a `docker` ecosystem entry to `.github/dependabot.yml` to receive weekly digest-update PRs.
- **Weekly CI scan** — Trivy scheduled scan (see [Section 8](#8-automated-vulnerability-scanning)) catches CVEs in base images between updates.

---

## 4. Linux Capabilities

### What Are Capabilities
Linux breaks root privileges into granular units. Key ones relevant to containers:

| Capability | What it allows | Typical need |
|------------|----------------|--------------|
| `CAP_CHOWN` | Change file ownership | Rarely needed at runtime |
| `CAP_NET_BIND_SERVICE` | Bind ports < 1024 | Only if app binds to port 80/443 |
| `CAP_SYS_PTRACE` | Attach debuggers | Never in production |
| `CAP_SYS_ADMIN` | Broad kernel operations | **Never** — nearly root-equivalent |
| `CAP_SETUID` / `CAP_SETGID` | Change UID/GID | Not needed with non-root user |

### Default vs Dropped
Docker adds ~14 capabilities by default. Drop all and add back only what is required:

```bash
# docker run
docker run --cap-drop ALL --cap-add NET_BIND_SERVICE myimage
```

### docker-compose Example

```yaml
services:
  api:
    image: myimage
    cap_drop:
      - ALL
    cap_add:
      - NET_BIND_SERVICE   # only if binding to port < 1024
    security_opt:
      - no-new-privileges:true
```

`no-new-privileges` prevents the process from gaining capabilities via `setuid` binaries.

---

## 5. Docker Daemon Socket Protection

### ⚠️ NEVER EXPOSE `/var/run/docker.sock`

Mounting the Docker daemon socket into a container is equivalent to giving that container **unrestricted root access to the host**:

```yaml
# ❌ NEVER DO THIS
volumes:
  - /var/run/docker.sock:/var/run/docker.sock
```

### Attack Vectors
- **Container escape** — any process inside the container can spawn a privileged container and mount the host filesystem.
- **Host takeover** — `docker run -v /:/host --privileged busybox chroot /host` gives full root on the host from inside the container.
- **Lateral movement** — access to the socket allows reading secrets from other containers and their environment variables.

### Safe Alternatives

| Alternative | How | Trade-offs |
|-------------|-----|-----------|
| **Rootless Docker** | Run `dockerd` as a non-root user | Best isolation; some storage driver limits |
| **Docker-in-Docker sidecar** (DinD) | Separate privileged DinD container, communicate via TCP | Isolated from host daemon; still privileged |
| **Kaniko** | Build images without Docker daemon | CI/CD build use-case only |
| **Buildah** | Rootless OCI image builds | Linux only |

```yaml
# ✅ DinD sidecar pattern for CI
services:
  ci-runner:
    image: my-ci-runner
    environment:
      DOCKER_HOST: tcp://docker:2376
      DOCKER_TLS_CERTDIR: /certs
    volumes:
      - docker-certs:/certs/client:ro
  docker:
    image: docker:dind
    privileged: true
    environment:
      DOCKER_TLS_CERTDIR: /certs
    volumes:
      - docker-certs:/certs

volumes:
  docker-certs:
```

---

## 6. .dockerignore

### Why It Matters
Without `.dockerignore`, `COPY . .` sends **everything** to the build context including:
- `.env` files with secrets
- Private keys and certificates
- The entire `.git` history
- Test fixtures and coverage reports

This bloats the image and can leak secrets into layers that are visible via `docker history`.

### Template

```
# Secrets and credentials
.env
.env.*
*.pem
*.key
*.p12
*.pfx
secrets/
credentials/
.aws/
.ssh/

# Git
.git/
.gitignore
.gitattributes

# CI/CD and GitHub
.github/
.circleci/
.travis.yml
Jenkinsfile

# Tests
tests/
test/
*_test.py
*.test.js
coverage/
.coverage
htmlcov/
.pytest_cache/
__pycache__/
*.pyc

# Documentation
docs/
*.md
!README.md

# Development tooling
.vscode/
.idea/
*.swp
.DS_Store
Makefile

# Dependencies (re-installed in build)
node_modules/
venv/
.venv/
__pycache__/

# Build artifacts
dist/
build/
*.egg-info/
.tox/
```

---

## 7. Healthchecks

### Why Critical for Production
- **Restart policies** — `restart: unless-stopped` combined with a HEALTHCHECK lets Docker restart only unhealthy containers, not all containers.
- **Load balancers** — orchestrators (Kubernetes, ECS, Swarm) only route traffic to **healthy** instances.
- **Deployment safety** — rolling deployments wait for a new container to report healthy before terminating old ones.

### Exec-Form vs Shell-Form

```dockerfile
# ✅ Exec-form — preferred (no shell process, signal handling works correctly)
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD ["curl", "-f", "http://localhost:8000/health"]

# ⚠️ Shell-form — spawns /bin/sh; avoid in distroless or shell-less images
HEALTHCHECK CMD curl -f http://localhost:8000/health || exit 1
```

### Configuration Reference

| Option | Default | Meaning |
|--------|---------|---------|
| `--interval` | 30s | Time between checks |
| `--timeout` | 30s | Maximum time for a single check |
| `--start-period` | 0s | Grace period for slow startup |
| `--retries` | 3 | Consecutive failures before `unhealthy` |

### Examples

```dockerfile
# Web application
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Database (PostgreSQL)
HEALTHCHECK --interval=10s --timeout=5s --start-period=30s --retries=5 \
    CMD pg_isready -U "$POSTGRES_USER" -d "$POSTGRES_DB" || exit 1

# Background worker (check a sentinel file updated by the process)
HEALTHCHECK --interval=60s --timeout=5s --start-period=10s --retries=3 \
    CMD test $(( $(date +%s) - $(stat -c %Y /tmp/worker.heartbeat) )) -lt 120 || exit 1
```

---

## 8. Automated Vulnerability Scanning

### Trivy Integration in CI/CD

[Trivy](https://github.com/aquasecurity/trivy) scans container images, filesystems, and IaC for CVEs, misconfigurations, secrets, and SBOM generation.

### GitHub Actions Workflow Template

```yaml
name: Container Security Scan

on:
  push:
    branches: [main, master]
  pull_request:
    branches: [main, master]
  schedule:
    - cron: '0 2 * * 1'   # Weekly Monday 2 AM UTC

permissions:
  contents: read
  security-events: write

jobs:
  trivy-scan:
    name: Trivy Vulnerability Scan
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Build image
        run: docker build -t ${{ github.repository }}:${{ github.sha }} .

      - name: Run Trivy (SARIF upload)
        uses: aquasecurity/trivy-action@master
        with:
          image-ref: ${{ github.repository }}:${{ github.sha }}
          format: sarif
          output: trivy-results.sarif
          severity: CRITICAL,HIGH
          exit-code: '1'          # Fail the build on CRITICAL/HIGH

      - name: Upload SARIF to GitHub Security tab
        uses: github/codeql-action/upload-sarif@v3
        if: always()
        with:
          sarif_file: trivy-results.sarif
          category: trivy-container-scan
```

See [`.github/workflows/security-scan-universal.yml`](../.github/workflows/security-scan-universal.yml) for the full multi-scanner workflow used in this repo.

### Fail-on-Severity Thresholds

| Environment | Recommended threshold |
|-------------|----------------------|
| Production | `CRITICAL` only (block deploy) |
| Staging | `CRITICAL,HIGH` |
| PR checks | `CRITICAL,HIGH,MEDIUM` |

### Scheduled Scanning
Use the `schedule` trigger (example above) so images that haven't changed are still re-scanned against the latest CVE database. A newly disclosed CVE in your pinned base image will surface within 24 hours.

---

## 9. Read-Only Root Filesystem

### Security Benefits
- Prevents an attacker from writing backdoors, cron jobs, or modified binaries to disk.
- Limits the damage of a Remote Code Execution (RCE) vulnerability — the attacker cannot persist changes.
- Detects misconfigured applications that write to unexpected paths.

### docker-compose Usage

```yaml
services:
  api:
    image: myimage
    read_only: true
    tmpfs:
      - /tmp:size=64m,mode=1777
      - /var/run:size=10m
    volumes:
      - app-logs:/app/logs   # named volume for intentional writes
```

### tmpfs Mounts for Writable Dirs

| Path | Purpose | Recommended size |
|------|---------|-----------------|
| `/tmp` | Temporary files | 64–256 MB |
| `/var/run` | PID files, sockets | 10 MB |
| `/var/cache` | Application cache | App-dependent |

In Kubernetes, use `emptyDir` medium `Memory`:

```yaml
volumes:
  - name: tmp
    emptyDir:
      medium: Memory
      sizeLimit: 64Mi
```

---

## 10. Security Scorecard Matrix

Use this table to audit Docker security posture across repositories. Update it during security reviews.

| Repo | Multi-Stage | Non-Root | Pinned Digest | Healthcheck | .dockerignore | Cap Drop | Read-Only FS | Score |
|------|-------------|----------|---------------|-------------|---------------|----------|--------------|-------|
| System-Design-Engineering-Universal-Reference | ✅ | ✅ | ⚠️ | ✅ | ⚠️ | ⚠️ | ⚠️ | B |
| ha-ble-mqtt-bridge | ⚠️ | ⚠️ | ⚠️ | ⚠️ | ⚠️ | ⚠️ | ⚠️ | — |
| ha-iot-stack | ⚠️ | ⚠️ | ⚠️ | ⚠️ | ⚠️ | ⚠️ | ⚠️ | — |
| NullPointVector | ⚠️ | ⚠️ | ⚠️ | ⚠️ | ⚠️ | ⚠️ | ⚠️ | — |
| finops-cost-control-as-code | ⚠️ | ⚠️ | ⚠️ | ⚠️ | ⚠️ | ⚠️ | ⚠️ | — |
| JOB-APPLICATION-AGENT | ⚠️ | ⚠️ | ⚠️ | ⚠️ | ⚠️ | ⚠️ | ⚠️ | — |
| popsmirror | ⚠️ | ⚠️ | ⚠️ | ⚠️ | ⚠️ | ⚠️ | ⚠️ | — |
| TF2S3-migration | ⚠️ | ⚠️ | ⚠️ | ⚠️ | ⚠️ | ⚠️ | ⚠️ | — |

**Score legend:**
- **A** — All 7 controls implemented
- **B** — 5–6 controls
- **C** — 3–4 controls
- **D** — 1–2 controls
- **F** — None

✅ Implemented · ⚠️ Not yet / Needs review · ❌ Confirmed absent

---

## Attack Surface Comparison

### Before vs After Hardening

| Metric | Before (unoptimized) | After (hardened) | Improvement |
|--------|---------------------|------------------|-------------|
| Image size | ~900 MB | ~150 MB | **83% smaller** |
| CVEs (CRITICAL+HIGH) | 47 | 3 | **94% reduction** |
| Running as root | Yes | No | ✅ |
| Build tools in runtime | Yes (gcc, make, pip) | No | ✅ |
| Secrets in layers | Possible | Blocked by `.dockerignore` | ✅ |
| Attack vectors | Full filesystem RW | Read-only + tmpfs only | ✅ |

### What Drives the Size Reduction
1. **Multi-stage build** — removes build-time packages and pip cache (~400 MB)
2. **`-slim` base** — removes locales, documentation, and optional system packages (~300 MB)
3. **`--no-install-recommends`** — skips apt recommended packages (~50 MB)
4. **`.dockerignore`** — excludes tests, docs, `.git` from build context (no layer bloat)

---

## Production Checklist

Run this checklist before pushing any Docker image to a production registry:

- [ ] **Non-root user** — `USER` directive uses UID > 1000; confirmed with `docker inspect`
- [ ] **Base image pinned to digest** — `FROM image@sha256:...` not `:latest`
- [ ] **Multi-stage build** — final image contains no compiler, pip, or build tools
- [ ] **No secrets in layers** — `docker history --no-trunc <image>` shows no tokens, passwords, or keys
- [ ] **Healthcheck configured** — `HEALTHCHECK` instruction present with appropriate interval/timeout
- [ ] **Capabilities dropped** — `--cap-drop ALL` in compose/run, `cap_add` only for required caps
- [ ] **Trivy scan passing** — no CRITICAL or HIGH CVEs; SARIF uploaded to GitHub Security tab
- [ ] **`.dockerignore` blocking secrets** — `.env`, `*.pem`, `.git/`, `secrets/` all excluded
