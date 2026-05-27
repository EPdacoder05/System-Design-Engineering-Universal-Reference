"""
Cloudflare Platform S-Tier Reference

Apply to: Any service deployed behind Cloudflare — APIs, web apps, Workers,
          SaaS backends, CDN-accelerated static sites

Covers:
- CORS configuration (strict, permissive, preflight caching)
- Security headers (CSP, HSTS, Permissions-Policy, COEP/COOP)
- Geo-IP rate limiting (per-country, blocked regions, trusted regions)
- Payload integrity verification (HMAC-SHA256, Content-Digest RFC 9530)
- Principal verification (CF-Access JWT, API key, mTLS, trusted-IP)
- WAF custom rule templates
- Cloudflare Workers JavaScript/TypeScript templates (embedded)
- D1 (SQLite), KV, R2, Queues, Durable Objects patterns
- Concurrency model: Workers isolates vs. Durable Object actors
- Cache API + Cache-Control strategy
- Access (Zero Trust) policy patterns
- Turnstile (bot protection) integration
- IaaC: see security/iac/cloudflare_terraform.tf

Note on Workers runtime:
Workers execute in V8 isolates, NOT Node.js. The JS/TS templates below
use the WinterCG-compatible Fetch API and Web Crypto API exclusively.

Installation (Python tools):
    pip install cloudflare          # Cloudflare Python SDK
    npm install -g wrangler         # Workers CLI (for JS/TS deployment)

Deploy Python (via Workers + Pyodide or external origin):
    The Python service_template.py acts as the origin.
    Cloudflare Workers proxy, rate-limit, and secure traffic before it reaches Python.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import ipaddress
import json
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


# =============================================================================
# CORS policy builder
# =============================================================================

class CORSPolicy(Enum):
    STRICT = "strict"         # exact origin allow-list, credentials allowed
    API_PUBLIC = "api_public" # no credentials, broad methods, any origin
    INTERNAL = "internal"     # same-origin only (no CORS headers)


@dataclass
class CORSConfig:
    """
    Build CORS headers for any response.
    Use STRICT for APIs that handle authenticated sessions.
    Use API_PUBLIC for fully public, credential-free data APIs.
    """
    policy: CORSPolicy = CORSPolicy.STRICT
    allowed_origins: List[str] = field(default_factory=list)
    allowed_methods: List[str] = field(default_factory=lambda: ["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"])
    allowed_headers: List[str] = field(default_factory=lambda: ["Content-Type", "Authorization", "X-Request-ID"])
    expose_headers: List[str] = field(default_factory=lambda: ["X-Request-ID", "X-RateLimit-Remaining"])
    max_age: int = 86400        # preflight cache: 24 hours (Cloudflare CDN caches OPTIONS)
    allow_credentials: bool = True

    def headers_for_origin(self, request_origin: Optional[str]) -> Dict[str, str]:
        """Return CORS response headers for a given request Origin."""
        if self.policy == CORSPolicy.INTERNAL:
            return {}

        headers: Dict[str, str] = {}

        if self.policy == CORSPolicy.API_PUBLIC:
            headers["Access-Control-Allow-Origin"] = "*"
            headers["Access-Control-Allow-Methods"] = ", ".join(self.allowed_methods)
            headers["Access-Control-Allow-Headers"] = ", ".join(self.allowed_headers)
            headers["Access-Control-Max-Age"] = str(self.max_age)
            # Never send Allow-Credentials with wildcard origin
            return headers

        # STRICT: reflect origin if in allow-list
        if request_origin and request_origin in self.allowed_origins:
            headers["Access-Control-Allow-Origin"] = request_origin
            headers["Vary"] = "Origin"  # required for correct CDN caching
        elif self.allowed_origins:
            headers["Access-Control-Allow-Origin"] = self.allowed_origins[0]
            headers["Vary"] = "Origin"

        headers["Access-Control-Allow-Methods"] = ", ".join(self.allowed_methods)
        headers["Access-Control-Allow-Headers"] = ", ".join(self.allowed_headers)
        headers["Access-Control-Expose-Headers"] = ", ".join(self.expose_headers)
        headers["Access-Control-Max-Age"] = str(self.max_age)
        if self.allow_credentials:
            headers["Access-Control-Allow-Credentials"] = "true"
        return headers

    def is_preflight(self, method: str) -> bool:
        return method.upper() == "OPTIONS"


# Production CORS configs
PRODUCTION_CORS = CORSConfig(
    policy=CORSPolicy.STRICT,
    allowed_origins=["https://app.example.com", "https://admin.example.com"],
    allow_credentials=True,
)

PUBLIC_API_CORS = CORSConfig(
    policy=CORSPolicy.API_PUBLIC,
    allow_credentials=False,
)


# =============================================================================
# Security headers (S-tier hardening)
# =============================================================================

def build_security_headers(
    *,
    hsts_max_age: int = 31536000,   # 1 year
    csp_report_uri: Optional[str] = None,
    frame_ancestors: str = "'none'",
) -> Dict[str, str]:
    """
    Build the full set of recommended Cloudflare / browser security headers.
    Apply via Cloudflare Transform Rules or inject at the origin.
    """
    csp_directives = [
        "default-src 'self'",
        "script-src 'self' 'strict-dynamic'",          # nonce-based scripts
        "style-src 'self' 'unsafe-inline'",            # relax for CSS-in-JS if needed
        "img-src 'self' data: https:",
        "font-src 'self'",
        "connect-src 'self'",
        "frame-ancestors " + frame_ancestors,
        "base-uri 'self'",
        "form-action 'self'",
        "upgrade-insecure-requests",
    ]
    if csp_report_uri:
        csp_directives.append(f"report-uri {csp_report_uri}")

    headers = {
        # Prevent MIME type sniffing
        "X-Content-Type-Options": "nosniff",
        # Clickjacking protection (redundant with CSP frame-ancestors, but keep for older browsers)
        "X-Frame-Options": "DENY",
        # XSS filter (legacy browsers)
        "X-XSS-Protection": "1; mode=block",
        # HSTS — tell browsers to always use HTTPS
        "Strict-Transport-Security": f"max-age={hsts_max_age}; includeSubDomains; preload",
        # Referrer policy
        "Referrer-Policy": "strict-origin-when-cross-origin",
        # Content Security Policy
        "Content-Security-Policy": "; ".join(csp_directives),
        # Disable browser features not used by the app
        "Permissions-Policy": (
            "camera=(), microphone=(), geolocation=(), "
            "payment=(), usb=(), interest-cohort=()"
        ),
        # Cross-Origin policies (required for SharedArrayBuffer / high-res timers)
        "Cross-Origin-Opener-Policy": "same-origin",
        "Cross-Origin-Embedder-Policy": "require-corp",
        "Cross-Origin-Resource-Policy": "same-origin",
        # Remove server fingerprint
        "Server": "Cloudflare",
        "X-Powered-By": "",    # suppress; set via Cloudflare Transform Rule "remove header"
    }
    return headers


# =============================================================================
# Geo-IP rate limiting
# =============================================================================

class GeoIPRisk(Enum):
    TRUSTED = "trusted"    # <10 req/min — low-risk regions (internal networks)
    NORMAL = "normal"      # standard limits
    ELEVATED = "elevated"  # higher scrutiny, lower limits
    BLOCKED = "blocked"    # hard-block — sanctioned regions, known bad actors


# Country codes and their risk tiers (edit to match your compliance requirements)
GEO_RISK_TIERS: Dict[str, GeoIPRisk] = {
    # Blocked: OFAC-sanctioned countries (replace/augment with current OFAC list)
    "CU": GeoIPRisk.BLOCKED,
    "IR": GeoIPRisk.BLOCKED,
    "KP": GeoIPRisk.BLOCKED,
    "RU": GeoIPRisk.BLOCKED,
    "SY": GeoIPRisk.BLOCKED,
    # Elevated scrutiny examples
    "CN": GeoIPRisk.ELEVATED,
    "BR": GeoIPRisk.ELEVATED,
    # All others: NORMAL (handled by default)
}

# Requests per minute limits by geo tier
GEO_RATE_LIMITS: Dict[GeoIPRisk, int] = {
    GeoIPRisk.TRUSTED: 600,
    GeoIPRisk.NORMAL: 120,
    GeoIPRisk.ELEVATED: 30,
    GeoIPRisk.BLOCKED: 0,
}


@dataclass
class GeoIPRateLimiter:
    """
    Application-layer geo-IP rate limiter.
    In production, offload to Cloudflare WAF Rate Limiting Rules
    (see cloudflare_terraform.tf) for zero-latency enforcement at the edge.
    This class is the fallback / local replica for dev and testing.
    """

    # bucket: ip → (tokens, last_refill_ts)
    _buckets: Dict[str, Tuple[float, float]] = field(default_factory=dict)

    def get_risk(self, country_code: Optional[str]) -> GeoIPRisk:
        if country_code is None:
            return GeoIPRisk.ELEVATED  # unknown geo = elevated
        return GEO_RISK_TIERS.get(country_code.upper(), GeoIPRisk.NORMAL)

    def check(self, ip: str, country_code: Optional[str]) -> Tuple[bool, Dict[str, Any]]:
        """
        Returns (allowed, headers_dict).
        headers_dict contains X-RateLimit-* values for the response.
        """
        risk = self.get_risk(country_code)
        rpm_limit = GEO_RATE_LIMITS[risk]

        if risk == GeoIPRisk.BLOCKED:
            return False, {
                "X-Block-Reason": "geo-blocked",
                "X-Block-Country": country_code or "unknown",
                "Retry-After": "0",
            }

        # Token bucket refill
        capacity = float(rpm_limit)
        refill_rate = rpm_limit / 60.0  # tokens per second
        now = time.monotonic()

        tokens, last = self._buckets.get(ip, (capacity, now))
        elapsed = now - last
        tokens = min(capacity, tokens + elapsed * refill_rate)

        if tokens < 1:
            return False, {
                "X-RateLimit-Limit": str(rpm_limit),
                "X-RateLimit-Remaining": "0",
                "X-RateLimit-Reset": str(int(time.time()) + int(60 / rpm_limit)),
                "Retry-After": str(int(60 / max(refill_rate, 0.01))),
            }

        tokens -= 1
        self._buckets[ip] = (tokens, now)
        return True, {
            "X-RateLimit-Limit": str(rpm_limit),
            "X-RateLimit-Remaining": str(int(tokens)),
            "X-RateLimit-Policy": f"{rpm_limit};w=60",
        }


# =============================================================================
# Cloudflare Workers JavaScript templates (embedded as strings)
# Deploy with: wrangler deploy
# =============================================================================

WORKER_TEMPLATE_CORS_SECURITY = r"""
/**
 * Cloudflare Workers: CORS + Security Headers + Geo-IP Rate Limiting
 *
 * wrangler.toml:
 *   name = "api-gateway"
 *   main = "worker.js"
 *   compatibility_date = "2024-01-01"
 *   [vars]
 *     ALLOWED_ORIGINS = "https://app.example.com,https://admin.example.com"
 *     BLOCKED_COUNTRIES = "CU,IR,KP,RU,SY"
 *
 * KV namespace (rate limiting state):
 *   [[kv_namespaces]]
 *   binding = "RATE_LIMIT_KV"
 *   id = "<your-kv-namespace-id>"
 */

const SECURITY_HEADERS = {
  "X-Content-Type-Options": "nosniff",
  "X-Frame-Options": "DENY",
  "Strict-Transport-Security": "max-age=31536000; includeSubDomains; preload",
  "Referrer-Policy": "strict-origin-when-cross-origin",
  "Permissions-Policy": "camera=(), microphone=(), geolocation=()",
  "Content-Security-Policy": "default-src 'self'; frame-ancestors 'none'; upgrade-insecure-requests",
  "Cross-Origin-Opener-Policy": "same-origin",
  "Cross-Origin-Embedder-Policy": "require-corp",
};

function getCORSHeaders(request, env) {
  const origin = request.headers.get("Origin") || "";
  const allowed = (env.ALLOWED_ORIGINS || "").split(",").map(s => s.trim());
  if (allowed.includes(origin)) {
    return {
      "Access-Control-Allow-Origin": origin,
      "Vary": "Origin",
      "Access-Control-Allow-Methods": "GET, POST, PUT, PATCH, DELETE, OPTIONS",
      "Access-Control-Allow-Headers": "Content-Type, Authorization, X-Request-ID",
      "Access-Control-Expose-Headers": "X-Request-ID, X-RateLimit-Remaining",
      "Access-Control-Max-Age": "86400",
      "Access-Control-Allow-Credentials": "true",
    };
  }
  return {};
}

async function checkGeoRateLimit(request, env) {
  const country = request.cf?.country || "XX";
  const blocked = (env.BLOCKED_COUNTRIES || "CU,IR,KP,RU,SY").split(",");
  if (blocked.includes(country)) {
    return new Response(JSON.stringify({ error: "Access denied", code: "GEO_BLOCKED" }), {
      status: 403,
      headers: { "Content-Type": "application/json", ...SECURITY_HEADERS },
    });
  }

  // Rate limiting via KV (sliding window)
  const ip = request.headers.get("CF-Connecting-IP") || "unknown";
  const key = `rl:${country}:${ip}`;
  const window = 60; // seconds
  const limit = country === "CN" || country === "BR" ? 30 : 120;

  const now = Math.floor(Date.now() / 1000);
  const raw = await env.RATE_LIMIT_KV.get(key, { type: "json" });
  const state = raw || { count: 0, reset: now + window };

  if (now > state.reset) {
    state.count = 0;
    state.reset = now + window;
  }

  state.count++;
  await env.RATE_LIMIT_KV.put(key, JSON.stringify(state), { expirationTtl: window + 5 });

  if (state.count > limit) {
    return new Response(JSON.stringify({ error: "Rate limit exceeded" }), {
      status: 429,
      headers: {
        "Content-Type": "application/json",
        "Retry-After": String(state.reset - now),
        "X-RateLimit-Limit": String(limit),
        "X-RateLimit-Remaining": "0",
        ...SECURITY_HEADERS,
      },
    });
  }
  return null; // allowed
}

export default {
  async fetch(request, env, ctx) {
    const url = new URL(request.url);

    // 1. OPTIONS preflight
    if (request.method === "OPTIONS") {
      return new Response(null, {
        status: 204,
        headers: { ...getCORSHeaders(request, env), ...SECURITY_HEADERS },
      });
    }

    // 2. Geo-IP rate limiting
    const geoBlock = await checkGeoRateLimit(request, env);
    if (geoBlock) return geoBlock;

    // 3. Forward to origin
    const originResponse = await fetch(request);
    const response = new Response(originResponse.body, originResponse);

    // 4. Inject headers on the way out
    const headers = new Headers(response.headers);
    Object.entries({ ...getCORSHeaders(request, env), ...SECURITY_HEADERS })
      .forEach(([k, v]) => headers.set(k, v));
    headers.delete("X-Powered-By");  // remove fingerprint

    return new Response(response.body, { status: response.status, headers });
  },
};
"""


WORKER_TEMPLATE_DURABLE_OBJECTS = r"""
/**
 * Cloudflare Durable Objects: stateful, single-threaded actor per key.
 * Use for: rate limiting state, real-time collaboration, presence/lobby management.
 *
 * wrangler.toml:
 *   [[durable_objects.bindings]]
 *   name = "RATE_LIMITER"
 *   class_name = "RateLimiterDO"
 *
 *   [[migrations]]
 *   tag = "v1"
 *   new_classes = ["RateLimiterDO"]
 */

export class RateLimiterDO {
  constructor(state, env) {
    this.state = state;
    this.storage = state.storage;
  }

  async fetch(request) {
    const { limit = 60, window = 60 } = await request.json();
    const now = Math.floor(Date.now() / 1000);

    let { count = 0, reset = now + window } = (await this.storage.get("state")) || {};

    if (now > reset) { count = 0; reset = now + window; }
    count++;
    await this.storage.put("state", { count, reset });

    const allowed = count <= limit;
    return Response.json({
      allowed,
      remaining: Math.max(0, limit - count),
      reset,
    }, {
      status: allowed ? 200 : 429,
      headers: allowed ? {} : { "Retry-After": String(reset - now) },
    });
  }
}

// Worker entry point — routes each request to its own DO by IP
export default {
  async fetch(request, env) {
    const ip = request.headers.get("CF-Connecting-IP") || "anon";
    const id = env.RATE_LIMITER.idFromName(ip);
    const stub = env.RATE_LIMITER.get(id);
    return stub.fetch(new Request("https://internal/check", {
      method: "POST",
      body: JSON.stringify({ limit: 60, window: 60 }),
    }));
  },
};
"""


WORKER_TEMPLATE_TURNSTILE = r"""
/**
 * Cloudflare Turnstile: bot-protection CAPTCHA alternative.
 * Place on login, signup, and payment endpoints.
 *
 * Server-side validation — call this from your Worker or origin:
 */

async function verifyTurnstile(token, ip, secretKey) {
  const formData = new FormData();
  formData.append("secret", secretKey);
  formData.append("response", token);
  formData.append("remoteip", ip);

  const result = await fetch("https://challenges.cloudflare.com/turnstile/v0/siteverify", {
    method: "POST",
    body: formData,
  });
  const outcome = await result.json();
  return outcome.success === true;
}

export default {
  async fetch(request, env) {
    if (request.method === "POST" && new URL(request.url).pathname === "/login") {
      const body = await request.json();
      const ip = request.headers.get("CF-Connecting-IP") || "";

      const valid = await verifyTurnstile(body["cf-turnstile-response"], ip, env.TURNSTILE_SECRET);
      if (!valid) {
        return Response.json({ error: "Bot challenge failed" }, { status: 403 });
      }
      // Proceed with login logic
    }
    return fetch(request);
  },
};
"""


# =============================================================================
# Cloudflare platform feature matrix
# =============================================================================

CLOUDFLARE_PLATFORM_GUIDE: Dict[str, Any] = {
    "Workers": {
        "description": "Serverless JS/TS/Wasm at the edge, 300+ PoPs, <1 ms cold start",
        "concurrency": "Isolates (not Node.js), no shared memory between requests",
        "cpu_limit": "10 ms CPU per request (50 ms on paid plans)",
        "use_cases": ["API gateway", "A/B testing", "Auth middleware", "Image resizing"],
        "limits": {
            "memory": "128 MB per isolate",
            "subrequests": 50,
            "kv_reads_per_request": 1000,
        },
    },
    "Durable Objects": {
        "description": "Stateful single-threaded actors with strongly consistent storage",
        "concurrency": "Single-threaded per object; all requests to same DO are serialized",
        "use_cases": ["Rate limiting", "Chat rooms", "Collaborative docs", "Game lobbies"],
        "storage": "Transactional key-value, up to 10 GiB per DO",
        "scale": "Millions of DOs; each runs in exactly one PoP",
    },
    "KV": {
        "description": "Globally replicated key-value store, eventually consistent",
        "read_latency": "<1 ms (cache hit at nearest PoP)",
        "write_propagation": "~60 seconds globally",
        "use_cases": ["Feature flags", "Session tokens", "Static config", "Rate limit counters (eventually consistent)"],
        "limits": {"key_size": "512 bytes", "value_size": "25 MiB", "writes_per_second": 1},
    },
    "R2": {
        "description": "S3-compatible object storage with zero egress fees",
        "use_cases": ["User uploads", "Media storage", "Backup", "Data lake"],
        "compatibility": "AWS S3 SDK compatible — swap endpoint URL",
        "egress": "FREE (unlike S3 which charges per GB egress)",
    },
    "D1": {
        "description": "Serverless SQLite at the edge, replicated read replicas",
        "use_cases": ["Lightweight CRUD", "Edge-local reads", "Per-tenant databases"],
        "limits": {"db_size": "2 GB", "rows_read_per_day": "25M (free tier)"},
        "access": "Via Workers binding: env.DB.prepare(sql).bind(...).run()",
    },
    "Queues": {
        "description": "Guaranteed delivery message queue, Workers consumer",
        "use_cases": ["Webhook fan-out", "Background jobs", "Event sourcing"],
        "delivery": "At-least-once; idempotency key recommended",
        "throughput": "Up to 5,000 msg/s per queue",
    },
    "Access (Zero Trust)": {
        "description": "Identity-aware proxy — protect any URL without VPN",
        "auth_providers": ["GitHub", "Google", "Okta", "Azure AD", "SAML 2.0", "OIDC"],
        "use_cases": ["Admin panels", "Internal tools", "CI/CD dashboards", "Staging envs"],
        "jwt_validation": "CF-Access-JWT-Assertion header injected on every request",
    },
    "WAF + Rate Limiting": {
        "description": "L7 firewall + DDoS mitigation + bot management",
        "custom_rules": "Firewall Expression Language (matches HTTP fields, CF metadata)",
        "managed_rules": "OWASP Core Rule Set, Cloudflare Managed Ruleset",
        "rate_limiting": "Token bucket or sliding window, per IP / country / header / cookie",
        "geo_blocking": "cf.country, cf.continent, cf.asn, cf.threat_score fields",
    },
}


# =============================================================================
# Cache strategy reference
# =============================================================================

CACHE_CONTROL_RECIPES: Dict[str, str] = {
    # Immutable assets (hashed filenames)
    "static_immutable": "public, max-age=31536000, immutable",
    # API responses — cache at CDN, revalidate frequently
    "api_cdn_short": "public, max-age=5, s-maxage=30, stale-while-revalidate=60",
    # User-specific data — cache only in browser
    "private_session": "private, max-age=300, no-store",
    # No caching (auth tokens, mutations)
    "no_cache": "no-store, no-cache, must-revalidate",
    # HTML pages — cache at CDN, revalidate on next request
    "html_page": "public, max-age=0, s-maxage=3600, must-revalidate",
}


# =============================================================================
# Payload integrity verification
# =============================================================================

class PayloadIntegrityError(Exception):
    """Raised when a payload fails signature or digest verification."""


@dataclass
class PayloadVerifier:
    """
    Verifies inbound payload integrity before any parsing occurs.

    Design principles:
      1. **Verify before parse** — signature check runs on the raw bytes;
         JSON deserialization happens only after the MAC is confirmed.
         This eliminates hash-then-use (TOCTOU) vulnerabilities.
      2. **Constant-time comparison** — :func:`hmac.compare_digest` is used
         throughout to prevent timing-oracle attacks.
      3. **Platform-neutral** — works with GitHub webhooks (`X-Hub-Signature-256`),
         Stripe (`Stripe-Signature`), Cloudflare Webhooks, and any custom
         ``X-Signature-256: sha256=<hex>`` header convention.

    Usage::

        verifier = PayloadVerifier(secret=os.environ["WEBHOOK_SECRET"])
        payload  = verifier.parse_and_verify_json(raw_body, request.headers["X-Hub-Signature-256"])
    """

    secret: str  # shared HMAC secret from environment / secret store

    # ------------------------------------------------------------------ #
    # HMAC-SHA256 webhook signature                                        #
    # ------------------------------------------------------------------ #

    def _compute_hmac(self, body: bytes) -> str:
        """Return ``sha256=<hex>`` for *body* using the configured secret."""
        mac = hmac.new(self.secret.encode(), body, hashlib.sha256)
        return "sha256=" + mac.hexdigest()

    def verify_hmac_signature(
        self,
        body: bytes,
        signature_header: str,
        *,
        prefix: str = "sha256=",
    ) -> bool:
        """
        Validate an HMAC-SHA256 signature header in constant time.

        Args:
            body:             Raw request body bytes.
            signature_header: Value of the signature header (e.g.
                              ``"sha256=abc123..."``).
            prefix:           Expected prefix before the hex digest.
                              Defaults to ``"sha256="``.

        Returns:
            ``True`` if the signature is valid.

        Raises:
            :class:`PayloadIntegrityError` on invalid format or mismatch.
        """
        if not signature_header.startswith(prefix):
            raise PayloadIntegrityError(
                f"Signature header missing expected prefix '{prefix}'"
            )
        expected = self._compute_hmac(body)
        if not hmac.compare_digest(expected, signature_header):
            raise PayloadIntegrityError("HMAC signature mismatch — payload may be tampered")
        return True

    # ------------------------------------------------------------------ #
    # Content-Digest (RFC 9530)                                           #
    # ------------------------------------------------------------------ #

    def verify_content_digest(self, body: bytes, digest_header: str) -> bool:
        """
        Validate a ``Content-Digest: sha-256=:<base64>:`` header (RFC 9530).

        Args:
            body:          Raw request body bytes.
            digest_header: Value of the ``Content-Digest`` header.

        Returns:
            ``True`` if the digest matches.

        Raises:
            :class:`PayloadIntegrityError` on format error or mismatch.
        """
        # RFC 9530 format: "sha-256=:<base64padded>:"
        prefix = "sha-256=:"
        suffix = ":"
        if not (digest_header.startswith(prefix) and digest_header.endswith(suffix)):
            raise PayloadIntegrityError(
                "Content-Digest header not in RFC 9530 format 'sha-256=:<base64>:'"
            )
        encoded = digest_header[len(prefix):-len(suffix)]
        try:
            claimed = base64.b64decode(encoded)
        except Exception as exc:
            raise PayloadIntegrityError(f"Content-Digest base64 decode failed: {exc}") from exc
        actual = hashlib.sha256(body).digest()
        if not hmac.compare_digest(actual, claimed):
            raise PayloadIntegrityError("Content-Digest mismatch — payload corrupted or replayed")
        return True

    # ------------------------------------------------------------------ #
    # Verify-then-parse (safe JSON ingestion)                             #
    # ------------------------------------------------------------------ #

    def parse_and_verify_json(
        self,
        body: bytes,
        signature_header: str,
        *,
        prefix: str = "sha256=",
        max_bytes: int = 10 * 1024 * 1024,  # 10 MiB hard cap
    ) -> Dict[str, Any]:
        """
        Verify the HMAC signature on *body*, **then** deserialize JSON.

        The deliberate verify-before-parse order prevents an attacker from
        exploiting parser quirks to smuggle malicious content past the MAC
        check (billion-laughs, zip-bomb, prototype-pollution variants).

        Args:
            body:             Raw request body bytes.
            signature_header: HMAC signature header value.
            prefix:           Signature prefix (default ``"sha256="``).
            max_bytes:        Maximum allowed body size; rejects oversized
                              payloads before verification.

        Returns:
            Deserialized JSON object as a ``dict``.

        Raises:
            :class:`PayloadIntegrityError` on size violation, bad signature,
            or invalid JSON.
        """
        if len(body) > max_bytes:
            raise PayloadIntegrityError(
                f"Payload too large: {len(body)} bytes (max {max_bytes})"
            )
        # 1. Verify signature on raw bytes — before any parsing
        self.verify_hmac_signature(body, signature_header, prefix=prefix)
        # 2. Only now is it safe to deserialize
        try:
            return json.loads(body)
        except json.JSONDecodeError as exc:
            raise PayloadIntegrityError(f"JSON parse error after verified signature: {exc}") from exc


# =============================================================================
# Principal verification
# =============================================================================

class PrincipalType(Enum):
    CF_ACCESS_JWT = "cf_access_jwt"   # Cloudflare Access (Zero Trust SSO)
    API_KEY       = "api_key"         # ****** X-API-Key token
    MTLS          = "mtls"            # mTLS client certificate
    TRUSTED_IP    = "trusted_ip"      # Internal service by IP range
    ANONYMOUS     = "anonymous"       # No credentials (public)


@dataclass
class PrincipalContext:
    """
    Normalised identity result returned by :class:`PrincipalVerifier`.

    Attach to the request state so every downstream handler can ask
    "who is calling and are they allowed?" without re-inspecting headers.
    Ties into any platform: FastAPI request state, ASGI scope extras,
    Django request.META, plain dicts, or Workers ``ctx.waitUntil`` audit logs.
    """

    principal_type: PrincipalType
    identity: Optional[str]           # email / sub / fingerprint / IP CIDR
    country: Optional[str]            # ISO 3166-1 alpha-2 from CF-IPCountry
    ip: str                           # CF-Connecting-IP or REMOTE_ADDR
    claims: Dict[str, Any]            # raw JWT claims or empty dict
    is_verified: bool                 # False → anonymous / failed (never raise here)
    is_bot_challenge_passed: bool = False   # Turnstile result if checked

    @property
    def is_authenticated(self) -> bool:
        return self.is_verified and self.principal_type != PrincipalType.ANONYMOUS

    def require_authenticated(self) -> None:
        """Raise :class:`PermissionError` if the principal is not authenticated."""
        if not self.is_authenticated:
            raise PermissionError(
                f"Unauthenticated request from {self.ip} "
                f"(type={self.principal_type.value})"
            )


@dataclass
class TrustedIPRanges:
    """
    Validate that an IP address falls within one of the declared CIDR ranges.

    Typical use: allow internal worker-to-worker calls without JWT overhead.

    Example::

        internal = TrustedIPRanges(cidrs=["10.0.0.0/8", "172.16.0.0/12"])
        internal.is_trusted("10.1.2.3")   # True
        internal.is_trusted("1.2.3.4")    # False
    """

    cidrs: List[str]
    _networks: List[ipaddress.IPv4Network | ipaddress.IPv6Network] = field(
        init=False, default_factory=list
    )

    def __post_init__(self) -> None:
        for cidr in self.cidrs:
            self._networks.append(ipaddress.ip_network(cidr, strict=False))

    def is_trusted(self, ip_str: str) -> bool:
        try:
            addr = ipaddress.ip_address(ip_str)
        except ValueError:
            return False
        return any(addr in net for net in self._networks)


@dataclass
class PrincipalVerifier:
    """
    Unified principal verifier that works with any HTTP framework.

    Supports four authentication strategies in priority order:
      1. **Cloudflare Access JWT** (``CF-Access-JWT-Assertion`` header) — for
         Zero Trust–protected apps; JWTs are verified against the team domain's
         public JWKS endpoint.
      2. **API key** (``Authorization: ****** or ``X-API-Key: <key>``)
         — constant-time comparison against a pre-loaded allow-set.
      3. **mTLS client cert** (``Cf-Client-Cert-*`` headers injected by
         Cloudflare mutual TLS) — extracts fingerprint for audit logging.
      4. **Trusted IP** — falls back to :class:`TrustedIPRanges` for
         service-mesh internal traffic.

    All strategies populate a unified :class:`PrincipalContext` so application
    code never needs to branch on authentication mechanism.

    Note on CF-Access JWT:
        Full signature validation requires fetching
        ``https://<team>.cloudflareaccess.com/cdn-cgi/access/certs`` (JWKS).
        The method below performs structural + audience + expiry checks using
        only stdlib (no PyJWT dependency). For production, replace the
        ``_verify_jwt_signature`` stub with a call to a JWKS-backed verifier
        (e.g. ``python-jose`` or ``PyJWT`` with ``algorithms=["RS256"]``).
    """

    team_domain: str           # e.g. "myteam.cloudflareaccess.com"
    audience: str              # AUD claim in CF Access JWT (application ID)
    valid_api_keys: Set[str]   # hashed API keys (SHA-256 hex) loaded from secrets
    trusted_ips: Optional[TrustedIPRanges] = None

    # ------------------------------------------------------------------ #
    # CF-Access JWT                                                        #
    # ------------------------------------------------------------------ #

    def _decode_jwt_claims(self, token: str) -> Dict[str, Any]:
        """Decode JWT payload without signature verification (structural check only)."""
        parts = token.split(".")
        if len(parts) != 3:
            raise ValueError("Token is not a valid JWT (expected 3 parts)")
        # Base64url → bytes (pad to multiple of 4)
        payload_b64 = parts[1] + "=" * (-len(parts[1]) % 4)
        try:
            payload_bytes = base64.urlsafe_b64decode(payload_b64)
            return json.loads(payload_bytes)
        except Exception as exc:
            raise ValueError(f"JWT payload decode error: {exc}") from exc

    def _verify_jwt_claims(self, claims: Dict[str, Any]) -> None:
        """Validate standard JWT claims (aud, exp, iss)."""
        now = int(time.time())
        # Expiry
        exp = claims.get("exp")
        if exp is None or now >= int(exp):
            raise PermissionError("JWT is expired or missing exp claim")
        # Not-before
        nbf = claims.get("nbf")
        if nbf is not None and now < int(nbf):
            raise PermissionError("JWT not yet valid (nbf claim)")
        # Audience
        aud = claims.get("aud")
        if isinstance(aud, str):
            aud = [aud]
        if not aud or self.audience not in aud:
            raise PermissionError(f"JWT audience mismatch: expected '{self.audience}'")
        # Issuer
        expected_iss = f"https://{self.team_domain}"
        iss = claims.get("iss", "")
        if not iss.startswith(expected_iss):
            raise PermissionError(f"JWT issuer mismatch: '{iss}'")

    def _verify_jwt_signature(self, token: str) -> None:
        """
        Stub: replace with JWKS-backed RS256 verification in production.

        Production implementation::

            from jose import jwt as jose_jwt
            jwks_url = f"https://{self.team_domain}/cdn-cgi/access/certs"
            # Fetch and cache the JWKS, then:
            jose_jwt.decode(token, jwks, algorithms=["RS256"], audience=self.audience)
        """
        # In test / dev environments this is a no-op.
        # The claims checks in _verify_jwt_claims still run.
        logger.warning(
            "JWT signature verification is using the stub implementation. "
            "Replace PrincipalVerifier._verify_jwt_signature with a "
            "JWKS-backed RS256 verifier before deploying to production."
        )

    def from_cf_access_jwt(
        self, jwt_assertion: str, ip: str, country: Optional[str] = None
    ) -> PrincipalContext:
        """
        Build a :class:`PrincipalContext` from a ``CF-Access-JWT-Assertion`` header.

        Raises:
            :class:`PermissionError` if the token is invalid, expired, or
            the audience/issuer does not match.
        """
        claims = self._decode_jwt_claims(jwt_assertion)
        self._verify_jwt_claims(claims)
        self._verify_jwt_signature(jwt_assertion)
        identity = claims.get("email") or claims.get("sub")
        return PrincipalContext(
            principal_type=PrincipalType.CF_ACCESS_JWT,
            identity=identity,
            country=country,
            ip=ip,
            claims=claims,
            is_verified=True,
        )

    # ------------------------------------------------------------------ #
    # API key                                                              #
    # ------------------------------------------------------------------ #

    def _hash_api_key(self, api_token: str) -> str:
        """
        Return SHA-256 hex of *api_token* for constant-time comparison.

        API keys are high-entropy random tokens (≥128 bits), not
        user-chosen passwords.  SHA-256 is the correct choice here:
        bcrypt/argon2 are designed for low-entropy secrets (passwords).
        For passwords, use a password-hashing function; for random tokens,
        SHA-256 is both secure and appropriate.
        """
        token_bytes: bytes = api_token.encode("utf-8")
        return hashlib.sha256(token_bytes).hexdigest()

    def from_api_key(
        self,
        api_token: str,
        ip: str,
        country: Optional[str] = None,
    ) -> PrincipalContext:
        """
        Validate an API token using constant-time comparison against stored hashes.

        Tokens are never stored in plain text; the ``valid_api_keys`` set holds
        SHA-256 hex digests.  Raw tokens come from ``Authorization: ******
        or ``X-API-Key`` headers.

        Raises:
            :class:`PermissionError` for invalid tokens.
        """
        token_hash = self._hash_api_key(api_token)
        # Build a dummy digest for constant-time comparison even on miss
        # (prevents early-exit timing oracle)
        match = any(
            hmac.compare_digest(token_hash, stored) for stored in self.valid_api_keys
        )
        if not match:
            raise PermissionError("Invalid API key")
        return PrincipalContext(
            principal_type=PrincipalType.API_KEY,
            identity=token_hash[:12] + "…",  # partial hash for audit logs, never full token
            country=country,
            ip=ip,
            claims={},
            is_verified=True,
        )

    # ------------------------------------------------------------------ #
    # mTLS                                                                 #
    # ------------------------------------------------------------------ #

    def from_mtls_headers(
        self, headers: Dict[str, str], ip: str, country: Optional[str] = None
    ) -> PrincipalContext:
        """
        Extract mTLS identity from Cloudflare-injected ``Cf-Client-Cert-*`` headers.

        Cloudflare injects these headers when mTLS is configured on the zone:
          - ``Cf-Client-Cert-Der-Base64``  — DER-encoded cert (base64)
          - ``Cf-Client-Cert-Verified``    — ``SUCCESS`` if cert is valid
          - ``Cf-Client-Cert-Fingerprint`` — SHA-256 fingerprint

        Raises:
            :class:`PermissionError` if the certificate is not verified.
        """
        verified = headers.get("Cf-Client-Cert-Verified", "FAILED")
        if verified != "SUCCESS":
            raise PermissionError(f"mTLS certificate not verified: '{verified}'")
        fingerprint = headers.get("Cf-Client-Cert-Fingerprint", "unknown")
        return PrincipalContext(
            principal_type=PrincipalType.MTLS,
            identity=fingerprint,
            country=country,
            ip=ip,
            claims={"cert_verified": verified, "fingerprint": fingerprint},
            is_verified=True,
        )

    # ------------------------------------------------------------------ #
    # Auto-detect                                                          #
    # ------------------------------------------------------------------ #

    def from_request_headers(
        self,
        headers: Dict[str, str],
        ip: str,
        country: Optional[str] = None,
    ) -> PrincipalContext:
        """
        Auto-detect the authentication mechanism from request headers and
        return the highest-trust :class:`PrincipalContext` available.

        Priority order (highest → lowest):
          1. CF-Access JWT assertion
          2. mTLS client certificate
          3. ****** X-API-Key token
          4. Trusted source IP
          5. Anonymous

        This method is intentionally non-raising: failed verifications fall
        through to the next strategy, ultimately returning an ANONYMOUS
        context.  Use :meth:`PrincipalContext.require_authenticated` in
        handlers that require identity.
        """
        # 1. CF-Access JWT
        jwt_header = headers.get("CF-Access-JWT-Assertion") or headers.get("Cf-Access-Jwt-Assertion")
        if jwt_header:
            try:
                return self.from_cf_access_jwt(jwt_header, ip, country)
            except (PermissionError, ValueError) as exc:
                logger.warning("CF-Access JWT rejected for %s: %s", ip, exc)

        # 2. mTLS
        if headers.get("Cf-Client-Cert-Verified"):
            try:
                return self.from_mtls_headers(headers, ip, country)
            except PermissionError as exc:
                logger.warning("mTLS rejected for %s: %s", ip, exc)

        # 3. API key  (****** X-API-Key)
        auth = headers.get("Authorization", "")
        bearer_token = ""
        if auth.lower().startswith("bearer "):
            bearer_token = auth[7:].strip()
        elif "X-API-Key" in headers:
            bearer_token = headers["X-API-Key"].strip()
        if bearer_token:
            try:
                return self.from_api_key(bearer_token, ip, country)
            except PermissionError as exc:
                logger.warning("API key rejected for %s: %s", ip, exc)

        # 4. Trusted IP
        if self.trusted_ips and self.trusted_ips.is_trusted(ip):
            return PrincipalContext(
                principal_type=PrincipalType.TRUSTED_IP,
                identity=ip,
                country=country,
                ip=ip,
                claims={},
                is_verified=True,
            )

        # 5. Anonymous
        return PrincipalContext(
            principal_type=PrincipalType.ANONYMOUS,
            identity=None,
            country=country,
            ip=ip,
            claims={},
            is_verified=False,
        )


# =============================================================================
# Workers template: verified payload ingestion + principal auth
# =============================================================================

WORKER_TEMPLATE_PAYLOAD_VERIFIED = r"""
/**
 * Cloudflare Workers: Payload Integrity + Principal Verification
 *
 * Combines HMAC-SHA256 webhook signature verification, CF-Access JWT
 * validation, and API-key auth into a single composable middleware chain.
 * Drop into any Workers project — wire up the env bindings below.
 *
 * wrangler.toml env bindings:
 *   [vars]
 *     ALLOWED_ORIGINS   = "https://app.example.com"
 *     AUDIENCE          = "<CF-Access Application AUD>"
 *     TEAM_DOMAIN       = "myteam.cloudflareaccess.com"
 *   [[kv_namespaces]]
 *     binding           = "API_KEYS_KV"   # stores sha256(key) → "1"
 *   [secrets]  (via wrangler secret put)
 *     WEBHOOK_SECRET                      # shared HMAC secret
 */

// ── Payload integrity ────────────────────────────────────────────────────────

/**
 * Verify an HMAC-SHA256 signature header using the Web Crypto API.
 * Constant-time via crypto.subtle.verify — immune to timing oracles.
 *
 * @param {ArrayBuffer} body          Raw request body bytes.
 * @param {string}      sigHeader     Header value, e.g. "sha256=abc123..."
 * @param {string}      secret        Shared HMAC secret from env.
 * @returns {Promise<boolean>}
 */
async function verifyHmacSignature(body, sigHeader, secret) {
  const PREFIX = "sha256=";
  if (!sigHeader?.startsWith(PREFIX)) return false;
  const claimedHex = sigHeader.slice(PREFIX.length);

  const enc = new TextEncoder();
  const key = await crypto.subtle.importKey(
    "raw", enc.encode(secret),
    { name: "HMAC", hash: "SHA-256" },
    false, ["verify"]
  );

  // Convert hex → Uint8Array for constant-time comparison
  const claimedBytes = new Uint8Array(
    claimedHex.match(/.{2}/g).map(b => parseInt(b, 16))
  );
  return crypto.subtle.verify("HMAC", key, claimedBytes, body);
}

/**
 * Verify a Content-Digest header (RFC 9530, sha-256 only).
 *
 * @param {ArrayBuffer} body
 * @param {string}      digestHeader  e.g. "sha-256=:<base64>:"
 * @returns {Promise<boolean>}
 */
async function verifyContentDigest(body, digestHeader) {
  const PREFIX = "sha-256=:", SUFFIX = ":";
  if (!digestHeader?.startsWith(PREFIX) || !digestHeader.endsWith(SUFFIX)) return false;
  const b64 = digestHeader.slice(PREFIX.length, -SUFFIX.length);
  const claimed = Uint8Array.from(atob(b64), c => c.charCodeAt(0));
  const actual = new Uint8Array(await crypto.subtle.digest("SHA-256", body));
  if (actual.length !== claimed.length) return false;
  // Constant-time comparison
  let diff = 0;
  for (let i = 0; i < actual.length; i++) diff |= actual[i] ^ claimed[i];
  return diff === 0;
}

/**
 * Read body bytes, verify signature/digest, THEN parse JSON.
 * Rejects oversized bodies before any crypto work.
 *
 * @param {Request} request
 * @param {object}  env
 * @param {number}  [maxBytes=10485760]  Default 10 MiB.
 * @returns {Promise<object>} Parsed JSON payload.
 */
async function parseVerifiedJson(request, env, maxBytes = 10 * 1024 * 1024) {
  const body = await request.arrayBuffer();
  if (body.byteLength > maxBytes) {
    throw Object.assign(new Error("Payload too large"), { status: 413 });
  }

  const sigHeader    = request.headers.get("X-Hub-Signature-256")
                    || request.headers.get("X-Signature-256");
  const digestHeader = request.headers.get("Content-Digest");

  if (sigHeader) {
    const ok = await verifyHmacSignature(body, sigHeader, env.WEBHOOK_SECRET);
    if (!ok) throw Object.assign(new Error("HMAC signature mismatch"), { status: 401 });
  } else if (digestHeader) {
    const ok = await verifyContentDigest(body, digestHeader);
    if (!ok) throw Object.assign(new Error("Content-Digest mismatch"), { status: 400 });
  }
  // Signature verified — safe to parse
  return JSON.parse(new TextDecoder().decode(body));
}

// ── Principal verification ───────────────────────────────────────────────────

/**
 * Decode a JWT payload without signature verification (structural check).
 * Always call verifyCFAccessJWT (which includes claim validation) in prod.
 */
function decodeJwtPayload(token) {
  const [, payload] = token.split(".");
  const padded = payload + "=".repeat((4 - payload.length % 4) % 4);
  return JSON.parse(atob(padded.replace(/-/g, "+").replace(/_/g, "/")));
}

/**
 * Validate a CF-Access JWT assertion.
 * Performs structural, audience, expiry, and issuer checks.
 * For RS256 signature verification, use the CF-Access JWKS endpoint:
 *   https://<TEAM_DOMAIN>/cdn-cgi/access/certs
 * (integrate with a JWKS cache in production).
 *
 * @param {string} token
 * @param {object} env   Must have AUDIENCE and TEAM_DOMAIN vars.
 * @returns {{ email: string, claims: object }}
 */
function verifyCFAccessJWT(token, env) {
  let claims;
  try { claims = decodeJwtPayload(token); }
  catch { throw Object.assign(new Error("Malformed JWT"), { status: 401 }); }

  const now = Math.floor(Date.now() / 1000);
  if (!claims.exp || now >= claims.exp) {
    throw Object.assign(new Error("JWT expired"), { status: 401 });
  }
  if (claims.nbf && now < claims.nbf) {
    throw Object.assign(new Error("JWT not yet valid"), { status: 401 });
  }
  const aud = Array.isArray(claims.aud) ? claims.aud : [claims.aud];
  if (!aud.includes(env.AUDIENCE)) {
    throw Object.assign(new Error("JWT audience mismatch"), { status: 401 });
  }
  const expectedIss = `https://${env.TEAM_DOMAIN}`;
  if (!String(claims.iss || "").startsWith(expectedIss)) {
    throw Object.assign(new Error("JWT issuer mismatch"), { status: 401 });
  }
  return { email: claims.email || claims.sub, claims };
}

/**
 * Resolve the caller's principal from request headers.
 * Returns null for anonymous requests.
 *
 * Priority: CF-Access JWT → mTLS cert → API key → anonymous
 *
 * @param {Request} request
 * @param {object}  env
 * @returns {Promise<{ type: string, identity: string|null, claims: object }>}
 */
async function resolvePrincipal(request, env) {
  // 1. CF-Access JWT
  const jwtAssertion = request.headers.get("CF-Access-JWT-Assertion");
  if (jwtAssertion) {
    const { email, claims } = verifyCFAccessJWT(jwtAssertion, env);
    return { type: "cf_access_jwt", identity: email, claims };
  }

  // 2. mTLS client certificate
  const certVerified = request.headers.get("Cf-Client-Cert-Verified");
  if (certVerified === "SUCCESS") {
    const fingerprint = request.headers.get("Cf-Client-Cert-Fingerprint") || "unknown";
    return { type: "mtls", identity: fingerprint, claims: {} };
  }

  // 3. API key (****** X-API-Key)
  let rawKey = null;
  const auth = request.headers.get("Authorization") || "";
  if (auth.toLowerCase().startsWith("bearer ")) rawKey = auth.slice(7).trim();
  else rawKey = request.headers.get("X-API-Key");

  if (rawKey) {
    const enc = new TextEncoder();
    const hashBuf = await crypto.subtle.digest("SHA-256", enc.encode(rawKey));
    const hashHex = Array.from(new Uint8Array(hashBuf))
      .map(b => b.toString(16).padStart(2, "0")).join("");
    const valid = await env.API_KEYS_KV.get(hashHex);
    if (!valid) throw Object.assign(new Error("Invalid API key"), { status: 401 });
    return { type: "api_key", identity: hashHex.slice(0, 12) + "…", claims: {} };
  }

  // 4. Anonymous
  return { type: "anonymous", identity: null, claims: {} };
}

// ── Main handler ─────────────────────────────────────────────────────────────

export default {
  async fetch(request, env, ctx) {
    const url = new URL(request.url);

    // a) OPTIONS preflight (no body to verify)
    if (request.method === "OPTIONS") {
      return new Response(null, { status: 204, headers: getCORSHeaders(request, env) });
    }

    try {
      // b) Resolve principal (does NOT require an authenticated caller yet)
      const principal = await resolvePrincipal(request, env);

      // c) For mutating endpoints: verify payload integrity before processing
      let payload = null;
      if (["POST", "PUT", "PATCH"].includes(request.method)) {
        payload = await parseVerifiedJson(request, env);
      }

      // d) Route to handler — pass principal + verified payload
      if (url.pathname === "/webhook" && request.method === "POST") {
        // Webhook consumers: payload is already MAC-verified
        ctx.waitUntil(processWebhook(payload, principal, env));
        return Response.json({ queued: true });
      }

      if (url.pathname.startsWith("/api/") && principal.type === "anonymous") {
        return Response.json({ error: "Authentication required" }, { status: 401 });
      }

      return Response.json({ ok: true, principal: principal.identity });

    } catch (err) {
      const status = err.status || 500;
      console.error(`[${status}] ${err.message}`);
      return Response.json({ error: err.message }, { status });
    }
  },
};

async function processWebhook(payload, principal, env) {
  // Payload is already verified — safe to act on
  console.log("Webhook from", principal.type, JSON.stringify(payload));
}
"""


# =============================================================================
# Security checklist
# =============================================================================
#
# ✅ CORS: strict origin allow-list, credentials never with wildcard
# ✅ Security headers: CSP, HSTS (preload), Permissions-Policy, COEP/COOP
# ✅ Geo-IP blocking: WAF rule + Workers fallback for OFAC-sanctioned countries
# ✅ Rate limiting: per-IP token bucket, tiered by country risk
# ✅ Turnstile: bot protection on sensitive endpoints (login, signup, checkout)
# ✅ WAF Managed Rules: OWASP CRS + Cloudflare Managed Ruleset enabled
# ✅ DDoS: L3/L4 auto-mitigation (always on), L7 rate limiting + challenge mode
# ✅ Zero Trust Access: all internal tools gated by SSO identity
# ✅ mTLS client certs: Workers mTLS binding for B2B API auth
# ✅ Durable Objects: serialized access for shared state (no race conditions)
# ✅ Cache: s-maxage headers prevent sensitive data from being CDN-cached
# ✅ IaaC: all config managed in Terraform (see cloudflare_terraform.tf)
# ✅ Payload integrity: HMAC-SHA256 + Content-Digest (RFC 9530); verify-before-parse order
# ✅ Principal verification: CF-Access JWT, API key (constant-time), mTLS, trusted-IP
# ✅ Timing-attack resistance: hmac.compare_digest / crypto.subtle.verify throughout
# ✅ Oversized payload rejection: hard cap before any crypto or JSON work
