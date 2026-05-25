"""
Cloudflare Platform S-Tier Reference

Apply to: Any service deployed behind Cloudflare — APIs, web apps, Workers,
          SaaS backends, CDN-accelerated static sites

Covers:
- CORS configuration (strict, permissive, preflight caching)
- Security headers (CSP, HSTS, Permissions-Policy, COEP/COOP)
- Geo-IP rate limiting (per-country, blocked regions, trusted regions)
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

import hashlib
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
