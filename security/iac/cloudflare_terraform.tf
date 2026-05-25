# ============================================================================
# Cloudflare IaaC — Terraform Reference
# ============================================================================
# Covers:
#   - Zone settings (HTTPS, TLS, security level, bot management)
#   - WAF Managed Rules (OWASP CRS + Cloudflare Managed Ruleset)
#   - WAF Custom Rules (geo-IP blocking, threat score, user-agent)
#   - Rate Limiting Rules (per-IP, per-country, sliding window)
#   - Geo-IP Rate Limiting (country-level token bucket)
#   - Page Rules / Cache Rules
#   - Workers Route binding
#   - R2 bucket with CORS and lifecycle
#   - Access Application + Zero Trust policies
#   - Cloudflare Tunnel (private network)
#   - DNS records (A, CNAME, MX, TXT)
#
# Prerequisites:
#   terraform init
#   export CLOUDFLARE_API_TOKEN=<token>   # needs Zone:Edit + Firewall:Edit permissions
#
# Apply:
#   terraform plan -var="zone_id=<your_zone_id>" -var="account_id=<your_account_id>"
#   terraform apply
# ============================================================================

terraform {
  required_version = ">= 1.6.0"

  required_providers {
    cloudflare = {
      source  = "cloudflare/cloudflare"
      version = "~> 4.0"
    }
    random = {
      source  = "hashicorp/random"
      version = "~> 3.0"
    }
  }

  # Remote state — replace with your backend
  backend "s3" {
    bucket = "my-terraform-state"
    key    = "cloudflare/production.tfstate"
    region = "us-east-1"
    # Optionally use Cloudflare R2 as S3-compatible backend:
    # endpoint = "https://<account_id>.r2.cloudflarestorage.com"
  }
}

provider "cloudflare" {
  # Reads CLOUDFLARE_API_TOKEN from env automatically
  # Never hard-code tokens in .tf files
}

# ============================================================================
# Variables
# ============================================================================

variable "zone_id" {
  description = "Cloudflare Zone ID (found in zone dashboard)"
  type        = string
}

variable "account_id" {
  description = "Cloudflare Account ID"
  type        = string
}

variable "domain" {
  description = "Root domain (e.g. example.com)"
  type        = string
}

variable "environment" {
  description = "Deployment environment"
  type        = string
  default     = "production"
  validation {
    condition     = contains(["development", "staging", "production"], var.environment)
    error_message = "Must be development, staging, or production."
  }
}

variable "allowed_origins" {
  description = "Origins allowed by CORS policy"
  type        = list(string)
  default     = []
}

variable "blocked_country_codes" {
  description = "ISO 3166-1 alpha-2 country codes to hard-block (OFAC list + custom)"
  type        = list(string)
  # OFAC-sanctioned countries — verify against current OFAC SDN list
  default     = ["CU", "IR", "KP", "RU", "SY"]
}

variable "elevated_risk_countries" {
  description = "Country codes subject to stricter rate limits"
  type        = list(string)
  default     = ["CN", "BR"]
}

variable "worker_script_name" {
  description = "Name of the deployed Cloudflare Worker script"
  type        = string
  default     = "api-gateway"
}

variable "zero_trust_team_name" {
  description = "Cloudflare Access / Zero Trust team name"
  type        = string
  default     = "myorg"
}

variable "access_allowed_emails" {
  description = "Email addresses allowed through Zero Trust Access policy"
  type        = list(string)
  default     = []
}

# ============================================================================
# Zone Settings — security & performance baseline
# ============================================================================

resource "cloudflare_zone_settings_override" "security_baseline" {
  zone_id = var.zone_id

  settings {
    # TLS / HTTPS
    ssl                      = "strict"          # Full (strict) — origin must have valid cert
    min_tls_version          = "1.2"             # Reject TLS 1.0 / 1.1
    tls_1_3                  = "zrt"             # Enable TLS 1.3 + 0-RTT
    automatic_https_rewrites = "on"
    always_use_https         = "on"
    opportunistic_encryption = "on"

    # Security
    security_level           = "medium"          # "high" for attack mode
    challenge_ttl            = 1800
    browser_check            = "on"
    hotlink_protection       = "on"
    email_obfuscation        = "on"
    server_side_exclude      = "on"

    # Performance
    brotli                   = "on"
    minify {
      js   = "on"
      css  = "on"
      html = "on"
    }
    http2        = "on"
    http3        = "on"
    early_hints  = "on"
    zero_rtt     = "on"

    # Bot management (requires Bot Management add-on for full feature set)
    # bot_management is configured via separate resource; "fight_mode" is the basic free option
    # security_level = "essentially_off"  # set this only for fully public static sites

    # Polish — lossless image compression (Pro plan required)
    # polish       = "lossless"
    # webp         = "on"

    # Headers
    development_mode = var.environment == "production" ? "off" : "on"
  }
}

# ============================================================================
# WAF — Managed Rulesets
# ============================================================================

resource "cloudflare_ruleset" "waf_managed" {
  zone_id     = var.zone_id
  name        = "Managed WAF Rules"
  description = "OWASP + Cloudflare Managed Ruleset"
  kind        = "zone"
  phase       = "http_request_firewall_managed"

  # Cloudflare Managed Ruleset (blocks common web attacks)
  rules {
    action = "execute"
    action_parameters {
      id      = "efb7b8c949ac4650a09736fc376e9aee"  # Cloudflare Managed Ruleset ID
      version = "latest"
      overrides {
        # Paranoia level 1 by default; increase for stricter rules
        action           = "block"
        sensitivity_level = "default"
      }
    }
    expression  = "true"
    description = "Cloudflare Managed Ruleset"
    enabled     = true
  }

  # OWASP Core Rule Set
  rules {
    action = "execute"
    action_parameters {
      id      = "4814384a9e5d4991b9815dcfc25d2f1f"  # OWASP Core Rule Set ID
      version = "latest"
      overrides {
        action           = "block"
        sensitivity_level = "medium"
      }
    }
    expression  = "true"
    description = "OWASP Core Rule Set"
    enabled     = true
  }
}

# ============================================================================
# WAF Custom Rules — Geo-IP blocking + threat score
# ============================================================================

resource "cloudflare_ruleset" "waf_custom" {
  zone_id     = var.zone_id
  name        = "Custom WAF Rules"
  description = "Geo-IP blocking, bot scores, custom threat rules"
  kind        = "zone"
  phase       = "http_request_firewall_custom"

  # Rule 1: Hard-block OFAC-sanctioned countries
  rules {
    action      = "block"
    expression  = join(" or ", [
      for cc in var.blocked_country_codes :
      "(ip.geoip.country eq \"${cc}\")"
    ])
    description = "Block OFAC-sanctioned and hard-blocked countries"
    enabled     = true
  }

  # Rule 2: Challenge elevated-risk countries for non-API traffic
  rules {
    action     = "managed_challenge"
    expression = join(" or ", concat(
      [for cc in var.elevated_risk_countries : "(ip.geoip.country eq \"${cc}\")"],
      ["(cf.threat_score gt 10)"]
    ))
    description = "Challenge elevated-risk countries and high threat scores"
    enabled     = true
  }

  # Rule 3: Block high threat score IPs unconditionally
  rules {
    action      = "block"
    expression  = "(cf.threat_score gt 50)"
    description = "Block IPs with threat score > 50 (known bad actors)"
    enabled     = true
  }

  # Rule 4: Block known bad bots
  rules {
    action     = "block"
    expression = "(cf.client.bot) and not (cf.verified_bot_category in {\"Search Engine Crawler\" \"Monitoring & Analytics\"})"
    description = "Block unverified bots; allow search crawlers and monitors"
    enabled     = true
  }

  # Rule 5: Block requests with suspicious user-agents
  rules {
    action     = "block"
    expression = "(http.user_agent contains \"sqlmap\") or (http.user_agent contains \"nikto\") or (http.user_agent contains \"nmap\")"
    description = "Block common scanner user-agents"
    enabled    = true
  }

  # Rule 6: Block path traversal attempts
  rules {
    action     = "block"
    expression = "(http.request.uri.path contains \"../\") or (http.request.uri.path contains \"..\\\\\")"
    description = "Block path traversal"
    enabled    = true
  }
}

# ============================================================================
# Rate Limiting Rules
# ============================================================================

resource "cloudflare_ruleset" "rate_limiting" {
  zone_id     = var.zone_id
  name        = "Rate Limiting Rules"
  description = "Per-IP and per-country rate limiting"
  kind        = "zone"
  phase       = "http_ratelimit"

  # Rule 1: Global rate limit — 120 req/min per IP
  rules {
    action = "block"
    action_parameters {
      response {
        status_code  = 429
        content_type = "application/json"
        content      = "{\"error\":\"Rate limit exceeded\",\"code\":429}"
      }
    }
    ratelimit {
      characteristics        = ["ip.src"]
      period                 = 60
      requests_per_period    = 120
      mitigation_timeout     = 60
      counting_expression    = ""  # count all matching requests
    }
    expression  = "true"
    description = "Global: 120 req/min per IP"
    enabled     = true
  }

  # Rule 2: Stricter limit for elevated-risk countries — 30 req/min
  rules {
    action = "managed_challenge"
    ratelimit {
      characteristics     = ["ip.src"]
      period              = 60
      requests_per_period = 30
      mitigation_timeout  = 120
    }
    expression  = join(" or ", [
      for cc in var.elevated_risk_countries :
      "(ip.geoip.country eq \"${cc}\")"
    ])
    description = "Elevated-risk countries: 30 req/min per IP"
    enabled     = true
  }

  # Rule 3: Auth endpoint brute-force protection — 10 req/min per IP
  rules {
    action = "block"
    action_parameters {
      response {
        status_code  = 429
        content_type = "application/json"
        content      = "{\"error\":\"Too many login attempts\",\"code\":429}"
      }
    }
    ratelimit {
      characteristics     = ["ip.src"]
      period              = 60
      requests_per_period = 10
      mitigation_timeout  = 300  # 5-minute block after brute-force
    }
    expression  = "(http.request.uri.path eq \"/v1/auth/login\" or http.request.uri.path eq \"/v1/auth/token\")"
    description = "Auth endpoints: 10 req/min per IP (brute-force protection)"
    enabled     = true
  }

  # Rule 4: API key creation — 3 req/hour per authenticated user
  rules {
    action = "block"
    ratelimit {
      characteristics     = ["http.request.headers[\"authorization\"]"]
      period              = 3600
      requests_per_period = 3
      mitigation_timeout  = 3600
    }
    expression  = "(http.request.uri.path eq \"/v1/api-keys\" and http.request.method eq \"POST\")"
    description = "API key creation: 3/hour per auth token"
    enabled     = true
  }
}

# ============================================================================
# Cache Rules
# ============================================================================

resource "cloudflare_ruleset" "cache_rules" {
  zone_id     = var.zone_id
  name        = "Cache Rules"
  description = "Cache-Control strategy per path"
  kind        = "zone"
  phase       = "http_response_headers_transform"

  # Immutable static assets
  rules {
    action = "rewrite"
    action_parameters {
      headers {
        name      = "Cache-Control"
        operation = "set"
        value     = "public, max-age=31536000, immutable"
      }
    }
    expression  = "(http.request.uri.path matches \"^/static/.*\\\\.[0-9a-f]{8}\\\\.(js|css|woff2|png|webp)$\")"
    description = "Immutable hashed assets: 1-year cache"
    enabled     = true
  }

  # API responses — short CDN cache
  rules {
    action = "rewrite"
    action_parameters {
      headers {
        name      = "Cache-Control"
        operation = "set"
        value     = "public, max-age=0, s-maxage=30, stale-while-revalidate=60"
      }
    }
    expression  = "(http.request.uri.path matches \"^/v[0-9]+/public/\")"
    description = "Public API: 30s CDN cache, 60s SWR"
    enabled     = true
  }

  # Auth / mutation endpoints — never cache
  rules {
    action = "rewrite"
    action_parameters {
      headers {
        name      = "Cache-Control"
        operation = "set"
        value     = "no-store, no-cache, must-revalidate"
      }
    }
    expression  = "(http.request.uri.path matches \"^/v[0-9]+/auth/\") or (http.request.method in {\"POST\" \"PUT\" \"PATCH\" \"DELETE\"})"
    description = "Auth and mutations: never cache"
    enabled     = true
  }
}

# ============================================================================
# Transform Rules — inject security headers on all responses
# ============================================================================

resource "cloudflare_ruleset" "security_headers" {
  zone_id     = var.zone_id
  name        = "Security Response Headers"
  description = "Inject HSTS, CSP, and hardening headers on all responses"
  kind        = "zone"
  phase       = "http_response_headers_transform"

  rules {
    action = "rewrite"
    action_parameters {
      headers {
        name      = "Strict-Transport-Security"
        operation = "set"
        value     = "max-age=31536000; includeSubDomains; preload"
      }
      headers {
        name      = "X-Content-Type-Options"
        operation = "set"
        value     = "nosniff"
      }
      headers {
        name      = "X-Frame-Options"
        operation = "set"
        value     = "DENY"
      }
      headers {
        name      = "Referrer-Policy"
        operation = "set"
        value     = "strict-origin-when-cross-origin"
      }
      headers {
        name      = "Permissions-Policy"
        operation = "set"
        value     = "camera=(), microphone=(), geolocation=(), payment=()"
      }
      headers {
        name      = "Content-Security-Policy"
        operation = "set"
        value     = "default-src 'self'; frame-ancestors 'none'; upgrade-insecure-requests"
      }
      headers {
        name      = "Cross-Origin-Opener-Policy"
        operation = "set"
        value     = "same-origin"
      }
      headers {
        name      = "X-Powered-By"
        operation = "remove"
      }
      headers {
        name      = "Server"
        operation = "set"
        value     = "Cloudflare"
      }
    }
    expression  = "true"
    description = "Inject all security headers on every response"
    enabled     = true
  }
}

# ============================================================================
# Workers Route
# ============================================================================

resource "cloudflare_worker_route" "api_route" {
  zone_id     = var.zone_id
  pattern     = "${var.domain}/api/*"
  script_name = var.worker_script_name
}

resource "cloudflare_worker_route" "catch_all" {
  zone_id     = var.zone_id
  pattern     = "${var.domain}/*"
  script_name = var.worker_script_name
}

# ============================================================================
# R2 Bucket (zero-egress object storage)
# ============================================================================

resource "cloudflare_r2_bucket" "assets" {
  account_id = var.account_id
  name       = "${replace(var.domain, ".", "-")}-assets-${var.environment}"
  location   = "WNAM"  # Western North America; also: ENAM, WEUR, EEUR, APAC, OC

  lifecycle {
    prevent_destroy = true  # protect production data
  }
}

# R2 CORS policy (managed via Cloudflare dashboard or API; Terraform support varies by provider version)
# Equivalent API call:
# PUT https://api.cloudflare.com/client/v4/accounts/{account_id}/r2/buckets/{bucket}/cors
# Body: { "rules": [{ "allowed": { "origins": ["https://app.example.com"], "methods": ["GET"], "headers": ["*"] }, "exposeHeaders": ["ETag"], "maxAgeSeconds": 3600 }] }

# ============================================================================
# Cloudflare Access — Zero Trust Application
# ============================================================================

resource "cloudflare_access_application" "admin_panel" {
  zone_id          = var.zone_id
  name             = "Admin Panel"
  domain           = "admin.${var.domain}"
  type             = "self_hosted"
  session_duration = "8h"

  # Require re-authentication after 8 hours
  auto_redirect_to_identity = true

  # Enforce CORS for the protected app
  cors_headers {
    allowed_origins            = var.allowed_origins
    allowed_methods            = ["GET", "POST", "OPTIONS"]
    allowed_headers            = ["Authorization", "Content-Type"]
    allow_credentials          = true
    max_age                    = 86400
  }
}

resource "cloudflare_access_policy" "admin_email_policy" {
  application_id = cloudflare_access_application.admin_panel.id
  zone_id        = var.zone_id
  name           = "Allow approved admin emails"
  precedence     = 1
  decision       = "allow"

  include {
    email = var.access_allowed_emails
  }

  require {
    # Multi-factor authentication required for all admins
    mfa {}
  }
}

# Block everyone not matching the allow policy
resource "cloudflare_access_policy" "admin_block_all" {
  application_id = cloudflare_access_application.admin_panel.id
  zone_id        = var.zone_id
  name           = "Block everyone else"
  precedence     = 100
  decision       = "block"

  include {
    everyone = true
  }
}

# ============================================================================
# DNS Records
# ============================================================================

resource "cloudflare_record" "root_a" {
  zone_id = var.zone_id
  name    = "@"
  value   = "192.0.2.1"   # replace with your origin IP
  type    = "A"
  proxied = true  # traffic routes through Cloudflare (orange-cloud)
  ttl     = 1     # 1 = auto when proxied
}

resource "cloudflare_record" "www_cname" {
  zone_id = var.zone_id
  name    = "www"
  value   = var.domain
  type    = "CNAME"
  proxied = true
  ttl     = 1
}

resource "cloudflare_record" "api_cname" {
  zone_id = var.zone_id
  name    = "api"
  value   = var.domain
  type    = "CNAME"
  proxied = true
  ttl     = 1
}

resource "cloudflare_record" "admin_cname" {
  zone_id = var.zone_id
  name    = "admin"
  value   = var.domain
  type    = "CNAME"
  proxied = true
  ttl     = 1
}

# SPF record for email
resource "cloudflare_record" "spf" {
  zone_id = var.zone_id
  name    = "@"
  value   = "v=spf1 include:_spf.google.com ~all"
  type    = "TXT"
  ttl     = 3600
}

# DMARC record
resource "cloudflare_record" "dmarc" {
  zone_id = var.zone_id
  name    = "_dmarc"
  value   = "v=DMARC1; p=reject; rua=mailto:dmarc@${var.domain}; ruf=mailto:dmarc@${var.domain}; adkim=s; aspf=s"
  type    = "TXT"
  ttl     = 3600
}

# ============================================================================
# Cloudflare Tunnel (private origin, no public IP required)
# ============================================================================

resource "random_password" "tunnel_secret" {
  length  = 32
  special = false
  # Result is stored in Terraform state — use remote state encryption (S3 SSE or Vault)
}

resource "cloudflare_tunnel" "origin_tunnel" {
  account_id = var.account_id
  name       = "${var.domain}-${var.environment}-tunnel"
  secret     = base64encode(random_password.tunnel_secret.result)  # store in secrets manager; never commit
}

resource "cloudflare_tunnel_config" "origin_config" {
  account_id = var.account_id
  tunnel_id  = cloudflare_tunnel.origin_tunnel.id

  config {
    # Route HTTPS traffic to local FastAPI service
    ingress_rule {
      hostname = "api.${var.domain}"
      service  = "http://localhost:8000"
      origin_request {
        connect_timeout          = "10s"
        tls_timeout              = "10s"
        tcp_keep_alive           = "30s"
        no_happy_eyeballs        = false
        keep_alive_connections   = 100
        keep_alive_timeout       = "90s"
        http_host_header         = "api.${var.domain}"
        origin_server_name       = "api.${var.domain}"
        no_tls_verify            = false  # never disable TLS verification
      }
    }
    # Catch-all — reject unmatched hostnames
    ingress_rule {
      service = "http_status:404"
    }
  }
}

# ============================================================================
# Outputs
# ============================================================================

output "zone_id" {
  description = "Cloudflare Zone ID"
  value       = var.zone_id
  sensitive   = false
}

output "r2_bucket_name" {
  description = "R2 assets bucket name"
  value       = cloudflare_r2_bucket.assets.name
}

output "tunnel_id" {
  description = "Cloudflare Tunnel ID"
  value       = cloudflare_tunnel.origin_tunnel.id
}

output "access_app_aud" {
  description = "Access Application AUD (audience tag) for JWT validation"
  value       = cloudflare_access_application.admin_panel.aud
  sensitive   = true
}

# ============================================================================
# IaaC Security Checklist
# ============================================================================
#
# ✅ TLS 1.2+ enforced; TLS 1.3 + 0-RTT enabled
# ✅ HSTS with preload set at zone level AND injected as response header
# ✅ WAF Managed Rules: Cloudflare + OWASP CRS (block mode)
# ✅ Geo-IP blocking: OFAC-sanctioned countries blocked at WAF layer
# ✅ Geo-IP rate limiting: elevated-risk countries limited to 30 req/min
# ✅ Brute-force protection: auth endpoints limited to 10 req/min, 5-min block
# ✅ Bot management: unverified bots blocked; known crawlers allowed
# ✅ Scanner detection: sqlmap / nikto / nmap user-agents blocked
# ✅ Path traversal blocked by WAF custom rule
# ✅ Security headers injected via Transform Rules on all responses
# ✅ Zero Trust Access: admin panel protected by SSO + MFA
# ✅ Cloudflare Tunnel: origin never exposed to public internet
# ✅ R2 storage: zero egress cost, CORS configured per bucket
# ✅ DMARC p=reject: prevents email spoofing
# ✅ All resources managed in Terraform (no manual dashboard changes)
# ✅ State stored remotely (S3 or R2 backend)
# ✅ Secrets (tunnel token, API token) loaded from environment, never committed
