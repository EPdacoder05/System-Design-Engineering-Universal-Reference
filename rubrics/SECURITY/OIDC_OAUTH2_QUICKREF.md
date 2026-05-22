# OIDC vs OAuth2 Quick Reference

## Canonical Definitions
- **OAuth2**: Delegated authorization framework (access tokens + scopes).
- **OIDC**: Authentication and identity layer built on OAuth2 (ID token + identity claims).

## Decision Rule
- Use **OAuth2 only** when you need service/API authorization without end-user identity.
- Use **OIDC + OAuth2** when user login and verified identity claims are required.

## Minimum Security Checks
- [ ] PKCE for public clients
- [ ] `iss` validation
- [ ] `aud` validation
- [ ] Expiry and rotation handling for tokens/keys
- [ ] Least-privilege scopes
- [ ] Signature verification (JWKS) with key rotation support

## Baseline Anti-Patterns to Avoid
- Treating ID tokens as API access tokens
- Skipping audience validation
- Over-broad scopes by default
- Long-lived tokens without rotation/revocation strategy
