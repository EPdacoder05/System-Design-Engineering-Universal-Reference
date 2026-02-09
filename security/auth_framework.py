"""
Authentication & Authorization Framework

Apply to: APIs, web apps, internal tools, admin panels
No hardcoded secrets - all from environment variables

Features:
- JWT token creation, validation, refresh
- API key generation and validation with rate limiting
- RBAC with hierarchical permission checking
- Token rotation mechanism
- MFA TOTP generation and verification
"""

import os
import secrets
import time
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Set
from collections import defaultdict
import hashlib
import hmac

try:
    from jose import JWTError, jwt
    import pyotp
except ImportError:
    print("Install dependencies: pip install python-jose[cryptography] pyotp")


# Configuration - MUST be set via environment variables
JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", "CHANGE_ME_IN_PRODUCTION")
JWT_ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")
JWT_ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("JWT_ACCESS_TOKEN_EXPIRE_MINUTES", "30"))
JWT_REFRESH_TOKEN_EXPIRE_DAYS = int(os.getenv("JWT_REFRESH_TOKEN_EXPIRE_DAYS", "7"))


# ============================================================================
# JWT Token Management
# ============================================================================


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """
    Create JWT access token.
    
    Apply to: User authentication, API access tokens
    
    Args:
        data: Payload to encode (e.g., {"sub": user_id, "role": "admin"})
        expires_delta: Token expiration time (default: 30 minutes)
    
    Returns:
        Encoded JWT token string
    
    Example:
        token = create_access_token({"sub": "user123", "role": "admin"})
    """
    to_encode = data.copy()
    
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=JWT_ACCESS_TOKEN_EXPIRE_MINUTES)
    
    to_encode.update({
        "exp": expire,
        "iat": datetime.utcnow(),
        "type": "access"
    })
    
    encoded_jwt = jwt.encode(to_encode, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)
    return encoded_jwt


def create_refresh_token(data: dict) -> str:
    """
    Create JWT refresh token (longer expiration).
    
    Apply to: Token rotation, mobile apps, long-lived sessions
    
    Args:
        data: Payload to encode (usually just user ID)
    
    Returns:
        Encoded refresh token
    
    Example:
        refresh = create_refresh_token({"sub": "user123"})
    """
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(days=JWT_REFRESH_TOKEN_EXPIRE_DAYS)
    
    to_encode.update({
        "exp": expire,
        "iat": datetime.utcnow(),
        "type": "refresh"
    })
    
    encoded_jwt = jwt.encode(to_encode, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)
    return encoded_jwt


def verify_token(token: str, expected_type: str = "access") -> Optional[Dict]:
    """
    Verify and decode JWT token.
    
    Apply to: API authentication middleware, protected routes
    
    Args:
        token: JWT token string
        expected_type: Expected token type ("access" or "refresh")
    
    Returns:
        Decoded payload if valid, None if invalid
    
    Example:
        payload = verify_token(token)
        if payload:
            user_id = payload["sub"]
    """
    try:
        payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
        
        # Verify token type
        if payload.get("type") != expected_type:
            return None
        
        return payload
    except JWTError:
        return None


def refresh_access_token(refresh_token: str) -> Optional[str]:
    """
    Generate new access token from refresh token.
    
    Apply to: Token rotation without re-authentication
    
    Args:
        refresh_token: Valid refresh token
    
    Returns:
        New access token if refresh valid, None otherwise
    
    Example:
        new_token = refresh_access_token(old_refresh_token)
    """
    payload = verify_token(refresh_token, expected_type="refresh")
    
    if not payload:
        return None
    
    # Create new access token with same user data
    new_token_data = {"sub": payload["sub"]}
    if "role" in payload:
        new_token_data["role"] = payload["role"]
    
    return create_access_token(new_token_data)


# ============================================================================
# API Key Management
# ============================================================================


class APIKeyManager:
    """
    API key generation and validation with rate limiting.
    
    Apply to: Third-party integrations, service-to-service auth, webhook signatures
    
    Example:
        manager = APIKeyManager()
        api_key, secret = manager.generate_api_key(user_id="user123")
        
        # Validate incoming request
        if manager.validate_api_key(api_key, secret):
            print("Valid API key")
    """
    
    def __init__(self):
        # In-memory storage (use Redis/database in production)
        self._api_keys: Dict[str, Dict] = {}
        self._rate_limits: Dict[str, List[float]] = defaultdict(list)
    
    def generate_api_key(self, user_id: str, name: str = "") -> tuple[str, str]:
        """
        Generate new API key and secret.
        
        Args:
            user_id: User/service identifier
            name: Human-readable name for the key
        
        Returns:
            Tuple of (api_key, api_secret)
        """
        api_key = f"sk_{secrets.token_urlsafe(32)}"
        api_secret = secrets.token_urlsafe(48)
        
        # Hash the secret before storing (never store plaintext)
        secret_hash = hashlib.sha256(api_secret.encode()).hexdigest()
        
        self._api_keys[api_key] = {
            "user_id": user_id,
            "secret_hash": secret_hash,
            "name": name,
            "created_at": datetime.utcnow(),
            "last_used": None,
            "rate_limit_per_minute": 60  # Default: 60 requests/minute
        }
        
        return api_key, api_secret
    
    def validate_api_key(self, api_key: str, api_secret: str) -> bool:
        """
        Validate API key and secret.
        
        Args:
            api_key: The API key
            api_secret: The API secret
        
        Returns:
            True if valid and rate limit not exceeded
        """
        if api_key not in self._api_keys:
            return False
        
        key_data = self._api_keys[api_key]
        
        # Verify secret
        secret_hash = hashlib.sha256(api_secret.encode()).hexdigest()
        if not hmac.compare_digest(secret_hash, key_data["secret_hash"]):
            return False
        
        # Check rate limit
        if not self._check_rate_limit(api_key, key_data["rate_limit_per_minute"]):
            return False
        
        # Update last used
        key_data["last_used"] = datetime.utcnow()
        return True
    
    def _check_rate_limit(self, api_key: str, limit_per_minute: int) -> bool:
        """Check if request is within rate limit."""
        now = time.time()
        minute_ago = now - 60
        
        # Remove old entries
        self._rate_limits[api_key] = [
            ts for ts in self._rate_limits[api_key] if ts > minute_ago
        ]
        
        # Check limit
        if len(self._rate_limits[api_key]) >= limit_per_minute:
            return False
        
        # Add current request
        self._rate_limits[api_key].append(now)
        return True
    
    def revoke_api_key(self, api_key: str) -> bool:
        """Revoke an API key."""
        if api_key in self._api_keys:
            del self._api_keys[api_key]
            if api_key in self._rate_limits:
                del self._rate_limits[api_key]
            return True
        return False


# ============================================================================
# Role-Based Access Control (RBAC)
# ============================================================================


class RBACManager:
    """
    Role-Based Access Control with hierarchical permissions.
    
    Apply to: Multi-tenant apps, admin panels, internal tools
    
    Example:
        rbac = RBACManager()
        rbac.add_role("admin", ["users:read", "users:write", "users:delete"])
        rbac.add_role("editor", ["users:read", "users:write"])
        
        if rbac.has_permission("admin", "users:delete"):
            # Allow operation
            pass
    """
    
    def __init__(self):
        self._roles: Dict[str, Set[str]] = {}
        self._user_roles: Dict[str, Set[str]] = {}
        
        # Define role hierarchy (child inherits from parent)
        self._role_hierarchy = {
            "admin": ["editor", "viewer"],
            "editor": ["viewer"],
            "viewer": []
        }
    
    def add_role(self, role: str, permissions: List[str]):
        """Define a role with permissions."""
        self._roles[role] = set(permissions)
    
    def assign_role_to_user(self, user_id: str, role: str):
        """Assign role to user."""
        if user_id not in self._user_roles:
            self._user_roles[user_id] = set()
        self._user_roles[user_id].add(role)
    
    def has_permission(self, user_id: str, permission: str) -> bool:
        """
        Check if user has permission (including inherited from role hierarchy).
        
        Args:
            user_id: User identifier
            permission: Permission string (e.g., "users:write")
        
        Returns:
            True if user has permission
        """
        if user_id not in self._user_roles:
            return False
        
        # Get all roles (including inherited)
        all_roles = self._get_all_roles(user_id)
        
        # Check if any role has the permission
        for role in all_roles:
            if role in self._roles and permission in self._roles[role]:
                return True
        
        return False
    
    def _get_all_roles(self, user_id: str) -> Set[str]:
        """Get all roles including inherited ones."""
        direct_roles = self._user_roles.get(user_id, set())
        all_roles = set(direct_roles)
        
        # Add inherited roles
        for role in direct_roles:
            if role in self._role_hierarchy:
                all_roles.update(self._role_hierarchy[role])
        
        return all_roles


# ============================================================================
# Multi-Factor Authentication (MFA)
# ============================================================================


def generate_totp_secret() -> str:
    """
    Generate TOTP secret for MFA.
    
    Apply to: User account security, admin access
    
    Returns:
        Base32-encoded secret
    
    Example:
        secret = generate_totp_secret()
        # Store secret in user's record
        # Generate QR code: pyotp.totp.TOTP(secret).provisioning_uri("user@example.com", issuer_name="MyApp")
    """
    return pyotp.random_base32()


def verify_totp_token(secret: str, token: str) -> bool:
    """
    Verify TOTP token (6-digit code).
    
    Apply to: Login flow, sensitive operations
    
    Args:
        secret: User's TOTP secret
        token: 6-digit code from authenticator app
    
    Returns:
        True if token is valid
    
    Example:
        if verify_totp_token(user.mfa_secret, user_input_code):
            # Allow access
            pass
    """
    totp = pyotp.TOTP(secret)
    return totp.verify(token, valid_window=1)  # Allow 30-second window


def get_current_totp_token(secret: str) -> str:
    """
    Get current TOTP token (for testing/debugging).
    
    Args:
        secret: TOTP secret
    
    Returns:
        Current 6-digit token
    """
    totp = pyotp.TOTP(secret)
    return totp.now()


# ============================================================================
# Example Usage
# ============================================================================


if __name__ == "__main__":
    print("=== Authentication Framework Demo ===\n")
    
    # 1. JWT Tokens
    print("1. JWT Token Example:")
    access_token = create_access_token({"sub": "user123", "role": "admin"})
    print(f"Access Token: {access_token[:50]}...")
    
    refresh_token = create_refresh_token({"sub": "user123"})
    print(f"Refresh Token: {refresh_token[:50]}...")
    
    payload = verify_token(access_token)
    print(f"Decoded Payload: {payload}\n")
    
    # 2. API Keys
    print("2. API Key Example:")
    api_manager = APIKeyManager()
    api_key, api_secret = api_manager.generate_api_key(user_id="service1", name="Payment Service")
    print(f"API Key: {api_key}")
    print(f"API Secret: {api_secret}")
    
    is_valid = api_manager.validate_api_key(api_key, api_secret)
    print(f"Validation: {is_valid}\n")
    
    # 3. RBAC
    print("3. RBAC Example:")
    rbac = RBACManager()
    rbac.add_role("admin", ["users:read", "users:write", "users:delete"])
    rbac.add_role("editor", ["users:read", "users:write"])
    rbac.add_role("viewer", ["users:read"])
    
    rbac.assign_role_to_user("alice", "admin")
    rbac.assign_role_to_user("bob", "editor")
    
    print(f"Alice can delete users: {rbac.has_permission('alice', 'users:delete')}")
    print(f"Bob can delete users: {rbac.has_permission('bob', 'users:delete')}")
    print(f"Bob can read users: {rbac.has_permission('bob', 'users:read')}\n")
    
    # 4. MFA
    print("4. MFA Example:")
    mfa_secret = generate_totp_secret()
    print(f"MFA Secret: {mfa_secret}")
    
    current_token = get_current_totp_token(mfa_secret)
    print(f"Current TOTP Token: {current_token}")
    
    is_valid_mfa = verify_totp_token(mfa_secret, current_token)
    print(f"Token Valid: {is_valid_mfa}")
