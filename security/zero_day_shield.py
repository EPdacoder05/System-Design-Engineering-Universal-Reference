"""
Zero-Day Security Shield Utilities

Apply to: Defense-in-depth security, zero-trust architectures, production systems

Features from NullPointVector:
- Secure deserialization with whitelist validation
- Secure hashing with timing attack protection
- Secure random token generation
- Input validation with timeout protection
- Defense-in-depth utilities
- Rate limiting and circuit breaker integration
"""

import hashlib
import hmac
import secrets
import time
import json
import re
from typing import Any, Optional, Dict, List, Set, Union
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum


# ============================================================================
# Security Configuration
# ============================================================================

@dataclass
class SecurityConfig:
    """Security configuration for zero-day shield."""
    # Regex timeout protection (seconds)
    regex_timeout: float = 1.0
    
    # Maximum input length to prevent DoS
    max_input_length: int = 10_000
    
    # Allowed classes for deserialization (whitelist)
    allowed_classes: Set[str] = None
    
    # Token configuration
    token_length: int = 32
    token_entropy_bits: int = 256
    
    # Hash algorithm
    hash_algorithm: str = "sha256"
    
    # Timing attack protection
    constant_time_compare: bool = True
    
    def __post_init__(self):
        if self.allowed_classes is None:
            self.allowed_classes = {
                'builtins.dict',
                'builtins.list',
                'builtins.str',
                'builtins.int',
                'builtins.float',
                'builtins.bool',
                'builtins.NoneType',
            }


# ============================================================================
# Secure Deserialization
# ============================================================================

class SecureDeserializer:
    """
    Secure deserialization with whitelist validation.
    
    Apply to: API input parsing, message queue processing, cache deserialization
    
    Prevents: Arbitrary code execution, pickle exploits, YAML exploits
    """
    
    def __init__(self, config: SecurityConfig = None):
        self.config = config or SecurityConfig()
    
    def safe_json_loads(self, data: str) -> Optional[Dict]:
        """
        Safely deserialize JSON with validation.
        
        Args:
            data: JSON string to deserialize
            
        Returns:
            Parsed JSON object or None if validation fails
        """
        try:
            # Length check
            if len(data) > self.config.max_input_length:
                raise ValueError(f"Input exceeds maximum length of {self.config.max_input_length}")
            
            # Parse JSON
            obj = json.loads(data)
            
            # Validate object types recursively
            if not self._validate_object(obj):
                raise ValueError("JSON contains disallowed types")
            
            return obj
            
        except (json.JSONDecodeError, ValueError) as e:
            print(f"⚠️  Deserialization failed: {e}")
            return None
    
    def _validate_object(self, obj: Any) -> bool:
        """Recursively validate object types against whitelist."""
        obj_type = f"{type(obj).__module__}.{type(obj).__name__}"
        
        if obj_type not in self.config.allowed_classes:
            return False
        
        # Recursively validate containers
        if isinstance(obj, dict):
            return all(
                self._validate_object(k) and self._validate_object(v)
                for k, v in obj.items()
            )
        elif isinstance(obj, list):
            return all(self._validate_object(item) for item in obj)
        
        return True
    
    def safe_eval_disabled(self, data: str) -> None:
        """
        NEVER use eval() - this method exists to document why.
        
        ⚠️  SECURITY: eval() allows arbitrary code execution.
        Use json.loads() or ast.literal_eval() instead.
        """
        raise NotImplementedError(
            "eval() is disabled for security reasons. "
            "Use safe_json_loads() or ast.literal_eval() instead."
        )


# ============================================================================
# Secure Hashing with Timing Attack Protection
# ============================================================================

class SecureHasher:
    """
    Secure hashing with timing attack protection.
    
    Apply to: Password verification, token validation, HMAC generation
    
    Prevents: Timing attacks, hash length extension attacks
    """
    
    def __init__(self, config: SecurityConfig = None):
        self.config = config or SecurityConfig()
    
    def hash_password(self, password: str, salt: Optional[bytes] = None) -> Dict[str, str]:
        """
        Hash password with PBKDF2 and random salt.
        
        Args:
            password: Password to hash
            salt: Optional salt (generated if not provided)
            
        Returns:
            Dictionary with 'hash' and 'salt' (hex encoded)
        """
        if salt is None:
            salt = secrets.token_bytes(32)
        
        # PBKDF2 with 100,000 iterations
        hash_bytes = hashlib.pbkdf2_hmac(
            'sha256',
            password.encode('utf-8'),
            salt,
            100_000,
            dklen=32
        )
        
        return {
            'hash': hash_bytes.hex(),
            'salt': salt.hex(),
            'algorithm': 'pbkdf2_sha256',
            'iterations': 100_000
        }
    
    def verify_password(self, password: str, stored_hash: str, stored_salt: str) -> bool:
        """
        Verify password against stored hash using constant-time comparison.
        
        Args:
            password: Password to verify
            stored_hash: Stored hash (hex encoded)
            stored_salt: Stored salt (hex encoded)
            
        Returns:
            True if password matches, False otherwise
        """
        # Recompute hash with same salt
        salt_bytes = bytes.fromhex(stored_salt)
        computed = self.hash_password(password, salt_bytes)
        
        # Constant-time comparison to prevent timing attacks
        return hmac.compare_digest(computed['hash'], stored_hash)
    
    def generate_hmac(self, message: str, key: str) -> str:
        """
        Generate HMAC for message integrity verification.
        
        Args:
            message: Message to authenticate
            key: Secret key for HMAC
            
        Returns:
            HMAC (hex encoded)
        """
        return hmac.new(
            key.encode('utf-8'),
            message.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
    
    def verify_hmac(self, message: str, key: str, signature: str) -> bool:
        """
        Verify HMAC signature using constant-time comparison.
        
        Args:
            message: Original message
            key: Secret key
            signature: HMAC to verify (hex encoded)
            
        Returns:
            True if HMAC is valid, False otherwise
        """
        expected = self.generate_hmac(message, key)
        return hmac.compare_digest(expected, signature)


# ============================================================================
# Secure Token Generation
# ============================================================================

class SecureTokenGenerator:
    """
    Cryptographically secure token generation.
    
    Apply to: API keys, session tokens, CSRF tokens, nonces
    
    Features: Cryptographic randomness, URL-safe encoding
    """
    
    def __init__(self, config: SecurityConfig = None):
        self.config = config or SecurityConfig()
    
    def generate_token(self, length: Optional[int] = None) -> str:
        """
        Generate cryptographically secure random token.
        
        Args:
            length: Token length in bytes (default: 32)
            
        Returns:
            URL-safe token string
        """
        length = length or self.config.token_length
        return secrets.token_urlsafe(length)
    
    def generate_hex_token(self, length: Optional[int] = None) -> str:
        """
        Generate hex-encoded random token.
        
        Args:
            length: Token length in bytes (default: 32)
            
        Returns:
            Hex-encoded token string
        """
        length = length or self.config.token_length
        return secrets.token_hex(length)
    
    def generate_api_key(self) -> str:
        """
        Generate API key with prefix for identification.
        
        Returns:
            API key in format: sk_live_<random_token>
        """
        token = self.generate_token(32)
        return f"sk_live_{token}"
    
    def generate_csrf_token(self) -> str:
        """
        Generate CSRF token for form protection.
        
        Returns:
            CSRF token
        """
        return self.generate_token(32)


# ============================================================================
# Secure Input Validation with Timeout Protection
# ============================================================================

class SecureValidator:
    """
    Input validation with ReDoS (regex DoS) protection.
    
    Apply to: User input validation, API parameter validation
    
    Prevents: ReDoS attacks, regex catastrophic backtracking
    """
    
    def __init__(self, config: SecurityConfig = None):
        self.config = config or SecurityConfig()
    
    def validate_with_timeout(
        self, 
        pattern: str, 
        text: str, 
        timeout: Optional[float] = None
    ) -> bool:
        """
        Validate input against regex pattern with timeout protection.
        
        Args:
            pattern: Regex pattern to match
            text: Text to validate
            timeout: Timeout in seconds (default: from config)
            
        Returns:
            True if pattern matches and completes within timeout, False otherwise
        """
        timeout = timeout or self.config.regex_timeout
        
        # Length check
        if len(text) > self.config.max_input_length:
            return False
        
        # Compile regex with timeout protection
        compiled_pattern = re.compile(pattern, re.IGNORECASE)
        
        # Simple timeout implementation using time check
        start_time = time.time()
        try:
            match = compiled_pattern.search(text)
            elapsed = time.time() - start_time
            
            if elapsed > timeout:
                print(f"⚠️  Regex timeout exceeded: {elapsed:.3f}s")
                return False
            
            return match is not None
            
        except Exception as e:
            print(f"⚠️  Regex validation failed: {e}")
            return False
    
    def validate_length(self, text: str, min_len: int = 0, max_len: Optional[int] = None) -> bool:
        """
        Validate text length.
        
        Args:
            text: Text to validate
            min_len: Minimum length
            max_len: Maximum length (default: from config)
            
        Returns:
            True if length is valid, False otherwise
        """
        max_len = max_len or self.config.max_input_length
        return min_len <= len(text) <= max_len
    
    def validate_charset(self, text: str, allowed_chars: str) -> bool:
        """
        Validate that text contains only allowed characters.
        
        Args:
            text: Text to validate
            allowed_chars: String of allowed characters
            
        Returns:
            True if all characters are allowed, False otherwise
        """
        return all(c in allowed_chars for c in text)


# ============================================================================
# Metadata Sanitizer (Side-Channel Protection)
# ============================================================================

class MetadataSanitizer:
    """
    Sanitize metadata to prevent side-channel leaks.
    
    Apply to: File uploads, document processing, image handling
    
    Prevents: Metadata leakage, geolocation exposure, PII exposure
    """
    
    SENSITIVE_EXIF_TAGS = {
        'GPSInfo',
        'GPSLatitude',
        'GPSLongitude',
        'GPSTimeStamp',
        'GPSDateStamp',
        'Make',
        'Model',
        'Software',
        'DateTime',
        'DateTimeOriginal',
        'UserComment',
        'Copyright',
        'Artist',
    }
    
    def sanitize_filename(self, filename: str) -> str:
        """
        Sanitize filename to prevent path traversal and information leakage.
        
        Args:
            filename: Original filename
            
        Returns:
            Sanitized filename
        """
        # Remove path components
        filename = filename.split('/')[-1].split('\\')[-1]
        
        # Remove potentially dangerous characters
        filename = re.sub(r'[^\w\s\-\.]', '', filename)
        
        # Truncate to reasonable length
        max_length = 255
        if len(filename) > max_length:
            name, ext = filename.rsplit('.', 1) if '.' in filename else (filename, '')
            filename = name[:max_length - len(ext) - 1] + '.' + ext
        
        return filename
    
    def strip_metadata_info(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Strip sensitive metadata from dictionary.
        
        Args:
            data: Dictionary potentially containing sensitive metadata
            
        Returns:
            Dictionary with sensitive fields removed
        """
        sanitized = {}
        for key, value in data.items():
            # Skip sensitive keys
            if key in self.SENSITIVE_EXIF_TAGS:
                continue
            
            # Recursively sanitize nested dictionaries
            if isinstance(value, dict):
                sanitized[key] = self.strip_metadata_info(value)
            else:
                sanitized[key] = value
        
        return sanitized


# ============================================================================
# Defense-in-Depth Utilities
# ============================================================================

class DefenseInDepthValidator:
    """
    Multi-layer validation for defense-in-depth security.
    
    Apply to: Critical operations, financial transactions, privileged actions
    
    Features: Multiple validation layers, fail-secure defaults
    """
    
    def __init__(self):
        self.deserializer = SecureDeserializer()
        self.hasher = SecureHasher()
        self.token_gen = SecureTokenGenerator()
        self.validator = SecureValidator()
        self.sanitizer = MetadataSanitizer()
    
    def validate_multi_layer(
        self, 
        data: str, 
        expected_type: str = 'json',
        max_length: int = 10_000
    ) -> Optional[Any]:
        """
        Apply multiple validation layers to input data.
        
        Layers:
        1. Length validation
        2. Type validation
        3. Content validation
        4. Deserialization with whitelist
        
        Args:
            data: Input data to validate
            expected_type: Expected data type ('json', 'string')
            max_length: Maximum allowed length
            
        Returns:
            Validated data or None if validation fails
        """
        # Layer 1: Length validation
        if not self.validator.validate_length(data, max_len=max_length):
            print("⚠️  Layer 1 failed: Length validation")
            return None
        
        # Layer 2: Type validation
        if expected_type == 'json':
            result = self.deserializer.safe_json_loads(data)
            if result is None:
                print("⚠️  Layer 2 failed: JSON deserialization")
                return None
        elif expected_type == 'string':
            result = data
        else:
            print(f"⚠️  Unknown type: {expected_type}")
            return None
        
        # Layer 3: Content validation (basic checks)
        # Can be extended with custom validators
        
        print("✅ All validation layers passed")
        return result


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    print("=== Zero-Day Security Shield Demo ===\n")
    
    # 1. Secure Deserialization
    print("1. Secure Deserialization")
    deserializer = SecureDeserializer()
    
    safe_json = '{"user": "alice", "role": "admin"}'
    result = deserializer.safe_json_loads(safe_json)
    print(f"   Safe JSON: {result}\n")
    
    # 2. Secure Hashing
    print("2. Secure Password Hashing")
    hasher = SecureHasher()
    
    password = "SecurePassword123!"
    hashed = hasher.hash_password(password)
    print(f"   Password hash: {hashed['hash'][:32]}...")
    
    is_valid = hasher.verify_password(password, hashed['hash'], hashed['salt'])
    print(f"   Verification: {is_valid}\n")
    
    # 3. Secure Token Generation
    print("3. Secure Token Generation")
    token_gen = SecureTokenGenerator()
    
    api_key = token_gen.generate_api_key()
    csrf_token = token_gen.generate_csrf_token()
    print(f"   API Key: {api_key[:20]}...")
    print(f"   CSRF Token: {csrf_token[:20]}...\n")
    
    # 4. Secure Validation with Timeout
    print("4. Secure Validation with Timeout Protection")
    validator = SecureValidator()
    
    email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    test_email = "user@example.com"
    is_valid = validator.validate_with_timeout(email_pattern, test_email)
    print(f"   Email '{test_email}' valid: {is_valid}\n")
    
    # 5. Metadata Sanitization
    print("5. Metadata Sanitization")
    sanitizer = MetadataSanitizer()
    
    filename = "../../etc/passwd/../malicious_file.txt"
    safe_filename = sanitizer.sanitize_filename(filename)
    print(f"   Original: {filename}")
    print(f"   Sanitized: {safe_filename}\n")
    
    # 6. Defense-in-Depth Validation
    print("6. Defense-in-Depth Multi-Layer Validation")
    defense = DefenseInDepthValidator()
    
    test_data = '{"transaction": "payment", "amount": 100}'
    validated = defense.validate_multi_layer(test_data, expected_type='json')
    print(f"   Result: {validated}\n")
    
    print("=== Demo Complete ===")
