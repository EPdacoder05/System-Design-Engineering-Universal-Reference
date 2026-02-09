"""
Encryption & Cryptographic Operations Module

Apply to: Data encryption, secure storage, password hashing, secure token generation

Features:
- AES-256-GCM encryption/decryption with authenticated encryption
- SHA-256/SHA-512 hashing functions
- PBKDF2 password hashing with salt (configurable iterations)
- Secure random token generation (hex, URL-safe, bytes)
- Base64 encoding/decoding utilities
- Key derivation functions (PBKDF2, HKDF)
- Fernet symmetric encryption (high-level interface)
"""

import os
import secrets
import base64
import hashlib
import hmac
from typing import Optional, Tuple, Union

try:
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    from cryptography.hazmat.primitives.kdf.hkdf import HKDF
    from cryptography.fernet import Fernet
    from cryptography.hazmat.backends import default_backend
except ImportError:
    print("Install dependencies: pip install cryptography")


# Configuration - Can be overridden via environment variables
DEFAULT_PBKDF2_ITERATIONS = int(os.getenv("PBKDF2_ITERATIONS", "480000"))  # Industry standard (OWASP minimum)
DEFAULT_SALT_LENGTH = 32  # bytes
DEFAULT_TOKEN_LENGTH = 32  # bytes


# ============================================================================
# AES-256-GCM Encryption/Decryption
# ============================================================================


def generate_aes_key() -> bytes:
    """
    Generate a secure 256-bit AES key.
    
    Apply to: Creating encryption keys for AES-256-GCM
    
    Returns:
        32-byte key suitable for AES-256
        
    Example:
        >>> key = generate_aes_key()
        >>> len(key)
        32
    """
    return secrets.token_bytes(32)


def aes_gcm_encrypt(plaintext: Union[str, bytes], key: bytes, associated_data: Optional[bytes] = None) -> Tuple[bytes, bytes]:
    """
    Encrypt data using AES-256-GCM (authenticated encryption).
    
    Apply to: Encrypting sensitive data (credentials, PII, API keys, database fields)
    
    Args:
        plaintext: Data to encrypt (string or bytes)
        key: 256-bit (32-byte) encryption key
        associated_data: Optional additional authenticated data (AAD) - not encrypted but authenticated
        
    Returns:
        Tuple of (ciphertext, nonce) - both must be stored to decrypt
        
    Example:
        >>> key = generate_aes_key()
        >>> ciphertext, nonce = aes_gcm_encrypt("secret data", key)
        >>> decrypted = aes_gcm_decrypt(ciphertext, key, nonce)
        >>> decrypted
        b'secret data'
    """
    if isinstance(plaintext, str):
        plaintext = plaintext.encode('utf-8')
    
    if len(key) != 32:
        raise ValueError("Key must be 32 bytes (256 bits) for AES-256")
    
    # Generate a random 96-bit nonce (12 bytes is standard for GCM)
    nonce = secrets.token_bytes(12)
    
    aesgcm = AESGCM(key)
    ciphertext = aesgcm.encrypt(nonce, plaintext, associated_data)
    
    return ciphertext, nonce


def aes_gcm_decrypt(ciphertext: bytes, key: bytes, nonce: bytes, associated_data: Optional[bytes] = None) -> bytes:
    """
    Decrypt data using AES-256-GCM.
    
    Apply to: Decrypting data encrypted with aes_gcm_encrypt()
    
    Args:
        ciphertext: Encrypted data
        key: 256-bit (32-byte) encryption key (same as used for encryption)
        nonce: Nonce used during encryption
        associated_data: AAD used during encryption (if any)
        
    Returns:
        Decrypted plaintext as bytes
        
    Raises:
        cryptography.exceptions.InvalidTag: If authentication fails (data tampered or wrong key)
        
    Example:
        >>> key = generate_aes_key()
        >>> ciphertext, nonce = aes_gcm_encrypt("secret", key)
        >>> plaintext = aes_gcm_decrypt(ciphertext, key, nonce)
        >>> plaintext.decode('utf-8')
        'secret'
    """
    if len(key) != 32:
        raise ValueError("Key must be 32 bytes (256 bits) for AES-256")
    
    aesgcm = AESGCM(key)
    plaintext = aesgcm.decrypt(nonce, ciphertext, associated_data)
    
    return plaintext


def aes_gcm_encrypt_combined(plaintext: Union[str, bytes], key: bytes, associated_data: Optional[bytes] = None) -> bytes:
    """
    Encrypt and return ciphertext with nonce prepended (single output).
    
    Apply to: When you need a single encrypted output (nonce + ciphertext combined)
    
    Args:
        plaintext: Data to encrypt
        key: 256-bit encryption key
        associated_data: Optional AAD
        
    Returns:
        Combined bytes: nonce (12 bytes) + ciphertext
        
    Example:
        >>> key = generate_aes_key()
        >>> encrypted = aes_gcm_encrypt_combined("secret", key)
        >>> decrypted = aes_gcm_decrypt_combined(encrypted, key)
        >>> decrypted.decode('utf-8')
        'secret'
    """
    ciphertext, nonce = aes_gcm_encrypt(plaintext, key, associated_data)
    return nonce + ciphertext


def aes_gcm_decrypt_combined(encrypted_data: bytes, key: bytes, associated_data: Optional[bytes] = None) -> bytes:
    """
    Decrypt data where nonce is prepended to ciphertext.
    
    Apply to: Decrypting data from aes_gcm_encrypt_combined()
    
    Args:
        encrypted_data: Combined nonce + ciphertext
        key: 256-bit encryption key
        associated_data: Optional AAD
        
    Returns:
        Decrypted plaintext as bytes
    """
    nonce = encrypted_data[:12]
    ciphertext = encrypted_data[12:]
    return aes_gcm_decrypt(ciphertext, key, nonce, associated_data)


# ============================================================================
# Hash Functions (SHA-256, SHA-512)
# ============================================================================


def sha256_hash(data: Union[str, bytes]) -> str:
    """
    Compute SHA-256 hash of data.
    
    Apply to: File integrity checks, content verification, checksums
    
    Args:
        data: Data to hash (string or bytes)
        
    Returns:
        Hex-encoded SHA-256 hash (64 characters)
        
    Example:
        >>> sha256_hash("hello world")
        'b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9'
    """
    if isinstance(data, str):
        data = data.encode('utf-8')
    return hashlib.sha256(data).hexdigest()


def sha512_hash(data: Union[str, bytes]) -> str:
    """
    Compute SHA-512 hash of data.
    
    Apply to: File integrity checks requiring higher security, digital signatures
    
    Args:
        data: Data to hash (string or bytes)
        
    Returns:
        Hex-encoded SHA-512 hash (128 characters)
        
    Example:
        >>> sha512_hash("hello world")
        '309ecc489c12d6eb4cc40f50c902f2b4d0ed77ee511a7c7a9bcd3ca86d4cd86f989dd35bc5ff499670da34255b45b0cfd830e81f605dcf7dc5542e93ae9cd76f'
    """
    if isinstance(data, str):
        data = data.encode('utf-8')
    return hashlib.sha512(data).hexdigest()


def hmac_sha256(data: Union[str, bytes], key: Union[str, bytes]) -> str:
    """
    Compute HMAC-SHA256 (keyed hash) of data.
    
    Apply to: Message authentication, API signatures, webhook signatures
    
    Args:
        data: Data to authenticate
        key: Secret key for HMAC
        
    Returns:
        Hex-encoded HMAC-SHA256 (64 characters)
        
    Example:
        >>> hmac_sha256("message", "secret_key")
        '8b5f48702995c1598c573db1e21866a9b825d4a794d61e45a65c86390e09f7d6'
    """
    if isinstance(data, str):
        data = data.encode('utf-8')
    if isinstance(key, str):
        key = key.encode('utf-8')
    
    return hmac.new(key, data, hashlib.sha256).hexdigest()


# ============================================================================
# Password Hashing (PBKDF2)
# ============================================================================


def hash_password(password: str, salt: Optional[bytes] = None, iterations: int = DEFAULT_PBKDF2_ITERATIONS) -> Tuple[bytes, bytes]:
    """
    Hash password using PBKDF2-HMAC-SHA256 with salt.
    
    Apply to: User password storage, credential hashing
    
    Args:
        password: Plain text password
        salt: Optional salt (generates random if None)
        iterations: Number of iterations (default: 480,000 - industry standard minimum)
        
    Returns:
        Tuple of (hashed_password, salt) - store both in database
        
    Example:
        >>> password_hash, salt = hash_password("user_password123")
        >>> # Store password_hash and salt in database
        >>> is_valid = verify_password("user_password123", password_hash, salt)
        >>> is_valid
        True
    """
    if salt is None:
        salt = secrets.token_bytes(DEFAULT_SALT_LENGTH)
    
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=iterations,
        backend=default_backend()
    )
    
    password_hash = kdf.derive(password.encode('utf-8'))
    return password_hash, salt


def verify_password(password: str, password_hash: bytes, salt: bytes, iterations: int = DEFAULT_PBKDF2_ITERATIONS) -> bool:
    """
    Verify password against stored hash.
    
    Apply to: User login verification, password checking
    
    Args:
        password: Plain text password to verify
        password_hash: Stored password hash
        salt: Salt used during hashing
        iterations: Number of iterations used (must match hashing)
        
    Returns:
        True if password matches, False otherwise
        
    Example:
        >>> password_hash, salt = hash_password("correct_password")
        >>> verify_password("correct_password", password_hash, salt)
        True
        >>> verify_password("wrong_password", password_hash, salt)
        False
    """
    try:
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=iterations,
            backend=default_backend()
        )
        kdf.verify(password.encode('utf-8'), password_hash)
        return True
    except Exception:
        return False


def hash_password_combined(password: str, iterations: int = DEFAULT_PBKDF2_ITERATIONS) -> str:
    """
    Hash password and return combined string (iterations$salt$hash in base64).
    
    Apply to: Simplified password storage (single field in database)
    
    Args:
        password: Plain text password
        iterations: Number of PBKDF2 iterations
        
    Returns:
        Combined string: "iterations$salt_b64$hash_b64"
        
    Example:
        >>> combined = hash_password_combined("password123")
        >>> is_valid = verify_password_combined("password123", combined)
        >>> is_valid
        True
    """
    password_hash, salt = hash_password(password, iterations=iterations)
    
    salt_b64 = base64.b64encode(salt).decode('utf-8')
    hash_b64 = base64.b64encode(password_hash).decode('utf-8')
    
    return f"{iterations}${salt_b64}${hash_b64}"


def verify_password_combined(password: str, combined_hash: str) -> bool:
    """
    Verify password against combined hash string.
    
    Apply to: Password verification with combined storage format
    
    Args:
        password: Plain text password to verify
        combined_hash: Combined hash from hash_password_combined()
        
    Returns:
        True if password matches, False otherwise
    """
    try:
        parts = combined_hash.split('$')
        if len(parts) != 3:
            return False
        
        iterations = int(parts[0])
        salt = base64.b64decode(parts[1])
        stored_hash = base64.b64decode(parts[2])
        
        return verify_password(password, stored_hash, salt, iterations)
    except Exception:
        return False


# ============================================================================
# Secure Random Token Generation
# ============================================================================


def generate_token_hex(length: int = DEFAULT_TOKEN_LENGTH) -> str:
    """
    Generate cryptographically secure random token (hex encoded).
    
    Apply to: Session tokens, API keys, CSRF tokens, password reset tokens
    
    Args:
        length: Number of random bytes (output will be 2x this length in hex)
        
    Returns:
        Hex-encoded random token
        
    Example:
        >>> token = generate_token_hex(32)
        >>> len(token)
        64
    """
    return secrets.token_hex(length)


def generate_token_urlsafe(length: int = DEFAULT_TOKEN_LENGTH) -> str:
    """
    Generate URL-safe random token (base64 encoded).
    
    Apply to: URL tokens, file names, API keys that appear in URLs
    
    Args:
        length: Number of random bytes
        
    Returns:
        URL-safe base64 encoded token
        
    Example:
        >>> token = generate_token_urlsafe(32)
        >>> len(token) >= 40  # Base64 encoding increases length
        True
    """
    return secrets.token_urlsafe(length)


def generate_token_bytes(length: int = DEFAULT_TOKEN_LENGTH) -> bytes:
    """
    Generate random bytes (raw binary token).
    
    Apply to: Encryption keys, nonces, raw binary tokens
    
    Args:
        length: Number of random bytes
        
    Returns:
        Random bytes
        
    Example:
        >>> token = generate_token_bytes(32)
        >>> len(token)
        32
    """
    return secrets.token_bytes(length)


def generate_numeric_code(length: int = 6) -> str:
    """
    Generate cryptographically secure numeric code.
    
    Apply to: OTP codes, PIN codes, verification codes
    
    Args:
        length: Number of digits
        
    Returns:
        Numeric string of specified length
        
    Example:
        >>> code = generate_numeric_code(6)
        >>> len(code)
        6
        >>> code.isdigit()
        True
    """
    return ''.join(str(secrets.randbelow(10)) for _ in range(length))


# ============================================================================
# Base64 Encoding Utilities
# ============================================================================


def base64_encode(data: Union[str, bytes]) -> str:
    """
    Encode data to base64 string.
    
    Apply to: Encoding binary data for text transmission, JSON APIs
    
    Args:
        data: Data to encode (string or bytes)
        
    Returns:
        Base64 encoded string
        
    Example:
        >>> base64_encode("hello world")
        'aGVsbG8gd29ybGQ='
    """
    if isinstance(data, str):
        data = data.encode('utf-8')
    return base64.b64encode(data).decode('utf-8')


def base64_decode(encoded: str) -> bytes:
    """
    Decode base64 string to bytes.
    
    Apply to: Decoding base64 data from APIs, configurations
    
    Args:
        encoded: Base64 encoded string
        
    Returns:
        Decoded bytes
        
    Example:
        >>> data = base64_decode('aGVsbG8gd29ybGQ=')
        >>> data.decode('utf-8')
        'hello world'
    """
    return base64.b64decode(encoded)


def base64_urlsafe_encode(data: Union[str, bytes]) -> str:
    """
    Encode data to URL-safe base64 (no padding).
    
    Apply to: Tokens in URLs, filename-safe encoding
    
    Args:
        data: Data to encode
        
    Returns:
        URL-safe base64 string (+ and / replaced with - and _)
    """
    if isinstance(data, str):
        data = data.encode('utf-8')
    return base64.urlsafe_b64encode(data).decode('utf-8').rstrip('=')


def base64_urlsafe_decode(encoded: str) -> bytes:
    """
    Decode URL-safe base64 string.
    
    Apply to: Decoding URL-safe tokens
    
    Args:
        encoded: URL-safe base64 string
        
    Returns:
        Decoded bytes
    """
    # Add padding if needed
    padding = 4 - (len(encoded) % 4)
    if padding != 4:
        encoded += '=' * padding
    return base64.urlsafe_b64decode(encoded)


# ============================================================================
# Key Derivation Functions
# ============================================================================


def derive_key_pbkdf2(password: Union[str, bytes], salt: bytes, length: int = 32, iterations: int = DEFAULT_PBKDF2_ITERATIONS) -> bytes:
    """
    Derive cryptographic key from password using PBKDF2.
    
    Apply to: Deriving encryption keys from passwords, key stretching
    
    Args:
        password: Password or passphrase
        salt: Random salt
        length: Desired key length in bytes (32 for AES-256)
        iterations: Number of iterations
        
    Returns:
        Derived key bytes
        
    Example:
        >>> salt = secrets.token_bytes(32)
        >>> key = derive_key_pbkdf2("user_password", salt, length=32)
        >>> len(key)
        32
    """
    if isinstance(password, str):
        password = password.encode('utf-8')
    
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=length,
        salt=salt,
        iterations=iterations,
        backend=default_backend()
    )
    
    return kdf.derive(password)


def derive_key_hkdf(input_key: bytes, length: int = 32, salt: Optional[bytes] = None, info: Optional[bytes] = None) -> bytes:
    """
    Derive key using HKDF (HMAC-based Key Derivation Function).
    
    Apply to: Key derivation from existing key material, multi-key derivation
    
    Args:
        input_key: Input key material (IKM)
        length: Desired output key length
        salt: Optional salt (random if None)
        info: Optional context/application info
        
    Returns:
        Derived key bytes
        
    Example:
        >>> master_key = secrets.token_bytes(32)
        >>> encryption_key = derive_key_hkdf(master_key, info=b"encryption")
        >>> signing_key = derive_key_hkdf(master_key, info=b"signing")
    """
    if salt is None:
        salt = secrets.token_bytes(16)
    
    hkdf = HKDF(
        algorithm=hashes.SHA256(),
        length=length,
        salt=salt,
        info=info,
        backend=default_backend()
    )
    
    return hkdf.derive(input_key)


# ============================================================================
# Fernet Symmetric Encryption (High-Level)
# ============================================================================


def generate_fernet_key() -> bytes:
    """
    Generate Fernet encryption key.
    
    Apply to: Simple symmetric encryption with built-in authentication
    
    Returns:
        Fernet key (base64 encoded 32-byte key)
        
    Example:
        >>> key = generate_fernet_key()
        >>> encrypted = fernet_encrypt("secret data", key)
        >>> decrypted = fernet_decrypt(encrypted, key)
        >>> decrypted
        'secret data'
    """
    return Fernet.generate_key()


def fernet_encrypt(data: Union[str, bytes], key: bytes) -> bytes:
    """
    Encrypt data using Fernet (symmetric encryption with timestamp).
    
    Apply to: Simple authenticated encryption, time-limited tokens
    
    Args:
        data: Data to encrypt
        key: Fernet key from generate_fernet_key()
        
    Returns:
        Encrypted data (includes timestamp and signature)
        
    Example:
        >>> key = generate_fernet_key()
        >>> encrypted = fernet_encrypt("sensitive data", key)
        >>> decrypted = fernet_decrypt(encrypted, key)
    """
    if isinstance(data, str):
        data = data.encode('utf-8')
    
    f = Fernet(key)
    return f.encrypt(data)


def fernet_decrypt(encrypted_data: bytes, key: bytes, ttl: Optional[int] = None) -> str:
    """
    Decrypt Fernet encrypted data.
    
    Apply to: Decrypting Fernet encrypted data, verifying token age
    
    Args:
        encrypted_data: Encrypted data from fernet_encrypt()
        key: Fernet key used for encryption
        ttl: Optional time-to-live in seconds (raises error if token is older)
        
    Returns:
        Decrypted data as string
        
    Raises:
        cryptography.fernet.InvalidToken: If decryption fails or TTL exceeded
        
    Example:
        >>> key = generate_fernet_key()
        >>> encrypted = fernet_encrypt("data", key)
        >>> fernet_decrypt(encrypted, key, ttl=3600)  # Must be < 1 hour old
        'data'
    """
    f = Fernet(key)
    decrypted = f.decrypt(encrypted_data, ttl=ttl)
    return decrypted.decode('utf-8')


# ============================================================================
# Example Usage
# ============================================================================


if __name__ == "__main__":
    print("=" * 80)
    print("ENCRYPTION MODULE - Example Usage")
    print("=" * 80)
    
    # 1. AES-256-GCM Encryption
    print("\n1. AES-256-GCM Encryption:")
    print("-" * 40)
    aes_key = generate_aes_key()
    print(f"Generated AES-256 key: {aes_key.hex()[:40]}...")
    
    plaintext = "Sensitive user data: SSN 123-45-6789"
    ciphertext, nonce = aes_gcm_encrypt(plaintext, aes_key)
    print(f"Plaintext: {plaintext}")
    print(f"Ciphertext: {ciphertext.hex()[:40]}...")
    print(f"Nonce: {nonce.hex()}")
    
    decrypted = aes_gcm_decrypt(ciphertext, aes_key, nonce)
    print(f"Decrypted: {decrypted.decode('utf-8')}")
    print(f"Match: {decrypted.decode('utf-8') == plaintext}")
    
    # Combined encryption
    combined = aes_gcm_encrypt_combined(plaintext, aes_key)
    decrypted_combined = aes_gcm_decrypt_combined(combined, aes_key)
    print(f"Combined encryption/decryption: {decrypted_combined.decode('utf-8')}")
    
    # 2. Hashing Functions
    print("\n2. SHA-256 and SHA-512 Hashing:")
    print("-" * 40)
    data = "Important document content"
    sha256 = sha256_hash(data)
    sha512 = sha512_hash(data)
    print(f"Data: {data}")
    print(f"SHA-256: {sha256}")
    print(f"SHA-512: {sha512[:64]}...")
    
    # 3. Password Hashing
    print("\n3. PBKDF2 Password Hashing:")
    print("-" * 40)
    password = "UserSecurePassword123!"
    password_hash, salt = hash_password(password, iterations=100000)  # Reduced for demo
    print(f"Password: {password}")
    print(f"Salt: {salt.hex()[:40]}...")
    print(f"Hash: {password_hash.hex()[:40]}...")
    
    # Verify correct password
    is_valid = verify_password(password, password_hash, salt, iterations=100000)
    print(f"Verify correct password: {is_valid}")
    
    # Verify wrong password
    is_valid_wrong = verify_password("WrongPassword", password_hash, salt, iterations=100000)
    print(f"Verify wrong password: {is_valid_wrong}")
    
    # Combined format
    combined_hash = hash_password_combined(password, iterations=100000)
    print(f"Combined format: {combined_hash[:50]}...")
    is_valid_combined = verify_password_combined(password, combined_hash)
    print(f"Verify combined format: {is_valid_combined}")
    
    # 4. Token Generation
    print("\n4. Secure Token Generation:")
    print("-" * 40)
    token_hex = generate_token_hex(16)
    token_urlsafe = generate_token_urlsafe(16)
    token_bytes = generate_token_bytes(16)
    numeric_code = generate_numeric_code(6)
    
    print(f"Hex token (16 bytes): {token_hex}")
    print(f"URL-safe token (16 bytes): {token_urlsafe}")
    print(f"Bytes token (16 bytes): {token_bytes.hex()}")
    print(f"Numeric code (6 digits): {numeric_code}")
    
    # 5. Base64 Encoding
    print("\n5. Base64 Encoding:")
    print("-" * 40)
    original = "Hello, World! 🔐"
    encoded = base64_encode(original)
    decoded = base64_decode(encoded)
    print(f"Original: {original}")
    print(f"Base64 encoded: {encoded}")
    print(f"Decoded: {decoded.decode('utf-8')}")
    
    encoded_urlsafe = base64_urlsafe_encode(original)
    decoded_urlsafe = base64_urlsafe_decode(encoded_urlsafe)
    print(f"URL-safe base64: {encoded_urlsafe}")
    print(f"Decoded: {decoded_urlsafe.decode('utf-8')}")
    
    # 6. Key Derivation
    print("\n6. Key Derivation Functions:")
    print("-" * 40)
    passphrase = "my secure passphrase"
    salt_kdf = secrets.token_bytes(32)
    
    # PBKDF2
    derived_key = derive_key_pbkdf2(passphrase, salt_kdf, length=32, iterations=100000)
    print(f"Passphrase: {passphrase}")
    print(f"PBKDF2 derived key: {derived_key.hex()[:40]}...")
    
    # HKDF
    master_key = secrets.token_bytes(32)
    encryption_key = derive_key_hkdf(master_key, info=b"encryption")
    signing_key = derive_key_hkdf(master_key, info=b"signing")
    print(f"Master key: {master_key.hex()[:40]}...")
    print(f"HKDF encryption key: {encryption_key.hex()[:40]}...")
    print(f"HKDF signing key: {signing_key.hex()[:40]}...")
    
    # 7. Fernet Encryption
    print("\n7. Fernet Symmetric Encryption:")
    print("-" * 40)
    fernet_key = generate_fernet_key()
    print(f"Fernet key: {fernet_key.decode('utf-8')[:40]}...")
    
    fernet_plaintext = "Confidential information"
    fernet_encrypted = fernet_encrypt(fernet_plaintext, fernet_key)
    print(f"Plaintext: {fernet_plaintext}")
    print(f"Encrypted: {fernet_encrypted.decode('utf-8')[:40]}...")
    
    fernet_decrypted = fernet_decrypt(fernet_encrypted, fernet_key)
    print(f"Decrypted: {fernet_decrypted}")
    print(f"Match: {fernet_decrypted == fernet_plaintext}")
    
    print("\n" + "=" * 80)
    print("All encryption operations completed successfully!")
    print("=" * 80)
