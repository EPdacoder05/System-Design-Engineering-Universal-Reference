"""
Production Configuration Management with Pydantic

Apply to: Any production application requiring configuration management, environment-based
settings, secret management, microservices, containerized applications, cloud deployments.

This module provides:
- Environment-aware configuration (dev/staging/prod)
- Automatic .env file loading
- Type-safe configuration with validation
- Secret generation utilities
- Database and Redis URL construction
- API key validation
- Zero hardcoded secrets policy

Example .env file template:
----------------------------
# Application Environment
APP_ENV=development  # development, staging, production
APP_NAME=my-service
DEBUG=true

# Server Configuration
HOST=0.0.0.0
PORT=8000
WORKERS=4
RELOAD=true

# Security Secrets (NEVER commit these!)
SECRET_KEY=your-secret-key-here
JWT_SECRET=your-jwt-secret-here
API_KEY=your-api-key-here
ENCRYPTION_KEY=your-32-byte-base64-encoded-key

# Database Configuration
DB_HOST=localhost
DB_PORT=5432
DB_USER=postgres
DB_PASSWORD=postgres
DB_NAME=myapp
DB_POOL_SIZE=20
DB_MAX_OVERFLOW=10

# Redis Configuration
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_PASSWORD=
REDIS_DB=0
REDIS_MAX_CONNECTIONS=50

# External APIs
STRIPE_API_KEY=sk_test_...
SENDGRID_API_KEY=SG....
AWS_ACCESS_KEY_ID=AKIA...
AWS_SECRET_ACCESS_KEY=...
AWS_REGION=us-east-1

# Feature Flags
ENABLE_CACHING=true
ENABLE_RATE_LIMITING=true
ENABLE_METRICS=true

# Monitoring
SENTRY_DSN=https://...@sentry.io/...
LOG_LEVEL=INFO
"""

import secrets
import string
from base64 import b64encode
from enum import Enum
from typing import Any, Dict, List, Optional
from urllib.parse import quote_plus

from pydantic import (
    BaseModel,
    Field,
    field_validator,
    model_validator,
)
from pydantic_settings import BaseSettings, SettingsConfigDict


class Environment(str, Enum):
    """Application environment types."""

    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"


class LogLevel(str, Enum):
    """Logging levels."""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class DatabaseConfig(BaseModel):
    """
    Database configuration with URL construction.
    
    Apply to: PostgreSQL, MySQL, any SQL database connection configuration.
    """

    host: str = "localhost"
    port: int = 5432
    user: str = "postgres"
    password: str = "postgres"
    name: str = "app"
    driver: str = "postgresql+asyncpg"
    pool_size: int = 20
    max_overflow: int = 10
    pool_pre_ping: bool = True
    echo: bool = False

    def get_url(self, async_driver: bool = True) -> str:
        """
        Construct database URL from components.
        
        Args:
            async_driver: Use async driver (postgresql+asyncpg vs postgresql+psycopg2)
            
        Returns:
            Database connection URL
        """
        driver = self.driver if async_driver else self.driver.replace("+asyncpg", "+psycopg2")
        password = quote_plus(self.password)
        return f"{driver}://{self.user}:{password}@{self.host}:{self.port}/{self.name}"

    def get_sqlalchemy_config(self) -> Dict[str, Any]:
        """
        Get SQLAlchemy engine configuration.
        
        Returns:
            Dictionary of engine configuration parameters
        """
        return {
            "pool_size": self.pool_size,
            "max_overflow": self.max_overflow,
            "pool_pre_ping": self.pool_pre_ping,
            "echo": self.echo,
        }


class RedisConfig(BaseModel):
    """
    Redis configuration with URL construction.
    
    Apply to: Caching, session storage, pub/sub, task queues, rate limiting.
    """

    host: str = "localhost"
    port: int = 6379
    password: Optional[str] = None
    db: int = 0
    max_connections: int = 50
    socket_timeout: float = 5.0
    socket_connect_timeout: float = 5.0
    decode_responses: bool = False

    def get_url(self) -> str:
        """
        Construct Redis URL from components.
        
        Returns:
            Redis connection URL
        """
        if self.password:
            password = quote_plus(self.password)
            return f"redis://:{password}@{self.host}:{self.port}/{self.db}"
        return f"redis://{self.host}:{self.port}/{self.db}"

    def get_connection_params(self) -> Dict[str, Any]:
        """
        Get Redis connection parameters.
        
        Returns:
            Dictionary of connection parameters
        """
        params = {
            "host": self.host,
            "port": self.port,
            "db": self.db,
            "max_connections": self.max_connections,
            "socket_timeout": self.socket_timeout,
            "socket_connect_timeout": self.socket_connect_timeout,
            "decode_responses": self.decode_responses,
        }
        if self.password:
            params["password"] = self.password
        return params


class SecurityConfig(BaseModel):
    """
    Security configuration with secret validation.
    
    Apply to: Authentication, authorization, encryption, API security.
    """

    secret_key: str = Field(..., min_length=32)
    jwt_secret: str = Field(..., min_length=32)
    jwt_algorithm: str = "HS256"
    jwt_expiration_minutes: int = 60
    api_key: Optional[str] = Field(None, min_length=32)
    encryption_key: Optional[str] = Field(None, min_length=32)
    allowed_hosts: List[str] = Field(default_factory=lambda: ["*"])
    cors_origins: List[str] = Field(default_factory=lambda: ["*"])
    rate_limit_per_minute: int = 60

    @field_validator("secret_key", "jwt_secret", "api_key", "encryption_key")
    @classmethod
    def validate_secret_strength(cls, v: Optional[str]) -> Optional[str]:
        """Validate secret strength."""
        if v and len(v) < 32:
            raise ValueError("Secret must be at least 32 characters long")
        return v

    def validate_api_key(self, provided_key: str) -> bool:
        """
        Validate provided API key against configured key.
        
        Args:
            provided_key: API key to validate
            
        Returns:
            True if valid, False otherwise
        """
        if not self.api_key:
            return False
        return secrets.compare_digest(self.api_key, provided_key)


class Settings(BaseSettings):
    """
    Main application settings with environment-aware configuration.
    
    Apply to: FastAPI, Flask, Django, any Python application requiring configuration.
    
    Features:
    - Automatic .env file loading
    - Environment variable override
    - Type validation
    - Nested configuration models
    - Secret generation
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        env_nested_delimiter="__",
        extra="ignore",
    )

    # Application
    app_name: str = "app"
    app_env: Environment = Environment.DEVELOPMENT
    debug: bool = False
    version: str = "1.0.0"

    # Server
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 4
    reload: bool = False

    # Security
    secret_key: str = Field(default_factory=lambda: generate_secret_key())
    jwt_secret: str = Field(default_factory=lambda: generate_secret_key())
    jwt_algorithm: str = "HS256"
    jwt_expiration_minutes: int = 60
    api_key: Optional[str] = None
    encryption_key: Optional[str] = None

    # Database
    db_host: str = "localhost"
    db_port: int = 5432
    db_user: str = "postgres"
    db_password: str = "postgres"
    db_name: str = "app"
    db_pool_size: int = 20
    db_max_overflow: int = 10

    # Redis
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_password: Optional[str] = None
    redis_db: int = 0
    redis_max_connections: int = 50

    # External Services
    stripe_api_key: Optional[str] = None
    sendgrid_api_key: Optional[str] = None
    aws_access_key_id: Optional[str] = None
    aws_secret_access_key: Optional[str] = None
    aws_region: str = "us-east-1"

    # Feature Flags
    enable_caching: bool = True
    enable_rate_limiting: bool = True
    enable_metrics: bool = True

    # Monitoring
    sentry_dsn: Optional[str] = None
    log_level: LogLevel = LogLevel.INFO

    # CORS
    cors_origins: List[str] = Field(default_factory=lambda: ["*"])
    cors_allow_credentials: bool = True
    cors_allow_methods: List[str] = Field(default_factory=lambda: ["*"])
    cors_allow_headers: List[str] = Field(default_factory=lambda: ["*"])

    @model_validator(mode="after")
    def validate_production_settings(self) -> "Settings":
        """Validate production-specific requirements."""
        if self.app_env == Environment.PRODUCTION:
            if self.debug:
                raise ValueError("DEBUG must be False in production")
            if self.secret_key == "changeme" or len(self.secret_key) < 32:
                raise ValueError("Strong SECRET_KEY required in production")
            if self.jwt_secret == "changeme" or len(self.jwt_secret) < 32:
                raise ValueError("Strong JWT_SECRET required in production")
            if "*" in self.cors_origins:
                raise ValueError("Wildcard CORS origins not allowed in production")
        return self

    @property
    def is_development(self) -> bool:
        """Check if running in development mode."""
        return self.app_env == Environment.DEVELOPMENT

    @property
    def is_staging(self) -> bool:
        """Check if running in staging mode."""
        return self.app_env == Environment.STAGING

    @property
    def is_production(self) -> bool:
        """Check if running in production mode."""
        return self.app_env == Environment.PRODUCTION

    def get_database_config(self) -> DatabaseConfig:
        """
        Get database configuration object.
        
        Returns:
            DatabaseConfig instance
        """
        return DatabaseConfig(
            host=self.db_host,
            port=self.db_port,
            user=self.db_user,
            password=self.db_password,
            name=self.db_name,
            pool_size=self.db_pool_size,
            max_overflow=self.db_max_overflow,
            echo=self.debug,
        )

    def get_redis_config(self) -> RedisConfig:
        """
        Get Redis configuration object.
        
        Returns:
            RedisConfig instance
        """
        return RedisConfig(
            host=self.redis_host,
            port=self.redis_port,
            password=self.redis_password,
            db=self.redis_db,
            max_connections=self.redis_max_connections,
        )

    def get_security_config(self) -> SecurityConfig:
        """
        Get security configuration object.
        
        Returns:
            SecurityConfig instance
        """
        return SecurityConfig(
            secret_key=self.secret_key,
            jwt_secret=self.jwt_secret,
            jwt_algorithm=self.jwt_algorithm,
            jwt_expiration_minutes=self.jwt_expiration_minutes,
            api_key=self.api_key,
            encryption_key=self.encryption_key,
            cors_origins=self.cors_origins,
        )


# Secret Generation Utilities

def generate_secret_key(length: int = 64) -> str:
    """
    Generate a cryptographically secure secret key.
    
    Apply to: JWT secrets, session keys, CSRF tokens, any security-sensitive random strings.
    
    Args:
        length: Length of the secret key (minimum 32)
        
    Returns:
        Secure random string
        
    Example:
        >>> secret = generate_secret_key(64)
        >>> len(secret)
        64
    """
    if length < 32:
        raise ValueError("Secret key must be at least 32 characters")

    alphabet = string.ascii_letters + string.digits + string.punctuation
    return "".join(secrets.choice(alphabet) for _ in range(length))


def generate_api_key(prefix: str = "sk", length: int = 32) -> str:
    """
    Generate an API key with optional prefix.
    
    Apply to: External API authentication, service-to-service authentication, API tokens.
    
    Args:
        prefix: Prefix for the API key (e.g., 'sk', 'pk', 'api')
        length: Length of the random part (minimum 32)
        
    Returns:
        API key string
        
    Example:
        >>> api_key = generate_api_key("pk", 32)
        >>> api_key.startswith("pk_")
        True
    """
    if length < 32:
        raise ValueError("API key must be at least 32 characters")

    random_part = secrets.token_urlsafe(length)
    return f"{prefix}_{random_part}"


def generate_encryption_key() -> str:
    """
    Generate a base64-encoded 32-byte encryption key.
    
    Apply to: AES encryption, data at rest encryption, field-level encryption.
    
    Returns:
        Base64-encoded encryption key
        
    Example:
        >>> key = generate_encryption_key()
        >>> len(key) >= 32
        True
    """
    key_bytes = secrets.token_bytes(32)
    return b64encode(key_bytes).decode("utf-8")


def validate_database_url(url: str) -> bool:
    """
    Validate database URL format.
    
    Args:
        url: Database URL to validate
        
    Returns:
        True if valid, False otherwise
    """
    valid_schemes = ["postgresql", "postgresql+asyncpg", "postgresql+psycopg2", "mysql"]
    try:
        scheme = url.split("://")[0]
        return scheme in valid_schemes
    except Exception:
        return False


def validate_redis_url(url: str) -> bool:
    """
    Validate Redis URL format.
    
    Args:
        url: Redis URL to validate
        
    Returns:
        True if valid, False otherwise
    """
    try:
        return url.startswith("redis://") or url.startswith("rediss://")
    except Exception:
        return False


# Global settings instance (singleton pattern)
_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """
    Get global settings instance (singleton).
    
    Apply to: Dependency injection, FastAPI dependencies, global configuration access.
    
    Returns:
        Settings instance
        
    Example:
        >>> settings = get_settings()
        >>> settings.app_name
        'app'
    """
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def reload_settings() -> Settings:
    """
    Reload settings from environment (useful for testing).
    
    Returns:
        New Settings instance
    """
    global _settings
    _settings = Settings()
    return _settings


# Example usage
if __name__ == "__main__":
    print("=" * 80)
    print("PRODUCTION CONFIGURATION MANAGEMENT - EXAMPLE USAGE")
    print("=" * 80)

    print("\n1. SECRET GENERATION EXAMPLES")
    print("-" * 80)

    # Generate various secrets
    secret_key = generate_secret_key(64)
    print(f"Secret Key (64 chars): {secret_key[:20]}... [truncated]")

    api_key = generate_api_key("sk", 32)
    print(f"API Key: {api_key[:30]}... [truncated]")

    encryption_key = generate_encryption_key()
    print(f"Encryption Key (base64): {encryption_key[:30]}... [truncated]")

    print("\n2. SETTINGS INITIALIZATION")
    print("-" * 80)

    # Load settings (will use .env if present, otherwise defaults)
    settings = get_settings()

    print(f"App Name: {settings.app_name}")
    print(f"Environment: {settings.app_env.value}")
    print(f"Debug Mode: {settings.debug}")
    print(f"Host: {settings.host}:{settings.port}")
    print(f"Workers: {settings.workers}")
    print(f"Log Level: {settings.log_level.value}")

    print("\n3. ENVIRONMENT CHECKS")
    print("-" * 80)

    print(f"Is Development: {settings.is_development}")
    print(f"Is Staging: {settings.is_staging}")
    print(f"Is Production: {settings.is_production}")

    print("\n4. DATABASE CONFIGURATION")
    print("-" * 80)

    db_config = settings.get_database_config()
    print(f"Database URL: {db_config.get_url()}")
    print(f"Pool Size: {db_config.pool_size}")
    print(f"Max Overflow: {db_config.max_overflow}")

    sqlalchemy_config = db_config.get_sqlalchemy_config()
    print("\nSQLAlchemy Config:")
    for key, value in sqlalchemy_config.items():
        print(f"  {key}: {value}")

    print("\n5. REDIS CONFIGURATION")
    print("-" * 80)

    redis_config = settings.get_redis_config()
    print(f"Redis URL: {redis_config.get_url()}")
    print(f"Max Connections: {redis_config.max_connections}")

    print("\n6. SECURITY CONFIGURATION")
    print("-" * 80)

    security_config = settings.get_security_config()
    print(f"Secret Key Length: {len(security_config.secret_key)} chars")
    print(f"JWT Algorithm: {security_config.jwt_algorithm}")
    print(f"JWT Expiration: {security_config.jwt_expiration_minutes} minutes")
    print(f"CORS Origins: {security_config.cors_origins}")

    # Test API key validation if configured
    if security_config.api_key:
        test_key = security_config.api_key
        print("\nAPI Key Validation:")
        print(f"  Valid key: {security_config.validate_api_key(test_key)}")
        print(f"  Invalid key: {security_config.validate_api_key('wrong_key')}")

    print("\n7. FEATURE FLAGS")
    print("-" * 80)

    print(f"Caching Enabled: {settings.enable_caching}")
    print(f"Rate Limiting Enabled: {settings.enable_rate_limiting}")
    print(f"Metrics Enabled: {settings.enable_metrics}")

    print("\n8. EXTERNAL SERVICES")
    print("-" * 80)

    print(f"AWS Region: {settings.aws_region}")
    print(f"Stripe API Key Configured: {settings.stripe_api_key is not None}")
    print(f"SendGrid API Key Configured: {settings.sendgrid_api_key is not None}")
    print(f"Sentry DSN Configured: {settings.sentry_dsn is not None}")

    print("\n9. URL VALIDATION")
    print("-" * 80)

    db_url = db_config.get_url()
    redis_url = redis_config.get_url()

    print(f"Database URL valid: {validate_database_url(db_url)}")
    print(f"Redis URL valid: {validate_redis_url(redis_url)}")

    print("\n10. EXAMPLE .ENV FILE CONTENT")
    print("-" * 80)

    env_template = f"""
# Generated example .env file
APP_ENV=development
APP_NAME={settings.app_name}
DEBUG=true

# Security (NEVER commit real secrets!)
SECRET_KEY={generate_secret_key(64)}
JWT_SECRET={generate_secret_key(64)}
API_KEY={generate_api_key("sk", 32)}
ENCRYPTION_KEY={generate_encryption_key()}

# Database
DB_HOST={settings.db_host}
DB_PORT={settings.db_port}
DB_USER={settings.db_user}
DB_PASSWORD=your-secure-password
DB_NAME={settings.db_name}

# Redis
REDIS_HOST={settings.redis_host}
REDIS_PORT={settings.redis_port}
REDIS_DB={settings.redis_db}
    """.strip()

    print(env_template)

    print("\n" + "=" * 80)
    print("CONFIGURATION EXAMPLE COMPLETE")
    print("=" * 80)
    print("\nNOTE: Never commit .env files or hardcode secrets in production!")
    print("Use environment variables, secret managers, or vault systems.")
