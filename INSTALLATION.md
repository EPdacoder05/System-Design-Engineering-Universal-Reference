# 🚀 Installation & Quick Start Guide

## Prerequisites

- Python 3.10+ (3.11+ recommended)
- pip package manager

## Installation

### Option 1: Clone and Install All Dependencies

```bash
git clone https://github.com/EPdacoder05/System-Design-Engineering-Universal-Reference.git
cd System-Design-Engineering-Universal-Reference
pip install -r requirements.txt
```

### Option 2: Selective Installation (Recommended)

Install only what you need:

```bash
# For security modules
pip install python-jose[cryptography] passlib[bcrypt] pyotp cryptography

# For API development
pip install fastapi uvicorn[standard] pydantic pydantic-settings

# For database modules
pip install sqlalchemy[asyncio] asyncpg psycopg2-binary pgvector

# For performance/caching
pip install redis pandas numpy

# For ML modules
pip install scikit-learn joblib pandas numpy

# For monitoring
pip install structlog prometheus-client

# For testing
pip install pytest pytest-asyncio pytest-cov httpx faker
```

## Quick Usage Examples

### 1. Security - Attack Detection

```python
from security.input_validator import detect_attack_patterns

user_input = "' OR '1'='1"
attacks = detect_attack_patterns(user_input)
if attacks:
    print(f"⚠️ Attack detected: {attacks[0]['type']}")
```

### 2. Security - JWT Authentication

```python
from security.auth_framework import create_access_token, verify_token

# Create token
token = create_access_token({"sub": "user123", "role": "admin"})

# Verify token
payload = verify_token(token)
print(f"User ID: {payload['sub']}")
```

### 3. Performance - Multi-Tier Caching

```python
from performance.caching import InMemoryCache

cache = InMemoryCache(max_size=1000, default_ttl=300)
cache.set("user:123", {"name": "John", "email": "john@example.com"})
user = cache.get("user:123")
```

### 4. API - FastAPI Template

```bash
# Run the production FastAPI server
cd api
uvicorn service_template:app --reload

# Visit http://localhost:8000/docs for interactive API docs
```

### 5. ML - Anomaly Detection

```python
from ml.anomaly_detector import AnomalyDetector
import numpy as np

detector = AnomalyDetector(threshold=3.0)
data = np.array([10, 11, 12, 10, 11, 12, 11, 10])
detector.fit(data)

result = detector.detect_anomaly(100.0)
print(f"Anomaly: {result['is_anomaly']}, Severity: {result['classification']}")
```

### 6. Database - Async SQLAlchemy

```python
from database.connection import DatabaseManager
import asyncio

async def example():
    db = DatabaseManager()
    await db.init()
    
    async with db.get_session() as session:
        # Your database operations here
        pass
    
    await db.close()

asyncio.run(example())
```

## Copy-Paste Workflow

The beauty of this library is modularity. Copy only what you need:

```bash
# Copy security module to your project
cp security/auth_framework.py ~/my-project/

# Copy API template
cp api/service_template.py ~/my-project/

# Copy CI/CD workflow
cp cicd/test-pipeline.yml ~/my-project/.github/workflows/
```

## Environment Variables

Create a `.env` file for configuration:

```bash
# JWT Configuration
JWT_SECRET_KEY=your-secret-key-here
JWT_ALGORITHM=HS256
JWT_ACCESS_TOKEN_EXPIRE_MINUTES=30

# Database
DATABASE_URL=postgresql+asyncpg://user:pass@localhost/dbname
DB_POOL_SIZE=20
DB_MAX_OVERFLOW=10

# Redis
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0

# Application
APP_ENV=production
DEBUG=false
```

## Running Tests

```bash
# Install test dependencies
pip install pytest pytest-asyncio pytest-cov

# Run all tests
pytest

# Run with coverage
pytest --cov=. --cov-report=html
```

## CI/CD Setup

### GitHub Actions

Copy the workflow files to your repository:

```bash
mkdir -p .github/workflows
cp cicd/test-pipeline.yml .github/workflows/
cp cicd/security-scan.yml .github/workflows/
```

### Dependabot

```bash
cp cicd/dependabot.yml .github/
```

### Docker

```bash
# Build Docker image
docker build -f cicd/Dockerfile -t my-app:latest .

# Run container
docker run -p 8000:8000 my-app:latest
```

## Troubleshooting

### Import Errors

If you get import errors, ensure you're in the correct directory:

```python
import sys
sys.path.insert(0, '/path/to/System-Design-Engineering-Universal-Reference')
from security.auth_framework import create_access_token
```

### Database Connection Issues

Ensure PostgreSQL is running and credentials are correct:

```bash
# Test connection
psql -h localhost -U username -d database_name
```

### Redis Connection Issues

Ensure Redis is running:

```bash
# Start Redis
redis-server

# Test connection
redis-cli ping
```

## Support

This is a reference library - adapt it to your needs!

- Review the comprehensive [README.md](README.md)
- Check [TRADEOFFS.md](TRADEOFFS.md) for architecture decisions
- See [PROJECT_SUMMARY.md](PROJECT_SUMMARY.md) for full documentation

## License

MIT License - use freely in any project, commercial or personal.

---

**Ready to ship!** 🚀
