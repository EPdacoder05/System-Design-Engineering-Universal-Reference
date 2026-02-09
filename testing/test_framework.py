"""
Comprehensive Pytest Testing Patterns and Best Practices

Apply to: Unit tests, integration tests, API tests

This module demonstrates:
- Fixture patterns (session, function, module scope)
- Factory pattern for test data generation
- Async test helpers and fixtures
- Mock/patch patterns for external services
- Database test fixtures with transaction rollback
- API client test fixtures
- Parametrize examples for data-driven tests
- Complete working examples

Dependencies:
    pip install pytest pytest-asyncio pytest-cov faker httpx pytest-mock

Coverage Configuration (pytest.ini):
```ini
[pytest]
minversion = 6.0
addopts = 
    -ra
    -q
    --strict-markers
    --cov=src
    --cov-report=term-missing
    --cov-report=html
    --cov-report=xml
    --cov-fail-under=80
    -p no:warnings
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
asyncio_mode = auto
markers =
    slow: marks tests as slow
    integration: marks tests as integration tests
    unit: marks tests as unit tests
    asyncio: marks tests as async
```
"""

import asyncio
import json
from datetime import datetime, timedelta
from decimal import Decimal
from typing import AsyncGenerator, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import httpx
import pytest
from faker import Faker

# =============================================================================
# FIXTURE PATTERNS - Different Scopes
# =============================================================================


@pytest.fixture(scope="session")
def faker_instance() -> Faker:
    """
    Session-scoped fixture - created once per test session.
    
    Apply to: All test types
    Use for: Expensive resources that can be shared across all tests
    """
    return Faker()


@pytest.fixture(scope="module")
def db_connection():
    """
    Module-scoped fixture - created once per test module.
    
    Apply to: Integration tests, database tests
    Use for: Database connections, external service connections
    """
    # Simulate database connection
    connection = {"host": "localhost", "port": 5432, "connected": True}
    print("\n[SETUP] Database connection established")
    
    yield connection
    
    # Teardown
    connection["connected"] = False
    print("\n[TEARDOWN] Database connection closed")


@pytest.fixture(scope="function")
def clean_database(db_connection):
    """
    Function-scoped fixture - created for each test function.
    
    Apply to: Integration tests requiring clean state
    Use for: Database cleanup, isolated test state
    """
    # Setup: Clean database before test
    print("\n[SETUP] Cleaning database")
    
    yield db_connection
    
    # Teardown: Rollback or clean after test
    print("\n[TEARDOWN] Rolling back database changes")


@pytest.fixture
def current_timestamp() -> datetime:
    """
    Function-scoped fixture (default scope).
    
    Apply to: All test types
    """
    return datetime.utcnow()


# =============================================================================
# FACTORY PATTERN - Test Data Generation
# =============================================================================


class UserFactory:
    """
    Factory for creating test user data.
    
    Apply to: Unit tests, integration tests
    """
    
    def __init__(self, faker: Faker):
        self.faker = faker
        self._id_counter = 1
    
    def create(
        self,
        user_id: Optional[int] = None,
        email: Optional[str] = None,
        username: Optional[str] = None,
        is_active: bool = True,
        **kwargs
    ) -> Dict:
        """Create a user with optional overrides."""
        if user_id is None:
            user_id = self._id_counter
            self._id_counter += 1
        
        return {
            "id": user_id,
            "email": email or self.faker.email(),
            "username": username or self.faker.user_name(),
            "first_name": self.faker.first_name(),
            "last_name": self.faker.last_name(),
            "is_active": is_active,
            "created_at": datetime.utcnow().isoformat(),
            **kwargs
        }
    
    def create_batch(self, count: int, **kwargs) -> List[Dict]:
        """Create multiple users."""
        return [self.create(**kwargs) for _ in range(count)]


class ProductFactory:
    """
    Factory for creating test product data.
    
    Apply to: E-commerce tests, inventory tests
    """
    
    def __init__(self, faker: Faker):
        self.faker = faker
        self._id_counter = 1
    
    def create(
        self,
        product_id: Optional[int] = None,
        name: Optional[str] = None,
        price: Optional[Decimal] = None,
        in_stock: bool = True,
        **kwargs
    ) -> Dict:
        """Create a product with optional overrides."""
        if product_id is None:
            product_id = self._id_counter
            self._id_counter += 1
        
        return {
            "id": product_id,
            "name": name or self.faker.catch_phrase(),
            "description": self.faker.text(max_nb_chars=200),
            "price": float(price) if price else round(self.faker.random.uniform(10, 1000), 2),
            "sku": self.faker.ean13(),
            "in_stock": in_stock,
            "quantity": self.faker.random_int(0, 100) if in_stock else 0,
            "category": self.faker.word(),
            "created_at": datetime.utcnow().isoformat(),
            **kwargs
        }
    
    def create_batch(self, count: int, **kwargs) -> List[Dict]:
        """Create multiple products."""
        return [self.create(**kwargs) for _ in range(count)]


@pytest.fixture
def user_factory(faker_instance: Faker) -> UserFactory:
    """
    Fixture providing user factory.
    
    Apply to: All tests needing user data
    """
    return UserFactory(faker_instance)


@pytest.fixture
def product_factory(faker_instance: Faker) -> ProductFactory:
    """
    Fixture providing product factory.
    
    Apply to: All tests needing product data
    """
    return ProductFactory(faker_instance)


# =============================================================================
# ASYNC TEST HELPERS AND FIXTURES
# =============================================================================


@pytest.fixture
async def async_client() -> AsyncGenerator[httpx.AsyncClient, None]:
    """
    Async fixture for HTTP client.
    
    Apply to: API integration tests, async tests
    """
    async with httpx.AsyncClient(base_url="http://testserver") as client:
        yield client


@pytest.fixture
async def async_db_session():
    """
    Async database session with automatic rollback.
    
    Apply to: Async database tests
    """
    # Simulate async database session
    session = {
        "id": "async_session_123",
        "transaction_active": True,
        "queries": []
    }
    
    print("\n[SETUP] Async database session started")
    
    yield session
    
    # Rollback transaction
    session["transaction_active"] = False
    print(f"\n[TEARDOWN] Async session rolled back ({len(session['queries'])} queries)")


async def async_operation(delay: float = 0.1) -> str:
    """
    Helper function for async operations.
    
    Apply to: Async unit tests
    """
    await asyncio.sleep(delay)
    return "async_result"


@pytest.fixture
def event_loop():
    """
    Create event loop for async tests.
    
    Apply to: Async tests
    """
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


# =============================================================================
# MOCK/PATCH PATTERNS - External Services
# =============================================================================


@pytest.fixture
def mock_api_client():
    """
    Mock for external API client.
    
    Apply to: Unit tests, integration tests with external APIs
    """
    mock = Mock()
    mock.get.return_value = {
        "status": "success",
        "data": {"id": 1, "name": "Test Data"}
    }
    mock.post.return_value = {
        "status": "created",
        "id": 1
    }
    return mock


@pytest.fixture
def mock_redis_client():
    """
    Mock Redis client for caching tests.
    
    Apply to: Caching tests, integration tests
    """
    mock = Mock()
    storage = {}
    
    def get_side_effect(key):
        return storage.get(key)
    
    def set_side_effect(key, value, ex=None):
        storage[key] = value
        return True
    
    def delete_side_effect(key):
        return storage.pop(key, None)
    
    mock.get.side_effect = get_side_effect
    mock.set.side_effect = set_side_effect
    mock.delete.side_effect = delete_side_effect
    mock.exists.side_effect = lambda key: key in storage
    
    return mock


@pytest.fixture
def mock_database():
    """
    Mock database with in-memory storage.
    
    Apply to: Unit tests, database isolation
    """
    class MockDatabase:
        def __init__(self):
            self.users = {}
            self.products = {}
            self._user_id = 1
            self._product_id = 1
        
        def insert_user(self, user_data: Dict) -> Dict:
            user_id = self._user_id
            self._user_id += 1
            user = {**user_data, "id": user_id}
            self.users[user_id] = user
            return user
        
        def get_user(self, user_id: int) -> Optional[Dict]:
            return self.users.get(user_id)
        
        def update_user(self, user_id: int, update_data: Dict) -> Optional[Dict]:
            if user_id in self.users:
                self.users[user_id].update(update_data)
                return self.users[user_id]
            return None
        
        def delete_user(self, user_id: int) -> bool:
            return self.users.pop(user_id, None) is not None
        
        def clear(self):
            self.users.clear()
            self.products.clear()
    
    db = MockDatabase()
    yield db
    db.clear()


@pytest.fixture
async def mock_async_api_client():
    """
    Async mock for external API client.
    
    Apply to: Async API tests
    """
    mock = AsyncMock()
    mock.get.return_value = httpx.Response(
        200,
        json={"status": "success", "data": {"id": 1}},
    )
    mock.post.return_value = httpx.Response(
        201,
        json={"status": "created", "id": 1},
    )
    return mock


# =============================================================================
# DATABASE TEST FIXTURES WITH TRANSACTION ROLLBACK
# =============================================================================


@pytest.fixture
def db_transaction(db_connection):
    """
    Database fixture with automatic transaction rollback.
    
    Apply to: Integration tests requiring database isolation
    """
    # Begin transaction
    transaction = {
        "id": "txn_123",
        "connection": db_connection,
        "savepoint": "sp1",
        "operations": []
    }
    print("\n[SETUP] Database transaction started")
    
    yield transaction
    
    # Rollback transaction
    transaction["operations"].clear()
    print(f"\n[TEARDOWN] Database transaction rolled back")


@pytest.fixture
def populated_database(db_transaction, user_factory, product_factory):
    """
    Database pre-populated with test data.
    
    Apply to: Integration tests needing existing data
    """
    # Populate database with test data
    users = user_factory.create_batch(5)
    products = product_factory.create_batch(10)
    
    db_data = {
        "users": {user["id"]: user for user in users},
        "products": {product["id"]: product for product in products},
        "transaction": db_transaction
    }
    
    print(f"\n[SETUP] Database populated: {len(users)} users, {len(products)} products")
    
    yield db_data
    
    # Cleanup handled by db_transaction rollback
    print("\n[TEARDOWN] Database data cleared via rollback")


# =============================================================================
# API CLIENT TEST FIXTURES
# =============================================================================


class MockFastAPITestClient:
    """
    Mock FastAPI TestClient.
    
    Apply to: FastAPI application tests
    """
    
    def __init__(self, base_url: str = "http://testserver"):
        self.base_url = base_url
        self.headers = {}
    
    def get(self, url: str, **kwargs):
        """Mock GET request."""
        return Mock(
            status_code=200,
            json=lambda: {"message": "GET success", "url": url},
            headers={}
        )
    
    def post(self, url: str, json=None, **kwargs):
        """Mock POST request."""
        return Mock(
            status_code=201,
            json=lambda: {"message": "POST success", "data": json},
            headers={"Location": f"{url}/1"}
        )
    
    def put(self, url: str, json=None, **kwargs):
        """Mock PUT request."""
        return Mock(
            status_code=200,
            json=lambda: {"message": "PUT success", "data": json},
            headers={}
        )
    
    def delete(self, url: str, **kwargs):
        """Mock DELETE request."""
        return Mock(
            status_code=204,
            json=lambda: None,
            headers={}
        )


@pytest.fixture
def api_client():
    """
    API test client fixture.
    
    Apply to: API integration tests
    """
    return MockFastAPITestClient()


@pytest.fixture
def authenticated_api_client(api_client):
    """
    Authenticated API client with JWT token.
    
    Apply to: Authenticated API tests
    """
    api_client.headers["Authorization"] = "Bearer test_jwt_token_123"
    return api_client


# =============================================================================
# PYTEST PARAMETRIZE EXAMPLES - Data-Driven Tests
# =============================================================================


@pytest.mark.parametrize("input,expected", [
    (1, 2),
    (2, 4),
    (3, 6),
    (10, 20),
])
def test_double_number(input, expected):
    """
    Basic parametrize example.
    
    Apply to: Unit tests with multiple input cases
    """
    assert input * 2 == expected


@pytest.mark.parametrize("email,is_valid", [
    ("user@example.com", True),
    ("user.name@example.co.uk", True),
    ("invalid.email", False),
    ("@example.com", False),
    ("user@", False),
])
def test_email_validation(email, is_valid):
    """
    Email validation with parametrize.
    
    Apply to: Validation tests
    """
    import re
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    assert bool(re.match(pattern, email)) == is_valid


@pytest.mark.parametrize("status_code,expected_result", [
    (200, "success"),
    (201, "created"),
    (400, "client_error"),
    (401, "unauthorized"),
    (404, "not_found"),
    (500, "server_error"),
])
def test_http_status_handler(status_code, expected_result):
    """
    HTTP status code handling.
    
    Apply to: API response handling tests
    """
    def handle_status(code):
        if 200 <= code < 300:
            return "success" if code == 200 else "created"
        elif 400 <= code < 500:
            if code == 401:
                return "unauthorized"
            elif code == 404:
                return "not_found"
            return "client_error"
        else:
            return "server_error"
    
    assert handle_status(status_code) == expected_result


@pytest.mark.parametrize("user_data", [
    {"username": "john", "email": "john@example.com", "age": 25},
    {"username": "jane", "email": "jane@example.com", "age": 30},
    {"username": "bob", "email": "bob@example.com", "age": 35},
])
def test_user_creation_with_dict(user_data, mock_database):
    """
    Parametrize with dictionary data.
    
    Apply to: Entity creation tests
    """
    user = mock_database.insert_user(user_data)
    assert user["username"] == user_data["username"]
    assert user["email"] == user_data["email"]
    assert "id" in user


# =============================================================================
# EXAMPLE TEST CLASSES - Complete Patterns
# =============================================================================


@pytest.mark.unit
class TestUserService:
    """
    Unit tests for user service.
    
    Apply to: Service layer unit tests
    """
    
    def test_create_user(self, user_factory, mock_database):
        """Test user creation."""
        user_data = user_factory.create()
        created_user = mock_database.insert_user(user_data)
        
        assert created_user["id"] is not None
        assert created_user["email"] == user_data["email"]
        assert created_user["username"] == user_data["username"]
    
    def test_get_user(self, user_factory, mock_database):
        """Test user retrieval."""
        user_data = user_factory.create()
        created_user = mock_database.insert_user(user_data)
        
        retrieved_user = mock_database.get_user(created_user["id"])
        assert retrieved_user == created_user
    
    def test_update_user(self, user_factory, mock_database):
        """Test user update."""
        user_data = user_factory.create()
        created_user = mock_database.insert_user(user_data)
        
        update_data = {"email": "newemail@example.com"}
        updated_user = mock_database.update_user(created_user["id"], update_data)
        
        assert updated_user["email"] == "newemail@example.com"
        assert updated_user["username"] == created_user["username"]
    
    def test_delete_user(self, user_factory, mock_database):
        """Test user deletion."""
        user_data = user_factory.create()
        created_user = mock_database.insert_user(user_data)
        
        deleted = mock_database.delete_user(created_user["id"])
        assert deleted is True
        
        retrieved_user = mock_database.get_user(created_user["id"])
        assert retrieved_user is None
    
    @pytest.mark.parametrize("is_active", [True, False])
    def test_user_active_status(self, user_factory, mock_database, is_active):
        """Test user with different active status."""
        user_data = user_factory.create(is_active=is_active)
        created_user = mock_database.insert_user(user_data)
        
        assert created_user["is_active"] == is_active


@pytest.mark.integration
class TestAPIEndpoints:
    """
    Integration tests for API endpoints.
    
    Apply to: API integration tests
    """
    
    def test_get_users_endpoint(self, api_client):
        """Test GET /users endpoint."""
        response = api_client.get("/api/v1/users")
        
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
    
    def test_create_user_endpoint(self, api_client, user_factory):
        """Test POST /users endpoint."""
        user_data = user_factory.create()
        response = api_client.post("/api/v1/users", json=user_data)
        
        assert response.status_code == 201
        data = response.json()
        assert data["data"] == user_data
    
    def test_authenticated_endpoint(self, authenticated_api_client):
        """Test authenticated endpoint."""
        response = authenticated_api_client.get("/api/v1/profile")
        
        assert response.status_code == 200
        assert "Authorization" in authenticated_api_client.headers
    
    def test_update_user_endpoint(self, api_client, user_factory):
        """Test PUT /users/{id} endpoint."""
        user_data = user_factory.create()
        update_data = {"email": "updated@example.com"}
        
        response = api_client.put("/api/v1/users/1", json=update_data)
        
        assert response.status_code == 200
        data = response.json()
        assert data["data"] == update_data
    
    def test_delete_user_endpoint(self, api_client):
        """Test DELETE /users/{id} endpoint."""
        response = api_client.delete("/api/v1/users/1")
        
        assert response.status_code == 204


@pytest.mark.asyncio
class TestAsyncOperations:
    """
    Async operation tests.
    
    Apply to: Async service tests, async API tests
    """
    
    async def test_async_operation(self):
        """Test basic async operation."""
        result = await async_operation(0.01)
        assert result == "async_result"
    
    async def test_async_api_call(self, mock_async_api_client):
        """Test async API call."""
        response = await mock_async_api_client.get("https://api.example.com/data")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
    
    async def test_async_database_operation(self, async_db_session):
        """Test async database operation."""
        # Simulate async query
        async_db_session["queries"].append("SELECT * FROM users")
        
        assert async_db_session["transaction_active"] is True
        assert len(async_db_session["queries"]) == 1
    
    async def test_concurrent_operations(self):
        """Test concurrent async operations."""
        tasks = [async_operation(0.01) for _ in range(5)]
        results = await asyncio.gather(*tasks)
        
        assert len(results) == 5
        assert all(r == "async_result" for r in results)
    
    async def test_async_http_client(self, async_client):
        """Test async HTTP client."""
        # Note: This would make real requests to http://testserver
        # In practice, you'd mock the responses
        pass


class TestWithMocks:
    """
    Tests demonstrating mock patterns.
    
    Apply to: Unit tests with external dependencies
    """
    
    def test_with_api_mock(self, mock_api_client):
        """Test with mocked API client."""
        result = mock_api_client.get("/data")
        
        assert result["status"] == "success"
        assert "data" in result
        mock_api_client.get.assert_called_once_with("/data")
    
    def test_with_redis_mock(self, mock_redis_client):
        """Test with mocked Redis client."""
        # Set value
        mock_redis_client.set("key1", "value1")
        
        # Get value
        value = mock_redis_client.get("key1")
        assert value == "value1"
        
        # Check exists
        exists = mock_redis_client.exists("key1")
        assert exists is True
        
        # Delete
        mock_redis_client.delete("key1")
        value = mock_redis_client.get("key1")
        assert value is None
    
    @patch('httpx.get')
    def test_with_patch_decorator(self, mock_get):
        """Test using @patch decorator."""
        mock_get.return_value = Mock(
            status_code=200,
            json=lambda: {"data": "test"}
        )
        
        # Your code that uses httpx.get
        import httpx as httpx_module
        response = httpx_module.get("https://api.example.com")
        
        assert response.status_code == 200
        assert response.json()["data"] == "test"
    
    def test_with_context_manager_patch(self):
        """Test using patch as context manager."""
        with patch('httpx.post') as mock_post:
            mock_post.return_value = Mock(
                status_code=201,
                json=lambda: {"id": 1, "created": True}
            )
            
            import httpx as httpx_module
            response = httpx_module.post("https://api.example.com", json={"data": "test"})
            
            assert response.status_code == 201
            assert response.json()["created"] is True


@pytest.mark.integration
class TestDatabaseIntegration:
    """
    Database integration tests with transaction rollback.
    
    Apply to: Database integration tests
    """
    
    def test_with_clean_database(self, clean_database):
        """Test with clean database."""
        assert clean_database["connected"] is True
    
    def test_with_transaction(self, db_transaction):
        """Test with database transaction."""
        # Simulate database operations
        db_transaction["operations"].append("INSERT INTO users VALUES (...)")
        db_transaction["operations"].append("UPDATE users SET ...")
        
        assert len(db_transaction["operations"]) == 2
        # Transaction will be rolled back automatically
    
    def test_with_populated_database(self, populated_database):
        """Test with pre-populated database."""
        assert len(populated_database["users"]) == 5
        assert len(populated_database["products"]) == 10
        
        # Access user data
        user_id = list(populated_database["users"].keys())[0]
        user = populated_database["users"][user_id]
        assert "email" in user
        assert "username" in user


class TestFactoryPatterns:
    """
    Tests demonstrating factory pattern usage.
    
    Apply to: Tests requiring test data generation
    """
    
    def test_user_factory_basic(self, user_factory):
        """Test basic user creation."""
        user = user_factory.create()
        
        assert "id" in user
        assert "email" in user
        assert "username" in user
        assert user["is_active"] is True
    
    def test_user_factory_with_overrides(self, user_factory):
        """Test user creation with overrides."""
        user = user_factory.create(
            email="custom@example.com",
            username="customuser",
            is_active=False
        )
        
        assert user["email"] == "custom@example.com"
        assert user["username"] == "customuser"
        assert user["is_active"] is False
    
    def test_user_factory_batch(self, user_factory):
        """Test batch user creation."""
        users = user_factory.create_batch(10)
        
        assert len(users) == 10
        assert all("id" in user for user in users)
        
        # Check unique IDs
        user_ids = [user["id"] for user in users]
        assert len(set(user_ids)) == 10
    
    def test_product_factory(self, product_factory):
        """Test product creation."""
        product = product_factory.create()
        
        assert "id" in product
        assert "name" in product
        assert "price" in product
        assert product["in_stock"] is True
    
    def test_product_factory_out_of_stock(self, product_factory):
        """Test out of stock product."""
        product = product_factory.create(in_stock=False)
        
        assert product["in_stock"] is False
        assert product["quantity"] == 0
    
    def test_product_factory_custom_price(self, product_factory):
        """Test product with custom price."""
        product = product_factory.create(price=Decimal("99.99"))
        
        assert product["price"] == 99.99


@pytest.mark.slow
class TestPerformance:
    """
    Performance tests (marked as slow).
    
    Apply to: Performance testing, load testing
    """
    
    def test_batch_user_creation_performance(self, user_factory, mock_database):
        """Test performance of batch user creation."""
        import time
        
        start_time = time.time()
        users = user_factory.create_batch(1000)
        
        for user in users:
            mock_database.insert_user(user)
        
        elapsed_time = time.time() - start_time
        
        assert len(mock_database.users) == 1000
        assert elapsed_time < 5.0  # Should complete in under 5 seconds
    
    def test_concurrent_database_access(self, mock_database, user_factory):
        """Test concurrent database access."""
        users = user_factory.create_batch(100)
        
        for user in users:
            mock_database.insert_user(user)
        
        # Simulate concurrent reads
        for user_id in range(1, 101):
            user = mock_database.get_user(user_id)
            assert user is not None


# =============================================================================
# ADVANCED PATTERNS
# =============================================================================


@pytest.fixture
def temp_file(tmp_path):
    """
    Temporary file fixture using pytest's tmp_path.
    
    Apply to: File I/O tests
    """
    file_path = tmp_path / "test_file.json"
    file_path.write_text(json.dumps({"test": "data"}))
    yield file_path
    # Cleanup handled automatically by tmp_path


@pytest.fixture
def monkeypatch_env(monkeypatch):
    """
    Environment variable patching.
    
    Apply to: Configuration tests
    """
    monkeypatch.setenv("API_KEY", "test_key_123")
    monkeypatch.setenv("DATABASE_URL", "postgresql://test:test@localhost/test")
    return monkeypatch


class TestAdvancedPatterns:
    """
    Advanced testing patterns.
    
    Apply to: Complex testing scenarios
    """
    
    def test_with_temp_file(self, temp_file):
        """Test with temporary file."""
        content = json.loads(temp_file.read_text())
        assert content["test"] == "data"
    
    def test_with_env_vars(self, monkeypatch_env):
        """Test with environment variables."""
        import os
        assert os.getenv("API_KEY") == "test_key_123"
    
    def test_exception_handling(self, user_factory):
        """Test exception handling."""
        with pytest.raises(KeyError):
            user = user_factory.create()
            _ = user["nonexistent_key"]
    
    def test_warning_handling(self):
        """Test warning handling."""
        import warnings
        
        with pytest.warns(UserWarning):
            warnings.warn("This is a test warning", UserWarning)
    
    @pytest.mark.xfail(reason="Feature not implemented yet")
    def test_expected_failure(self):
        """Test expected to fail."""
        assert False
    
    @pytest.mark.skip(reason="Skipping for demonstration")
    def test_skipped(self):
        """Test to be skipped."""
        pass
    
    @pytest.mark.parametrize("value", [1, 2, 3])
    def test_with_fixture_and_parametrize(self, value, user_factory):
        """Test combining fixtures and parametrize."""
        users = user_factory.create_batch(value)
        assert len(users) == value


# =============================================================================
# FIXTURE DEPENDENCIES AND COMPOSITION
# =============================================================================


@pytest.fixture
def base_config():
    """Base configuration."""
    return {
        "app_name": "TestApp",
        "version": "1.0.0"
    }


@pytest.fixture
def extended_config(base_config):
    """Extended configuration depending on base_config."""
    return {
        **base_config,
        "database": "postgresql://localhost/test",
        "cache": "redis://localhost:6379"
    }


@pytest.fixture
def app_context(extended_config, mock_database, mock_redis_client):
    """
    Complete application context with all dependencies.
    
    Apply to: Integration tests requiring full app context
    """
    return {
        "config": extended_config,
        "database": mock_database,
        "cache": mock_redis_client,
        "initialized": True
    }


class TestFixtureComposition:
    """
    Tests demonstrating fixture composition.
    
    Apply to: Complex integration tests
    """
    
    def test_with_base_config(self, base_config):
        """Test with base configuration."""
        assert base_config["app_name"] == "TestApp"
    
    def test_with_extended_config(self, extended_config):
        """Test with extended configuration."""
        assert "database" in extended_config
        assert extended_config["app_name"] == "TestApp"
    
    def test_with_full_app_context(self, app_context):
        """Test with full application context."""
        assert app_context["initialized"] is True
        assert app_context["database"] is not None
        assert app_context["cache"] is not None


if __name__ == "__main__":
    """
    Run tests directly or with pytest:
    
    # Run all tests
    pytest test_framework.py -v
    
    # Run with coverage
    pytest test_framework.py --cov=. --cov-report=html
    
    # Run specific marker
    pytest test_framework.py -m unit
    pytest test_framework.py -m integration
    pytest test_framework.py -m "not slow"
    
    # Run specific test class
    pytest test_framework.py::TestUserService -v
    
    # Run specific test
    pytest test_framework.py::TestUserService::test_create_user -v
    
    # Run with keyword filter
    pytest test_framework.py -k "user" -v
    
    # Run async tests
    pytest test_framework.py -m asyncio -v
    """
    pytest.main([__file__, "-v", "--tb=short"])
