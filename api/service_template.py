"""
Production-Ready FastAPI Service Template

Apply to: REST APIs, microservices, internal services, public APIs

Features:
- Request ID middleware for distributed tracing
- CORS configuration for cross-origin requests
- Security headers (CSP, HSTS, X-Frame-Options, X-Content-Type-Options)
- Health check endpoints with dependency verification
- Structured JSON error responses
- Request/response logging middleware
- Rate limiting middleware
- API versioning pattern (/v1/)
- OpenAPI/Swagger documentation

Installation:
    pip install fastapi uvicorn python-multipart

Run:
    uvicorn service_template:app --host 0.0.0.0 --port 8000 --reload

Production:
    uvicorn service_template:app --host 0.0.0.0 --port 8000 --workers 4

Test endpoints:
    curl http://localhost:8000/health
    curl http://localhost:8000/ready
    curl http://localhost:8000/v1/users
    curl -X POST http://localhost:8000/v1/users -H "Content-Type: application/json" -d '{"name":"John","email":"john@example.com"}'
"""

import logging
import time
import uuid
from collections import defaultdict
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, Request, Response, HTTPException, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, EmailStr, Field
from starlette.middleware.base import BaseHTTPMiddleware

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# =============================================================================
# Configuration
# =============================================================================

class Settings:
    """Application settings"""
    APP_NAME = "FastAPI Service Template"
    APP_VERSION = "1.0.0"
    APP_DESCRIPTION = "Production-ready FastAPI template with enterprise features"
    API_V1_PREFIX = "/v1"
    
    # CORS settings
    CORS_ORIGINS = ["*"]  # In production, specify exact origins
    CORS_ALLOW_CREDENTIALS = True
    CORS_ALLOW_METHODS = ["*"]
    CORS_ALLOW_HEADERS = ["*"]
    
    # Rate limiting settings
    RATE_LIMIT_REQUESTS = 100  # requests per minute
    RATE_LIMIT_WINDOW = 60  # seconds
    
    # Security headers
    SECURITY_HEADERS = {
        "X-Content-Type-Options": "nosniff",
        "X-Frame-Options": "DENY",
        "X-XSS-Protection": "1; mode=block",
        "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
        "Content-Security-Policy": "default-src 'self'",
    }

settings = Settings()

# =============================================================================
# Models
# =============================================================================

class ErrorResponse(BaseModel):
    """Standard error response model"""
    error: str
    message: str
    request_id: Optional[str] = None
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    path: Optional[str] = None

class HealthResponse(BaseModel):
    """Health check response model"""
    status: str
    timestamp: str
    version: str
    request_id: Optional[str] = None

class ReadinessResponse(BaseModel):
    """Readiness check response model"""
    status: str
    timestamp: str
    checks: Dict[str, str]
    request_id: Optional[str] = None

class User(BaseModel):
    """User model"""
    id: Optional[int] = None
    name: str = Field(..., min_length=1, max_length=100)
    email: EmailStr
    created_at: Optional[str] = None

class UserResponse(BaseModel):
    """User response model"""
    data: User
    request_id: str

class UsersListResponse(BaseModel):
    """Users list response model"""
    data: List[User]
    count: int
    request_id: str

# =============================================================================
# Middleware
# =============================================================================

class RequestIDMiddleware(BaseHTTPMiddleware):
    """Add unique request ID to each request for distributed tracing"""
    
    async def dispatch(self, request: Request, call_next):
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        
        return response

class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Add security headers to all responses"""
    
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        
        for header, value in settings.SECURITY_HEADERS.items():
            response.headers[header] = value
        
        return response

class LoggingMiddleware(BaseHTTPMiddleware):
    """Log all requests and responses"""
    
    async def dispatch(self, request: Request, call_next):
        request_id = getattr(request.state, "request_id", "unknown")
        start_time = time.time()
        
        # Log request
        logger.info(
            f"Request: {request.method} {request.url.path} | "
            f"Request-ID: {request_id} | "
            f"Client: {request.client.host if request.client else 'unknown'}"
        )
        
        try:
            response = await call_next(request)
            process_time = time.time() - start_time
            
            # Log response
            logger.info(
                f"Response: {request.method} {request.url.path} | "
                f"Status: {response.status_code} | "
                f"Duration: {process_time:.3f}s | "
                f"Request-ID: {request_id}"
            )
            
            response.headers["X-Process-Time"] = str(process_time)
            return response
            
        except Exception as e:
            process_time = time.time() - start_time
            logger.error(
                f"Error: {request.method} {request.url.path} | "
                f"Exception: {str(e)} | "
                f"Duration: {process_time:.3f}s | "
                f"Request-ID: {request_id}"
            )
            raise

class RateLimitMiddleware(BaseHTTPMiddleware):
    """Simple in-memory rate limiting middleware"""
    
    def __init__(self, app, requests_per_window: int = 100, window_seconds: int = 60):
        super().__init__(app)
        self.requests_per_window = requests_per_window
        self.window_seconds = window_seconds
        self.clients: Dict[str, List[float]] = defaultdict(list)
    
    async def dispatch(self, request: Request, call_next):
        # Skip rate limiting for health checks
        if request.url.path in ["/health", "/ready", "/docs", "/redoc", "/openapi.json"]:
            return await call_next(request)
        
        client_ip = request.client.host if request.client else "unknown"
        current_time = time.time()
        
        # Clean old requests outside the time window
        self.clients[client_ip] = [
            req_time for req_time in self.clients[client_ip]
            if current_time - req_time < self.window_seconds
        ]
        
        # Check rate limit
        if len(self.clients[client_ip]) >= self.requests_per_window:
            logger.warning(
                f"Rate limit exceeded for client: {client_ip} | "
                f"Requests: {len(self.clients[client_ip])}/{self.requests_per_window}"
            )
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={
                    "error": "rate_limit_exceeded",
                    "message": f"Rate limit exceeded. Maximum {self.requests_per_window} requests per {self.window_seconds} seconds.",
                    "request_id": getattr(request.state, "request_id", None),
                    "timestamp": datetime.utcnow().isoformat(),
                }
            )
        
        # Add current request
        self.clients[client_ip].append(current_time)
        
        response = await call_next(request)
        response.headers["X-RateLimit-Limit"] = str(self.requests_per_window)
        response.headers["X-RateLimit-Remaining"] = str(
            self.requests_per_window - len(self.clients[client_ip])
        )
        
        return response

# =============================================================================
# Lifespan Events
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle startup and shutdown events"""
    # Startup
    logger.info(f"Starting {settings.APP_NAME} v{settings.APP_VERSION}")
    logger.info("Initializing dependencies...")
    
    # Initialize resources here (database connections, cache, etc.)
    app.state.startup_time = datetime.utcnow()
    
    yield
    
    # Shutdown
    logger.info(f"Shutting down {settings.APP_NAME}")
    # Cleanup resources here

# =============================================================================
# Application Setup
# =============================================================================

app = FastAPI(
    title=settings.APP_NAME,
    description=settings.APP_DESCRIPTION,
    version=settings.APP_VERSION,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    openapi_tags=[
        {
            "name": "health",
            "description": "Health check and readiness endpoints",
        },
        {
            "name": "users",
            "description": "User management operations (demo)",
        },
    ],
)

# Add middleware (order matters - first added = outermost layer)
app.add_middleware(RateLimitMiddleware, requests_per_window=settings.RATE_LIMIT_REQUESTS, window_seconds=settings.RATE_LIMIT_WINDOW)
app.add_middleware(SecurityHeadersMiddleware)
app.add_middleware(LoggingMiddleware)
app.add_middleware(RequestIDMiddleware)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=settings.CORS_ALLOW_CREDENTIALS,
    allow_methods=settings.CORS_ALLOW_METHODS,
    allow_headers=settings.CORS_ALLOW_HEADERS,
)

# =============================================================================
# Exception Handlers
# =============================================================================

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions with structured error response"""
    request_id = getattr(request.state, "request_id", None)
    
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": f"http_error_{exc.status_code}",
            "message": exc.detail,
            "request_id": request_id,
            "timestamp": datetime.utcnow().isoformat(),
            "path": str(request.url.path),
        }
    )

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle validation errors with structured error response"""
    request_id = getattr(request.state, "request_id", None)
    
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "error": "validation_error",
            "message": "Request validation failed",
            "details": exc.errors(),
            "request_id": request_id,
            "timestamp": datetime.utcnow().isoformat(),
            "path": str(request.url.path),
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle unexpected exceptions with structured error response"""
    request_id = getattr(request.state, "request_id", None)
    
    logger.exception(f"Unhandled exception: {str(exc)} | Request-ID: {request_id}")
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "internal_server_error",
            "message": "An unexpected error occurred",
            "request_id": request_id,
            "timestamp": datetime.utcnow().isoformat(),
            "path": str(request.url.path),
        }
    )

# =============================================================================
# Health Check Endpoints
# =============================================================================

@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["health"],
    summary="Basic health check",
    description="Returns service health status without checking dependencies"
)
async def health_check(request: Request) -> HealthResponse:
    """Basic health check endpoint - always returns 200 if service is running"""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.utcnow().isoformat(),
        version=settings.APP_VERSION,
        request_id=getattr(request.state, "request_id", None),
    )

@app.get(
    "/ready",
    response_model=ReadinessResponse,
    tags=["health"],
    summary="Readiness check",
    description="Returns service readiness status with dependency checks"
)
async def readiness_check(request: Request) -> ReadinessResponse:
    """
    Readiness check endpoint - verifies all dependencies are available
    
    In production, check actual dependencies:
    - Database connectivity
    - Cache availability
    - External service health
    """
    checks = {}
    all_healthy = True
    
    # Simulate database check
    try:
        # Replace with actual database ping
        # await database.ping()
        checks["database"] = "healthy"
    except Exception as e:
        checks["database"] = f"unhealthy: {str(e)}"
        all_healthy = False
    
    # Simulate cache check
    try:
        # Replace with actual cache ping
        # await cache.ping()
        checks["cache"] = "healthy"
    except Exception as e:
        checks["cache"] = f"unhealthy: {str(e)}"
        all_healthy = False
    
    status_code = status.HTTP_200_OK if all_healthy else status.HTTP_503_SERVICE_UNAVAILABLE
    
    return JSONResponse(
        status_code=status_code,
        content={
            "status": "ready" if all_healthy else "not_ready",
            "timestamp": datetime.utcnow().isoformat(),
            "checks": checks,
            "request_id": getattr(request.state, "request_id", None),
        }
    )

# =============================================================================
# API v1 Endpoints (Example)
# =============================================================================

# In-memory storage for demo purposes
users_db: Dict[int, User] = {}
user_id_counter = 1

@app.get(
    f"{settings.API_V1_PREFIX}/users",
    response_model=UsersListResponse,
    tags=["users"],
    summary="List all users",
    description="Retrieve a list of all users in the system"
)
async def list_users(request: Request) -> UsersListResponse:
    """Get all users"""
    users_list = list(users_db.values())
    
    return UsersListResponse(
        data=users_list,
        count=len(users_list),
        request_id=getattr(request.state, "request_id", "unknown"),
    )

@app.post(
    f"{settings.API_V1_PREFIX}/users",
    response_model=UserResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["users"],
    summary="Create a new user",
    description="Create a new user with the provided information"
)
async def create_user(user: User, request: Request) -> UserResponse:
    """Create a new user"""
    global user_id_counter
    
    user.id = user_id_counter
    user.created_at = datetime.utcnow().isoformat()
    users_db[user_id_counter] = user
    user_id_counter += 1
    
    logger.info(f"Created user: {user.id} | Email: {user.email}")
    
    return UserResponse(
        data=user,
        request_id=getattr(request.state, "request_id", "unknown"),
    )

@app.get(
    f"{settings.API_V1_PREFIX}/users/{{user_id}}",
    response_model=UserResponse,
    tags=["users"],
    summary="Get user by ID",
    description="Retrieve a specific user by their ID"
)
async def get_user(user_id: int, request: Request) -> UserResponse:
    """Get a specific user by ID"""
    if user_id not in users_db:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"User with ID {user_id} not found"
        )
    
    return UserResponse(
        data=users_db[user_id],
        request_id=getattr(request.state, "request_id", "unknown"),
    )

@app.put(
    f"{settings.API_V1_PREFIX}/users/{{user_id}}",
    response_model=UserResponse,
    tags=["users"],
    summary="Update user",
    description="Update an existing user's information"
)
async def update_user(user_id: int, user_update: User, request: Request) -> UserResponse:
    """Update an existing user"""
    if user_id not in users_db:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"User with ID {user_id} not found"
        )
    
    existing_user = users_db[user_id]
    existing_user.name = user_update.name
    existing_user.email = user_update.email
    
    logger.info(f"Updated user: {user_id}")
    
    return UserResponse(
        data=existing_user,
        request_id=getattr(request.state, "request_id", "unknown"),
    )

@app.delete(
    f"{settings.API_V1_PREFIX}/users/{{user_id}}",
    status_code=status.HTTP_204_NO_CONTENT,
    tags=["users"],
    summary="Delete user",
    description="Delete a user from the system"
)
async def delete_user(user_id: int, request: Request):
    """Delete a user"""
    if user_id not in users_db:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"User with ID {user_id} not found"
        )
    
    del users_db[user_id]
    logger.info(f"Deleted user: {user_id}")
    
    return Response(status_code=status.HTTP_204_NO_CONTENT)

# =============================================================================
# Root Endpoint
# =============================================================================

@app.get("/", include_in_schema=False)
async def root():
    """Root endpoint - redirect to documentation"""
    return {
        "message": f"Welcome to {settings.APP_NAME}",
        "version": settings.APP_VERSION,
        "docs": "/docs",
        "health": "/health",
        "ready": "/ready",
    }

# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "service_template:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
