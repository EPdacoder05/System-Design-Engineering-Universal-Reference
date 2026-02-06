"""
Distributed System Service Patterns

Apply to: Microservices, service mesh, resilient APIs

This module provides production-ready patterns for building resilient distributed systems:
- Circuit Breaker: Prevents cascading failures
- Retry with Exponential Backoff: Handles transient failures
- Fan-out/Fan-in: Parallel execution and aggregation
- Saga Pattern: Distributed transaction management
- Service Discovery: Dynamic service location

All patterns use async/await with httpx for high-performance HTTP operations.
"""

import asyncio
import random
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
from functools import wraps
import logging

try:
    import httpx
except ImportError:
    raise ImportError("httpx is required. Install with: pip install httpx")


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# CIRCUIT BREAKER PATTERN
# ============================================================================

class CircuitState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"      # Normal operation, requests pass through
    OPEN = "open"          # Failure threshold reached, requests blocked
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class CircuitBreakerConfig:
    """Circuit breaker configuration"""
    failure_threshold: int = 5  # Failures before opening circuit
    success_threshold: int = 2  # Successes in half-open to close
    timeout: float = 60.0  # Seconds to wait before half-open
    excluded_exceptions: Tuple[type, ...] = ()  # Exceptions that don't count as failures


class CircuitBreaker:
    """
    Circuit Breaker pattern implementation.
    
    Prevents cascading failures by blocking requests when a service is unavailable.
    Three states: CLOSED (normal), OPEN (blocking), HALF_OPEN (testing recovery).
    """
    
    def __init__(self, config: Optional[CircuitBreakerConfig] = None):
        self.config = config or CircuitBreakerConfig()
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: Optional[float] = None
        self._lock = asyncio.Lock()
    
    async def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection"""
        async with self._lock:
            if self.state == CircuitState.OPEN:
                if self._should_attempt_reset():
                    logger.info("Circuit transitioning to HALF_OPEN")
                    self.state = CircuitState.HALF_OPEN
                else:
                    raise CircuitBreakerOpenError(
                        f"Circuit breaker is OPEN. Last failure: {self.last_failure_time}"
                    )
        
        try:
            result = await func(*args, **kwargs)
            await self._on_success()
            return result
        except Exception as e:
            if not isinstance(e, self.config.excluded_exceptions):
                await self._on_failure()
            raise
    
    async def _on_success(self):
        """Handle successful call"""
        async with self._lock:
            self.failure_count = 0
            
            if self.state == CircuitState.HALF_OPEN:
                self.success_count += 1
                logger.info(f"HALF_OPEN success count: {self.success_count}")
                
                if self.success_count >= self.config.success_threshold:
                    logger.info("Circuit transitioning to CLOSED")
                    self.state = CircuitState.CLOSED
                    self.success_count = 0
    
    async def _on_failure(self):
        """Handle failed call"""
        async with self._lock:
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            if self.state == CircuitState.HALF_OPEN:
                logger.warning("Failure in HALF_OPEN, reopening circuit")
                self.state = CircuitState.OPEN
                self.success_count = 0
            elif self.failure_count >= self.config.failure_threshold:
                logger.error(f"Failure threshold reached ({self.failure_count}), opening circuit")
                self.state = CircuitState.OPEN
    
    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt recovery"""
        if self.last_failure_time is None:
            return True
        return time.time() - self.last_failure_time >= self.config.timeout
    
    def get_state(self) -> Dict[str, Any]:
        """Get current circuit breaker state"""
        return {
            "state": self.state.value,
            "failure_count": self.failure_count,
            "success_count": self.success_count,
            "last_failure_time": self.last_failure_time,
        }


class CircuitBreakerOpenError(Exception):
    """Raised when circuit breaker is open"""
    pass


def circuit_breaker(config: Optional[CircuitBreakerConfig] = None):
    """Decorator for circuit breaker pattern"""
    cb = CircuitBreaker(config)
    
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            return await cb.call(func, *args, **kwargs)
        wrapper.circuit_breaker = cb
        return wrapper
    return decorator


# ============================================================================
# RETRY WITH EXPONENTIAL BACKOFF + JITTER
# ============================================================================

@dataclass
class RetryConfig:
    """Retry configuration"""
    max_attempts: int = 3
    base_delay: float = 1.0  # Initial delay in seconds
    max_delay: float = 60.0  # Maximum delay in seconds
    exponential_base: float = 2.0  # Multiplier for each retry
    jitter: bool = True  # Add randomness to prevent thundering herd
    retryable_exceptions: Tuple[type, ...] = (Exception,)
    retryable_status_codes: Set[int] = field(default_factory=lambda: {408, 429, 500, 502, 503, 504})


class RetryExhaustedError(Exception):
    """Raised when all retry attempts are exhausted"""
    pass


async def retry_with_backoff(
    func: Callable,
    config: Optional[RetryConfig] = None,
    *args,
    **kwargs
) -> Any:
    """
    Retry function with exponential backoff and jitter.
    
    Handles transient failures with increasing delays between attempts.
    Jitter prevents thundering herd problem in distributed systems.
    """
    config = config or RetryConfig()
    last_exception = None
    
    for attempt in range(config.max_attempts):
        try:
            result = await func(*args, **kwargs)
            
            # Check for HTTP status codes if result is httpx.Response
            if isinstance(result, httpx.Response):
                if result.status_code in config.retryable_status_codes:
                    raise httpx.HTTPStatusError(
                        f"Retryable status code: {result.status_code}",
                        request=result.request,
                        response=result,
                    )
            
            return result
            
        except config.retryable_exceptions as e:
            last_exception = e
            
            if attempt < config.max_attempts - 1:
                delay = min(
                    config.base_delay * (config.exponential_base ** attempt),
                    config.max_delay
                )
                
                if config.jitter:
                    delay = delay * (0.5 + random.random())
                
                logger.warning(
                    f"Attempt {attempt + 1}/{config.max_attempts} failed: {e}. "
                    f"Retrying in {delay:.2f}s..."
                )
                await asyncio.sleep(delay)
            else:
                logger.error(f"All {config.max_attempts} retry attempts exhausted")
    
    raise RetryExhaustedError(
        f"Failed after {config.max_attempts} attempts"
    ) from last_exception


def with_retry(config: Optional[RetryConfig] = None):
    """Decorator for retry with exponential backoff"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            return await retry_with_backoff(func, config, *args, **kwargs)
        return wrapper
    return decorator


# ============================================================================
# FAN-OUT / FAN-IN PATTERN
# ============================================================================

@dataclass
class FanOutResult:
    """Result from fan-out operation"""
    successful: List[Tuple[int, Any]]  # (index, result)
    failed: List[Tuple[int, Exception]]  # (index, exception)
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate"""
        total = len(self.successful) + len(self.failed)
        return len(self.successful) / total if total > 0 else 0.0
    
    def get_results(self) -> List[Any]:
        """Get all successful results in order"""
        sorted_results = sorted(self.successful, key=lambda x: x[0])
        return [result for _, result in sorted_results]


class FanOutFanIn:
    """
    Fan-out/Fan-in pattern for parallel execution.
    
    Distributes work across multiple workers (fan-out) and aggregates
    results (fan-in). Useful for parallel API calls, batch processing.
    """
    
    def __init__(self, max_concurrency: Optional[int] = None):
        """
        Initialize fan-out/fan-in executor.
        
        Args:
            max_concurrency: Maximum concurrent tasks (None = unlimited)
        """
        self.max_concurrency = max_concurrency
        self.semaphore = asyncio.Semaphore(max_concurrency) if max_concurrency else None
    
    async def execute(
        self,
        tasks: List[Callable],
        fail_fast: bool = False,
        timeout: Optional[float] = None,
    ) -> FanOutResult:
        """
        Execute multiple tasks in parallel and aggregate results.
        
        Args:
            tasks: List of async callable tasks
            fail_fast: If True, cancel remaining tasks on first failure
            timeout: Maximum time to wait for all tasks
        
        Returns:
            FanOutResult with successful and failed results
        """
        async def execute_task(index: int, task: Callable) -> Tuple[int, Any]:
            if self.semaphore:
                async with self.semaphore:
                    return index, await task()
            return index, await task()
        
        coroutines = [execute_task(i, task) for i, task in enumerate(tasks)]
        
        try:
            if fail_fast:
                # Use gather with return_exceptions=False to fail fast
                results = await asyncio.wait_for(
                    asyncio.gather(*coroutines),
                    timeout=timeout
                )
                return FanOutResult(
                    successful=results,
                    failed=[]
                )
            else:
                # Collect all results, including exceptions
                results = await asyncio.wait_for(
                    asyncio.gather(*coroutines, return_exceptions=True),
                    timeout=timeout
                )
                
                successful = []
                failed = []
                
                for i, result in enumerate(results):
                    if isinstance(result, Exception):
                        failed.append((i, result))
                    else:
                        successful.append(result)
                
                return FanOutResult(successful=successful, failed=failed)
                
        except asyncio.TimeoutError as e:
            logger.error(f"Fan-out operation timed out after {timeout}s")
            raise


async def fan_out(
    tasks: List[Callable],
    max_concurrency: Optional[int] = None,
    **kwargs
) -> FanOutResult:
    """Convenience function for fan-out/fan-in pattern"""
    executor = FanOutFanIn(max_concurrency)
    return await executor.execute(tasks, **kwargs)


# ============================================================================
# SAGA PATTERN (ORCHESTRATION-BASED)
# ============================================================================

@dataclass
class SagaStep:
    """Single step in a saga transaction"""
    name: str
    action: Callable  # Forward action
    compensation: Callable  # Rollback action
    timeout: Optional[float] = 30.0


class SagaStatus(Enum):
    """Saga execution status"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    COMPENSATING = "compensating"
    FAILED = "failed"
    COMPENSATED = "compensated"


@dataclass
class SagaResult:
    """Result of saga execution"""
    status: SagaStatus
    completed_steps: List[str]
    failed_step: Optional[str] = None
    error: Optional[Exception] = None
    compensated_steps: List[str] = field(default_factory=list)


class SagaOrchestrator:
    """
    Saga pattern for distributed transactions (orchestration-based).
    
    Manages long-running transactions across multiple services with
    compensating actions for rollback. Ensures eventual consistency.
    """
    
    def __init__(self, saga_id: str):
        self.saga_id = saga_id
        self.steps: List[SagaStep] = []
        self.status = SagaStatus.PENDING
    
    def add_step(
        self,
        name: str,
        action: Callable,
        compensation: Callable,
        timeout: Optional[float] = 30.0
    ) -> 'SagaOrchestrator':
        """Add a step to the saga"""
        self.steps.append(SagaStep(name, action, compensation, timeout))
        return self
    
    async def execute(self) -> SagaResult:
        """
        Execute the saga with automatic compensation on failure.
        
        Returns:
            SagaResult with execution details
        """
        self.status = SagaStatus.IN_PROGRESS
        completed_steps = []
        step_results = {}
        
        logger.info(f"[Saga {self.saga_id}] Starting execution with {len(self.steps)} steps")
        
        try:
            # Execute forward actions
            for step in self.steps:
                logger.info(f"[Saga {self.saga_id}] Executing step: {step.name}")
                
                try:
                    result = await asyncio.wait_for(
                        step.action(),
                        timeout=step.timeout
                    )
                    step_results[step.name] = result
                    completed_steps.append(step.name)
                    logger.info(f"[Saga {self.saga_id}] Step {step.name} completed")
                    
                except Exception as e:
                    logger.error(f"[Saga {self.saga_id}] Step {step.name} failed: {e}")
                    # Compensate completed steps in reverse order
                    compensated = await self._compensate(completed_steps, step_results)
                    
                    return SagaResult(
                        status=SagaStatus.COMPENSATED if compensated else SagaStatus.FAILED,
                        completed_steps=completed_steps,
                        failed_step=step.name,
                        error=e,
                        compensated_steps=compensated,
                    )
            
            self.status = SagaStatus.COMPLETED
            logger.info(f"[Saga {self.saga_id}] Completed successfully")
            
            return SagaResult(
                status=SagaStatus.COMPLETED,
                completed_steps=completed_steps,
            )
            
        except Exception as e:
            logger.error(f"[Saga {self.saga_id}] Unexpected error: {e}")
            return SagaResult(
                status=SagaStatus.FAILED,
                completed_steps=completed_steps,
                error=e,
            )
    
    async def _compensate(
        self,
        completed_steps: List[str],
        step_results: Dict[str, Any]
    ) -> List[str]:
        """Execute compensating actions in reverse order"""
        self.status = SagaStatus.COMPENSATING
        compensated = []
        
        logger.warning(f"[Saga {self.saga_id}] Starting compensation")
        
        # Find steps to compensate (in reverse order)
        steps_to_compensate = [
            step for step in reversed(self.steps)
            if step.name in completed_steps
        ]
        
        for step in steps_to_compensate:
            try:
                logger.info(f"[Saga {self.saga_id}] Compensating step: {step.name}")
                result = step_results.get(step.name)
                await asyncio.wait_for(
                    step.compensation(result),
                    timeout=step.timeout
                )
                compensated.append(step.name)
                logger.info(f"[Saga {self.saga_id}] Compensated step: {step.name}")
                
            except Exception as e:
                logger.error(
                    f"[Saga {self.saga_id}] Compensation failed for {step.name}: {e}"
                )
                # Continue compensating other steps even if one fails
        
        return compensated


# ============================================================================
# SERVICE DISCOVERY PATTERN
# ============================================================================

@dataclass
class ServiceInstance:
    """Represents a service instance"""
    service_name: str
    host: str
    port: int
    metadata: Dict[str, Any] = field(default_factory=dict)
    health_check_url: Optional[str] = None
    last_heartbeat: float = field(default_factory=time.time)
    
    @property
    def url(self) -> str:
        """Get full service URL"""
        return f"http://{self.host}:{self.port}"
    
    def is_healthy(self, timeout: float = 30.0) -> bool:
        """Check if instance is healthy based on heartbeat"""
        return time.time() - self.last_heartbeat < timeout


class ServiceRegistry:
    """
    Service Discovery pattern implementation.
    
    Maintains a registry of available service instances for dynamic
    service location. Supports health checking and load balancing.
    """
    
    def __init__(self, health_check_interval: float = 30.0):
        self.services: Dict[str, List[ServiceInstance]] = {}
        self.health_check_interval = health_check_interval
        self._lock = asyncio.Lock()
    
    async def register(self, instance: ServiceInstance):
        """Register a service instance"""
        async with self._lock:
            if instance.service_name not in self.services:
                self.services[instance.service_name] = []
            
            # Update existing or add new
            existing = [
                i for i in self.services[instance.service_name]
                if i.host == instance.host and i.port == instance.port
            ]
            
            if existing:
                existing[0].last_heartbeat = time.time()
                existing[0].metadata.update(instance.metadata)
            else:
                self.services[instance.service_name].append(instance)
            
            logger.info(
                f"Registered service: {instance.service_name} at {instance.url}"
            )
    
    async def deregister(self, service_name: str, host: str, port: int):
        """Deregister a service instance"""
        async with self._lock:
            if service_name in self.services:
                self.services[service_name] = [
                    i for i in self.services[service_name]
                    if not (i.host == host and i.port == port)
                ]
                logger.info(f"Deregistered service: {service_name} at {host}:{port}")
    
    async def discover(
        self,
        service_name: str,
        healthy_only: bool = True
    ) -> List[ServiceInstance]:
        """Discover instances of a service"""
        async with self._lock:
            instances = self.services.get(service_name, [])
            
            if healthy_only:
                instances = [i for i in instances if i.is_healthy(self.health_check_interval)]
            
            return instances
    
    async def get_instance(
        self,
        service_name: str,
        strategy: str = "round_robin"
    ) -> Optional[ServiceInstance]:
        """
        Get a service instance using load balancing strategy.
        
        Args:
            service_name: Name of the service
            strategy: Load balancing strategy (round_robin, random)
        
        Returns:
            ServiceInstance or None if no healthy instances
        """
        instances = await self.discover(service_name, healthy_only=True)
        
        if not instances:
            logger.warning(f"No healthy instances found for service: {service_name}")
            return None
        
        if strategy == "random":
            return random.choice(instances)
        elif strategy == "round_robin":
            # Simple round-robin (in production, use more sophisticated approach)
            return instances[0]
        else:
            return instances[0]
    
    async def health_check(self, client: httpx.AsyncClient):
        """Perform health checks on all registered services"""
        async with self._lock:
            for service_name, instances in self.services.items():
                for instance in instances[:]:  # Copy list to allow removal
                    if not instance.is_healthy(self.health_check_interval):
                        logger.warning(
                            f"Instance unhealthy: {service_name} at {instance.url}"
                        )
                        continue
                    
                    if instance.health_check_url:
                        try:
                            response = await client.get(
                                instance.health_check_url,
                                timeout=5.0
                            )
                            if response.status_code != 200:
                                logger.warning(
                                    f"Health check failed for {instance.url}: "
                                    f"status {response.status_code}"
                                )
                        except Exception as e:
                            logger.error(
                                f"Health check error for {instance.url}: {e}"
                            )
    
    async def start_health_check_loop(self, client: httpx.AsyncClient):
        """Start periodic health check loop"""
        while True:
            try:
                await self.health_check(client)
                await asyncio.sleep(self.health_check_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health check loop error: {e}")
    
    def get_all_services(self) -> Dict[str, int]:
        """Get summary of all registered services"""
        return {
            service_name: len(instances)
            for service_name, instances in self.services.items()
        }


# ============================================================================
# COMPREHENSIVE USAGE EXAMPLES
# ============================================================================

async def example_circuit_breaker():
    """Example: Circuit Breaker pattern"""
    print("\n" + "="*70)
    print("CIRCUIT BREAKER PATTERN EXAMPLE")
    print("="*70)
    
    config = CircuitBreakerConfig(
        failure_threshold=3,
        success_threshold=2,
        timeout=5.0
    )
    
    @circuit_breaker(config)
    async def unreliable_api_call(should_fail: bool = False):
        """Simulated API call that may fail"""
        if should_fail:
            raise httpx.RequestError("Service unavailable")
        return {"status": "success", "data": "Hello, World!"}
    
    # Test circuit breaker behavior
    print("\n1. Testing normal operation (CLOSED state):")
    try:
        result = await unreliable_api_call(should_fail=False)
        print(f"   ✓ Success: {result}")
        print(f"   Circuit state: {unreliable_api_call.circuit_breaker.get_state()}")
    except Exception as e:
        print(f"   ✗ Error: {e}")
    
    print("\n2. Triggering failures to open circuit:")
    for i in range(4):
        try:
            await unreliable_api_call(should_fail=True)
        except Exception as e:
            print(f"   Attempt {i+1}: {type(e).__name__}")
    
    print(f"   Circuit state: {unreliable_api_call.circuit_breaker.get_state()}")
    
    print("\n3. Attempting call with open circuit:")
    try:
        await unreliable_api_call(should_fail=False)
    except CircuitBreakerOpenError as e:
        print(f"   ✗ Blocked: {e}")
    
    print("\n4. Waiting for timeout and testing HALF_OPEN state:")
    await asyncio.sleep(5.5)
    try:
        result = await unreliable_api_call(should_fail=False)
        print(f"   ✓ Success in HALF_OPEN: {result}")
        result = await unreliable_api_call(should_fail=False)
        print(f"   ✓ Circuit closed again: {result}")
        print(f"   Final state: {unreliable_api_call.circuit_breaker.get_state()}")
    except Exception as e:
        print(f"   ✗ Error: {e}")


async def example_retry_backoff():
    """Example: Retry with exponential backoff + jitter"""
    print("\n" + "="*70)
    print("RETRY WITH EXPONENTIAL BACKOFF + JITTER EXAMPLE")
    print("="*70)
    
    attempt_count = {"count": 0}
    
    @with_retry(RetryConfig(max_attempts=4, base_delay=0.5, jitter=True))
    async def flaky_service(fail_times: int = 2):
        """Simulated flaky service that fails then succeeds"""
        attempt_count["count"] += 1
        print(f"   Attempt {attempt_count['count']}")
        
        if attempt_count["count"] <= fail_times:
            raise httpx.RequestError("Temporary network error")
        
        return {"status": "success", "attempt": attempt_count["count"]}
    
    print("\n1. Service fails twice then succeeds:")
    try:
        result = await flaky_service(fail_times=2)
        print(f"   ✓ Final result: {result}")
    except Exception as e:
        print(f"   ✗ Failed: {e}")
    
    print("\n2. Service fails all attempts:")
    attempt_count["count"] = 0
    try:
        result = await flaky_service(fail_times=10)
    except RetryExhaustedError as e:
        print(f"   ✗ All retries exhausted: {e}")


async def example_fan_out_fan_in():
    """Example: Fan-out/Fan-in parallel execution"""
    print("\n" + "="*70)
    print("FAN-OUT / FAN-IN PATTERN EXAMPLE")
    print("="*70)
    
    async def fetch_user_data(user_id: int):
        """Simulate fetching user data"""
        await asyncio.sleep(random.uniform(0.1, 0.5))
        if user_id == 3:
            raise ValueError(f"User {user_id} not found")
        return {"user_id": user_id, "name": f"User{user_id}", "active": True}
    
    print("\n1. Parallel API calls with partial failures:")
    tasks = [lambda uid=i: fetch_user_data(uid) for i in range(1, 6)]
    
    result = await fan_out(tasks, max_concurrency=3, fail_fast=False)
    
    print(f"   Success rate: {result.success_rate:.1%}")
    print(f"   Successful results: {len(result.successful)}")
    print(f"   Failed results: {len(result.failed)}")
    
    for idx, data in result.successful:
        print(f"   ✓ Task {idx}: {data}")
    
    for idx, error in result.failed:
        print(f"   ✗ Task {idx}: {error}")
    
    print("\n2. Aggregated results:")
    successful_data = result.get_results()
    print(f"   Total users fetched: {len(successful_data)}")


async def example_saga_pattern():
    """Example: Saga pattern for distributed transactions"""
    print("\n" + "="*70)
    print("SAGA PATTERN (ORCHESTRATION) EXAMPLE")
    print("="*70)
    
    # Simulate distributed transaction state
    state = {
        "order": None,
        "payment": None,
        "inventory": None,
    }
    
    async def create_order():
        """Step 1: Create order"""
        await asyncio.sleep(0.1)
        state["order"] = {"order_id": "ORD-123", "status": "created"}
        print("   ✓ Order created:", state["order"])
        return state["order"]
    
    async def compensate_order(order_data):
        """Rollback: Cancel order"""
        await asyncio.sleep(0.1)
        state["order"]["status"] = "cancelled"
        print("   ↶ Order cancelled:", state["order"])
    
    async def process_payment():
        """Step 2: Process payment"""
        await asyncio.sleep(0.1)
        state["payment"] = {"payment_id": "PAY-456", "status": "completed"}
        print("   ✓ Payment processed:", state["payment"])
        return state["payment"]
    
    async def compensate_payment(payment_data):
        """Rollback: Refund payment"""
        await asyncio.sleep(0.1)
        state["payment"]["status"] = "refunded"
        print("   ↶ Payment refunded:", state["payment"])
    
    async def reserve_inventory(should_fail: bool = False):
        """Step 3: Reserve inventory"""
        await asyncio.sleep(0.1)
        if should_fail:
            raise Exception("Inventory not available")
        state["inventory"] = {"reservation_id": "INV-789", "status": "reserved"}
        print("   ✓ Inventory reserved:", state["inventory"])
        return state["inventory"]
    
    async def compensate_inventory(inventory_data):
        """Rollback: Release inventory"""
        await asyncio.sleep(0.1)
        if inventory_data:
            state["inventory"]["status"] = "released"
            print("   ↶ Inventory released:", state["inventory"])
    
    print("\n1. Successful saga execution:")
    saga1 = SagaOrchestrator("order-saga-001")
    saga1.add_step("create_order", create_order, compensate_order)
    saga1.add_step("process_payment", process_payment, compensate_payment)
    saga1.add_step("reserve_inventory", 
                   lambda: reserve_inventory(False), 
                   compensate_inventory)
    
    result1 = await saga1.execute()
    print(f"\n   Result: {result1.status.value}")
    print(f"   Completed steps: {result1.completed_steps}")
    
    print("\n2. Saga with failure and compensation:")
    state = {"order": None, "payment": None, "inventory": None}  # Reset
    
    saga2 = SagaOrchestrator("order-saga-002")
    saga2.add_step("create_order", create_order, compensate_order)
    saga2.add_step("process_payment", process_payment, compensate_payment)
    saga2.add_step("reserve_inventory", 
                   lambda: reserve_inventory(True),  # This will fail
                   compensate_inventory)
    
    result2 = await saga2.execute()
    print(f"\n   Result: {result2.status.value}")
    print(f"   Completed steps: {result2.completed_steps}")
    print(f"   Failed step: {result2.failed_step}")
    print(f"   Compensated steps: {result2.compensated_steps}")


async def example_service_discovery():
    """Example: Service discovery pattern"""
    print("\n" + "="*70)
    print("SERVICE DISCOVERY PATTERN EXAMPLE")
    print("="*70)
    
    registry = ServiceRegistry(health_check_interval=30.0)
    
    print("\n1. Registering service instances:")
    instances = [
        ServiceInstance("user-service", "10.0.1.1", 8001, 
                       {"region": "us-east", "version": "1.0"}),
        ServiceInstance("user-service", "10.0.1.2", 8001, 
                       {"region": "us-west", "version": "1.0"}),
        ServiceInstance("order-service", "10.0.2.1", 8002,
                       {"region": "us-east", "version": "2.0"}),
    ]
    
    for instance in instances:
        await registry.register(instance)
        print(f"   ✓ Registered: {instance.service_name} at {instance.url}")
    
    print("\n2. Discovering services:")
    summary = registry.get_all_services()
    for service_name, count in summary.items():
        print(f"   {service_name}: {count} instances")
    
    print("\n3. Finding specific service instances:")
    user_instances = await registry.discover("user-service")
    print(f"   Found {len(user_instances)} user-service instances:")
    for inst in user_instances:
        print(f"   - {inst.url} (region: {inst.metadata.get('region')})")
    
    print("\n4. Load balancing - getting instance:")
    instance = await registry.get_instance("user-service", strategy="random")
    if instance:
        print(f"   Selected instance: {instance.url}")
    
    print("\n5. Deregistering an instance:")
    await registry.deregister("user-service", "10.0.1.1", 8001)
    remaining = await registry.discover("user-service")
    print(f"   Remaining user-service instances: {len(remaining)}")


async def example_combined_patterns():
    """Example: Combining multiple patterns"""
    print("\n" + "="*70)
    print("COMBINED PATTERNS EXAMPLE")
    print("="*70)
    print("Demonstrating Circuit Breaker + Retry + Service Discovery\n")
    
    # Setup service registry
    registry = ServiceRegistry()
    await registry.register(
        ServiceInstance("api-service", "api.example.com", 443)
    )
    
    # Circuit breaker configuration
    cb_config = CircuitBreakerConfig(failure_threshold=2, timeout=3.0)
    
    # Retry configuration
    retry_config = RetryConfig(max_attempts=3, base_delay=0.5)
    
    @circuit_breaker(cb_config)
    @with_retry(retry_config)
    async def resilient_api_call(endpoint: str, fail: bool = False):
        """API call with circuit breaker and retry"""
        instance = await registry.get_instance("api-service")
        if not instance:
            raise Exception("No service instance available")
        
        print(f"   Calling {instance.url}{endpoint}")
        
        if fail:
            raise httpx.RequestError("Network error")
        
        return {"endpoint": endpoint, "status": "success"}
    
    print("1. Successful call with all patterns:")
    try:
        result = await resilient_api_call("/users", fail=False)
        print(f"   ✓ Result: {result}\n")
    except Exception as e:
        print(f"   ✗ Error: {e}\n")
    
    print("2. Failed calls demonstrate retry and circuit breaker:")
    for i in range(3):
        try:
            await resilient_api_call("/orders", fail=True)
        except Exception as e:
            print(f"   Attempt {i+1}: {type(e).__name__}")


async def main():
    """Run all examples"""
    print("\n" + "="*70)
    print("DISTRIBUTED SYSTEM SERVICE PATTERNS - COMPREHENSIVE EXAMPLES")
    print("="*70)
    print("\nApply to: Microservices, service mesh, resilient APIs")
    print("\nThese patterns help build production-ready distributed systems")
    print("with fault tolerance, resilience, and scalability.\n")
    
    try:
        await example_circuit_breaker()
        await example_retry_backoff()
        await example_fan_out_fan_in()
        await example_saga_pattern()
        await example_service_discovery()
        await example_combined_patterns()
        
        print("\n" + "="*70)
        print("ALL EXAMPLES COMPLETED SUCCESSFULLY")
        print("="*70)
        print("\nKey Takeaways:")
        print("- Circuit Breaker: Prevents cascade failures, fast-fails when service down")
        print("- Retry + Backoff: Handles transient failures, jitter prevents thundering herd")
        print("- Fan-out/Fan-in: Parallel execution with aggregation, improves throughput")
        print("- Saga Pattern: Distributed transactions with compensation for consistency")
        print("- Service Discovery: Dynamic service location with health checking")
        print("\nCombine these patterns for production-ready microservices!")
        
    except Exception as e:
        logger.error(f"Example execution failed: {e}", exc_info=True)


if __name__ == "__main__":
    asyncio.run(main())
