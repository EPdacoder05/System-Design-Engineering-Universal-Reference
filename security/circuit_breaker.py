"""
Circuit Breaker Pattern for Resilience

Apply to: Microservices, external API calls, database connections, distributed systems

Features from NullPointVector:
- Prevents cascading failures
- Automatic failure detection and recovery
- Configurable thresholds and timeouts
- Half-open state for gradual recovery
- Metrics tracking and monitoring integration
- Thread-safe implementation
"""

import time
import threading
from enum import Enum
from typing import Callable, Any, Optional, Dict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import deque


# ============================================================================
# Circuit Breaker States
# ============================================================================

class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"        # Normal operation, requests pass through
    OPEN = "open"            # Failure threshold exceeded, requests blocked
    HALF_OPEN = "half_open"  # Testing if service has recovered


# ============================================================================
# Circuit Breaker Configuration
# ============================================================================

@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker behavior."""
    # Failure threshold before opening circuit
    failure_threshold: int = 5
    
    # Time window for counting failures (seconds)
    failure_window: int = 60
    
    # Timeout before attempting recovery (seconds)
    recovery_timeout: int = 30
    
    # Success threshold to close circuit from half-open (seconds)
    success_threshold: int = 2
    
    # Request timeout (seconds)
    request_timeout: float = 10.0
    
    # Maximum number of failures to track
    max_tracked_failures: int = 100
    
    # Call timeout (seconds)
    call_timeout: Optional[float] = None


# ============================================================================
# Circuit Breaker Exception
# ============================================================================

class CircuitBreakerOpenException(Exception):
    """Raised when circuit breaker is open and requests are blocked."""
    pass


# ============================================================================
# Circuit Breaker Implementation
# ============================================================================

class CircuitBreaker:
    """
    Circuit Breaker pattern implementation.
    
    Apply to: External API calls, database connections, microservice communication
    
    States:
    - CLOSED: Normal operation, all requests pass through
    - OPEN: Too many failures, all requests blocked immediately
    - HALF_OPEN: Testing recovery, limited requests allowed
    
    Example:
        >>> circuit_breaker = CircuitBreaker(name="payment_api")
        >>> result = circuit_breaker.call(make_payment_api_call, amount=100)
    """
    
    def __init__(
        self, 
        name: str,
        config: Optional[CircuitBreakerConfig] = None
    ):
        """
        Initialize circuit breaker.
        
        Args:
            name: Identifier for this circuit breaker
            config: Configuration (uses defaults if not provided)
        """
        self.name = name
        self.config = config or CircuitBreakerConfig()
        
        # State management
        self._state = CircuitState.CLOSED
        self._lock = threading.RLock()
        
        # Failure tracking
        self._failures = deque(maxlen=self.config.max_tracked_failures)
        self._last_failure_time: Optional[float] = None
        self._opened_at: Optional[float] = None
        
        # Success tracking in half-open state
        self._consecutive_successes = 0
        
        # Metrics
        self._total_calls = 0
        self._total_successes = 0
        self._total_failures = 0
        self._total_rejected = 0
    
    @property
    def state(self) -> CircuitState:
        """Get current circuit breaker state."""
        with self._lock:
            return self._state
    
    @property
    def metrics(self) -> Dict[str, Any]:
        """Get circuit breaker metrics."""
        with self._lock:
            return {
                'name': self.name,
                'state': self._state.value,
                'total_calls': self._total_calls,
                'total_successes': self._total_successes,
                'total_failures': self._total_failures,
                'total_rejected': self._total_rejected,
                'success_rate': (
                    self._total_successes / self._total_calls 
                    if self._total_calls > 0 else 0
                ),
                'failure_rate': (
                    self._total_failures / self._total_calls 
                    if self._total_calls > 0 else 0
                ),
                'recent_failures': len(self._failures),
                'consecutive_successes': self._consecutive_successes,
            }
    
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute function through circuit breaker.
        
        Args:
            func: Function to execute
            *args: Positional arguments for function
            **kwargs: Keyword arguments for function
            
        Returns:
            Result of function execution
            
        Raises:
            CircuitBreakerOpenException: If circuit is open
            Exception: Any exception raised by the function
        """
        with self._lock:
            self._total_calls += 1
            
            # Check if circuit should transition states
            self._check_state_transition()
            
            # If circuit is open, reject immediately
            if self._state == CircuitState.OPEN:
                self._total_rejected += 1
                raise CircuitBreakerOpenException(
                    f"Circuit breaker '{self.name}' is OPEN. "
                    f"Service unavailable due to repeated failures."
                )
        
        # Execute function
        try:
            start_time = time.time()
            
            # Apply timeout if configured
            if self.config.call_timeout:
                # Note: This is a simplified timeout implementation
                # In production, use asyncio or threading.Timer
                result = func(*args, **kwargs)
                elapsed = time.time() - start_time
                
                if elapsed > self.config.call_timeout:
                    raise TimeoutError(
                        f"Call exceeded timeout of {self.config.call_timeout}s"
                    )
            else:
                result = func(*args, **kwargs)
            
            # Success: record and potentially close circuit
            self._record_success()
            return result
            
        except Exception as e:
            # Failure: record and potentially open circuit
            self._record_failure(e)
            raise
    
    def _check_state_transition(self) -> None:
        """Check if circuit breaker should transition to a different state."""
        current_time = time.time()
        
        if self._state == CircuitState.OPEN:
            # Check if recovery timeout has passed
            if (self._opened_at and 
                current_time - self._opened_at >= self.config.recovery_timeout):
                print(f"🔄 Circuit breaker '{self.name}': OPEN -> HALF_OPEN")
                self._state = CircuitState.HALF_OPEN
                self._consecutive_successes = 0
        
        elif self._state == CircuitState.CLOSED:
            # Check if we should open due to failures
            recent_failures = self._count_recent_failures(current_time)
            if recent_failures >= self.config.failure_threshold:
                print(f"🚨 Circuit breaker '{self.name}': CLOSED -> OPEN "
                      f"({recent_failures} failures in window)")
                self._state = CircuitState.OPEN
                self._opened_at = current_time
    
    def _count_recent_failures(self, current_time: float) -> int:
        """Count failures within the failure window."""
        window_start = current_time - self.config.failure_window
        return sum(1 for timestamp in self._failures if timestamp >= window_start)
    
    def _record_success(self) -> None:
        """Record successful call and update state."""
        with self._lock:
            self._total_successes += 1
            
            if self._state == CircuitState.HALF_OPEN:
                self._consecutive_successes += 1
                
                # If enough successes, close the circuit
                if self._consecutive_successes >= self.config.success_threshold:
                    print(f"✅ Circuit breaker '{self.name}': HALF_OPEN -> CLOSED")
                    self._state = CircuitState.CLOSED
                    self._consecutive_successes = 0
                    self._failures.clear()
    
    def _record_failure(self, exception: Exception) -> None:
        """Record failed call and update state."""
        with self._lock:
            self._total_failures += 1
            current_time = time.time()
            self._failures.append(current_time)
            self._last_failure_time = current_time
            
            # If in half-open, immediately open circuit on failure
            if self._state == CircuitState.HALF_OPEN:
                print(f"❌ Circuit breaker '{self.name}': HALF_OPEN -> OPEN "
                      f"(failure during recovery)")
                self._state = CircuitState.OPEN
                self._opened_at = current_time
                self._consecutive_successes = 0
    
    def force_open(self) -> None:
        """Manually open the circuit breaker."""
        with self._lock:
            print(f"🔴 Circuit breaker '{self.name}': Manually opened")
            self._state = CircuitState.OPEN
            self._opened_at = time.time()
    
    def force_close(self) -> None:
        """Manually close the circuit breaker."""
        with self._lock:
            print(f"🟢 Circuit breaker '{self.name}': Manually closed")
            self._state = CircuitState.CLOSED
            self._consecutive_successes = 0
            self._failures.clear()
    
    def reset(self) -> None:
        """Reset circuit breaker to initial state."""
        with self._lock:
            print(f"🔄 Circuit breaker '{self.name}': Reset")
            self._state = CircuitState.CLOSED
            self._failures.clear()
            self._last_failure_time = None
            self._opened_at = None
            self._consecutive_successes = 0
            self._total_calls = 0
            self._total_successes = 0
            self._total_failures = 0
            self._total_rejected = 0


# ============================================================================
# Circuit Breaker Decorator
# ============================================================================

def circuit_breaker(
    name: str,
    config: Optional[CircuitBreakerConfig] = None
) -> Callable:
    """
    Decorator to apply circuit breaker pattern to a function.
    
    Example:
        >>> @circuit_breaker(name="api_call", config=CircuitBreakerConfig(failure_threshold=3))
        >>> def call_external_api():
        >>>     return requests.get("https://api.example.com/data")
    """
    cb = CircuitBreaker(name=name, config=config)
    
    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs) -> Any:
            return cb.call(func, *args, **kwargs)
        return wrapper
    return decorator


# ============================================================================
# Circuit Breaker Registry for Multiple Services
# ============================================================================

class CircuitBreakerRegistry:
    """
    Registry to manage multiple circuit breakers.
    
    Apply to: Systems with multiple external dependencies
    
    Features:
    - Centralized management of multiple circuit breakers
    - Bulk operations (get all metrics, reset all)
    - Thread-safe
    """
    
    def __init__(self):
        self._breakers: Dict[str, CircuitBreaker] = {}
        self._lock = threading.RLock()
    
    def get_or_create(
        self, 
        name: str, 
        config: Optional[CircuitBreakerConfig] = None
    ) -> CircuitBreaker:
        """
        Get existing circuit breaker or create new one.
        
        Args:
            name: Circuit breaker identifier
            config: Configuration (only used if creating new)
            
        Returns:
            CircuitBreaker instance
        """
        with self._lock:
            if name not in self._breakers:
                self._breakers[name] = CircuitBreaker(name=name, config=config)
            return self._breakers[name]
    
    def get(self, name: str) -> Optional[CircuitBreaker]:
        """Get circuit breaker by name."""
        with self._lock:
            return self._breakers.get(name)
    
    def get_all_metrics(self) -> Dict[str, Dict[str, Any]]:
        """Get metrics for all circuit breakers."""
        with self._lock:
            return {
                name: breaker.metrics
                for name, breaker in self._breakers.items()
            }
    
    def reset_all(self) -> None:
        """Reset all circuit breakers."""
        with self._lock:
            for breaker in self._breakers.values():
                breaker.reset()
    
    def get_unhealthy(self) -> Dict[str, CircuitBreaker]:
        """Get all circuit breakers in OPEN or HALF_OPEN state."""
        with self._lock:
            return {
                name: breaker
                for name, breaker in self._breakers.items()
                if breaker.state in (CircuitState.OPEN, CircuitState.HALF_OPEN)
            }


# Global registry instance
_global_registry = CircuitBreakerRegistry()


def get_circuit_breaker(
    name: str, 
    config: Optional[CircuitBreakerConfig] = None
) -> CircuitBreaker:
    """
    Get circuit breaker from global registry.
    
    Args:
        name: Circuit breaker identifier
        config: Configuration (only used if creating new)
        
    Returns:
        CircuitBreaker instance
    """
    return _global_registry.get_or_create(name, config)


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    print("=== Circuit Breaker Pattern Demo ===\n")
    
    # Simulated external service
    call_count = 0
    
    def unreliable_service():
        """Simulated service that fails intermittently."""
        global call_count
        call_count += 1
        
        # Fail for first 5 calls, then succeed
        if call_count <= 5:
            print(f"   ❌ Service call {call_count} failed")
            raise Exception("Service unavailable")
        else:
            print(f"   ✅ Service call {call_count} succeeded")
            return {"status": "success", "data": "sample_data"}
    
    # Create circuit breaker with low thresholds for demo
    config = CircuitBreakerConfig(
        failure_threshold=3,
        failure_window=10,
        recovery_timeout=5,
        success_threshold=2
    )
    
    circuit = CircuitBreaker(name="demo_service", config=config)
    
    # Simulate calls
    print("1. Testing Circuit Breaker Behavior")
    print("-" * 50)
    
    for i in range(12):
        try:
            time.sleep(1)  # Pause between calls
            result = circuit.call(unreliable_service)
            print(f"   Call {i+1}: Success - {result}\n")
        except CircuitBreakerOpenException as e:
            print(f"   Call {i+1}: Blocked - {e}\n")
        except Exception as e:
            print(f"   Call {i+1}: Failed - {e}\n")
    
    # Show metrics
    print("\n2. Circuit Breaker Metrics")
    print("-" * 50)
    metrics = circuit.metrics
    for key, value in metrics.items():
        print(f"   {key}: {value}")
    
    # Using decorator
    print("\n\n3. Using Circuit Breaker Decorator")
    print("-" * 50)
    
    @circuit_breaker(name="decorated_service", config=config)
    def my_api_call():
        return {"result": "success"}
    
    try:
        result = my_api_call()
        print(f"   Decorator result: {result}")
    except Exception as e:
        print(f"   Decorator failed: {e}")
    
    # Registry example
    print("\n\n4. Circuit Breaker Registry")
    print("-" * 50)
    
    # Create multiple circuit breakers
    payment_cb = get_circuit_breaker("payment_api")
    user_cb = get_circuit_breaker("user_api")
    inventory_cb = get_circuit_breaker("inventory_api")
    
    # Get all metrics
    all_metrics = _global_registry.get_all_metrics()
    print(f"   Total circuit breakers: {len(all_metrics)}")
    for name, metrics in all_metrics.items():
        print(f"   - {name}: state={metrics['state']}")
    
    print("\n=== Demo Complete ===")
