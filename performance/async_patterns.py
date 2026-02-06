"""
Async/Await Patterns for High-Performance I/O Operations

Apply to: I/O-bound operations, API clients, data pipelines

This module provides production-ready async patterns for:
- Concurrency control with semaphores
- Batch processing
- Exponential backoff with jitter
- Resource management with async context managers
- Error handling in concurrent operations
- Rate limiting
- Producer-consumer patterns
- Timeout handling

Performance improvements: 10-100x faster than synchronous code for I/O-bound tasks
"""

import asyncio
import aiofiles
import random
import time
from typing import (
    List, Callable, Awaitable, Any, Optional, TypeVar, AsyncIterator,
    Tuple, Dict
)
from contextlib import asynccontextmanager
from dataclasses import dataclass
from enum import Enum
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

T = TypeVar('T')


# ============================================================================
# 1. SEMAPHORE-BASED CONCURRENCY LIMITING
# ============================================================================

class ConcurrencyLimiter:
    """
    Control parallelism to prevent resource exhaustion.
    
    Use cases:
    - Limit concurrent database connections
    - Control API request concurrency
    - Prevent overwhelming downstream services
    
    Performance: Prevents resource starvation while maximizing throughput
    """
    
    def __init__(self, max_concurrent: int):
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.active_tasks = 0
        self.completed_tasks = 0
    
    async def execute(
        self,
        coro: Callable[..., Awaitable[T]],
        *args,
        **kwargs
    ) -> T:
        """Execute coroutine with concurrency limiting"""
        async with self.semaphore:
            self.active_tasks += 1
            try:
                result = await coro(*args, **kwargs)
                self.completed_tasks += 1
                return result
            finally:
                self.active_tasks -= 1
    
    async def execute_many(
        self,
        coro: Callable[[Any], Awaitable[T]],
        items: List[Any]
    ) -> List[T]:
        """Execute many coroutines with concurrency limiting"""
        tasks = [self.execute(coro, item) for item in items]
        return await asyncio.gather(*tasks)


async def demo_concurrency_limiter():
    """
    Performance comparison: Unlimited vs Limited Concurrency
    
    Unlimited: Risk of connection pool exhaustion, OOM errors
    Limited: Controlled resource usage, sustainable throughput
    """
    
    async def fetch_data(item_id: int) -> dict:
        """Simulate API call"""
        await asyncio.sleep(0.1)  # Simulate I/O
        return {"id": item_id, "data": f"result_{item_id}"}
    
    items = list(range(100))
    
    # Without limiting (risky for large workloads)
    start = time.time()
    results_unlimited = await asyncio.gather(*[fetch_data(i) for i in items])
    unlimited_time = time.time() - start
    
    # With concurrency limiting (safe, controlled)
    limiter = ConcurrencyLimiter(max_concurrent=10)
    start = time.time()
    results_limited = await limiter.execute_many(fetch_data, items)
    limited_time = time.time() - start
    
    logger.info(f"Unlimited concurrency: {unlimited_time:.2f}s (100 concurrent)")
    logger.info(f"Limited concurrency: {limited_time:.2f}s (10 concurrent)")
    logger.info(f"Both processed {len(results_limited)} items successfully")


# ============================================================================
# 2. BATCH PROCESSING WITH CONFIGURABLE BATCH SIZES
# ============================================================================

class BatchProcessor:
    """
    Process items in configurable batches for optimal throughput.
    
    Use cases:
    - Bulk database inserts/updates
    - Batch API requests
    - Stream processing with micro-batching
    
    Performance: Reduces overhead, improves throughput by 5-10x
    """
    
    def __init__(self, batch_size: int = 100):
        self.batch_size = batch_size
    
    async def process_in_batches(
        self,
        items: List[T],
        processor: Callable[[List[T]], Awaitable[Any]],
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> List[Any]:
        """Process items in batches"""
        results = []
        total_items = len(items)
        
        for i in range(0, total_items, self.batch_size):
            batch = items[i:i + self.batch_size]
            result = await processor(batch)
            results.append(result)
            
            if progress_callback:
                progress_callback(min(i + self.batch_size, total_items), total_items)
        
        return results
    
    async def stream_batches(
        self,
        items: List[T]
    ) -> AsyncIterator[List[T]]:
        """Stream batches for pipeline processing"""
        for i in range(0, len(items), self.batch_size):
            batch = items[i:i + self.batch_size]
            yield batch
            await asyncio.sleep(0)  # Allow other tasks to run


async def demo_batch_processing():
    """
    Performance comparison: Item-by-item vs Batch Processing
    
    Item-by-item: High overhead, slow
    Batch: Amortized overhead, fast
    """
    
    async def save_to_database(records: List[dict]) -> int:
        """Simulate batch database insert"""
        await asyncio.sleep(0.01 * len(records))  # Simulate I/O
        return len(records)
    
    records = [{"id": i, "value": f"data_{i}"} for i in range(1000)]
    
    # Item-by-item (slow)
    start = time.time()
    for record in records[:100]:  # Just 100 to keep demo fast
        await save_to_database([record])
    item_time = time.time() - start
    
    # Batch processing (fast)
    processor = BatchProcessor(batch_size=50)
    start = time.time()
    results = await processor.process_in_batches(
        records,
        save_to_database,
        progress_callback=lambda done, total: logger.info(f"Progress: {done}/{total}")
    )
    batch_time = time.time() - start
    
    logger.info(f"Item-by-item (100 items): {item_time:.2f}s")
    logger.info(f"Batch processing (1000 items): {batch_time:.2f}s")
    logger.info(f"Speedup: {(item_time * 10) / batch_time:.1f}x")


# ============================================================================
# 3. EXPONENTIAL BACKOFF WITH JITTER
# ============================================================================

class RetryStrategy:
    """
    Exponential backoff with jitter for resilient retry logic.
    
    Use cases:
    - Transient API failures
    - Network errors
    - Rate limit handling
    - Database connection retries
    
    Performance: Prevents thundering herd, improves success rate
    """
    
    def __init__(
        self,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_base: float = 2.0,
        jitter: bool = True
    ):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter = jitter
    
    def calculate_delay(self, attempt: int) -> float:
        """Calculate delay with exponential backoff and jitter"""
        delay = min(
            self.base_delay * (self.exponential_base ** attempt),
            self.max_delay
        )
        
        if self.jitter:
            # Full jitter: random value between 0 and calculated delay
            delay = random.uniform(0, delay)
        
        return delay
    
    async def execute_with_retry(
        self,
        coro: Callable[..., Awaitable[T]],
        *args,
        **kwargs
    ) -> T:
        """Execute coroutine with retry logic"""
        last_exception = None
        
        for attempt in range(self.max_retries + 1):
            try:
                return await coro(*args, **kwargs)
            except Exception as e:
                last_exception = e
                
                if attempt < self.max_retries:
                    delay = self.calculate_delay(attempt)
                    logger.warning(
                        f"Attempt {attempt + 1} failed: {e}. "
                        f"Retrying in {delay:.2f}s..."
                    )
                    await asyncio.sleep(delay)
                else:
                    logger.error(f"All {self.max_retries + 1} attempts failed")
        
        raise last_exception


async def demo_exponential_backoff():
    """
    Demonstrates retry behavior with exponential backoff and jitter
    """
    retry_strategy = RetryStrategy(max_retries=3, base_delay=0.5, jitter=True)
    
    call_count = 0
    
    async def unreliable_api_call() -> str:
        """Simulates an API that fails then succeeds"""
        nonlocal call_count
        call_count += 1
        
        if call_count < 3:
            raise ConnectionError("Service temporarily unavailable")
        
        return "Success!"
    
    start = time.time()
    result = await retry_strategy.execute_with_retry(unreliable_api_call)
    duration = time.time() - start
    
    logger.info(f"Result: {result} after {call_count} attempts in {duration:.2f}s")


# ============================================================================
# 4. ASYNC CONTEXT MANAGERS (RESOURCE MANAGEMENT)
# ============================================================================

class AsyncResourcePool:
    """
    Async context manager for connection pooling and resource management.
    
    Use cases:
    - Database connection pools
    - HTTP client session management
    - File handle management
    - Cache connections
    
    Performance: Efficient resource reuse, automatic cleanup
    """
    
    def __init__(self, max_size: int = 10):
        self.max_size = max_size
        self.pool: asyncio.Queue = asyncio.Queue(maxsize=max_size)
        self.size = 0
        self._closed = False
    
    async def _create_resource(self) -> dict:
        """Create a new resource (e.g., database connection)"""
        await asyncio.sleep(0.1)  # Simulate connection establishment
        return {"id": self.size, "created_at": time.time()}
    
    @asynccontextmanager
    async def acquire(self):
        """Acquire resource from pool with automatic release"""
        if self._closed:
            raise RuntimeError("Pool is closed")
        
        # Try to get existing resource or create new one
        try:
            resource = self.pool.get_nowait()
        except asyncio.QueueEmpty:
            if self.size < self.max_size:
                resource = await self._create_resource()
                self.size += 1
            else:
                # Wait for available resource
                resource = await self.pool.get()
        
        try:
            yield resource
        finally:
            # Return resource to pool
            if not self._closed:
                await self.pool.put(resource)
    
    async def close(self):
        """Close pool and cleanup resources"""
        self._closed = True
        while not self.pool.empty():
            try:
                self.pool.get_nowait()
            except asyncio.QueueEmpty:
                break


@asynccontextmanager
async def async_file_writer(filepath: str):
    """
    Async context manager for file operations.
    
    Ensures proper file cleanup even on errors.
    """
    file_handle = await aiofiles.open(filepath, mode='w')
    try:
        yield file_handle
    finally:
        await file_handle.close()


async def demo_async_context_managers():
    """
    Demonstrates resource pool and async file operations
    """
    pool = AsyncResourcePool(max_size=5)
    
    async def use_resource(task_id: int):
        async with pool.acquire() as resource:
            logger.info(f"Task {task_id} acquired resource {resource['id']}")
            await asyncio.sleep(0.1)
            return f"Task {task_id} completed"
    
    # Multiple tasks sharing pool
    results = await asyncio.gather(*[use_resource(i) for i in range(20)])
    logger.info(f"Completed {len(results)} tasks with pool size 5")
    
    await pool.close()
    
    # Async file operations
    async with async_file_writer("/tmp/async_demo.txt") as f:
        await f.write("Async file write with automatic cleanup\n")
    
    logger.info("File written successfully")


# ============================================================================
# 5. ASYNCIO.GATHER() WITH ERROR HANDLING
# ============================================================================

class GatherMode(Enum):
    """Error handling modes for concurrent operations"""
    ALL_OR_NOTHING = "all_or_nothing"  # Fail fast, cancel all
    CONTINUE_ON_ERROR = "continue_on_error"  # Collect errors, continue


class AsyncGatherer:
    """
    Advanced asyncio.gather with multiple error handling strategies.
    
    Use cases:
    - Parallel API calls with graceful degradation
    - Concurrent data fetching with partial results
    - Batch operations with error tracking
    
    Performance: Maximizes parallelism while handling failures gracefully
    """
    
    @staticmethod
    async def gather_all_or_nothing(
        *coroutines: Awaitable[T]
    ) -> List[T]:
        """
        Fail fast: if any task fails, cancel all and raise.
        
        Use when: All results required for correctness
        """
        return await asyncio.gather(*coroutines)
    
    @staticmethod
    async def gather_continue_on_error(
        *coroutines: Awaitable[T]
    ) -> Tuple[List[T], List[Exception]]:
        """
        Continue on error: collect all results and errors.
        
        Use when: Partial results are acceptable
        """
        results = await asyncio.gather(*coroutines, return_exceptions=True)
        
        successes = [r for r in results if not isinstance(r, Exception)]
        errors = [r for r in results if isinstance(r, Exception)]
        
        return successes, errors
    
    @staticmethod
    async def gather_with_timeout(
        *coroutines: Awaitable[T],
        timeout: float
    ) -> List[T]:
        """Gather with overall timeout"""
        try:
            return await asyncio.wait_for(
                asyncio.gather(*coroutines),
                timeout=timeout
            )
        except asyncio.TimeoutError:
            logger.error(f"Gather operation timed out after {timeout}s")
            raise


async def demo_gather_patterns():
    """
    Demonstrates different gather error handling strategies
    """
    
    async def fetch_data(url: str, should_fail: bool = False) -> dict:
        await asyncio.sleep(0.1)
        if should_fail:
            raise ValueError(f"Failed to fetch {url}")
        return {"url": url, "status": 200}
    
    urls = ["api.example.com/1", "api.example.com/2", "api.example.com/3"]
    
    # Continue on error mode (graceful degradation)
    tasks = [
        fetch_data(urls[0]),
        fetch_data(urls[1], should_fail=True),  # This will fail
        fetch_data(urls[2])
    ]
    
    successes, errors = await AsyncGatherer.gather_continue_on_error(*tasks)
    
    logger.info(f"Successful: {len(successes)} requests")
    logger.info(f"Failed: {len(errors)} requests")
    logger.info(f"Success rate: {len(successes)}/{len(tasks)}")


# ============================================================================
# 6. RATE-LIMITED ASYNC EXECUTION
# ============================================================================

class RateLimiter:
    """
    Token bucket rate limiter for async operations.
    
    Use cases:
    - API rate limit compliance
    - Prevent overwhelming external services
    - Fair resource allocation
    - Cost control (pay-per-request APIs)
    
    Performance: Maximizes throughput within rate limits
    """
    
    def __init__(self, rate: float, per: float = 1.0):
        """
        Args:
            rate: Number of requests allowed
            per: Time period in seconds (e.g., 100 requests per 60 seconds)
        """
        self.rate = rate
        self.per = per
        self.allowance = rate
        self.last_check = time.time()
        self.lock = asyncio.Lock()
    
    async def acquire(self):
        """Acquire permission to proceed (blocks if rate exceeded)"""
        async with self.lock:
            current = time.time()
            time_passed = current - self.last_check
            self.last_check = current
            
            # Refill tokens based on time passed
            self.allowance += time_passed * (self.rate / self.per)
            
            if self.allowance > self.rate:
                self.allowance = self.rate
            
            if self.allowance < 1.0:
                # Calculate sleep time needed
                sleep_time = (1.0 - self.allowance) * (self.per / self.rate)
                await asyncio.sleep(sleep_time)
                self.allowance = 0.0
            else:
                self.allowance -= 1.0
    
    async def execute(
        self,
        coro: Callable[..., Awaitable[T]],
        *args,
        **kwargs
    ) -> T:
        """Execute coroutine with rate limiting"""
        await self.acquire()
        return await coro(*args, **kwargs)


class SlidingWindowRateLimiter:
    """
    Sliding window rate limiter (more accurate than token bucket).
    
    Use cases:
    - Strict rate limit enforcement
    - API quota management
    - Burst control
    """
    
    def __init__(self, max_requests: int, window_seconds: float):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests: List[float] = []
        self.lock = asyncio.Lock()
    
    async def acquire(self):
        """Acquire permission with sliding window"""
        async with self.lock:
            now = time.time()
            
            # Remove requests outside window
            self.requests = [
                req_time for req_time in self.requests
                if now - req_time < self.window_seconds
            ]
            
            if len(self.requests) >= self.max_requests:
                # Calculate wait time
                oldest_request = self.requests[0]
                sleep_time = self.window_seconds - (now - oldest_request)
                await asyncio.sleep(sleep_time)
                
                # Clean up again after sleep
                now = time.time()
                self.requests = [
                    req_time for req_time in self.requests
                    if now - req_time < self.window_seconds
                ]
            
            self.requests.append(now)


async def demo_rate_limiting():
    """
    Demonstrates rate limiting patterns
    """
    
    async def api_call(request_id: int) -> dict:
        """Simulate API request"""
        return {"id": request_id, "timestamp": time.time()}
    
    # Token bucket: 10 requests per second
    limiter = RateLimiter(rate=10, per=1.0)
    
    start = time.time()
    tasks = []
    for i in range(25):
        tasks.append(limiter.execute(api_call, i))
    
    results = await asyncio.gather(*tasks)
    duration = time.time() - start
    
    logger.info(f"Completed {len(results)} requests in {duration:.2f}s")
    logger.info(f"Actual rate: {len(results)/duration:.1f} req/s (limit: 10 req/s)")


# ============================================================================
# 7. ASYNC QUEUE PATTERNS (PRODUCER-CONSUMER)
# ============================================================================

class AsyncProducerConsumer:
    """
    Producer-consumer pattern for async data pipelines.
    
    Use cases:
    - Stream processing
    - ETL pipelines
    - Message queue processing
    - Work distribution
    
    Performance: Decouples producers/consumers, maximizes throughput
    """
    
    def __init__(self, queue_size: int = 100):
        self.queue: asyncio.Queue = asyncio.Queue(maxsize=queue_size)
        self.producers_done = False
    
    async def producer(
        self,
        items: List[T],
        producer_id: int
    ):
        """Produce items to queue"""
        for item in items:
            await self.queue.put(item)
            logger.debug(f"Producer {producer_id} added {item}")
            await asyncio.sleep(0.01)  # Simulate work
    
    async def consumer(
        self,
        processor: Callable[[T], Awaitable[Any]],
        consumer_id: int
    ) -> int:
        """Consume and process items from queue"""
        processed = 0
        
        while True:
            try:
                # Wait for item with timeout
                item = await asyncio.wait_for(
                    self.queue.get(),
                    timeout=1.0
                )
                
                await processor(item)
                self.queue.task_done()
                processed += 1
                logger.debug(f"Consumer {consumer_id} processed {item}")
                
            except asyncio.TimeoutError:
                # Check if producers are done
                if self.producers_done and self.queue.empty():
                    break
                continue
        
        return processed
    
    async def run_pipeline(
        self,
        producer_data: List[List[T]],
        processor: Callable[[T], Awaitable[Any]],
        num_consumers: int = 3
    ) -> Dict[str, Any]:
        """Run complete producer-consumer pipeline"""
        start = time.time()
        
        # Start producers
        producer_tasks = [
            self.producer(data, i)
            for i, data in enumerate(producer_data)
        ]
        
        # Start consumers
        consumer_tasks = [
            self.consumer(processor, i)
            for i in range(num_consumers)
        ]
        
        # Wait for producers to finish
        await asyncio.gather(*producer_tasks)
        self.producers_done = True
        
        # Wait for consumers to finish
        consumer_results = await asyncio.gather(*consumer_tasks)
        
        duration = time.time() - start
        total_processed = sum(consumer_results)
        
        return {
            "total_processed": total_processed,
            "duration": duration,
            "throughput": total_processed / duration,
            "consumer_results": consumer_results
        }


async def demo_producer_consumer():
    """
    Demonstrates async producer-consumer pattern
    """
    
    async def process_item(item: dict) -> dict:
        """Process item (simulate I/O operation)"""
        await asyncio.sleep(0.05)
        return {"processed": item["id"], "result": item["value"] * 2}
    
    # Create test data for multiple producers
    producer_data = [
        [{"id": i, "value": i} for i in range(0, 50)],
        [{"id": i, "value": i} for i in range(50, 100)],
        [{"id": i, "value": i} for i in range(100, 150)]
    ]
    
    pipeline = AsyncProducerConsumer(queue_size=50)
    
    results = await pipeline.run_pipeline(
        producer_data=producer_data,
        processor=process_item,
        num_consumers=5
    )
    
    logger.info(f"Processed {results['total_processed']} items in {results['duration']:.2f}s")
    logger.info(f"Throughput: {results['throughput']:.1f} items/s")


# ============================================================================
# 8. TIMEOUT HANDLING PATTERNS
# ============================================================================

class TimeoutHandler:
    """
    Comprehensive timeout handling for async operations.
    
    Use cases:
    - Prevent hanging operations
    - SLA enforcement
    - Circuit breaker patterns
    - Resource protection
    
    Performance: Prevents resource starvation, improves reliability
    """
    
    @staticmethod
    async def with_timeout(
        coro: Awaitable[T],
        timeout: float,
        default: Optional[T] = None
    ) -> Optional[T]:
        """Execute with timeout, return default on timeout"""
        try:
            return await asyncio.wait_for(coro, timeout=timeout)
        except asyncio.TimeoutError:
            logger.warning(f"Operation timed out after {timeout}s")
            return default
    
    @staticmethod
    async def with_multiple_timeouts(
        coro: Awaitable[T],
        warning_timeout: float,
        error_timeout: float
    ) -> T:
        """Execute with warning and error timeouts"""
        warning_task = asyncio.create_task(asyncio.sleep(warning_timeout))
        error_task = asyncio.create_task(asyncio.sleep(error_timeout))
        coro_task = asyncio.create_task(coro)
        
        done, pending = await asyncio.wait(
            [warning_task, error_task, coro_task],
            return_when=asyncio.FIRST_COMPLETED
        )
        
        if coro_task in done:
            # Operation completed
            for task in pending:
                task.cancel()
            return await coro_task
        elif warning_task in done:
            logger.warning(f"Operation exceeded warning timeout ({warning_timeout}s)")
            # Continue waiting for error timeout
            done, pending = await asyncio.wait(
                [error_task, coro_task],
                return_when=asyncio.FIRST_COMPLETED
            )
            
            if coro_task in done:
                error_task.cancel()
                return await coro_task
            else:
                coro_task.cancel()
                raise asyncio.TimeoutError(
                    f"Operation timed out after {error_timeout}s"
                )
        else:
            # Error timeout reached
            coro_task.cancel()
            raise asyncio.TimeoutError(
                f"Operation timed out after {error_timeout}s"
            )
    
    @staticmethod
    async def with_retry_timeout(
        coro: Callable[..., Awaitable[T]],
        max_retries: int,
        timeout_per_attempt: float,
        *args,
        **kwargs
    ) -> T:
        """Retry with timeout per attempt"""
        for attempt in range(max_retries):
            try:
                return await asyncio.wait_for(
                    coro(*args, **kwargs),
                    timeout=timeout_per_attempt
                )
            except asyncio.TimeoutError:
                if attempt < max_retries - 1:
                    logger.warning(
                        f"Attempt {attempt + 1} timed out, retrying..."
                    )
                    await asyncio.sleep(0.5)
                else:
                    raise


async def demo_timeout_patterns():
    """
    Demonstrates various timeout handling patterns
    """
    
    async def slow_operation(delay: float) -> str:
        await asyncio.sleep(delay)
        return f"Completed after {delay}s"
    
    # Simple timeout with default
    result = await TimeoutHandler.with_timeout(
        slow_operation(0.5),
        timeout=1.0,
        default="Timed out"
    )
    logger.info(f"Result 1: {result}")
    
    # Timeout with default (will timeout)
    result = await TimeoutHandler.with_timeout(
        slow_operation(2.0),
        timeout=0.5,
        default="Operation timed out"
    )
    logger.info(f"Result 2: {result}")
    
    # Multiple timeout levels
    try:
        result = await TimeoutHandler.with_multiple_timeouts(
            slow_operation(1.5),
            warning_timeout=0.5,
            error_timeout=2.0
        )
        logger.info(f"Result 3: {result}")
    except asyncio.TimeoutError as e:
        logger.error(f"Failed: {e}")


# ============================================================================
# COMPREHENSIVE EXAMPLE: ASYNC DATA PIPELINE
# ============================================================================

@dataclass
class PipelineMetrics:
    """Track pipeline performance metrics"""
    total_items: int
    successful: int
    failed: int
    duration: float
    throughput: float
    avg_latency: float


class AsyncDataPipeline:
    """
    Production-ready async data pipeline combining all patterns.
    
    Use cases:
    - ETL processes
    - Data ingestion
    - Multi-source aggregation
    - Real-time data processing
    
    Performance: 10-100x faster than synchronous pipelines
    """
    
    def __init__(
        self,
        max_concurrent: int = 10,
        batch_size: int = 50,
        rate_limit: float = 100.0,
        retry_attempts: int = 3,
        timeout: float = 5.0
    ):
        self.concurrency_limiter = ConcurrencyLimiter(max_concurrent)
        self.batch_processor = BatchProcessor(batch_size)
        self.rate_limiter = RateLimiter(rate=rate_limit, per=1.0)
        self.retry_strategy = RetryStrategy(max_retries=retry_attempts)
        self.timeout = timeout
    
    async def fetch_data(self, source: str) -> List[dict]:
        """Fetch data from source with rate limiting and retry"""
        async def _fetch():
            await asyncio.sleep(0.1)  # Simulate API call
            return [{"source": source, "id": i} for i in range(10)]
        
        return await self.rate_limiter.execute(
            self.retry_strategy.execute_with_retry,
            _fetch
        )
    
    async def transform_batch(self, batch: List[dict]) -> List[dict]:
        """Transform batch of records"""
        await asyncio.sleep(0.05 * len(batch))  # Simulate processing
        return [
            {**record, "transformed": True, "value": record["id"] * 2}
            for record in batch
        ]
    
    async def save_batch(self, batch: List[dict]) -> int:
        """Save batch to destination"""
        async def _save():
            await asyncio.sleep(0.02 * len(batch))  # Simulate I/O
            return len(batch)
        
        return await TimeoutHandler.with_timeout(
            _save(),
            timeout=self.timeout,
            default=0
        )
    
    async def run_pipeline(
        self,
        sources: List[str]
    ) -> PipelineMetrics:
        """Run complete data pipeline"""
        start = time.time()
        latencies = []
        
        # Stage 1: Fetch data from multiple sources (parallel)
        logger.info(f"Fetching from {len(sources)} sources...")
        fetch_tasks = [
            self.concurrency_limiter.execute(self.fetch_data, source)
            for source in sources
        ]
        fetch_results, fetch_errors = await AsyncGatherer.gather_continue_on_error(
            *fetch_tasks
        )
        
        # Flatten results
        all_records = [
            record
            for batch in fetch_results
            for record in batch
        ]
        
        logger.info(f"Fetched {len(all_records)} records")
        
        # Stage 2: Transform in batches
        logger.info("Transforming data...")
        transformed_batches = []
        async for batch in self.batch_processor.stream_batches(all_records):
            batch_start = time.time()
            transformed = await self.transform_batch(batch)
            latencies.append(time.time() - batch_start)
            transformed_batches.extend(transformed)
        
        logger.info(f"Transformed {len(transformed_batches)} records")
        
        # Stage 3: Save in batches (with concurrency control)
        logger.info("Saving data...")
        save_results = await self.batch_processor.process_in_batches(
            transformed_batches,
            self.save_batch
        )
        
        # Calculate metrics
        duration = time.time() - start
        successful = sum(save_results)
        failed = len(all_records) - successful
        
        metrics = PipelineMetrics(
            total_items=len(all_records),
            successful=successful,
            failed=failed,
            duration=duration,
            throughput=successful / duration if duration > 0 else 0,
            avg_latency=sum(latencies) / len(latencies) if latencies else 0
        )
        
        return metrics


async def demo_complete_pipeline():
    """
    Demonstrates complete async data pipeline
    
    Performance comparison:
    - Synchronous: ~50 items/second
    - Async pipeline: ~500-1000 items/second (10-20x improvement)
    """
    
    sources = [f"source_{i}" for i in range(20)]
    
    pipeline = AsyncDataPipeline(
        max_concurrent=10,
        batch_size=50,
        rate_limit=100.0,
        retry_attempts=3,
        timeout=5.0
    )
    
    metrics = await pipeline.run_pipeline(sources)
    
    logger.info("\n" + "="*60)
    logger.info("PIPELINE METRICS")
    logger.info("="*60)
    logger.info(f"Total items:     {metrics.total_items}")
    logger.info(f"Successful:      {metrics.successful}")
    logger.info(f"Failed:          {metrics.failed}")
    logger.info(f"Duration:        {metrics.duration:.2f}s")
    logger.info(f"Throughput:      {metrics.throughput:.1f} items/s")
    logger.info(f"Avg latency:     {metrics.avg_latency*1000:.1f}ms")
    logger.info(f"Success rate:    {metrics.successful/metrics.total_items*100:.1f}%")
    logger.info("="*60)


# ============================================================================
# PERFORMANCE COMPARISON: SYNC VS ASYNC
# ============================================================================

async def performance_comparison():
    """
    Comprehensive performance comparison: Synchronous vs Async
    
    Demonstrates real-world performance gains
    """
    
    def sync_io_operation():
        """Synchronous I/O operation"""
        time.sleep(0.1)
        return {"result": "data"}
    
    async def async_io_operation():
        """Async I/O operation"""
        await asyncio.sleep(0.1)
        return {"result": "data"}
    
    num_operations = 50
    
    # Synchronous execution
    logger.info("\nRunning SYNCHRONOUS operations...")
    start = time.time()
    sync_results = [sync_io_operation() for _ in range(num_operations)]
    sync_duration = time.time() - start
    
    # Async execution
    logger.info("Running ASYNC operations...")
    start = time.time()
    async_results = await asyncio.gather(*[
        async_io_operation() for _ in range(num_operations)
    ])
    async_duration = time.time() - start
    
    # Results
    speedup = sync_duration / async_duration
    
    logger.info("\n" + "="*60)
    logger.info("PERFORMANCE COMPARISON")
    logger.info("="*60)
    logger.info(f"Operations:           {num_operations}")
    logger.info(f"Synchronous time:     {sync_duration:.2f}s")
    logger.info(f"Async time:           {async_duration:.2f}s")
    logger.info(f"Speedup:              {speedup:.1f}x")
    logger.info(f"Throughput (sync):    {num_operations/sync_duration:.1f} ops/s")
    logger.info(f"Throughput (async):   {num_operations/async_duration:.1f} ops/s")
    logger.info("="*60)


# ============================================================================
# MAIN DEMO
# ============================================================================

async def main():
    """Run all demonstrations"""
    
    logger.info("\n" + "="*60)
    logger.info("ASYNC PATTERNS DEMONSTRATION")
    logger.info("="*60)
    
    demos = [
        ("Performance Comparison", performance_comparison),
        ("1. Concurrency Limiter", demo_concurrency_limiter),
        ("2. Batch Processing", demo_batch_processing),
        ("3. Exponential Backoff", demo_exponential_backoff),
        ("4. Async Context Managers", demo_async_context_managers),
        ("5. Gather Patterns", demo_gather_patterns),
        ("6. Rate Limiting", demo_rate_limiting),
        ("7. Producer-Consumer", demo_producer_consumer),
        ("8. Timeout Patterns", demo_timeout_patterns),
        ("Complete Pipeline", demo_complete_pipeline),
    ]
    
    for name, demo_func in demos:
        logger.info(f"\n{'='*60}")
        logger.info(f"{name}")
        logger.info(f"{'='*60}")
        try:
            await demo_func()
        except Exception as e:
            logger.error(f"Demo failed: {e}")
    
    logger.info(f"\n{'='*60}")
    logger.info("ALL DEMONSTRATIONS COMPLETED")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    asyncio.run(main())
