"""
Production Observability System

Apply to: Production monitoring, distributed systems, microservices, API services,
high-availability applications, SLA tracking, incident response, performance monitoring.

This module provides comprehensive observability features including:
- Structured JSON logging with correlation IDs and request context
- Prometheus metrics (counters, gauges, histograms)
- SLA tracking with uptime and latency percentiles
- Request tracing and context management
- Alert threshold definitions with severity levels
- Log level configuration for different environments

Features:
- Thread-safe and async-compatible
- Production-ready structured logging
- Multi-dimensional metrics
- Real-time SLA calculations
- Context propagation for distributed tracing
- Configurable alert thresholds
"""

import contextvars
import logging
import time
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, Iterator, List, Optional
from uuid import uuid4

import structlog
from prometheus_client import Counter, Gauge, Histogram, generate_latest

# Context variable for request tracing
request_context: contextvars.ContextVar[Dict[str, Any]] = contextvars.ContextVar(
    "request_context", default={}
)


class LogLevel(str, Enum):
    """Log levels for application logging."""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class AlertSeverity(str, Enum):
    """Alert severity levels."""

    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class AlertThreshold:
    """
    Alert threshold definition.
    
    Apply to: SLA monitoring, performance monitoring, error rate tracking,
    infrastructure monitoring, application health checks.
    """

    name: str
    metric: str
    threshold: float
    severity: AlertSeverity
    description: str
    comparison: str = ">"  # >, <, >=, <=, ==

    def check(self, value: float) -> bool:
        """Check if value breaches threshold."""
        if self.comparison == ">":
            return value > self.threshold
        elif self.comparison == "<":
            return value < self.threshold
        elif self.comparison == ">=":
            return value >= self.threshold
        elif self.comparison == "<=":
            return value <= self.threshold
        elif self.comparison == "==":
            return value == self.threshold
        return False


class StructuredLogger:
    """
    Structured JSON logger with correlation IDs and request context.
    
    Apply to: Production applications, microservices, distributed systems,
    audit logging, debugging, log aggregation (ELK, Splunk, CloudWatch).
    
    Features:
    - JSON structured logging for easy parsing
    - Correlation ID tracking across requests
    - Automatic request context binding
    - Multiple output formats (JSON, console)
    - Thread-safe context management
    """

    def __init__(self, service_name: str = "app", log_level: LogLevel = LogLevel.INFO):
        """
        Initialize structured logger.
        
        Args:
            service_name: Name of the service for log identification
            log_level: Minimum log level to output
        """
        self.service_name = service_name
        self._configure_structlog(log_level)
        self.logger = structlog.get_logger()

    def _configure_structlog(self, log_level: LogLevel) -> None:
        """Configure structlog with processors and formatting."""
        logging.basicConfig(
            format="%(message)s",
            level=getattr(logging, log_level.value),
        )

        structlog.configure(
            processors=[
                structlog.contextvars.merge_contextvars,
                structlog.processors.add_log_level,
                structlog.processors.StackInfoRenderer(),
                structlog.dev.set_exc_info,
                structlog.processors.TimeStamper(fmt="iso", utc=True),
                structlog.processors.JSONRenderer(),
            ],
            wrapper_class=structlog.make_filtering_bound_logger(
                getattr(logging, log_level.value)
            ),
            context_class=dict,
            logger_factory=structlog.PrintLoggerFactory(),
            cache_logger_on_first_use=True,
        )

    def bind_context(self, **kwargs: Any) -> None:
        """
        Bind context variables to all subsequent log messages.
        
        Args:
            **kwargs: Key-value pairs to add to log context
        """
        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(
            service=self.service_name,
            **kwargs,
        )

    def debug(self, message: str, **kwargs: Any) -> None:
        """Log debug message."""
        self.logger.debug(message, **kwargs)

    def info(self, message: str, **kwargs: Any) -> None:
        """Log info message."""
        self.logger.info(message, **kwargs)

    def warning(self, message: str, **kwargs: Any) -> None:
        """Log warning message."""
        self.logger.warning(message, **kwargs)

    def error(self, message: str, **kwargs: Any) -> None:
        """Log error message."""
        self.logger.error(message, **kwargs)

    def critical(self, message: str, **kwargs: Any) -> None:
        """Log critical message."""
        self.logger.critical(message, **kwargs)


class MetricsCollector:
    """
    Prometheus metrics collector for application monitoring.
    
    Apply to: Performance monitoring, request tracking, error rate monitoring,
    business metrics, resource utilization, capacity planning.
    
    Metrics types:
    - Counter: Monotonically increasing values (requests, errors)
    - Gauge: Values that go up and down (connections, queue size)
    - Histogram: Distribution of values (latency, request size)
    """

    def __init__(self, namespace: str = "app"):
        """
        Initialize metrics collector.
        
        Args:
            namespace: Namespace for all metrics (e.g., 'api', 'worker')
        """
        self.namespace = namespace
        self._counters: Dict[str, Counter] = {}
        self._gauges: Dict[str, Gauge] = {}
        self._histograms: Dict[str, Histogram] = {}

    def counter(
        self, name: str, description: str, labels: Optional[List[str]] = None
    ) -> Counter:
        """
        Get or create a counter metric.
        
        Args:
            name: Metric name
            description: Metric description
            labels: Label names for multi-dimensional metrics
            
        Returns:
            Prometheus Counter object
        """
        key = f"{self.namespace}_{name}"
        if key not in self._counters:
            self._counters[key] = Counter(
                key, description, labelnames=labels or []
            )
        return self._counters[key]

    def gauge(
        self, name: str, description: str, labels: Optional[List[str]] = None
    ) -> Gauge:
        """
        Get or create a gauge metric.
        
        Args:
            name: Metric name
            description: Metric description
            labels: Label names for multi-dimensional metrics
            
        Returns:
            Prometheus Gauge object
        """
        key = f"{self.namespace}_{name}"
        if key not in self._gauges:
            self._gauges[key] = Gauge(
                key, description, labelnames=labels or []
            )
        return self._gauges[key]

    def histogram(
        self,
        name: str,
        description: str,
        labels: Optional[List[str]] = None,
        buckets: Optional[List[float]] = None,
    ) -> Histogram:
        """
        Get or create a histogram metric.
        
        Args:
            name: Metric name
            description: Metric description
            labels: Label names for multi-dimensional metrics
            buckets: Histogram buckets (defaults to standard latency buckets)
            
        Returns:
            Prometheus Histogram object
        """
        key = f"{self.namespace}_{name}"
        if key not in self._histograms:
            # Default buckets for latency in seconds
            default_buckets = (
                0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0
            )
            self._histograms[key] = Histogram(
                key,
                description,
                labelnames=labels or [],
                buckets=buckets or default_buckets,
            )
        return self._histograms[key]

    def export_metrics(self) -> bytes:
        """
        Export metrics in Prometheus format.
        
        Returns:
            Metrics in Prometheus text format
        """
        return generate_latest()


@dataclass
class SLAMetrics:
    """SLA tracking metrics."""

    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    latencies: deque = field(default_factory=lambda: deque(maxlen=10000))
    uptime_start: datetime = field(default_factory=datetime.utcnow)
    downtime_duration: timedelta = field(default_factory=lambda: timedelta(0))
    last_downtime: Optional[datetime] = None

    @property
    def success_rate(self) -> float:
        """Calculate success rate percentage."""
        if self.total_requests == 0:
            return 100.0
        return (self.successful_requests / self.total_requests) * 100

    @property
    def error_rate(self) -> float:
        """Calculate error rate percentage."""
        return 100.0 - self.success_rate

    @property
    def uptime_percentage(self) -> float:
        """Calculate uptime percentage."""
        total_time = datetime.utcnow() - self.uptime_start
        if total_time.total_seconds() == 0:
            return 100.0
        uptime = total_time - self.downtime_duration
        return (uptime.total_seconds() / total_time.total_seconds()) * 100


class SLATracker:
    """
    SLA tracker for monitoring service level agreements.
    
    Apply to: Production APIs, SaaS applications, microservices, infrastructure
    monitoring, customer-facing services, uptime guarantees.
    
    Tracks:
    - Uptime percentage
    - Request success/error rates
    - Latency percentiles (p50, p90, p95, p99)
    - Availability windows
    """

    def __init__(self):
        """Initialize SLA tracker."""
        self.metrics = SLAMetrics()

    def record_request(
        self, success: bool, latency_seconds: float, endpoint: str = "default"
    ) -> None:
        """
        Record a request for SLA tracking.
        
        Args:
            success: Whether the request was successful
            latency_seconds: Request latency in seconds
            endpoint: API endpoint or service identifier
        """
        self.metrics.total_requests += 1
        if success:
            self.metrics.successful_requests += 1
        else:
            self.metrics.failed_requests += 1

        self.metrics.latencies.append(latency_seconds)

    def mark_downtime_start(self) -> None:
        """Mark the start of a downtime period."""
        self.metrics.last_downtime = datetime.utcnow()

    def mark_downtime_end(self) -> None:
        """Mark the end of a downtime period."""
        if self.metrics.last_downtime:
            downtime = datetime.utcnow() - self.metrics.last_downtime
            self.metrics.downtime_duration += downtime
            self.metrics.last_downtime = None

    def get_latency_percentiles(self) -> Dict[str, float]:
        """
        Calculate latency percentiles.
        
        Returns:
            Dictionary with p50, p90, p95, p99 percentiles in milliseconds
        """
        if not self.metrics.latencies:
            return {"p50": 0.0, "p90": 0.0, "p95": 0.0, "p99": 0.0}

        sorted_latencies = sorted(self.metrics.latencies)
        count = len(sorted_latencies)

        def percentile(p: float) -> float:
            index = int(count * p)
            return sorted_latencies[min(index, count - 1)] * 1000  # Convert to ms

        return {
            "p50": percentile(0.50),
            "p90": percentile(0.90),
            "p95": percentile(0.95),
            "p99": percentile(0.99),
        }

    def get_sla_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive SLA summary.
        
        Returns:
            Dictionary with all SLA metrics
        """
        percentiles = self.get_latency_percentiles()
        return {
            "uptime_percentage": round(self.metrics.uptime_percentage, 4),
            "success_rate": round(self.metrics.success_rate, 2),
            "error_rate": round(self.metrics.error_rate, 2),
            "total_requests": self.metrics.total_requests,
            "successful_requests": self.metrics.successful_requests,
            "failed_requests": self.metrics.failed_requests,
            "latency_percentiles_ms": percentiles,
            "uptime_start": self.metrics.uptime_start.isoformat(),
            "total_uptime_seconds": (
                datetime.utcnow() - self.metrics.uptime_start
            ).total_seconds(),
            "total_downtime_seconds": self.metrics.downtime_duration.total_seconds(),
        }


@contextmanager
def trace_request(
    logger: StructuredLogger,
    metrics: MetricsCollector,
    sla_tracker: SLATracker,
    operation: str,
    **context_kwargs: Any,
) -> Iterator[Dict[str, Any]]:
    """
    Context manager for tracing requests with automatic logging and metrics.
    
    Apply to: API endpoints, background jobs, database operations, external API calls,
    any operation requiring observability.
    
    Args:
        logger: Structured logger instance
        metrics: Metrics collector instance
        sla_tracker: SLA tracker instance
        operation: Name of the operation being traced
        **context_kwargs: Additional context to bind to logs
        
    Yields:
        Context dictionary with correlation_id and other bound values
        
    Example:
        with trace_request(logger, metrics, sla, "user_registration", user_id=123) as ctx:
            # Your code here
            pass
    """
    correlation_id = str(uuid4())
    start_time = time.time()

    # Bind request context
    context = {
        "correlation_id": correlation_id,
        "operation": operation,
        **context_kwargs,
    }
    logger.bind_context(**context)
    request_context.set(context)

    # Track request start
    request_counter = metrics.counter(
        "requests_total",
        "Total number of requests",
        labels=["operation", "status"],
    )
    active_requests = metrics.gauge(
        "active_requests",
        "Number of active requests",
        labels=["operation"],
    )

    active_requests.labels(operation=operation).inc()
    logger.info(f"{operation} started")

    success = False
    error_msg = None

    try:
        yield context
        success = True
        logger.info(f"{operation} completed successfully")
    except Exception as e:
        error_msg = str(e)
        logger.error(
            f"{operation} failed",
            error=error_msg,
            error_type=type(e).__name__,
        )
        raise
    finally:
        # Calculate duration
        duration = time.time() - start_time

        # Record metrics
        status = "success" if success else "error"
        request_counter.labels(operation=operation, status=status).inc()
        active_requests.labels(operation=operation).dec()

        latency_histogram = metrics.histogram(
            "request_duration_seconds",
            "Request duration in seconds",
            labels=["operation"],
        )
        latency_histogram.labels(operation=operation).observe(duration)

        # Track SLA
        sla_tracker.record_request(success, duration, operation)

        # Final log
        logger.info(
            f"{operation} finished",
            duration_seconds=round(duration, 3),
            success=success,
        )

        # Clear context
        structlog.contextvars.clear_contextvars()


# Predefined alert thresholds
DEFAULT_ALERT_THRESHOLDS = [
    AlertThreshold(
        name="high_error_rate",
        metric="error_rate",
        threshold=5.0,
        severity=AlertSeverity.CRITICAL,
        description="Error rate exceeds 5%",
        comparison=">",
    ),
    AlertThreshold(
        name="low_success_rate",
        metric="success_rate",
        threshold=95.0,
        severity=AlertSeverity.CRITICAL,
        description="Success rate below 95%",
        comparison="<",
    ),
    AlertThreshold(
        name="high_latency_p95",
        metric="latency_p95",
        threshold=500.0,
        severity=AlertSeverity.WARNING,
        description="P95 latency exceeds 500ms",
        comparison=">",
    ),
    AlertThreshold(
        name="high_latency_p99",
        metric="latency_p99",
        threshold=1000.0,
        severity=AlertSeverity.CRITICAL,
        description="P99 latency exceeds 1000ms",
        comparison=">",
    ),
    AlertThreshold(
        name="low_uptime",
        metric="uptime_percentage",
        threshold=99.9,
        severity=AlertSeverity.CRITICAL,
        description="Uptime below 99.9%",
        comparison="<",
    ),
]


def check_alert_thresholds(
    sla_summary: Dict[str, Any],
    thresholds: List[AlertThreshold] = DEFAULT_ALERT_THRESHOLDS,
) -> List[Dict[str, Any]]:
    """
    Check SLA metrics against alert thresholds.
    
    Apply to: Monitoring dashboards, alerting systems, incident detection,
    automated health checks.
    
    Args:
        sla_summary: SLA summary dictionary
        thresholds: List of alert thresholds to check
        
    Returns:
        List of triggered alerts with details
    """
    alerts = []

    for threshold in thresholds:
        # Map metric names to SLA summary keys
        metric_value = None

        if threshold.metric == "error_rate":
            metric_value = sla_summary["error_rate"]
        elif threshold.metric == "success_rate":
            metric_value = sla_summary["success_rate"]
        elif threshold.metric == "latency_p95":
            metric_value = sla_summary["latency_percentiles_ms"]["p95"]
        elif threshold.metric == "latency_p99":
            metric_value = sla_summary["latency_percentiles_ms"]["p99"]
        elif threshold.metric == "uptime_percentage":
            metric_value = sla_summary["uptime_percentage"]

        if metric_value is not None and threshold.check(metric_value):
            alerts.append(
                {
                    "name": threshold.name,
                    "severity": threshold.severity.value,
                    "description": threshold.description,
                    "metric": threshold.metric,
                    "current_value": metric_value,
                    "threshold": threshold.threshold,
                    "timestamp": datetime.utcnow().isoformat(),
                }
            )

    return alerts


# Example usage
if __name__ == "__main__":
    import random

    print("=" * 80)
    print("PRODUCTION OBSERVABILITY SYSTEM - EXAMPLE USAGE")
    print("=" * 80)

    # Initialize components
    logger = StructuredLogger(service_name="payment_service", log_level=LogLevel.INFO)
    metrics = MetricsCollector(namespace="payment")
    sla_tracker = SLATracker()

    print("\n1. STRUCTURED LOGGING EXAMPLE")
    print("-" * 80)

    # Manual logging with context
    logger.bind_context(user_id=12345, tenant_id="acme-corp")
    logger.info("User login successful", ip_address="192.168.1.1", method="oauth2")
    logger.warning(
        "Rate limit approaching",
        current_rate=95,
        limit=100,
        user_id=12345,
    )

    print("\n2. REQUEST TRACING EXAMPLE")
    print("-" * 80)

    # Simulate processing requests
    operations = ["process_payment", "validate_card", "send_notification"]

    for i in range(20):
        operation = random.choice(operations)
        # Simulate some failures
        will_fail = random.random() < 0.1  # 10% failure rate

        try:
            with trace_request(
                logger,
                metrics,
                sla_tracker,
                operation,
                transaction_id=f"txn_{i}",
                amount=round(random.uniform(10, 1000), 2),
            ) as ctx:
                # Simulate work with random latency
                time.sleep(random.uniform(0.01, 0.2))

                if will_fail:
                    raise ValueError("Simulated payment failure")

                # Log within traced context
                logger.debug("Processing step completed", step="validation")

        except ValueError:
            pass  # Already logged by trace_request

    print("\n3. METRICS COLLECTION EXAMPLE")
    print("-" * 80)

    # Manual metrics
    payment_counter = metrics.counter(
        "payments_processed",
        "Total payments processed",
        labels=["payment_method", "currency"],
    )
    payment_counter.labels(payment_method="credit_card", currency="USD").inc()
    payment_counter.labels(payment_method="paypal", currency="EUR").inc(5)

    active_connections = metrics.gauge(
        "database_connections",
        "Active database connections",
    )
    active_connections.set(42)

    print("Metrics recorded:")
    print("  - Payment counter (credit_card/USD): 1")
    print("  - Payment counter (paypal/EUR): 5")
    print("  - Active DB connections: 42")

    print("\n4. SLA TRACKING EXAMPLE")
    print("-" * 80)

    sla_summary = sla_tracker.get_sla_summary()
    print(f"Uptime: {sla_summary['uptime_percentage']:.4f}%")
    print(f"Success Rate: {sla_summary['success_rate']:.2f}%")
    print(f"Error Rate: {sla_summary['error_rate']:.2f}%")
    print(f"Total Requests: {sla_summary['total_requests']}")
    print(f"Successful: {sla_summary['successful_requests']}")
    print(f"Failed: {sla_summary['failed_requests']}")

    print("\nLatency Percentiles:")
    for percentile, value in sla_summary["latency_percentiles_ms"].items():
        print(f"  {percentile}: {value:.2f}ms")

    print("\n5. ALERT THRESHOLD CHECKING EXAMPLE")
    print("-" * 80)

    alerts = check_alert_thresholds(sla_summary)
    if alerts:
        print(f"⚠️  {len(alerts)} alert(s) triggered:")
        for alert in alerts:
            print(f"\n  [{alert['severity'].upper()}] {alert['name']}")
            print(f"  Description: {alert['description']}")
            print(
                f"  Current: {alert['current_value']:.2f}, "
                f"Threshold: {alert['threshold']}"
            )
    else:
        print("✓ All metrics within acceptable thresholds")

    print("\n6. PROMETHEUS METRICS EXPORT EXAMPLE")
    print("-" * 80)
    print("Exported metrics (first 500 chars):")
    print(metrics.export_metrics().decode()[:500] + "...")

    print("\n" + "=" * 80)
    print("OBSERVABILITY EXAMPLE COMPLETE")
    print("=" * 80)
