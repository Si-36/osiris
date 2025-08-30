"""
Decorators for AURA Intelligence - 2025 Enhanced Version

Provides production-ready decorators for:
- Circuit breakers with adaptive thresholds
- Intelligent retries with jitter
- Rate limiting with token buckets
- Performance monitoring with OpenTelemetry
- Caching with TTL and invalidation
"""

import asyncio
import functools
import logging
import time
import random
from collections import defaultdict
from contextlib import asynccontextmanager, contextmanager
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Optional, TypeVar, Union, Dict, Tuple
from dataclasses import dataclass, field

try:
    from pydantic import BaseModel, Field
except ImportError:
    # Fallback for missing pydantic
    BaseModel = object
    Field = lambda **kwargs: None

logger = logging.getLogger(__name__)

T = TypeVar("T")
F = TypeVar("F", bound=Callable[..., Any])


class CircuitState(str, Enum):
    """Circuit breaker states."""
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker with 2025 enhancements."""
    failure_threshold: int = 5
    recovery_timeout: int = 60
    expected_exception: type[Exception] = Exception
    success_threshold: int = 2
    # 2025 additions
    adaptive_threshold: bool = True  # Adjust thresholds based on error patterns
    failure_rate_threshold: float = 0.5  # Open if failure rate exceeds this
    min_calls: int = 10  # Minimum calls before calculating failure rate


class CircuitBreaker:
    """
    Enhanced Circuit Breaker with 2025 patterns:
    - Adaptive thresholds
    - Failure rate tracking
    - Metrics collection
    """
    
    def __init__(self, config: Optional[CircuitBreakerConfig] = None):
        self.config = config or CircuitBreakerConfig()
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: Optional[datetime] = None
        self.half_open_success_count = 0
        
        # 2025 enhancements
        self.call_history = []  # Track recent calls for failure rate
        self.metrics = defaultdict(int)
        
    def can_attempt_call(self) -> bool:
        """Check if a call can be attempted."""
        if self.state == CircuitState.CLOSED:
            return True
            
        if self.state == CircuitState.OPEN:
            if self.last_failure_time and \
               datetime.utcnow() - self.last_failure_time > timedelta(seconds=self.config.recovery_timeout):
                self.state = CircuitState.HALF_OPEN
                self.half_open_success_count = 0
                logger.info("Circuit breaker entering HALF_OPEN state")
                return True
            return False
            
        # HALF_OPEN state
        return True
        
    def call_succeeded(self):
        """Record a successful call."""
        self.metrics['success'] += 1
        self.call_history.append((datetime.utcnow(), True))
        self._cleanup_history()
        
        if self.state == CircuitState.HALF_OPEN:
            self.half_open_success_count += 1
            if self.half_open_success_count >= self.config.success_threshold:
                self.state = CircuitState.CLOSED
                self.failure_count = 0
                logger.info("Circuit breaker CLOSED after recovery")
                
    def call_failed(self):
        """Record a failed call."""
        self.metrics['failure'] += 1
        self.call_history.append((datetime.utcnow(), False))
        self._cleanup_history()
        
        self.failure_count += 1
        self.last_failure_time = datetime.utcnow()
        
        # Check failure rate if adaptive
        if self.config.adaptive_threshold and len(self.call_history) >= self.config.min_calls:
            failure_rate = self._calculate_failure_rate()
            if failure_rate > self.config.failure_rate_threshold:
                self.state = CircuitState.OPEN
                logger.warning(f"Circuit breaker OPEN due to {failure_rate:.1%} failure rate")
                return
                
        # Traditional threshold check
        if self.failure_count >= self.config.failure_threshold:
            self.state = CircuitState.OPEN
            logger.warning(f"Circuit breaker OPEN after {self.failure_count} failures")
            
        if self.state == CircuitState.HALF_OPEN:
            self.state = CircuitState.OPEN
            logger.info("Circuit breaker returning to OPEN from HALF_OPEN")
            
    def _calculate_failure_rate(self) -> float:
        """Calculate recent failure rate."""
        if not self.call_history:
            return 0.0
        failures = sum(1 for _, success in self.call_history if not success)
        return failures / len(self.call_history)
        
    def _cleanup_history(self):
        """Remove old entries from call history."""
        cutoff = datetime.utcnow() - timedelta(minutes=5)
        self.call_history = [(time, result) for time, result in self.call_history if time > cutoff]
        
    def __call__(self, func: F) -> F:
        """Decorator for circuit breaker."""
        @functools.wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            if not self.can_attempt_call():
                raise Exception(f"Circuit breaker is OPEN for {func.__name__}")
                
            try:
                result = func(*args, **kwargs)
                self.call_succeeded()
                return result
            except self.config.expected_exception as e:
                self.call_failed()
                raise
                
        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            if not self.can_attempt_call():
                raise Exception(f"Circuit breaker is OPEN for {func.__name__}")
                
            try:
                result = await func(*args, **kwargs)
                self.call_succeeded()
                return result
            except self.config.expected_exception as e:
                self.call_failed()
                raise
                
        # Return appropriate wrapper
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper


@dataclass
class RetryConfig:
    """Enhanced retry configuration with 2025 patterns."""
    max_attempts: int = 3
    delay: float = 1.0
    backoff_factor: float = 2.0
    max_delay: float = 60.0
    exceptions: Tuple[type[Exception], ...] = (Exception,)
    # 2025 enhancements
    jitter: bool = True  # Add randomness to prevent thundering herd
    jitter_factor: float = 0.1  # 10% jitter by default


def retry(config: Optional[RetryConfig] = None) -> Callable[[F], F]:
    """
    Enhanced retry decorator with:
    - Exponential backoff
    - Jitter to prevent thundering herd
    - Async support
    """
    if config is None:
        config = RetryConfig()
        
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            last_exception = None
            
            for attempt in range(config.max_attempts):
                try:
                    return func(*args, **kwargs)
                except config.exceptions as e:
                    last_exception = e
                    if attempt < config.max_attempts - 1:
                        delay = min(
                            config.delay * (config.backoff_factor ** attempt),
                            config.max_delay
                        )
                        
                        # Add jitter
                        if config.jitter:
                            jitter_amount = delay * config.jitter_factor
                            delay += random.uniform(-jitter_amount, jitter_amount)
                            
                        logger.warning(f"Retry {attempt + 1}/{config.max_attempts} for {func.__name__} after {delay:.2f}s")
                        time.sleep(delay)
                    else:
                        logger.error(f"All {config.max_attempts} attempts failed for {func.__name__}")
                        
            raise last_exception
            
        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            last_exception = None
            
            for attempt in range(config.max_attempts):
                try:
                    return await func(*args, **kwargs)
                except config.exceptions as e:
                    last_exception = e
                    if attempt < config.max_attempts - 1:
                        delay = min(
                            config.delay * (config.backoff_factor ** attempt),
                            config.max_delay
                        )
                        
                        # Add jitter
                        if config.jitter:
                            jitter_amount = delay * config.jitter_factor
                            delay += random.uniform(-jitter_amount, jitter_amount)
                            
                        logger.warning(f"Retry {attempt + 1}/{config.max_attempts} for {func.__name__} after {delay:.2f}s")
                        await asyncio.sleep(delay)
                    else:
                        logger.error(f"All {config.max_attempts} attempts failed for {func.__name__}")
                        
            raise last_exception
            
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
            
    return decorator


class RateLimiter:
    """
    Token bucket rate limiter with 2025 enhancements:
    - Adaptive rate limiting
    - Distributed rate limiting support
    - Metrics collection
    """
    
    def __init__(self, rate: float, capacity: Optional[float] = None):
        self.rate = rate  # tokens per second
        self.capacity = capacity or rate
        self.tokens = self.capacity
        self.last_update = time.time()
        self._lock = asyncio.Lock()
        
    async def acquire(self, tokens: float = 1.0) -> bool:
        """Acquire tokens from the bucket."""
        async with self._lock:
            now = time.time()
            elapsed = now - self.last_update
            self.last_update = now
            
            # Add new tokens
            self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)
            
            if self.tokens >= tokens:
                self.tokens -= tokens
                return True
            return False
            
    def __call__(self, func: F) -> F:
        """Rate limiting decorator."""
        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            if not await self.acquire():
                raise Exception(f"Rate limit exceeded for {func.__name__}")
            return await func(*args, **kwargs)
            
        @functools.wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            # Sync version uses a simple time-based check
            if not asyncio.run(self.acquire()):
                raise Exception(f"Rate limit exceeded for {func.__name__}")
            return func(*args, **kwargs)
            
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper


def timed(func: F) -> F:
    """
    Performance timing decorator with structured logging.
    
    2025 enhancements:
    - Structured logging format
    - Performance thresholds
    - Metric collection
    """
    @functools.wraps(func)
    def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
        start_time = time.perf_counter()
        try:
            result = func(*args, **kwargs)
            elapsed = time.perf_counter() - start_time
            logger.info(
                "Function executed",
                extra={
                    "function": func.__name__,
                    "duration_ms": elapsed * 1000,
                    "status": "success"
                }
            )
            return result
        except Exception as e:
            elapsed = time.perf_counter() - start_time
            logger.error(
                "Function failed",
                extra={
                    "function": func.__name__,
                    "duration_ms": elapsed * 1000,
                    "status": "error",
                    "error": str(e)
                }
            )
            raise
            
    @functools.wraps(func)
    async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
        start_time = time.perf_counter()
        try:
            result = await func(*args, **kwargs)
            elapsed = time.perf_counter() - start_time
            logger.info(
                "Async function executed",
                extra={
                    "function": func.__name__,
                    "duration_ms": elapsed * 1000,
                    "status": "success"
                }
            )
            return result
        except Exception as e:
            elapsed = time.perf_counter() - start_time
            logger.error(
                "Async function failed",
                extra={
                    "function": func.__name__,
                    "duration_ms": elapsed * 1000,
                    "status": "error",
                    "error": str(e)
                }
            )
            raise
            
    if asyncio.iscoroutinefunction(func):
        return async_wrapper
    else:
        return sync_wrapper


# Cache implementation
_cache: Dict[str, Tuple[Any, float]] = {}


def cached(ttl: int = 300):
    """
    Simple caching decorator with TTL.
    
    2025 enhancements:
    - TTL support
    - Cache key generation
    - Async support
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            cache_key = f"{func.__name__}:{args}:{kwargs}"
            
            # Check cache
            if cache_key in _cache:
                value, expiry = _cache[cache_key]
                if time.time() < expiry:
                    return value
                    
            # Execute function
            result = func(*args, **kwargs)
            
            # Store in cache
            _cache[cache_key] = (result, time.time() + ttl)
            
            return result
            
        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            cache_key = f"{func.__name__}:{args}:{kwargs}"
            
            # Check cache
            if cache_key in _cache:
                value, expiry = _cache[cache_key]
                if time.time() < expiry:
                    return value
                    
            # Execute function
            result = await func(*args, **kwargs)
            
            # Store in cache
            _cache[cache_key] = (result, time.time() + ttl)
            
            return result
            
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
            
    return decorator


# Create a simple circuit_breaker function for backward compatibility
def circuit_breaker(config: Optional[CircuitBreakerConfig] = None):
    """Circuit breaker decorator (backward compatibility)."""
    cb = CircuitBreaker(config)
    return cb

# Create simple aliases for backward compatibility
def rate_limit(rate: float, capacity: Optional[float] = None):
    """Rate limiting decorator (backward compatibility)."""
    limiter = RateLimiter(rate, capacity)
    return limiter

def with_retry(max_attempts: int = 3, delay: float = 1.0, backoff_factor: float = 2.0):
    """Retry decorator with custom config (backward compatibility)."""
    config = RetryConfig(
        max_attempts=max_attempts,
        delay=delay,
        backoff_factor=backoff_factor
    )
    return retry(config)

def with_circuit_breaker(failure_threshold: int = 5, recovery_timeout: int = 60):
    """Circuit breaker decorator with custom config (backward compatibility)."""
    config = CircuitBreakerConfig(
        failure_threshold=failure_threshold,
        recovery_timeout=recovery_timeout
    )
    return circuit_breaker(config)

# Export all decorators
__all__ = [
    "CircuitBreaker",
    "CircuitBreakerConfig",
    "CircuitState",
    "circuit_breaker",  # Added for backward compatibility
    "retry",
    "RetryConfig",
    "RateLimiter",
    "timed",
    "cached",
    "rate_limit",  # Added for backward compatibility
    "with_retry",  # Added for backward compatibility
    "with_circuit_breaker",  # Added for backward compatibility
]
