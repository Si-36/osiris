"""
Circuit Breaker for Store Resilience
====================================
Protects persistence stores from cascading failures with
adaptive thresholds and exponential backoff.
"""

import asyncio
from typing import Dict, Any, Optional, Callable, TypeVar, Generic
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import time
import logging
from functools import wraps

logger = logging.getLogger(__name__)

T = TypeVar('T')


class CircuitState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Blocking calls
    HALF_OPEN = "half_open"  # Testing recovery


@dataclass
class CircuitConfig:
    """Configuration for circuit breaker"""
    # Failure thresholds
    failure_threshold: int = 5
    failure_rate_threshold: float = 0.5
    min_calls_for_evaluation: int = 10
    
    # Timeouts
    timeout_seconds: float = 30.0
    recovery_timeout: float = 60.0
    half_open_max_calls: int = 3
    
    # Backoff
    backoff_factor: float = 2.0
    max_backoff: float = 300.0  # 5 minutes
    
    # Monitoring
    enable_metrics: bool = True
    log_failures: bool = True


@dataclass
class CircuitMetrics:
    """Metrics for circuit breaker decisions"""
    total_calls: int = 0
    success_count: int = 0
    failure_count: int = 0
    timeout_count: int = 0
    
    # Time windows
    last_failure_time: Optional[datetime] = None
    last_success_time: Optional[datetime] = None
    state_changed_at: datetime = field(default_factory=datetime.utcnow)
    consecutive_failures: int = 0
    
    # Performance tracking
    total_latency_ms: float = 0.0
    max_latency_ms: float = 0.0
    
    @property
    def failure_rate(self) -> float:
        """Calculate failure rate"""
        if self.total_calls == 0:
            return 0.0
        return (self.failure_count + self.timeout_count) / self.total_calls
        
    @property
    def avg_latency_ms(self) -> float:
        """Calculate average latency"""
        if self.success_count == 0:
            return 0.0
        return self.total_latency_ms / self.success_count
        
    def reset(self):
        """Reset metrics for new evaluation period"""
        self.total_calls = 0
        self.success_count = 0
        self.failure_count = 0
        self.timeout_count = 0
        self.consecutive_failures = 0
        self.total_latency_ms = 0.0
        self.max_latency_ms = 0.0


class StoreCircuitBreaker(Generic[T]):
    """
    Circuit breaker for persistence stores.
    Prevents cascading failures and provides automatic recovery.
    """
    
    def __init__(self, 
                 store_name: str,
                 config: Optional[CircuitConfig] = None,
                 fallback_func: Optional[Callable[..., T]] = None):
        self.store_name = store_name
        self.config = config or CircuitConfig()
        self.fallback_func = fallback_func
        
        # State management
        self.state = CircuitState.CLOSED
        self.metrics = CircuitMetrics()
        self._state_lock = asyncio.Lock()
        
        # Half-open state tracking
        self._half_open_calls = 0
        self._current_backoff = self.config.recovery_timeout
        
        logger.info(f"Circuit breaker initialized for store: {store_name}")
        
    async def call(self, func: Callable[..., T], *args, **kwargs) -> T:
        """Execute function with circuit breaker protection"""
        # Check if should allow call
        async with self._state_lock:
            if not await self._should_allow_call():
                return await self._handle_blocked_call(*args, **kwargs)
                
        # Execute with monitoring
        start_time = time.perf_counter()
        
        try:
            # Execute with timeout
            result = await asyncio.wait_for(
                func(*args, **kwargs),
                timeout=self.config.timeout_seconds
            )
            
            # Record success
            await self._record_success(start_time)
            return result
            
        except asyncio.TimeoutError as e:
            # Record timeout
            await self._record_timeout()
            
            if self.config.log_failures:
                logger.warning(f"Timeout in {self.store_name}: {func.__name__}")
                
            if self.fallback_func:
                return await self.fallback_func(*args, **kwargs)
            raise
            
        except Exception as e:
            # Record failure
            await self._record_failure(e)
            
            if self.config.log_failures:
                logger.error(f"Error in {self.store_name}: {func.__name__} - {e}")
                
            if self.fallback_func:
                return await self.fallback_func(*args, **kwargs)
            raise
            
    def protect(self, func: Callable[..., T]) -> Callable[..., T]:
        """Decorator to protect async functions with circuit breaker"""
        @wraps(func)
        async def wrapper(*args, **kwargs):
            return await self.call(func, *args, **kwargs)
        return wrapper
        
    async def _should_allow_call(self) -> bool:
        """Determine if call should be allowed (must be called with lock)"""
        if self.state == CircuitState.CLOSED:
            return True
            
        elif self.state == CircuitState.OPEN:
            # Check if should transition to half-open
            time_since_open = datetime.utcnow() - self.metrics.state_changed_at
            
            if time_since_open.total_seconds() > self._current_backoff:
                await self._transition_to(CircuitState.HALF_OPEN)
                return True
                
            return False
            
        else:  # HALF_OPEN
            # Allow limited calls for testing
            if self._half_open_calls < self.config.half_open_max_calls:
                self._half_open_calls += 1
                return True
            return False
            
    async def _record_success(self, start_time: float):
        """Record successful call"""
        latency_ms = (time.perf_counter() - start_time) * 1000
        
        async with self._state_lock:
            self.metrics.total_calls += 1
            self.metrics.success_count += 1
            self.metrics.last_success_time = datetime.utcnow()
            self.metrics.consecutive_failures = 0
            
            # Track latency
            self.metrics.total_latency_ms += latency_ms
            self.metrics.max_latency_ms = max(self.metrics.max_latency_ms, latency_ms)
            
            # State transitions
            if self.state == CircuitState.HALF_OPEN:
                if self._half_open_calls >= self.config.half_open_max_calls:
                    # All test calls succeeded
                    await self._transition_to(CircuitState.CLOSED)
                    self._current_backoff = self.config.recovery_timeout  # Reset backoff
                    
    async def _record_failure(self, error: Exception):
        """Record failed call"""
        async with self._state_lock:
            self.metrics.total_calls += 1
            self.metrics.failure_count += 1
            self.metrics.last_failure_time = datetime.utcnow()
            self.metrics.consecutive_failures += 1
            
            # State transitions
            if self.state == CircuitState.CLOSED:
                if await self._should_open():
                    await self._transition_to(CircuitState.OPEN)
                    
            elif self.state == CircuitState.HALF_OPEN:
                # Any failure in half-open goes back to open
                await self._transition_to(CircuitState.OPEN)
                # Increase backoff
                self._current_backoff = min(
                    self._current_backoff * self.config.backoff_factor,
                    self.config.max_backoff
                )
                
    async def _record_timeout(self):
        """Record timeout"""
        async with self._state_lock:
            self.metrics.total_calls += 1
            self.metrics.timeout_count += 1
            self.metrics.last_failure_time = datetime.utcnow()
            self.metrics.consecutive_failures += 1
            
            # Treat timeout as failure for state transitions
            if self.state in (CircuitState.CLOSED, CircuitState.HALF_OPEN):
                if await self._should_open():
                    await self._transition_to(CircuitState.OPEN)
                    
    async def _should_open(self) -> bool:
        """Check if breaker should open (must be called with lock)"""
        # Not enough calls
        if self.metrics.total_calls < self.config.min_calls_for_evaluation:
            # But check consecutive failures
            if self.metrics.consecutive_failures >= self.config.failure_threshold:
                return True
            return False
            
        # Check failure rate
        if self.metrics.failure_rate >= self.config.failure_rate_threshold:
            return True
            
        # Check absolute failure count
        if self.metrics.failure_count >= self.config.failure_threshold:
            return True
            
        return False
        
    async def _transition_to(self, new_state: CircuitState):
        """Transition to new state (must be called with lock)"""
        old_state = self.state
        self.state = new_state
        self.metrics.state_changed_at = datetime.utcnow()
        
        # Reset half-open counter
        if new_state == CircuitState.HALF_OPEN:
            self._half_open_calls = 0
            
        # Reset metrics on close
        if new_state == CircuitState.CLOSED:
            self.metrics.reset()
            
        logger.info(f"Circuit breaker {self.store_name}: {old_state.value} -> {new_state.value}")
        
    async def _handle_blocked_call(self, *args, **kwargs) -> T:
        """Handle call when circuit is open"""
        if self.fallback_func:
            logger.debug(f"Circuit open for {self.store_name}, using fallback")
            return await self.fallback_func(*args, **kwargs)
            
        raise RuntimeError(f"Circuit breaker {self.store_name} is OPEN")
        
    async def reset(self):
        """Manually reset the circuit breaker"""
        async with self._state_lock:
            await self._transition_to(CircuitState.CLOSED)
            self._current_backoff = self.config.recovery_timeout
            logger.info(f"Circuit breaker {self.store_name} manually reset")
            
    def get_status(self) -> Dict[str, Any]:
        """Get circuit breaker status"""
        return {
            'store_name': self.store_name,
            'state': self.state.value,
            'metrics': {
                'total_calls': self.metrics.total_calls,
                'success_count': self.metrics.success_count,
                'failure_count': self.metrics.failure_count,
                'timeout_count': self.metrics.timeout_count,
                'failure_rate': self.metrics.failure_rate,
                'avg_latency_ms': self.metrics.avg_latency_ms,
                'max_latency_ms': self.metrics.max_latency_ms,
                'consecutive_failures': self.metrics.consecutive_failures
            },
            'time_in_state': (datetime.utcnow() - self.metrics.state_changed_at).total_seconds(),
            'current_backoff': self._current_backoff if self.state == CircuitState.OPEN else 0
        }


class MultiStoreCircuitBreaker:
    """Manages circuit breakers for multiple stores"""
    
    def __init__(self, default_config: Optional[CircuitConfig] = None):
        self.default_config = default_config or CircuitConfig()
        self.breakers: Dict[str, StoreCircuitBreaker] = {}
        
    def get_or_create(self,
                     store_name: str,
                     config: Optional[CircuitConfig] = None,
                     fallback_func: Optional[Callable] = None) -> StoreCircuitBreaker:
        """Get existing or create new breaker"""
        if store_name not in self.breakers:
            breaker_config = config or self.default_config
            self.breakers[store_name] = StoreCircuitBreaker(
                store_name,
                breaker_config,
                fallback_func
            )
            
        return self.breakers[store_name]
        
    def get_all_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all breakers"""
        return {
            name: breaker.get_status()
            for name, breaker in self.breakers.items()
        }
        
    async def reset_all(self):
        """Reset all circuit breakers"""
        for breaker in self.breakers.values():
            await breaker.reset()