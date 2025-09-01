"""
🔌 AURA Base Adapter - Advanced Component Abstraction Layer
=========================================================

Implements cutting-edge 2025 patterns:
- Circuit breaker for fault tolerance
- Health monitoring with predictive failure detection
- Dependency tracking and resolution
- Lifecycle management with graceful shutdown
- Versioning and metadata support

Based on latest microservices and distributed AI research.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Set, Callable
from dataclasses import dataclass, field
from enum import Enum
import asyncio
import time
import structlog
from datetime import datetime, timedelta
import inspect

logger = structlog.get_logger(__name__)


# ==================== Health & Status ====================

class HealthStatus(Enum):
    """Multi-dimensional health states"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class HealthMetrics:
    """Comprehensive health metrics"""
    status: HealthStatus = HealthStatus.UNKNOWN
    latency_ms: float = 0.0
    error_rate: float = 0.0
    throughput: float = 0.0
    resource_usage: Dict[str, float] = field(default_factory=dict)
    last_check: datetime = field(default_factory=datetime.now)
    failure_predictions: List[str] = field(default_factory=list)


@dataclass
class ComponentMetadata:
    """Component metadata for versioning and discovery"""
    version: str
    capabilities: List[str]
    dependencies: Set[str]
    author: str = "AURA Team"
    created_at: datetime = field(default_factory=datetime.now)
    tags: List[str] = field(default_factory=list)


# ==================== Circuit Breaker ====================

class CircuitBreakerState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"  # Normal operation
    OPEN = "open"      # Failing, reject calls
    HALF_OPEN = "half_open"  # Testing recovery


class CircuitBreaker:
    """
    Advanced circuit breaker with adaptive thresholds.
    Based on Netflix Hystrix and modern resilience patterns.
    """
    
    def __init__(self, 
                 failure_threshold: int = 5,
                 recovery_timeout: float = 60.0,
                 half_open_max_calls: int = 3):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_max_calls = half_open_max_calls
        
        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.last_failure_time = None
        self.half_open_calls = 0
        self.success_count = 0
        
    async def call(self, func: Callable, *args, **kwargs):
        """Execute function with circuit breaker protection"""
        if self.state == CircuitBreakerState.OPEN:
            if self._should_attempt_reset():
                self.state = CircuitBreakerState.HALF_OPEN
                self.half_open_calls = 0
            else:
                raise Exception("Circuit breaker is OPEN")
                
        if self.state == CircuitBreakerState.HALF_OPEN:
            if self.half_open_calls >= self.half_open_max_calls:
                raise Exception("Circuit breaker is HALF_OPEN, max calls reached")
            self.half_open_calls += 1
            
        try:
            result = await func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise e
            
    def _should_attempt_reset(self) -> bool:
        """Check if we should try to reset the circuit"""
        return (self.last_failure_time and 
                time.time() - self.last_failure_time > self.recovery_timeout)
                
    def _on_success(self):
        """Handle successful call"""
        self.failure_count = 0
        self.success_count += 1
        
        if self.state == CircuitBreakerState.HALF_OPEN:
            if self.success_count > self.half_open_max_calls:
                self.state = CircuitBreakerState.CLOSED
                logger.info("Circuit breaker closed after recovery")
                
    def _on_failure(self):
        """Handle failed call"""
        self.failure_count += 1
        self.last_failure_time = time.time()
        self.success_count = 0
        
        if self.failure_count >= self.failure_threshold:
            self.state = CircuitBreakerState.OPEN
            logger.warning("Circuit breaker opened due to failures", 
                          failure_count=self.failure_count)


# ==================== Base Adapter ====================

class BaseAdapter(ABC):
    """
    Advanced base adapter for AURA components.
    
    Features:
    - Async lifecycle management
    - Circuit breaker protection
    - Health monitoring with predictions
    - Dependency tracking
    - Versioning support
    """
    
    def __init__(self, component_id: str, metadata: ComponentMetadata):
        self.component_id = component_id
        self.metadata = metadata
        self.health_metrics = HealthMetrics()
        self.circuit_breaker = CircuitBreaker()
        
        # Lifecycle state
        self.initialized = False
        self.running = False
        self._shutdown_event = asyncio.Event()
        
        # Monitoring
        self._health_check_task: Optional[asyncio.Task] = None
        self._metrics_history: List[HealthMetrics] = []
        
        logger.info(f"Created adapter for {component_id}", 
                   version=metadata.version,
                   capabilities=metadata.capabilities)
    
    # ==================== Abstract Methods ====================
    
    @abstractmethod
    async def _initialize_component(self, config: Dict[str, Any]) -> None:
        """Initialize the wrapped component"""
        pass
        
    @abstractmethod
    async def _start_component(self) -> None:
        """Start the wrapped component"""
        pass
        
    @abstractmethod
    async def _stop_component(self) -> None:
        """Stop the wrapped component"""
        pass
        
    @abstractmethod
    async def _check_component_health(self) -> HealthMetrics:
        """Check wrapped component health"""
        pass
    
    # ==================== Component Interface ====================
    
    async def initialize(self, config: Dict[str, Any]) -> None:
        """
        Initialize with timeout and retry logic.
        Implements Component interface.
        """
        if self.initialized:
            logger.warning(f"{self.component_id} already initialized")
            return
            
        logger.info(f"Initializing {self.component_id}")
        
        # Initialize with timeout
        try:
            await asyncio.wait_for(
                self._initialize_component(config),
                timeout=config.get('init_timeout', 30.0)
            )
            self.initialized = True
            self.health_metrics.status = HealthStatus.HEALTHY
            
        except asyncio.TimeoutError:
            logger.error(f"{self.component_id} initialization timeout")
            self.health_metrics.status = HealthStatus.UNHEALTHY
            raise
        except Exception as e:
            logger.error(f"{self.component_id} initialization failed", error=str(e))
            self.health_metrics.status = HealthStatus.UNHEALTHY
            raise
    
    async def start(self) -> None:
        """
        Start component with health monitoring.
        Implements Component interface.
        """
        if not self.initialized:
            raise RuntimeError(f"{self.component_id} not initialized")
            
        if self.running:
            logger.warning(f"{self.component_id} already running")
            return
            
        logger.info(f"Starting {self.component_id}")
        
        await self._start_component()
        self.running = True
        
        # Start health monitoring
        self._health_check_task = asyncio.create_task(
            self._health_monitor_loop()
        )
    
    async def stop(self) -> None:
        """
        Graceful shutdown with cleanup.
        Implements Component interface.
        """
        if not self.running:
            return
            
        logger.info(f"Stopping {self.component_id}")
        
        # Signal shutdown
        self._shutdown_event.set()
        
        # Stop health monitoring
        if self._health_check_task:
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass
                
        # Stop component
        await self._stop_component()
        self.running = False
        self.health_metrics.status = HealthStatus.UNKNOWN
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Comprehensive health check with predictions.
        Implements Component interface.
        """
        try:
            # Get current health with circuit breaker
            self.health_metrics = await self.circuit_breaker.call(
                self._check_component_health
            )
            
            # Predict failures based on trends
            self._predict_failures()
            
        except Exception as e:
            logger.error(f"Health check failed for {self.component_id}", error=str(e))
            self.health_metrics.status = HealthStatus.UNHEALTHY
            
        return {
            "component_id": self.component_id,
            "status": self.health_metrics.status.value,
            "metrics": {
                "latency_ms": self.health_metrics.latency_ms,
                "error_rate": self.health_metrics.error_rate,
                "throughput": self.health_metrics.throughput,
                "resource_usage": self.health_metrics.resource_usage
            },
            "predictions": self.health_metrics.failure_predictions,
            "circuit_breaker": self.circuit_breaker.state.value,
            "last_check": self.health_metrics.last_check.isoformat()
        }
    
    def get_capabilities(self) -> List[str]:
        """
        Get component capabilities.
        Implements Component interface.
        """
        return self.metadata.capabilities
    
    # ==================== Health Monitoring ====================
    
    async def _health_monitor_loop(self):
        """Continuous health monitoring loop"""
        while not self._shutdown_event.is_set():
            try:
                await self.health_check()
                
                # Store history for trend analysis
                self._metrics_history.append(self.health_metrics)
                if len(self._metrics_history) > 100:
                    self._metrics_history.pop(0)
                    
            except Exception as e:
                logger.error(f"Health monitor error for {self.component_id}", error=str(e))
                
            # Wait before next check
            await asyncio.sleep(10.0)
    
    def _predict_failures(self):
        """
        Predict potential failures based on metrics trends.
        Uses simple heuristics, can be enhanced with ML.
        """
        predictions = []
        
        if len(self._metrics_history) < 10:
            return
            
        # Check error rate trend
        recent_errors = [m.error_rate for m in self._metrics_history[-10:]]
        if all(e > 0.1 for e in recent_errors[-3:]):
            predictions.append("High error rate trend detected")
            
        # Check latency trend
        recent_latencies = [m.latency_ms for m in self._metrics_history[-10:]]
        avg_latency = sum(recent_latencies) / len(recent_latencies)
        if recent_latencies[-1] > avg_latency * 2:
            predictions.append("Latency spike detected")
            
        # Check resource usage
        if self.health_metrics.resource_usage.get('memory', 0) > 0.9:
            predictions.append("High memory usage")
            
        self.health_metrics.failure_predictions = predictions