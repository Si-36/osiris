"""
Cognitive Circuit Breaker with Adaptive Trust Scoring
====================================================
Protects system components with trust-based fallback mechanisms
Based on CrashBytes 2025 resilience patterns
"""

import asyncio
from typing import Dict, List, Any, Optional, Callable, Tuple
from dataclasses import dataclass, field
import time
from enum import Enum
from collections import deque
import numpy as np
import math

# AURA imports
from ...components.registry import get_registry, ComponentRole

import logging
logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Blocking calls
    HALF_OPEN = "half_open"  # Testing recovery


@dataclass
class BreakerId:
    """Circuit breaker identifier"""
    component: str
    service: str
    
    def __str__(self):
        return f"{self.component}:{self.service}"


@dataclass
class CircuitMetrics:
    """Metrics for circuit breaker decisions"""
    success_count: int = 0
    failure_count: int = 0
    timeout_count: int = 0
    
    # Latency tracking
    latencies: deque = field(default_factory=lambda: deque(maxlen=100))
    
    # Error tracking
    error_types: Dict[str, int] = field(default_factory=dict)
    
    # Time windows
    last_failure_time: Optional[float] = None
    last_success_time: Optional[float] = None
    state_changed_at: float = field(default_factory=time.time)
    
    @property
    def total_calls(self) -> int:
        return self.success_count + self.failure_count + self.timeout_count
        
    @property
    def failure_rate(self) -> float:
        if self.total_calls == 0:
            return 0.0
        return (self.failure_count + self.timeout_count) / self.total_calls
        
    @property
    def avg_latency(self) -> float:
        if not self.latencies:
            return 0.0
        return np.mean(self.latencies)
        
    @property
    def latency_p95(self) -> float:
        if not self.latencies:
            return 0.0
        return np.percentile(self.latencies, 95)


class AdaptiveTrustScorer:
    """
    Calculates trust scores for components based on historical performance
    Uses exponential decay and Bayesian inference
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or self._default_config()
        
        # Trust scores per breaker
        self.trust_scores: Dict[str, float] = {}
        
        # Historical data
        self.history: Dict[str, deque] = {}
        
        logger.info("Adaptive Trust Scorer initialized")
        
    def _default_config(self) -> Dict[str, Any]:
        return {
            'initial_trust': 0.8,
            'decay_factor': 0.95,
            'recovery_factor': 1.05,
            'min_trust': 0.1,
            'max_trust': 0.99,
            'history_window': 1000
        }
        
    def update_trust(self, 
                    breaker_id: str,
                    success: bool,
                    latency_ms: float,
                    expected_latency_ms: float = 100.0) -> float:
        """Update trust score based on outcome"""
        # Initialize if needed
        if breaker_id not in self.trust_scores:
            self.trust_scores[breaker_id] = self.config['initial_trust']
            self.history[breaker_id] = deque(maxlen=self.config['history_window'])
            
        current_trust = self.trust_scores[breaker_id]
        
        # Calculate performance score
        if success:
            # Latency-aware success score
            latency_ratio = latency_ms / expected_latency_ms
            if latency_ratio < 1.0:
                performance = 1.0  # Better than expected
            elif latency_ratio < 2.0:
                performance = 2.0 - latency_ratio  # Linear decay
            else:
                performance = 0.5  # Still success but slow
                
            # Update trust positively
            new_trust = current_trust * self.config['recovery_factor'] * performance
        else:
            # Failure decreases trust
            new_trust = current_trust * self.config['decay_factor']
            
        # Apply bounds
        new_trust = np.clip(
            new_trust,
            self.config['min_trust'],
            self.config['max_trust']
        )
        
        # Store update
        self.trust_scores[breaker_id] = new_trust
        self.history[breaker_id].append({
            'timestamp': time.time(),
            'success': success,
            'latency_ms': latency_ms,
            'trust_before': current_trust,
            'trust_after': new_trust
        })
        
        return new_trust
        
    def get_trust_score(self, breaker_id: str) -> float:
        """Get current trust score"""
        return self.trust_scores.get(breaker_id, self.config['initial_trust'])
        
    def should_allow_request(self, breaker_id: str) -> bool:
        """Probabilistic decision based on trust"""
        trust = self.get_trust_score(breaker_id)
        
        # Always allow if trust is high
        if trust > 0.9:
            return True
            
        # Never allow if trust is too low
        if trust < self.config['min_trust']:
            return False
            
        # Probabilistic decision
        return np.random.random() < trust
        
    def get_recovery_suggestions(self, breaker_id: str) -> Dict[str, Any]:
        """Suggest recovery strategies based on history"""
        if breaker_id not in self.history:
            return {'suggestion': 'No history available'}
            
        recent_history = list(self.history[breaker_id])[-20:]
        
        if not recent_history:
            return {'suggestion': 'Insufficient data'}
            
        # Analyze failure patterns
        failures = [h for h in recent_history if not h['success']]
        
        if len(failures) > len(recent_history) * 0.8:
            # Persistent failures
            return {
                'suggestion': 'persistent_failure',
                'recommended_action': 'extended_cooldown',
                'cooldown_minutes': 30
            }
        elif failures and all(h['latency_ms'] > 1000 for h in failures):
            # Timeout issues
            return {
                'suggestion': 'timeout_issues',
                'recommended_action': 'increase_timeout',
                'new_timeout_ms': 2000
            }
        else:
            # Intermittent issues
            return {
                'suggestion': 'intermittent_failure',
                'recommended_action': 'gradual_recovery',
                'test_rate': 0.1
            }


class CognitiveCircuitBreaker:
    """
    Adaptive circuit breaker with cognitive load awareness
    Implements trust-based recovery and intelligent fallbacks
    """
    
    def __init__(self, 
                 breaker_id: BreakerId,
                 config: Optional[Dict[str, Any]] = None):
        self.breaker_id = breaker_id
        self.config = config or self._default_config()
        
        # State management
        self.state = CircuitState.CLOSED
        self.metrics = CircuitMetrics()
        
        # Trust scoring
        self.trust_scorer = AdaptiveTrustScorer()
        
        # Fallback functions
        self.fallback_func: Optional[Callable] = None
        self.enhanced_fallback_func: Optional[Callable] = None
        
        # State transition callbacks
        self.state_listeners: List[Callable] = []
        
        # Component registry
        self.registry = get_registry()
        
        logger.info(f"Circuit breaker initialized for {breaker_id}")
        
    def _default_config(self) -> Dict[str, Any]:
        return {
            'failure_threshold': 5,
            'failure_rate_threshold': 0.5,
            'timeout_seconds': 30.0,
            'half_open_max_calls': 3,
            'recovery_timeout': 60.0,
            'min_calls_for_evaluation': 10
        }
        
    async def call(self,
                  func: Callable,
                  *args,
                  **kwargs) -> Any:
        """Execute function with circuit breaker protection"""
        # Check if should allow call
        if not self._should_allow_call():
            return await self._handle_blocked_call(*args, **kwargs)
            
        # Execute with monitoring
        start_time = time.perf_counter()
        
        try:
            # Execute with timeout
            result = await asyncio.wait_for(
                func(*args, **kwargs),
                timeout=self.config['timeout_seconds']
            )
            
            # Record success
            latency_ms = (time.perf_counter() - start_time) * 1000
            self._record_success(latency_ms)
            
            return result
            
        except asyncio.TimeoutError:
            # Record timeout
            self._record_timeout()
            
            if self.fallback_func:
                return await self.fallback_func(*args, **kwargs)
            raise
            
        except Exception as e:
            # Record failure
            self._record_failure(e)
            
            if self.fallback_func:
                return await self.fallback_func(*args, **kwargs)
            raise
            
    def _should_allow_call(self) -> bool:
        """Determine if call should be allowed"""
        if self.state == CircuitState.CLOSED:
            return True
            
        elif self.state == CircuitState.OPEN:
            # Check if should transition to half-open
            time_since_open = time.time() - self.metrics.state_changed_at
            
            if time_since_open > self.config['recovery_timeout']:
                # Use trust score for recovery decision
                if self.trust_scorer.should_allow_request(str(self.breaker_id)):
                    self._transition_to(CircuitState.HALF_OPEN)
                    return True
                    
            return False
            
        else:  # HALF_OPEN
            # Allow limited calls for testing
            recent_calls = (
                self.metrics.success_count + 
                self.metrics.failure_count + 
                self.metrics.timeout_count
            ) - self._state_start_calls
            
            return recent_calls < self.config['half_open_max_calls']
            
    def _record_success(self, latency_ms: float):
        """Record successful call"""
        self.metrics.success_count += 1
        self.metrics.latencies.append(latency_ms)
        self.metrics.last_success_time = time.time()
        
        # Update trust
        trust = self.trust_scorer.update_trust(
            str(self.breaker_id),
            success=True,
            latency_ms=latency_ms
        )
        
        # State transitions
        if self.state == CircuitState.HALF_OPEN:
            if self._should_close():
                self._transition_to(CircuitState.CLOSED)
                
    def _record_failure(self, error: Exception):
        """Record failed call"""
        self.metrics.failure_count += 1
        self.metrics.last_failure_time = time.time()
        
        # Track error types
        error_type = type(error).__name__
        self.metrics.error_types[error_type] = (
            self.metrics.error_types.get(error_type, 0) + 1
        )
        
        # Update trust
        self.trust_scorer.update_trust(
            str(self.breaker_id),
            success=False,
            latency_ms=float('inf')
        )
        
        # State transitions
        if self.state == CircuitState.CLOSED:
            if self._should_open():
                self._transition_to(CircuitState.OPEN)
        elif self.state == CircuitState.HALF_OPEN:
            self._transition_to(CircuitState.OPEN)
            
    def _record_timeout(self):
        """Record timeout"""
        self.metrics.timeout_count += 1
        self.metrics.last_failure_time = time.time()
        
        # Update trust (timeouts are bad)
        self.trust_scorer.update_trust(
            str(self.breaker_id),
            success=False,
            latency_ms=self.config['timeout_seconds'] * 1000
        )
        
        # State transitions
        if self.state in (CircuitState.CLOSED, CircuitState.HALF_OPEN):
            if self._should_open():
                self._transition_to(CircuitState.OPEN)
                
    def _should_open(self) -> bool:
        """Check if breaker should open"""
        # Not enough calls
        if self.metrics.total_calls < self.config['min_calls_for_evaluation']:
            return False
            
        # Check failure count
        recent_failures = self.metrics.failure_count + self.metrics.timeout_count
        if recent_failures >= self.config['failure_threshold']:
            return True
            
        # Check failure rate
        if self.metrics.failure_rate >= self.config['failure_rate_threshold']:
            return True
            
        return False
        
    def _should_close(self) -> bool:
        """Check if breaker should close (from half-open)"""
        # Need successful calls in half-open state
        calls_in_state = (
            self.metrics.success_count - self._state_start_calls
        )
        
        # All test calls succeeded
        return calls_in_state >= self.config['half_open_max_calls']
        
    def _transition_to(self, new_state: CircuitState):
        """Transition to new state"""
        old_state = self.state
        self.state = new_state
        self.metrics.state_changed_at = time.time()
        
        # Track metrics at state change
        self._state_start_calls = self.metrics.total_calls
        
        logger.info(f"Circuit breaker {self.breaker_id}: {old_state.value} -> {new_state.value}")
        
        # Notify listeners
        for listener in self.state_listeners:
            try:
                listener(self.breaker_id, old_state, new_state)
            except Exception as e:
                logger.error(f"State listener error: {e}")
                
    async def _handle_blocked_call(self, *args, **kwargs) -> Any:
        """Handle call when circuit is open"""
        # Try enhanced fallback first
        if self.enhanced_fallback_func:
            trust = self.trust_scorer.get_trust_score(str(self.breaker_id))
            
            # Use enhanced fallback if trust is moderate
            if trust > 0.3:
                try:
                    return await self.enhanced_fallback_func(*args, **kwargs)
                except Exception as e:
                    logger.warning(f"Enhanced fallback failed: {e}")
                    
        # Use basic fallback
        if self.fallback_func:
            return await self.fallback_func(*args, **kwargs)
            
        # No fallback available
        raise RuntimeError(f"Circuit breaker {self.breaker_id} is OPEN")
        
    def set_fallback(self, 
                    fallback: Callable,
                    enhanced_fallback: Optional[Callable] = None):
        """Set fallback functions"""
        self.fallback_func = fallback
        self.enhanced_fallback_func = enhanced_fallback
        
    def add_state_listener(self, listener: Callable):
        """Add state change listener"""
        self.state_listeners.append(listener)
        
    def get_status(self) -> Dict[str, Any]:
        """Get circuit breaker status"""
        trust_score = self.trust_scorer.get_trust_score(str(self.breaker_id))
        recovery_suggestions = self.trust_scorer.get_recovery_suggestions(
            str(self.breaker_id)
        )
        
        return {
            'breaker_id': str(self.breaker_id),
            'state': self.state.value,
            'metrics': {
                'total_calls': self.metrics.total_calls,
                'success_count': self.metrics.success_count,
                'failure_count': self.metrics.failure_count,
                'timeout_count': self.metrics.timeout_count,
                'failure_rate': self.metrics.failure_rate,
                'avg_latency_ms': self.metrics.avg_latency,
                'p95_latency_ms': self.metrics.latency_p95
            },
            'trust_score': trust_score,
            'recovery_suggestions': recovery_suggestions,
            'time_in_state': time.time() - self.metrics.state_changed_at
        }


class CircuitBreakerManager:
    """Manages multiple circuit breakers across system"""
    
    def __init__(self):
        self.breakers: Dict[str, CognitiveCircuitBreaker] = {}
        self.global_config = {}
        
    def get_or_create(self,
                     breaker_id: BreakerId,
                     config: Optional[Dict[str, Any]] = None) -> CognitiveCircuitBreaker:
        """Get existing or create new breaker"""
        key = str(breaker_id)
        
        if key not in self.breakers:
            breaker_config = {**self.global_config, **(config or {})}
            self.breakers[key] = CognitiveCircuitBreaker(breaker_id, breaker_config)
            
        return self.breakers[key]
        
    def get_all_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all breakers"""
        return {
            breaker_id: breaker.get_status()
            for breaker_id, breaker in self.breakers.items()
        }
        
    def get_open_breakers(self) -> List[str]:
        """Get list of open circuit breakers"""
        return [
            breaker_id
            for breaker_id, breaker in self.breakers.items()
            if breaker.state == CircuitState.OPEN
        ]