"""
Event Router - High-Performance Event Routing

Real-time event routing with TDA-aware decision making.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Callable, Set
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import json

try:
    from .event_patterns import PatternMatch, PatternPriority
except ImportError:
    # Fallback for standalone testing
    from event_patterns import PatternMatch, PatternPriority

logger = logging.getLogger(__name__)

class RoutingStrategy(Enum):
    """Event routing strategies"""
    ROUND_ROBIN = "round_robin"
    PRIORITY_BASED = "priority_based"
    LOAD_BALANCED = "load_balanced"
    TDA_AWARE = "tda_aware"

@dataclass
class EventHandler:
    """Event handler registration"""
    handler_id: str
    handler_func: Callable
    event_types: Set[str]
    priority: PatternPriority = PatternPriority.NORMAL
    max_concurrent: int = 10
    timeout_seconds: int = 30

@dataclass
class RoutingMetrics:
    """Routing performance metrics"""
    total_events: int = 0
    successful_routes: int = 0
    failed_routes: int = 0
    avg_latency_ms: float = 0.0
    handler_load: Dict[str, int] = field(default_factory=dict)

class EventRouter:
    """High-performance event router with TDA integration (40 lines)"""
    
    def __init__(self, strategy: RoutingStrategy = RoutingStrategy.TDA_AWARE):
        self.strategy = strategy
        self.handlers: Dict[str, EventHandler] = {}
        self.active_tasks: Dict[str, Set[asyncio.Task]] = {}
        self.metrics = RoutingMetrics()
        self.routing_queue: asyncio.Queue = asyncio.Queue()
        
        # TDA-aware routing
        self.tda_weights: Dict[str, float] = {}
        self.anomaly_threshold = 0.7
        
        # Performance optimization
        self.batch_size = 50
        self.batch_timeout = 0.1  # 100ms
        
        logger.info(f"Event router initialized with {strategy.value} strategy")
    
    def register_handler(self, handler: EventHandler) -> None:
        """Register event handler"""
        self.handlers[handler.handler_id] = handler
        self.active_tasks[handler.handler_id] = set()
        self.metrics.handler_load[handler.handler_id] = 0
        
        logger.info(f"Registered handler {handler.handler_id} for {len(handler.event_types)} event types")
    
    async def route_event(self, event: Dict[str, Any], 
                         pattern_matches: List[PatternMatch] = None) -> bool:
        """Route single event to appropriate handlers"""
        start_time = datetime.utcnow()
        self.metrics.total_events += 1
        
        try:
            # Determine routing based on strategy
            selected_handlers = self._select_handlers(event, pattern_matches)
            
            if not selected_handlers:
                logger.warning(f"No handlers found for event type: {event.get('type', 'unknown')}")
                return False
            
            # Route to selected handlers
            tasks = []
            for handler_id in selected_handlers:
                handler = self.handlers[handler_id]
                
                # Check concurrency limits
                if len(self.active_tasks[handler_id]) >= handler.max_concurrent:
                    logger.warning(f"Handler {handler_id} at max concurrency")
                    continue
                
                # Create handler task
                task = asyncio.create_task(
                    self._execute_handler(handler, event, pattern_matches)
                )
                tasks.append(task)
                self.active_tasks[handler_id].add(task)
            
            # Wait for all handlers (with timeout)
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
            
            # Update metrics
            latency = (datetime.utcnow() - start_time).total_seconds() * 1000
            self.metrics.successful_routes += 1
            self._update_latency_metric(latency)
            
            return True
            
        except Exception as e:
            logger.error(f"Event routing failed: {e}")
            self.metrics.failed_routes += 1
            return False
    
    def _select_handlers(self, event: Dict[str, Any], 
                        pattern_matches: List[PatternMatch] = None) -> List[str]:
        """Select handlers based on routing strategy"""
        event_type = event.get('type', '')
        
        # Find eligible handlers
        eligible_handlers = [
            handler_id for handler_id, handler in self.handlers.items()
            if event_type in handler.event_types or '*' in handler.event_types
        ]
        
        if not eligible_handlers:
            return []
        
        if self.strategy == RoutingStrategy.ROUND_ROBIN:
            return [eligible_handlers[self.metrics.total_events % len(eligible_handlers)]]
        
        elif self.strategy == RoutingStrategy.PRIORITY_BASED:
            # Sort by priority and pattern matches
            priority_scores = {}
            for handler_id in eligible_handlers:
                handler = self.handlers[handler_id]
                score = handler.priority.value  # Lower is higher priority
                
                # Boost score if pattern matches
                if pattern_matches:
                    for match in pattern_matches:
                        if match.pattern_id.startswith(handler_id):
                            score -= match.confidence
                
                priority_scores[handler_id] = score
            
            # Return highest priority handler
            best_handler = min(priority_scores.keys(), key=lambda h: priority_scores[h])
            return [best_handler]
        
        elif self.strategy == RoutingStrategy.LOAD_BALANCED:
            # Select handler with lowest current load
            load_scores = {
                handler_id: len(self.active_tasks[handler_id])
                for handler_id in eligible_handlers
            }
            best_handler = min(load_scores.keys(), key=lambda h: load_scores[h])
            return [best_handler]
        
        elif self.strategy == RoutingStrategy.TDA_AWARE:
            return self._tda_aware_selection(event, eligible_handlers, pattern_matches)
        
        return eligible_handlers[:1]  # Default: first eligible
    
    def _tda_aware_selection(self, event: Dict[str, Any], 
                           eligible_handlers: List[str],
                           pattern_matches: List[PatternMatch] = None) -> List[str]:
        """TDA-aware handler selection"""
        tda_score = event.get('tda_anomaly_score', 0.0)
        
        # High anomaly events get priority routing
        if tda_score >= self.anomaly_threshold:
            # Route to critical handlers first
            critical_handlers = [
                h for h in eligible_handlers
                if self.handlers[h].priority == PatternPriority.CRITICAL
            ]
            if critical_handlers:
                return critical_handlers
        
        # Normal TDA-aware routing
        handler_scores = {}
        for handler_id in eligible_handlers:
            handler = self.handlers[handler_id]
            
            # Base score from priority
            score = 1.0 / handler.priority.value
            
            # TDA weight adjustment
            tda_weight = self.tda_weights.get(handler_id, 1.0)
            score *= tda_weight
            
            # Pattern match boost
            if pattern_matches:
                for match in pattern_matches:
                    if match.tda_correlation:
                        score *= (1.0 + match.confidence)
            
            # Load balancing factor
            current_load = len(self.active_tasks[handler_id])
            max_load = handler.max_concurrent
            load_factor = 1.0 - (current_load / max_load)
            score *= load_factor
            
            handler_scores[handler_id] = score
        
        # Return top handler
        best_handler = max(handler_scores.keys(), key=lambda h: handler_scores[h])
        return [best_handler]
    
    async def _execute_handler(self, handler: EventHandler, event: Dict[str, Any],
                             pattern_matches: List[PatternMatch] = None) -> None:
        """Execute event handler with timeout"""
        task = asyncio.current_task()
        
        try:
            # Execute handler with timeout
            await asyncio.wait_for(
                handler.handler_func(event, pattern_matches),
                timeout=handler.timeout_seconds
            )
            
            # Update load metrics
            self.metrics.handler_load[handler.handler_id] += 1
            
        except asyncio.TimeoutError:
            logger.warning(f"Handler {handler.handler_id} timed out")
        except Exception as e:
            logger.error(f"Handler {handler.handler_id} failed: {e}")
        finally:
            # Clean up task tracking
            if task and task in self.active_tasks[handler.handler_id]:
                self.active_tasks[handler.handler_id].remove(task)
    
    def update_tda_weights(self, weights: Dict[str, float]) -> None:
        """Update TDA-based handler weights"""
        self.tda_weights.update(weights)
        logger.debug(f"Updated TDA weights for {len(weights)} handlers")
    
    def _update_latency_metric(self, latency_ms: float) -> None:
        """Update average latency metric"""
        if self.metrics.successful_routes == 1:
            self.metrics.avg_latency_ms = latency_ms
        else:
            # Exponential moving average
            alpha = 0.1
            self.metrics.avg_latency_ms = (
                alpha * latency_ms + 
                (1 - alpha) * self.metrics.avg_latency_ms
            )
    
    def get_routing_stats(self) -> Dict[str, Any]:
        """Get routing performance statistics"""
        total_events = self.metrics.total_events
        success_rate = (
            self.metrics.successful_routes / max(1, total_events) * 100
        )
        
        return {
            'strategy': self.strategy.value,
            'total_events': total_events,
            'success_rate': f"{success_rate:.1f}%",
            'avg_latency_ms': f"{self.metrics.avg_latency_ms:.2f}",
            'active_handlers': len(self.handlers),
            'handler_load': dict(self.metrics.handler_load),
            'current_active_tasks': {
                h: len(tasks) for h, tasks in self.active_tasks.items()
            }
        }