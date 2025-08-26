"""
Auto-Scaling System

TDA-aware automatic scaling based on load patterns and anomaly detection.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json

logger = logging.getLogger(__name__)

class ScalingAction(Enum):
    """Scaling actions"""
    SCALE_UP = "scale_up"
    SCALE_DOWN = "scale_down"
    NO_ACTION = "no_action"

class ScalingTrigger(Enum):
    """Scaling triggers"""
    CPU_UTILIZATION = "cpu_utilization"
    MEMORY_UTILIZATION = "memory_utilization"
    REQUEST_RATE = "request_rate"
    RESPONSE_TIME = "response_time"
    TDA_ANOMALY = "tda_anomaly"
    QUEUE_LENGTH = "queue_length"

@dataclass
class ScalingRule:
    """Auto-scaling rule definition"""
    rule_id: str
    trigger: ScalingTrigger
    threshold_up: float
    threshold_down: float
    action: ScalingAction
    cooldown_minutes: int = 5
    min_instances: int = 1
    max_instances: int = 10
    enabled: bool = True

@dataclass
class ScalingEvent:
    """Scaling event record"""
    event_id: str
    trigger: ScalingTrigger
    action: ScalingAction
    from_instances: int
    to_instances: int
    reason: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    tda_context: Optional[Dict[str, Any]] = None

class AutoScaler:
    """
    TDA-aware auto-scaling system.
    
    Automatically scales orchestration components based on load and anomalies.
    """
    
    def __init__(self, tda_integration: Optional[Any] = None):
        self.tda_integration = tda_integration
        self.scaling_rules: Dict[str, ScalingRule] = {}
        self.scaling_history: List[ScalingEvent] = []
        
        # Current state
        self.current_instances: Dict[str, int] = {}
        self.last_scaling_action: Dict[str, datetime] = {}
        
        # Scaling state
        self.is_scaling_enabled = True
        self.scaling_task: Optional[asyncio.Task] = None
        
        # Metrics tracking
        self.metrics: Dict[str, List[Dict[str, Any]]] = {}
        self.scaling_callbacks: List[Callable] = []
        
        # TDA-aware scaling
        self.tda_anomaly_threshold = 0.8
        self.tda_scaling_multiplier = 1.5
        
        self._setup_default_rules()
        
        logger.info("Auto Scaler initialized")
    
    def _setup_default_rules(self) -> None:
        """Setup default scaling rules"""
        pass
        default_rules = [
            ScalingRule(
                rule_id="cpu_scaling",
                trigger=ScalingTrigger.CPU_UTILIZATION,
                threshold_up=70.0,  # Scale up at 70% CPU
                threshold_down=30.0,  # Scale down at 30% CPU
                action=ScalingAction.SCALE_UP,
                cooldown_minutes=5,
                min_instances=2,
                max_instances=20
            ),
            ScalingRule(
                rule_id="memory_scaling",
                trigger=ScalingTrigger.MEMORY_UTILIZATION,
                threshold_up=80.0,  # Scale up at 80% memory
                threshold_down=40.0,  # Scale down at 40% memory
                action=ScalingAction.SCALE_UP,
                cooldown_minutes=5,
                min_instances=2,
                max_instances=15
            ),
            ScalingRule(
                rule_id="request_rate_scaling",
                trigger=ScalingTrigger.REQUEST_RATE,
                threshold_up=100.0,  # Scale up at 100 req/s
                threshold_down=20.0,  # Scale down at 20 req/s
                action=ScalingAction.SCALE_UP,
                cooldown_minutes=3,
                min_instances=1,
                max_instances=25
            ),
            ScalingRule(
                rule_id="response_time_scaling",
                trigger=ScalingTrigger.RESPONSE_TIME,
                threshold_up=200.0,  # Scale up at 200ms
                threshold_down=50.0,  # Scale down at 50ms
                action=ScalingAction.SCALE_UP,
                cooldown_minutes=2,
                min_instances=2,
                max_instances=30
            ),
            ScalingRule(
                rule_id="tda_anomaly_scaling",
                trigger=ScalingTrigger.TDA_ANOMALY,
                threshold_up=0.8,  # Scale up at 80% anomaly score
                threshold_down=0.3,  # Scale down at 30% anomaly score
                action=ScalingAction.SCALE_UP,
                cooldown_minutes=1,  # Fast response to anomalies
                min_instances=3,
                max_instances=50  # Allow aggressive scaling for anomalies
            )
        ]
        
        for rule in default_rules:
            self.scaling_rules[rule.rule_id] = rule
    
        async def start_scaling(self) -> None:
        """Start auto-scaling monitoring"""
        pass
        if self.scaling_task:
            return
        
        self.is_scaling_enabled = True
        self.scaling_task = asyncio.create_task(self._scaling_loop())
        
        logger.info("Auto-scaling started")
    
        async def stop_scaling(self) -> None:
        """Stop auto-scaling monitoring"""
        pass
        self.is_scaling_enabled = False
        
        if self.scaling_task:
            self.scaling_task.cancel()
            try:
                await self.scaling_task
            except asyncio.CancelledError:
        pass
        
        logger.info("Auto-scaling stopped")
    
    def add_scaling_rule(self, rule: ScalingRule) -> None:
        """Add custom scaling rule"""
        self.scaling_rules[rule.rule_id] = rule
        logger.info(f"Added scaling rule: {rule.rule_id}")
    
    def add_scaling_callback(self, callback: Callable) -> None:
        """Add callback for scaling events"""
        self.scaling_callbacks.append(callback)
    
    def record_metric(self, trigger: ScalingTrigger, value: float,
        component: str = "default") -> None:
        """Record metric for scaling decisions"""
        metric_key = f"{component}:{trigger.value}"
        
        if metric_key not in self.metrics:
            self.metrics[metric_key] = []
        
        self.metrics[metric_key].append({
            'value': value,
            'timestamp': datetime.utcnow()
        })
        
        # Keep only recent metrics (last hour)
        cutoff_time = datetime.utcnow() - timedelta(hours=1)
        self.metrics[metric_key] = [
            m for m in self.metrics[metric_key]
            if m['timestamp'] > cutoff_time
        ]
    
        async def _scaling_loop(self) -> None:
        """Main scaling monitoring loop"""
        pass
        while self.is_scaling_enabled:
        try:
            # Check all scaling rules
        for rule in self.scaling_rules.values():
        if rule.enabled:
            await self._evaluate_scaling_rule(rule)
                
        # Wait before next evaluation
        await asyncio.sleep(30)  # Check every 30 seconds
                
        except Exception as e:
        logger.error(f"Scaling loop error: {e}")
        await asyncio.sleep(60)  # Back off on error
    
        async def _evaluate_scaling_rule(self, rule: ScalingRule) -> None:
        """Evaluate single scaling rule"""
        # Check cooldown period
        if rule.rule_id in self.last_scaling_action:
            last_action = self.last_scaling_action[rule.rule_id]
            cooldown_period = timedelta(minutes=rule.cooldown_minutes)
            
            if datetime.utcnow() - last_action < cooldown_period:
                return  # Still in cooldown
        
        # Get current metric value
        current_value = self._get_current_metric_value(rule.trigger)
        if current_value is None:
            return
        
        # Determine scaling action
        scaling_action = self._determine_scaling_action(rule, current_value)
        
        if scaling_action != ScalingAction.NO_ACTION:
            await self._execute_scaling_action(rule, scaling_action, current_value)
    
    def _get_current_metric_value(self, trigger: ScalingTrigger) -> Optional[float]:
        """Get current value for scaling trigger"""
        # Look for metrics from any component
        relevant_metrics = []
        
        for metric_key, metric_history in self.metrics.items():
        if trigger.value in metric_key and metric_history:
            # Get recent average (last 5 minutes)
        cutoff_time = datetime.utcnow() - timedelta(minutes=5)
        recent_values = [
        m['value'] for m in metric_history
        if m['timestamp'] > cutoff_time
        ]
                
        if recent_values:
            avg_value = sum(recent_values) / len(recent_values)
        relevant_metrics.append(avg_value)
        
        if not relevant_metrics:
            return None
        
        # Return average across components
        return sum(relevant_metrics) / len(relevant_metrics)
    
    def _determine_scaling_action(self, rule: ScalingRule, current_value: float) -> ScalingAction:
        """Determine what scaling action to take"""
        current_instances = self.current_instances.get(rule.rule_id, rule.min_instances)
        
        # TDA anomaly enhancement
        tda_multiplier = 1.0
        if (rule.trigger == ScalingTrigger.TDA_ANOMALY and 
            current_value >= self.tda_anomaly_threshold):
            tda_multiplier = self.tda_scaling_multiplier
        
        adjusted_threshold_up = rule.threshold_up / tda_multiplier
        
        if current_value >= adjusted_threshold_up and current_instances < rule.max_instances:
            return ScalingAction.SCALE_UP
        elif current_value <= rule.threshold_down and current_instances > rule.min_instances:
            return ScalingAction.SCALE_DOWN
        else:
            return ScalingAction.NO_ACTION
    
        async def _execute_scaling_action(self, rule: ScalingRule, action: ScalingAction,
        metric_value: float) -> None:
        """Execute scaling action"""
        current_instances = self.current_instances.get(rule.rule_id, rule.min_instances)
        
        if action == ScalingAction.SCALE_UP:
            # Calculate scale up amount (TDA-aware)
            if rule.trigger == ScalingTrigger.TDA_ANOMALY:
                # Aggressive scaling for anomalies
                scale_factor = min(3, int(metric_value * 5))  # Up to 3x scaling
            else:
                scale_factor = 1
            
            new_instances = min(rule.max_instances, current_instances + scale_factor)
        
        elif action == ScalingAction.SCALE_DOWN:
            new_instances = max(rule.min_instances, current_instances - 1)
        
        else:
            return
        
        if new_instances == current_instances:
            return  # No change needed
        
        # Create scaling event
        event = ScalingEvent(
            event_id=f"scale_{rule.rule_id}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
            trigger=rule.trigger,
            action=action,
            from_instances=current_instances,
            to_instances=new_instances,
            reason=f"{rule.trigger.value} = {metric_value:.2f} (threshold: {rule.threshold_up if action == ScalingAction.SCALE_UP else rule.threshold_down})"
        )
        
        # Add TDA context if available
        if self.tda_integration and rule.trigger == ScalingTrigger.TDA_ANOMALY:
            event.tda_context = self._get_tda_context()
        
        # Update state
        self.current_instances[rule.rule_id] = new_instances
        self.last_scaling_action[rule.rule_id] = datetime.utcnow()
        self.scaling_history.append(event)
        
        logger.info(f"Scaling {action.value}: {rule.rule_id} from {current_instances} to {new_instances} instances")
        
        # Execute callbacks
        for callback in self.scaling_callbacks:
            try:
                await callback(event)
            except Exception as e:
                logger.error(f"Scaling callback failed: {e}")
    
    def _get_tda_context(self) -> Dict[str, Any]:
        """Get TDA context for scaling event"""
        pass
        # Mock TDA context
        return {
        'anomaly_score': 0.85,
        'anomaly_type': 'system_overload',
        'correlation_id': f"tda_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
        'recommended_action': 'scale_up_aggressive'
        }
    
    def get_scaling_status(self) -> Dict[str, Any]:
        """Get auto-scaling status"""
        pass
        # Calculate scaling efficiency
        total_events = len(self.scaling_history)
        if total_events > 0:
            scale_up_events = sum(1 for e in self.scaling_history if e.action == ScalingAction.SCALE_UP)
            scale_down_events = sum(1 for e in self.scaling_history if e.action == ScalingAction.SCALE_DOWN)
        else:
            scale_up_events = scale_down_events = 0
        
        return {
            'scaling_enabled': self.is_scaling_enabled,
            'active_rules': len([r for r in self.scaling_rules.values() if r.enabled]),
            'total_rules': len(self.scaling_rules),
            'current_instances': dict(self.current_instances),
            'scaling_events': {
                'total': total_events,
                'scale_up': scale_up_events,
                'scale_down': scale_down_events
            },
            'tda_integration': self.tda_integration is not None,
            'tda_anomaly_threshold': self.tda_anomaly_threshold,
            'recent_scaling_events': [
                {
                    'event_id': e.event_id,
                    'action': e.action.value,
                    'trigger': e.trigger.value,
                    'from_instances': e.from_instances,
                    'to_instances': e.to_instances,
                    'timestamp': e.timestamp.isoformat()
                }
                for e in self.scaling_history[-5:]  # Last 5 events
            ]
        }

# Factory function
    def create_auto_scaler(tda_integration: Optional[Any] = None) -> AutoScaler:
        """Create auto-scaler with optional TDA integration"""
        return AutoScaler(tda_integration=tda_integration)
