"""
⚡ Real Executor Agent 2025 - Action Execution with Failure Prevention
====================================================================

The Executor Agent is responsible for:
- Executing actions based on supervisor decisions
- Monitoring execution for failure patterns
- Implementing preventive measures
- Learning from execution outcomes

"Execute with intelligence, prevent with foresight"
"""

import asyncio
import time
import json
import hashlib
from typing import Dict, Any, Optional, List, Tuple, Callable
from enum import Enum
from dataclasses import dataclass, field
from collections import defaultdict, deque
from datetime import datetime, timezone
import numpy as np
import structlog

logger = structlog.get_logger(__name__)


# ==================== Core Types ====================

class ActionType(str, Enum):
    """Types of actions the executor can perform."""
    ISOLATE_AGENT = "isolate_agent"
    REDUCE_LOAD = "reduce_load"
    INCREASE_RESOURCES = "increase_resources"
    RESTART_COMPONENT = "restart_component"
    CHECKPOINT_STATE = "checkpoint_state"
    ROLLBACK_STATE = "rollback_state"
    PAUSE_WORKFLOW = "pause_workflow"
    RESUME_WORKFLOW = "resume_workflow"
    SCALE_HORIZONTALLY = "scale_horizontally"
    APPLY_THROTTLING = "apply_throttling"
    CLEAR_CACHE = "clear_cache"
    NOTIFY_HUMAN = "notify_human"


class ExecutionStatus(str, Enum):
    """Status of action execution."""
    PENDING = "pending"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"


@dataclass
class ActionContext:
    """Context for action execution."""
    action_id: str
    action_type: ActionType
    target: str  # Target component/agent
    parameters: Dict[str, Any]
    priority: float = 0.5
    timeout_seconds: float = 30.0
    retry_count: int = 0
    max_retries: int = 3
    initiated_by: str = "supervisor"
    reason: str = ""
    
    def can_retry(self) -> bool:
        """Check if action can be retried."""
        return self.retry_count < self.max_retries


@dataclass
class ExecutionResult:
    """Result of action execution."""
    action_id: str
    status: ExecutionStatus
    started_at: float
    completed_at: Optional[float] = None
    duration_ms: Optional[float] = None
    output: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    side_effects: List[str] = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=dict)
    
    def calculate_duration(self):
        """Calculate execution duration."""
        if self.completed_at and self.started_at:
            self.duration_ms = (self.completed_at - self.started_at) * 1000


@dataclass
class ExecutionPattern:
    """Pattern in execution history."""
    pattern_id: str
    pattern_type: str  # success_streak, failure_pattern, timeout_pattern
    actions: List[ActionType]
    success_rate: float
    avg_duration_ms: float
    occurrences: int = 1
    last_seen: float = field(default_factory=time.time)


# ==================== Execution Strategies ====================

class ExecutionStrategy:
    """Base class for execution strategies."""
    
    def __init__(self, name: str):
        self.name = name
        self.execution_count = 0
        self.success_count = 0
        
    async def execute(self, context: ActionContext) -> ExecutionResult:
        """Execute action with strategy."""
        raise NotImplementedError
        
    def get_success_rate(self) -> float:
        """Get success rate of strategy."""
        if self.execution_count == 0:
            return 0.0
        return self.success_count / self.execution_count


class SafeExecutionStrategy(ExecutionStrategy):
    """Safe execution with validation and rollback."""
    
    def __init__(self):
        super().__init__("safe_execution")
        self.checkpoints = {}
        
    async def execute(self, context: ActionContext) -> ExecutionResult:
        """Execute with safety checks."""
        self.execution_count += 1
        result = ExecutionResult(
            action_id=context.action_id,
            status=ExecutionStatus.EXECUTING,
            started_at=time.time()
        )
        
        try:
            # Pre-execution validation
            if not await self._validate_preconditions(context):
                result.status = ExecutionStatus.FAILED
                result.error = "Precondition validation failed"
                return result
            
            # Create checkpoint
            checkpoint_id = await self._create_checkpoint(context)
            
            # Execute action
            output = await self._execute_action(context)
            
            # Validate result
            if await self._validate_postconditions(context, output):
                result.status = ExecutionStatus.COMPLETED
                result.output = output
                self.success_count += 1
            else:
                # Rollback on validation failure
                await self._rollback_checkpoint(checkpoint_id)
                result.status = ExecutionStatus.FAILED
                result.error = "Postcondition validation failed"
                
        except asyncio.TimeoutError:
            result.status = ExecutionStatus.TIMEOUT
            result.error = f"Execution timeout after {context.timeout_seconds}s"
        except Exception as e:
            result.status = ExecutionStatus.FAILED
            result.error = str(e)
            
        result.completed_at = time.time()
        result.calculate_duration()
        return result
    
    async def _validate_preconditions(self, context: ActionContext) -> bool:
        """Validate conditions before execution."""
        # Check target availability
        if context.action_type == ActionType.RESTART_COMPONENT:
            # Ensure component exists and is stoppable
            return True  # Simplified for demo
        return True
    
    async def _create_checkpoint(self, context: ActionContext) -> str:
        """Create checkpoint before execution."""
        checkpoint_id = f"ckpt_{context.action_id}_{int(time.time())}"
        self.checkpoints[checkpoint_id] = {
            "context": context,
            "timestamp": time.time(),
            "state": {}  # Would capture actual state
        }
        return checkpoint_id
    
    async def _execute_action(self, context: ActionContext) -> Dict[str, Any]:
        """Execute the actual action."""
        # Simulate action execution
        await asyncio.sleep(0.1)  # Simulate work
        
        # Action-specific execution
        if context.action_type == ActionType.REDUCE_LOAD:
            return {"load_reduced": True, "new_capacity": 0.7}
        elif context.action_type == ActionType.ISOLATE_AGENT:
            return {"isolated": True, "connections_closed": 5}
        else:
            return {"executed": True}
    
    async def _validate_postconditions(self, context: ActionContext, output: Dict[str, Any]) -> bool:
        """Validate conditions after execution."""
        # Check expected outcomes
        if context.action_type == ActionType.REDUCE_LOAD:
            return output.get("load_reduced", False)
        return True
    
    async def _rollback_checkpoint(self, checkpoint_id: str):
        """Rollback to checkpoint on failure."""
        if checkpoint_id in self.checkpoints:
            # Would perform actual rollback
            logger.warning(f"Rolling back to checkpoint {checkpoint_id}")


class AdaptiveExecutionStrategy(ExecutionStrategy):
    """Adaptive execution that learns from outcomes."""
    
    def __init__(self):
        super().__init__("adaptive_execution")
        self.action_history = defaultdict(list)
        self.timing_model = {}
        
    async def execute(self, context: ActionContext) -> ExecutionResult:
        """Execute with adaptive adjustments."""
        self.execution_count += 1
        
        # Adapt timeout based on history
        adapted_timeout = self._adapt_timeout(context)
        context.timeout_seconds = adapted_timeout
        
        result = ExecutionResult(
            action_id=context.action_id,
            status=ExecutionStatus.EXECUTING,
            started_at=time.time()
        )
        
        try:
            # Execute with monitoring
            output = await asyncio.wait_for(
                self._monitored_execution(context),
                timeout=context.timeout_seconds
            )
            
            result.status = ExecutionStatus.COMPLETED
            result.output = output
            self.success_count += 1
            
            # Learn from success
            self._update_model(context, result)
            
        except asyncio.TimeoutError:
            result.status = ExecutionStatus.TIMEOUT
            result.error = f"Adaptive timeout after {adapted_timeout}s"
        except Exception as e:
            result.status = ExecutionStatus.FAILED
            result.error = str(e)
            
        result.completed_at = time.time()
        result.calculate_duration()
        
        # Record for learning
        self.action_history[context.action_type].append(result)
        
        return result
    
    async def _monitored_execution(self, context: ActionContext) -> Dict[str, Any]:
        """Execute with monitoring."""
        metrics = {
            "start_cpu": 0.5,  # Would get actual metrics
            "start_memory": 0.6
        }
        
        # Execute
        await asyncio.sleep(0.15)  # Simulate work
        
        metrics["end_cpu"] = 0.4
        metrics["end_memory"] = 0.55
        
        return {
            "executed": True,
            "metrics": metrics,
            "impact": "positive"
        }
    
    def _adapt_timeout(self, context: ActionContext) -> float:
        """Adapt timeout based on history."""
        history = self.action_history.get(context.action_type, [])
        if not history:
            return context.timeout_seconds
        
        # Calculate average duration of successful executions
        successful = [h for h in history if h.status == ExecutionStatus.COMPLETED]
        if successful:
            avg_duration = np.mean([h.duration_ms for h in successful if h.duration_ms])
            # Add 50% buffer
            return (avg_duration / 1000) * 1.5
        
        return context.timeout_seconds
    
    def _update_model(self, context: ActionContext, result: ExecutionResult):
        """Update timing model based on outcome."""
        if context.action_type not in self.timing_model:
            self.timing_model[context.action_type] = {
                "avg_duration": 0,
                "success_rate": 0,
                "sample_count": 0
            }
        
        model = self.timing_model[context.action_type]
        model["sample_count"] += 1
        
        # Update running average
        if result.duration_ms:
            prev_avg = model["avg_duration"]
            model["avg_duration"] = (prev_avg * (model["sample_count"] - 1) + result.duration_ms) / model["sample_count"]
        
        # Update success rate
        if result.status == ExecutionStatus.COMPLETED:
            prev_rate = model["success_rate"]
            model["success_rate"] = (prev_rate * (model["sample_count"] - 1) + 1) / model["sample_count"]


# ==================== Main Executor Agent ====================

class RealExecutorAgent:
    """
    Real Executor Agent with intelligent action execution.
    
    Features:
    - Multiple execution strategies
    - Failure pattern detection
    - Preventive action planning
    - Learning from outcomes
    - Resource-aware execution
    """
    
    def __init__(
        self,
        agent_id: str = "executor_001",
        enable_safe_mode: bool = True,
        enable_learning: bool = True
    ):
        self.agent_id = agent_id
        self.enable_safe_mode = enable_safe_mode
        self.enable_learning = enable_learning
        
        # Execution strategies
        self.strategies = {
            "safe": SafeExecutionStrategy(),
            "adaptive": AdaptiveExecutionStrategy()
        }
        
        # Execution tracking
        self.active_executions: Dict[str, ActionContext] = {}
        self.execution_history: deque = deque(maxlen=1000)
        self.execution_patterns: Dict[str, ExecutionPattern] = {}
        
        # Resource management
        self.resource_limits = {
            "max_concurrent": 10,
            "cpu_threshold": 0.8,
            "memory_threshold": 0.85
        }
        self.current_load = {
            "concurrent_count": 0,
            "cpu_usage": 0.0,
            "memory_usage": 0.0
        }
        
        # Learning components
        self.outcome_history = defaultdict(list)
        self.action_effectiveness = defaultdict(float)
        
        # Statistics
        self.stats = {
            "total_executed": 0,
            "successful": 0,
            "failed": 0,
            "timeout": 0,
            "patterns_detected": 0
        }
        
        logger.info(
            "Real Executor Agent initialized",
            agent_id=agent_id,
            safe_mode=enable_safe_mode,
            learning=enable_learning
        )
    
    async def execute_action(
        self,
        action_type: ActionType,
        target: str,
        parameters: Optional[Dict[str, Any]] = None,
        priority: float = 0.5,
        reason: str = ""
    ) -> ExecutionResult:
        """
        Execute an action with intelligent strategy selection.
        
        Args:
            action_type: Type of action to execute
            target: Target component/agent
            parameters: Action parameters
            priority: Execution priority (0-1)
            reason: Reason for execution
            
        Returns:
            Execution result with status and output
        """
        # Create action context
        context = ActionContext(
            action_id=self._generate_action_id(),
            action_type=action_type,
            target=target,
            parameters=parameters or {},
            priority=priority,
            reason=reason
        )
        
        # Check resource availability
        if not await self._check_resources():
            return ExecutionResult(
                action_id=context.action_id,
                status=ExecutionStatus.FAILED,
                started_at=time.time(),
                completed_at=time.time(),
                error="Resource limits exceeded"
            )
        
        # Select execution strategy
        strategy = self._select_strategy(context)
        
        # Track active execution
        self.active_executions[context.action_id] = context
        self.current_load["concurrent_count"] += 1
        
        try:
            # Execute with selected strategy
            result = await strategy.execute(context)
            
            # Post-execution processing
            await self._post_execution(context, result)
            
            # Learn from outcome
            if self.enable_learning:
                await self._learn_from_execution(context, result)
            
            # Detect patterns
            self._detect_execution_patterns(context, result)
            
            return result
            
        finally:
            # Clean up
            self.active_executions.pop(context.action_id, None)
            self.current_load["concurrent_count"] -= 1
    
    async def execute_prevention_plan(
        self,
        plan: Dict[str, Any]
    ) -> List[ExecutionResult]:
        """
        Execute a failure prevention plan.
        
        Args:
            plan: Prevention plan with actions
            
        Returns:
            List of execution results
        """
        results = []
        
        # Sort actions by priority
        actions = plan.get("immediate_actions", [])
        actions.sort(key=lambda a: a.get("priority", 0.5), reverse=True)
        
        # Execute actions in priority order
        for action in actions:
            result = await self.execute_action(
                action_type=ActionType(action["action"]),
                target=action["target"],
                parameters=action.get("params", {}),
                priority=action.get("priority", 0.5),
                reason=action.get("reason", "Prevention plan")
            )
            results.append(result)
            
            # Stop if critical action fails
            if result.status == ExecutionStatus.FAILED and action.get("critical", False):
                logger.error(
                    "Critical prevention action failed",
                    action=action["action"],
                    target=action["target"]
                )
                break
        
        # Update monitoring targets
        for monitor in plan.get("monitoring_targets", []):
            # Would set up actual monitoring
            logger.info(
                "Setting up monitoring",
                target=monitor["target"],
                threshold=monitor.get("threshold", 0.8)
            )
        
        return results
    
    async def cancel_action(self, action_id: str) -> bool:
        """Cancel an active action execution."""
        if action_id in self.active_executions:
            # Would implement actual cancellation
            logger.info(f"Cancelling action {action_id}")
            return True
        return False
    
    def get_execution_status(self, action_id: str) -> Optional[Dict[str, Any]]:
        """Get status of an action execution."""
        if action_id in self.active_executions:
            context = self.active_executions[action_id]
            return {
                "action_id": action_id,
                "status": "executing",
                "action_type": context.action_type.value,
                "target": context.target,
                "started_at": context.action_id  # Would track actual start time
            }
        
        # Check history
        for result in self.execution_history:
            if result.action_id == action_id:
                return {
                    "action_id": action_id,
                    "status": result.status.value,
                    "duration_ms": result.duration_ms,
                    "output": result.output
                }
        
        return None
    
    # ==================== Internal Methods ====================
    
    def _generate_action_id(self) -> str:
        """Generate unique action ID."""
        timestamp = int(time.time() * 1000)
        random_part = hashlib.md5(f"{timestamp}{self.agent_id}".encode()).hexdigest()[:8]
        return f"act_{timestamp}_{random_part}"
    
    async def _check_resources(self) -> bool:
        """Check if resources allow execution."""
        # Check concurrent execution limit
        if self.current_load["concurrent_count"] >= self.resource_limits["max_concurrent"]:
            logger.warning("Concurrent execution limit reached")
            return False
        
        # Check CPU usage (simulated)
        self.current_load["cpu_usage"] = 0.6  # Would get actual
        if self.current_load["cpu_usage"] > self.resource_limits["cpu_threshold"]:
            logger.warning("CPU threshold exceeded")
            return False
        
        # Check memory usage (simulated)
        self.current_load["memory_usage"] = 0.7  # Would get actual
        if self.current_load["memory_usage"] > self.resource_limits["memory_threshold"]:
            logger.warning("Memory threshold exceeded")
            return False
        
        return True
    
    def _select_strategy(self, context: ActionContext) -> ExecutionStrategy:
        """Select execution strategy based on context."""
        # High priority or critical actions use safe strategy
        if context.priority > 0.8 or self.enable_safe_mode:
            return self.strategies["safe"]
        
        # Use adaptive strategy for learning
        if self.enable_learning:
            return self.strategies["adaptive"]
        
        # Default to safe
        return self.strategies["safe"]
    
    async def _post_execution(self, context: ActionContext, result: ExecutionResult):
        """Post-execution processing."""
        # Update statistics
        self.stats["total_executed"] += 1
        
        if result.status == ExecutionStatus.COMPLETED:
            self.stats["successful"] += 1
        elif result.status == ExecutionStatus.FAILED:
            self.stats["failed"] += 1
        elif result.status == ExecutionStatus.TIMEOUT:
            self.stats["timeout"] += 1
        
        # Record in history
        self.execution_history.append(result)
        
        # Log execution
        logger.info(
            "Action executed",
            action_id=context.action_id,
            action_type=context.action_type.value,
            target=context.target,
            status=result.status.value,
            duration_ms=result.duration_ms
        )
    
    async def _learn_from_execution(self, context: ActionContext, result: ExecutionResult):
        """Learn from execution outcome."""
        # Record outcome
        self.outcome_history[context.action_type].append({
            "context": context,
            "result": result,
            "timestamp": time.time()
        })
        
        # Update effectiveness score
        if result.status == ExecutionStatus.COMPLETED:
            # Increase effectiveness
            current = self.action_effectiveness[context.action_type]
            self.action_effectiveness[context.action_type] = current * 0.9 + 0.1
        else:
            # Decrease effectiveness
            current = self.action_effectiveness[context.action_type]
            self.action_effectiveness[context.action_type] = current * 0.9
        
        # Analyze patterns in failures
        if result.status in [ExecutionStatus.FAILED, ExecutionStatus.TIMEOUT]:
            await self._analyze_failure_pattern(context, result)
    
    async def _analyze_failure_pattern(self, context: ActionContext, result: ExecutionResult):
        """Analyze failure patterns."""
        # Get recent failures
        recent_failures = [
            h for h in self.execution_history
            if h.status in [ExecutionStatus.FAILED, ExecutionStatus.TIMEOUT]
            and time.time() - (h.completed_at or time.time()) < 300  # Last 5 minutes
        ]
        
        if len(recent_failures) >= 3:
            # Potential pattern detected
            logger.warning(
                "Failure pattern detected",
                failure_count=len(recent_failures),
                action_type=context.action_type.value
            )
    
    def _detect_execution_patterns(self, context: ActionContext, result: ExecutionResult):
        """Detect patterns in execution history."""
        # Get recent executions of same type
        recent = [
            h for h in self.execution_history
            if hasattr(h, 'action_type') and h.action_type == context.action_type
        ][-10:]  # Last 10
        
        if len(recent) >= 5:
            # Calculate pattern metrics
            success_count = sum(1 for r in recent if r.status == ExecutionStatus.COMPLETED)
            success_rate = success_count / len(recent)
            
            durations = [r.duration_ms for r in recent if r.duration_ms]
            avg_duration = np.mean(durations) if durations else 0
            
            # Create or update pattern
            pattern_id = f"pattern_{context.action_type.value}"
            
            if pattern_id not in self.execution_patterns:
                pattern = ExecutionPattern(
                    pattern_id=pattern_id,
                    pattern_type="execution_history",
                    actions=[context.action_type],
                    success_rate=success_rate,
                    avg_duration_ms=avg_duration
                )
                self.execution_patterns[pattern_id] = pattern
                self.stats["patterns_detected"] += 1
            else:
                # Update existing pattern
                pattern = self.execution_patterns[pattern_id]
                pattern.success_rate = success_rate
                pattern.avg_duration_ms = avg_duration
                pattern.occurrences += 1
                pattern.last_seen = time.time()
    
    def get_effectiveness_report(self) -> Dict[str, Any]:
        """Get action effectiveness report."""
        report = {
            "overall_stats": self.stats,
            "action_effectiveness": {},
            "execution_patterns": {},
            "strategy_performance": {}
        }
        
        # Action effectiveness
        for action_type, score in self.action_effectiveness.items():
            report["action_effectiveness"][action_type.value] = {
                "score": score,
                "rating": "high" if score > 0.8 else "medium" if score > 0.5 else "low"
            }
        
        # Execution patterns
        for pattern_id, pattern in self.execution_patterns.items():
            report["execution_patterns"][pattern_id] = {
                "type": pattern.pattern_type,
                "success_rate": pattern.success_rate,
                "avg_duration_ms": pattern.avg_duration_ms,
                "occurrences": pattern.occurrences
            }
        
        # Strategy performance
        for name, strategy in self.strategies.items():
            report["strategy_performance"][name] = {
                "executions": strategy.execution_count,
                "success_rate": strategy.get_success_rate()
            }
        
        return report
    
    def get_resource_usage(self) -> Dict[str, Any]:
        """Get current resource usage."""
        return {
            "concurrent_executions": self.current_load["concurrent_count"],
            "cpu_usage": self.current_load["cpu_usage"],
            "memory_usage": self.current_load["memory_usage"],
            "limits": self.resource_limits
        }