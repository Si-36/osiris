"""
🧠 Real Working Supervisor Implementation
A production-ready supervisor that actually analyzes, decides, and learns.
"""

import asyncio
import json
import time
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Tuple
from enum import Enum
from dataclasses import dataclass, field
import numpy as np
from collections import defaultdict, deque
import hashlib


class DecisionType(str, Enum):
    """Types of supervisor decisions."""
    CONTINUE = "continue"      # Continue workflow execution
    ESCALATE = "escalate"      # Escalate to human/higher authority
    RETRY = "retry"            # Retry failed operation
    COMPLETE = "complete"      # Mark workflow as complete
    ABORT = "abort"            # Abort workflow
    DELEGATE = "delegate"      # Delegate to specialized agent
    WAIT = "wait"             # Wait for external input
    OPTIMIZE = "optimize"      # Optimize workflow path


@dataclass
class WorkflowMetrics:
    """Real-time workflow metrics."""
    total_steps: int = 0
    completed_steps: int = 0
    failed_steps: int = 0
    retried_steps: int = 0
    total_duration_ms: float = 0
    average_step_duration_ms: float = 0
    error_rate: float = 0
    success_rate: float = 0
    bottleneck_nodes: List[str] = field(default_factory=list)
    resource_usage: Dict[str, float] = field(default_factory=dict)
    
    def update(self, step_result: Dict[str, Any]):
        """Update metrics with step result."""
        self.total_steps += 1
        
        if step_result.get("success"):
            self.completed_steps += 1
        else:
            self.failed_steps += 1
            
        if step_result.get("retried"):
            self.retried_steps += 1
            
        # Update durations
        duration = step_result.get("duration_ms", 0)
        self.total_duration_ms += duration
        self.average_step_duration_ms = self.total_duration_ms / self.total_steps
        
        # Update rates
        self.error_rate = self.failed_steps / self.total_steps if self.total_steps > 0 else 0
        self.success_rate = self.completed_steps / self.total_steps if self.total_steps > 0 else 0
        
        # Identify bottlenecks (steps taking >2x average time)
        if duration > 2 * self.average_step_duration_ms and duration > 1000:
            node_name = step_result.get("node_name", "unknown")
            if node_name not in self.bottleneck_nodes:
                self.bottleneck_nodes.append(node_name)


@dataclass
class RiskAssessment:
    """Comprehensive risk assessment."""
    overall_score: float
    risk_factors: Dict[str, float] = field(default_factory=dict)
    mitigations: List[str] = field(default_factory=list)
    confidence: float = 0.0
    
    @property
    def risk_level(self) -> str:
        """Get risk level category."""
        if self.overall_score < 0.3:
            return "low"
        elif self.overall_score < 0.6:
            return "medium"
        elif self.overall_score < 0.8:
            return "high"
        else:
            return "critical"


class PatternDetector:
    """Detects patterns in workflow execution."""
    
    def __init__(self, window_size: int = 10):
        self.window_size = window_size
        self.pattern_history = defaultdict(lambda: deque(maxlen=window_size))
        self.known_patterns = {
            "retry_loop": lambda h: self._detect_retry_loop(h),
            "cascading_failure": lambda h: self._detect_cascading_failure(h),
            "performance_degradation": lambda h: self._detect_performance_degradation(h),
            "success_streak": lambda h: self._detect_success_streak(h),
            "oscillation": lambda h: self._detect_oscillation(h)
        }
    
    def update(self, node_name: str, result: Dict[str, Any]):
        """Update pattern history."""
        self.pattern_history[node_name].append({
            "success": result.get("success", False),
            "duration_ms": result.get("duration_ms", 0),
            "timestamp": time.time(),
            "error_type": result.get("error_type"),
            "retry_count": result.get("retry_count", 0)
        })
    
    def detect_patterns(self) -> Dict[str, List[str]]:
        """Detect patterns across all nodes."""
        detected_patterns = defaultdict(list)
        
        for node_name, history in self.pattern_history.items():
            if len(history) < 3:  # Need minimum history
                continue
                
            for pattern_name, detector_func in self.known_patterns.items():
                if detector_func(list(history)):
                    detected_patterns[pattern_name].append(node_name)
        
        return dict(detected_patterns)
    
    def _detect_retry_loop(self, history: List[Dict]) -> bool:
        """Detect if node is stuck in retry loop."""
        if len(history) < 3:
            return False
        
        # Check if last N attempts all have high retry counts
        recent_retries = [h.get("retry_count", 0) for h in history[-3:]]
        return all(r > 2 for r in recent_retries)
    
    def _detect_cascading_failure(self, history: List[Dict]) -> bool:
        """Detect cascading failures."""
        if len(history) < 4:
            return False
        
        # Check if failures are increasing
        recent_failures = [not h["success"] for h in history[-4:]]
        return sum(recent_failures) >= 3
    
    def _detect_performance_degradation(self, history: List[Dict]) -> bool:
        """Detect performance getting worse over time."""
        if len(history) < 5:
            return False
        
        durations = [h["duration_ms"] for h in history[-5:]]
        # Check if each duration is worse than previous
        degrading = all(durations[i] > durations[i-1] * 1.2 for i in range(1, len(durations)))
        return degrading
    
    def _detect_success_streak(self, history: List[Dict]) -> bool:
        """Detect consistent success pattern."""
        if len(history) < 5:
            return False
        
        return all(h["success"] for h in history[-5:])
    
    def _detect_oscillation(self, history: List[Dict]) -> bool:
        """Detect oscillating success/failure pattern."""
        if len(history) < 6:
            return False
        
        # Check for alternating pattern
        results = [h["success"] for h in history[-6:]]
        return all(results[i] != results[i+1] for i in range(len(results)-1))


class DecisionEngine:
    """Intelligent decision making engine."""
    
    def __init__(self):
        self.decision_history = deque(maxlen=100)
        self.decision_weights = {
            DecisionType.CONTINUE: 1.0,
            DecisionType.RETRY: 0.8,
            DecisionType.DELEGATE: 0.7,
            DecisionType.WAIT: 0.5,
            DecisionType.OPTIMIZE: 0.6,
            DecisionType.ESCALATE: 0.3,
            DecisionType.COMPLETE: 0.9,
            DecisionType.ABORT: 0.1
        }
        self.learning_rate = 0.01
    
    async def make_decision(
        self,
        state_analysis: Dict[str, Any],
        risk_assessment: RiskAssessment,
        patterns: Dict[str, List[str]],
        metrics: WorkflowMetrics
    ) -> Tuple[DecisionType, Dict[str, Any]]:
        """Make intelligent decision based on all inputs."""
        
        # Calculate decision scores
        scores = {}
        
        # CONTINUE - if things are going well
        if metrics.success_rate > 0.7 and risk_assessment.risk_level in ["low", "medium"]:
            scores[DecisionType.CONTINUE] = 0.8
        else:
            scores[DecisionType.CONTINUE] = 0.3
        
        # RETRY - if recent failure but not in retry loop
        if metrics.failed_steps > 0 and "retry_loop" not in patterns:
            scores[DecisionType.RETRY] = 0.6
        else:
            scores[DecisionType.RETRY] = 0.1
        
        # ESCALATE - if high risk or cascading failures
        if risk_assessment.risk_level in ["high", "critical"] or "cascading_failure" in patterns:
            scores[DecisionType.ESCALATE] = 0.9
        else:
            scores[DecisionType.ESCALATE] = 0.2
        
        # OPTIMIZE - if performance degradation detected
        if "performance_degradation" in patterns or metrics.bottleneck_nodes:
            scores[DecisionType.OPTIMIZE] = 0.7
        else:
            scores[DecisionType.OPTIMIZE] = 0.3
        
        # COMPLETE - if workflow objectives met
        completion_confidence = state_analysis.get("completion_confidence", 0)
        if completion_confidence > 0.8:
            scores[DecisionType.COMPLETE] = 0.9
        else:
            scores[DecisionType.COMPLETE] = 0.1
        
        # ABORT - if critical failures or stuck
        if metrics.error_rate > 0.8 or "retry_loop" in patterns:
            scores[DecisionType.ABORT] = 0.7
        else:
            scores[DecisionType.ABORT] = 0.1
        
        # Apply learned weights
        for decision_type, base_score in scores.items():
            scores[decision_type] = base_score * self.decision_weights.get(decision_type, 1.0)
        
        # Select decision with highest score
        decision = max(scores.items(), key=lambda x: x[1])[0]
        
        # Build reasoning
        reasoning = {
            "scores": scores,
            "selected": decision,
            "confidence": scores[decision],
            "factors": {
                "risk_level": risk_assessment.risk_level,
                "success_rate": metrics.success_rate,
                "patterns_detected": list(patterns.keys()),
                "bottlenecks": metrics.bottleneck_nodes
            }
        }
        
        # Record decision for learning
        self.decision_history.append({
            "decision": decision,
            "reasoning": reasoning,
            "timestamp": time.time()
        })
        
        return decision, reasoning
    
    def update_weights(self, decision: DecisionType, outcome: float):
        """Update decision weights based on outcome."""
        # Simple reinforcement learning update
        current_weight = self.decision_weights[decision]
        new_weight = current_weight + self.learning_rate * (outcome - 0.5)
        self.decision_weights[decision] = max(0.1, min(1.0, new_weight))


class RealSupervisor:
    """
    Production-ready supervisor with real implementation.
    
    Features:
    - Real state analysis
    - Pattern detection
    - Risk assessment
    - Intelligent decision making
    - Learning from outcomes
    - Performance optimization
    """
    
    def __init__(
        self,
        risk_threshold: float = 0.7,
        enable_learning: bool = True,
        enable_patterns: bool = True
    ):
        self.risk_threshold = risk_threshold
        self.enable_learning = enable_learning
        self.enable_patterns = enable_patterns
        
        # Core components
        self.metrics = WorkflowMetrics()
        self.pattern_detector = PatternDetector() if enable_patterns else None
        self.decision_engine = DecisionEngine()
        
        # State tracking
        self.workflow_history = {}
        self.active_workflows = {}
        
    async def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main supervisor entry point.
        
        Args:
            state: Current workflow state
            
        Returns:
            Updated state with supervisor decision
        """
        start_time = time.time()
        workflow_id = state.get("workflow_id", "unknown")
        
        try:
            # Track active workflow
            self.active_workflows[workflow_id] = {
                "start_time": start_time,
                "state": state
            }
            
            # Analyze state
            state_analysis = self._analyze_state(state)
            
            # Assess risk
            risk_assessment = self._assess_risk(state, state_analysis)
            
            # Detect patterns
            patterns = {}
            if self.enable_patterns and self.pattern_detector:
                patterns = self.pattern_detector.detect_patterns()
            
            # Make decision
            decision, reasoning = await self.decision_engine.make_decision(
                state_analysis,
                risk_assessment,
                patterns,
                self.metrics
            )
            
            # Build result
            duration_ms = (time.time() - start_time) * 1000
            
            result = {
                "success": True,
                "node_name": "supervisor",
                "decision": decision.value,
                "reasoning": reasoning,
                "risk_assessment": {
                    "score": risk_assessment.overall_score,
                    "level": risk_assessment.risk_level,
                    "factors": risk_assessment.risk_factors,
                    "mitigations": risk_assessment.mitigations
                },
                "patterns_detected": patterns,
                "metrics": {
                    "success_rate": self.metrics.success_rate,
                    "error_rate": self.metrics.error_rate,
                    "average_duration_ms": self.metrics.average_step_duration_ms,
                    "bottlenecks": self.metrics.bottleneck_nodes
                },
                "duration_ms": duration_ms,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
            # Update state
            updated_state = state.copy()
            updated_state["supervisor_decision"] = result
            updated_state["next_action"] = self._determine_next_action(decision, state_analysis)
            
            # Record for learning
            self.workflow_history[workflow_id] = {
                "decision": decision,
                "risk_assessment": risk_assessment,
                "timestamp": time.time()
            }
            
            return updated_state
            
        except Exception as e:
            # Handle errors gracefully
            return {
                **state,
                "supervisor_decision": {
                    "success": False,
                    "error": str(e),
                    "decision": DecisionType.ESCALATE.value,
                    "reasoning": {"error": "Exception in supervisor", "details": str(e)}
                }
            }
    
    def _analyze_state(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Perform comprehensive state analysis."""
        analysis = {
            "workflow_stage": self._determine_workflow_stage(state),
            "health_indicators": self._analyze_health_indicators(state),
            "completion_confidence": self._calculate_completion_confidence(state),
            "anomalies": self._detect_anomalies(state),
            "resource_status": self._analyze_resources(state)
        }
        
        # Update metrics with latest results
        if "step_results" in state:
            for result in state["step_results"]:
                self.metrics.update(result)
                
                # Update pattern detector
                if self.pattern_detector:
                    node_name = result.get("node_name", "unknown")
                    self.pattern_detector.update(node_name, result)
        
        return analysis
    
    def _determine_workflow_stage(self, state: Dict[str, Any]) -> str:
        """Determine current workflow stage."""
        steps_completed = len(state.get("completed_steps", []))
        total_steps = state.get("total_steps", 0)
        
        if steps_completed == 0:
            return "initialization"
        elif steps_completed < total_steps * 0.3:
            return "early"
        elif steps_completed < total_steps * 0.7:
            return "middle"
        elif steps_completed < total_steps:
            return "late"
        else:
            return "completion"
    
    def _analyze_health_indicators(self, state: Dict[str, Any]) -> Dict[str, float]:
        """Analyze workflow health indicators."""
        indicators = {}
        
        # Performance health
        if self.metrics.average_step_duration_ms > 0:
            performance_health = min(1.0, 1000 / self.metrics.average_step_duration_ms)
        else:
            performance_health = 1.0
        indicators["performance"] = performance_health
        
        # Error health
        indicators["error"] = 1.0 - self.metrics.error_rate
        
        # Progress health
        total_steps = state.get("total_steps", 1)
        completed_steps = len(state.get("completed_steps", []))
        indicators["progress"] = completed_steps / total_steps if total_steps > 0 else 0
        
        # Resource health
        resource_usage = state.get("resource_usage", {})
        if resource_usage:
            avg_usage = sum(resource_usage.values()) / len(resource_usage)
            indicators["resource"] = 1.0 - min(1.0, avg_usage)
        else:
            indicators["resource"] = 1.0
        
        return indicators
    
    def _calculate_completion_confidence(self, state: Dict[str, Any]) -> float:
        """Calculate confidence that workflow can complete successfully."""
        
        # Base confidence on success rate
        confidence = self.metrics.success_rate
        
        # Adjust based on stage
        stage = self._determine_workflow_stage(state)
        stage_multipliers = {
            "initialization": 0.5,
            "early": 0.7,
            "middle": 0.9,
            "late": 1.0,
            "completion": 1.1
        }
        confidence *= stage_multipliers.get(stage, 1.0)
        
        # Adjust based on recent trends
        recent_results = state.get("step_results", [])[-5:]
        if recent_results:
            recent_success_rate = sum(1 for r in recent_results if r.get("success")) / len(recent_results)
            confidence = (confidence + recent_success_rate) / 2
        
        return min(1.0, confidence)
    
    def _detect_anomalies(self, state: Dict[str, Any]) -> List[str]:
        """Detect anomalies in workflow execution."""
        anomalies = []
        
        # Duration anomalies
        if self.metrics.average_step_duration_ms > 0:
            recent_durations = [r.get("duration_ms", 0) for r in state.get("step_results", [])[-5:]]
            for duration in recent_durations:
                if duration > 3 * self.metrics.average_step_duration_ms:
                    anomalies.append("extreme_duration_spike")
                    break
        
        # Error pattern anomalies
        error_types = [r.get("error_type") for r in state.get("step_results", []) if not r.get("success")]
        if len(set(error_types)) == 1 and len(error_types) > 3:
            anomalies.append("repeated_error_type")
        
        # Resource anomalies
        resource_usage = state.get("resource_usage", {})
        for resource, usage in resource_usage.items():
            if usage > 0.9:
                anomalies.append(f"high_{resource}_usage")
        
        return anomalies
    
    def _analyze_resources(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze resource utilization."""
        resources = state.get("resource_usage", {})
        
        return {
            "current_usage": resources,
            "available": {k: 1.0 - v for k, v in resources.items()},
            "constrained": [k for k, v in resources.items() if v > 0.8]
        }
    
    def _assess_risk(self, state: Dict[str, Any], analysis: Dict[str, Any]) -> RiskAssessment:
        """Perform comprehensive risk assessment."""
        risk_factors = {}
        
        # Error rate risk
        risk_factors["error_rate"] = self.metrics.error_rate
        
        # Performance risk
        if self.metrics.average_step_duration_ms > 5000:  # >5 seconds average
            risk_factors["performance"] = min(1.0, self.metrics.average_step_duration_ms / 10000)
        
        # Resource risk
        constrained_resources = analysis["resource_status"]["constrained"]
        if constrained_resources:
            risk_factors["resource"] = len(constrained_resources) / 3.0  # Assume 3 main resources
        
        # Anomaly risk
        anomalies = analysis["anomalies"]
        if anomalies:
            risk_factors["anomaly"] = min(1.0, len(anomalies) / 5.0)
        
        # Pattern risk
        if self.pattern_detector:
            patterns = self.pattern_detector.detect_patterns()
            risky_patterns = ["retry_loop", "cascading_failure", "oscillation"]
            pattern_risks = [p for p in risky_patterns if p in patterns]
            if pattern_risks:
                risk_factors["pattern"] = len(pattern_risks) / len(risky_patterns)
        
        # Calculate overall risk
        if risk_factors:
            overall_score = sum(risk_factors.values()) / len(risk_factors)
        else:
            overall_score = 0.0
        
        # Determine mitigations
        mitigations = []
        if risk_factors.get("error_rate", 0) > 0.5:
            mitigations.append("increase_retry_delays")
            mitigations.append("add_circuit_breakers")
        
        if risk_factors.get("performance", 0) > 0.5:
            mitigations.append("scale_resources")
            mitigations.append("optimize_bottlenecks")
        
        if risk_factors.get("resource", 0) > 0.5:
            mitigations.append("resource_throttling")
            mitigations.append("load_balancing")
        
        return RiskAssessment(
            overall_score=overall_score,
            risk_factors=risk_factors,
            mitigations=mitigations,
            confidence=0.8  # High confidence in risk assessment
        )
    
    def _determine_next_action(self, decision: DecisionType, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Determine specific next action based on decision."""
        actions = {
            DecisionType.CONTINUE: {
                "action": "proceed_to_next_step",
                "params": {"skip_validation": False}
            },
            DecisionType.RETRY: {
                "action": "retry_last_step",
                "params": {"delay_ms": 1000, "max_attempts": 3}
            },
            DecisionType.ESCALATE: {
                "action": "notify_human_operator",
                "params": {"urgency": "high", "include_diagnostics": True}
            },
            DecisionType.OPTIMIZE: {
                "action": "optimize_workflow",
                "params": {"target_bottlenecks": self.metrics.bottleneck_nodes}
            },
            DecisionType.COMPLETE: {
                "action": "finalize_workflow",
                "params": {"generate_report": True}
            },
            DecisionType.ABORT: {
                "action": "stop_workflow",
                "params": {"cleanup": True, "save_state": True}
            },
            DecisionType.DELEGATE: {
                "action": "delegate_to_specialist",
                "params": {"specialist_type": "error_recovery"}
            },
            DecisionType.WAIT: {
                "action": "pause_workflow",
                "params": {"timeout_ms": 5000}
            }
        }
        
        return actions.get(decision, {"action": "unknown", "params": {}})
    
    def learn_from_outcome(self, workflow_id: str, outcome: float):
        """Learn from workflow outcome to improve future decisions."""
        if not self.enable_learning:
            return
        
        if workflow_id in self.workflow_history:
            history = self.workflow_history[workflow_id]
            decision = history["decision"]
            
            # Update decision weights
            self.decision_engine.update_weights(decision, outcome)
            
            # Clean up
            self.workflow_history.pop(workflow_id, None)
            self.active_workflows.pop(workflow_id, None)


# Example usage and testing
async def test_real_supervisor():
    """Test the real supervisor implementation."""
    
    # Create supervisor
    supervisor = RealSupervisor(
        risk_threshold=0.7,
        enable_learning=True,
        enable_patterns=True
    )
    
    # Simulate workflow state
    test_state = {
        "workflow_id": "test-workflow-123",
        "total_steps": 10,
        "completed_steps": ["step1", "step2", "step3"],
        "step_results": [
            {"node_name": "analyzer", "success": True, "duration_ms": 1200},
            {"node_name": "processor", "success": True, "duration_ms": 800},
            {"node_name": "validator", "success": False, "duration_ms": 2000, "error_type": "ValidationError"},
            {"node_name": "validator", "success": False, "duration_ms": 2200, "retry_count": 1, "error_type": "ValidationError"},
        ],
        "resource_usage": {
            "cpu": 0.75,
            "memory": 0.60,
            "network": 0.30
        }
    }
    
    # Run supervisor
    result = await supervisor(test_state)
    
    # Print results
    print(json.dumps(result["supervisor_decision"], indent=2))
    
    # Simulate learning from outcome
    supervisor.learn_from_outcome("test-workflow-123", 0.8)  # Good outcome


if __name__ == "__main__":
    # Run test
    asyncio.run(test_real_supervisor())