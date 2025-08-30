"""
🧠 Advanced Supervisor 2025 - Production Ready
State-of-the-art supervisor with real implementation, not mock code.
"""

import asyncio
import json
import time
import numpy as np
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Tuple
from enum import Enum
from dataclasses import dataclass, field
from collections import defaultdict, deque
import hashlib

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig
import structlog

from ..state import CollectiveState, NodeResult

logger = structlog.get_logger(__name__)


class DecisionType(str, Enum):
    """Enhanced decision types."""
    CONTINUE = "continue"
    ESCALATE = "escalate"
    RETRY = "retry"
    COMPLETE = "complete"
    ABORT = "abort"
    DELEGATE = "delegate"
    WAIT = "wait"
    OPTIMIZE = "optimize"
    CHECKPOINT = "checkpoint"
    ROLLBACK = "rollback"


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
    
    def update(self, step_result: Dict[str, Any]):
        """Update metrics with step result."""
        self.total_steps += 1
        
        if step_result.get("success"):
            self.completed_steps += 1
        else:
            self.failed_steps += 1
            
        if step_result.get("retried"):
            self.retried_steps += 1
            
        duration = step_result.get("duration_ms", 0)
        self.total_duration_ms += duration
        self.average_step_duration_ms = self.total_duration_ms / self.total_steps
        
        self.error_rate = self.failed_steps / self.total_steps if self.total_steps > 0 else 0
        self.success_rate = self.completed_steps / self.total_steps if self.total_steps > 0 else 0
        
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
    confidence: float = 0.8
    
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
    
    def update(self, node_name: str, result: Dict[str, Any]):
        """Update pattern history."""
        self.pattern_history[node_name].append({
            "success": result.get("success", False),
            "duration_ms": result.get("duration_ms", 0),
            "timestamp": time.time(),
            "retry_count": result.get("retry_count", 0)
        })
    
    def detect_patterns(self) -> Dict[str, List[str]]:
        """Detect patterns across all nodes."""
        patterns = defaultdict(list)
        
        for node_name, history in self.pattern_history.items():
            if len(history) >= 3:
                # Retry loop detection
                if all(h.get("retry_count", 0) > 2 for h in list(history)[-3:]):
                    patterns["retry_loop"].append(node_name)
                
                # Cascading failure detection
                recent_failures = [not h["success"] for h in list(history)[-4:]]
                if sum(recent_failures) >= 3:
                    patterns["cascading_failure"].append(node_name)
                
                # Performance degradation
                if len(history) >= 5:
                    durations = [h["duration_ms"] for h in list(history)[-5:]]
                    if all(durations[i] > durations[i-1] * 1.2 for i in range(1, len(durations))):
                        patterns["performance_degradation"].append(node_name)
        
        return dict(patterns)


class SupervisorNode:
    """
    Advanced Supervisor with real implementation.
    
    Features:
    - Real state analysis
    - Pattern detection
    - Risk assessment
    - Intelligent decision making
    - Performance optimization
    """
    
    def __init__(self, llm=None, risk_threshold: float = 0.7):
        self.llm = llm
        self.risk_threshold = risk_threshold
        self.name = "supervisor"
        
        # Core components
        self.metrics = WorkflowMetrics()
        self.pattern_detector = PatternDetector()
        
        # State tracking
        self.workflow_history = {}
    
    async def __call__(
        self,
        state: CollectiveState,
        config: Optional[RunnableConfig] = None
    ) -> Dict[str, Any]:
        """Execute supervisor decision logic."""
        start_time = time.time()
        workflow_id = state.get("workflow_id", "unknown")
        
        try:
            logger.info(
                "Advanced supervisor analyzing workflow",
                workflow_id=workflow_id,
                steps_completed=len(state.get("completed_steps", []))
            )
            
            # Analyze state
            state_analysis = self._analyze_state(state)
            
            # Assess risk
            risk_assessment = self._assess_risk(state, state_analysis)
            
            # Detect patterns
            patterns = self.pattern_detector.detect_patterns()
            
            # Make decision
            decision = self._make_decision(state_analysis, risk_assessment, patterns)
            
            # Build result
            duration_ms = (time.time() - start_time) * 1000
            
            decision_record = {
                "decision": decision.value,
                "reasoning": {
                    "risk_level": risk_assessment.risk_level,
                    "confidence": risk_assessment.confidence,
                    "patterns_detected": list(patterns.keys()),
                    "primary_factors": self._get_decision_factors(state_analysis, risk_assessment)
                },
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
            result = NodeResult(
                success=True,
                node_name=self.name,
                output=decision_record,
                duration_ms=duration_ms,
                next_node=self._determine_next_node(decision)
            )
            
            # Update state
            updates = {
                "supervisor_decisions": [decision_record],
                "current_step": f"supervisor_decided_{decision.value}",
                "risk_assessment": {
                    "score": risk_assessment.overall_score,
                    "level": risk_assessment.risk_level,
                    "factors": risk_assessment.risk_factors
                },
                "workflow_metrics": {
                    "success_rate": self.metrics.success_rate,
                    "error_rate": self.metrics.error_rate,
                    "bottlenecks": self.metrics.bottleneck_nodes
                }
            }
            
            # Add message
            message = AIMessage(
                content=f"Decision: {decision.value} (risk: {risk_assessment.risk_level}, confidence: {risk_assessment.confidence:.2f})",
                additional_kwargs={"node": self.name, "decision": decision.value}
            )
            updates["messages"] = [message]
            
            logger.info(
                "Supervisor decision complete",
                workflow_id=workflow_id,
                decision=decision.value,
                risk_level=risk_assessment.risk_level,
                duration_ms=duration_ms
            )
            
            return updates
            
        except Exception as e:
            logger.error("Supervisor error", error=str(e))
            return {
                "supervisor_decisions": [{
                    "decision": DecisionType.ESCALATE.value,
                    "error": str(e),
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }]
            }
    
    def _analyze_state(self, state: CollectiveState) -> Dict[str, Any]:
        """Perform comprehensive state analysis."""
        # Update metrics
        for result in state.get("step_results", []):
            self.metrics.update(result)
            node_name = result.get("node_name", "unknown")
            self.pattern_detector.update(node_name, result)
        
        analysis = {
            "workflow_progress": len(state.get("completed_steps", [])) / max(state.get("total_steps", 1), 1),
            "health_score": 1.0 - self.metrics.error_rate,
            "performance_score": 1.0 - min(1.0, self.metrics.average_step_duration_ms / 5000),
            "has_critical_errors": any(r.get("error_type") == "CriticalError" for r in state.get("step_results", [])),
            "resource_constraints": self._check_resource_constraints(state)
        }
        
        return analysis
    
    def _assess_risk(self, state: CollectiveState, analysis: Dict[str, Any]) -> RiskAssessment:
        """Assess workflow risk."""
        risk_factors = {}
        
        # Error risk
        if self.metrics.error_rate > 0:
            risk_factors["error_rate"] = min(1.0, self.metrics.error_rate * 2)
        
        # Performance risk
        if self.metrics.average_step_duration_ms > 3000:
            risk_factors["performance"] = min(1.0, self.metrics.average_step_duration_ms / 10000)
        
        # Resource risk
        resources = state.get("resource_usage", {})
        if any(usage > 0.8 for usage in resources.values()):
            risk_factors["resources"] = max(resources.values())
        
        # Critical error risk
        if analysis["has_critical_errors"]:
            risk_factors["critical_errors"] = 0.9
        
        # Calculate overall risk
        if risk_factors:
            overall_score = sum(risk_factors.values()) / len(risk_factors)
        else:
            overall_score = 0.0
        
        # Determine mitigations
        mitigations = []
        if risk_factors.get("error_rate", 0) > 0.5:
            mitigations.extend(["increase_retries", "add_error_handling"])
        if risk_factors.get("performance", 0) > 0.5:
            mitigations.extend(["optimize_bottlenecks", "increase_resources"])
        if risk_factors.get("resources", 0) > 0.5:
            mitigations.extend(["scale_horizontally", "implement_throttling"])
        
        return RiskAssessment(
            overall_score=overall_score,
            risk_factors=risk_factors,
            mitigations=mitigations,
            confidence=0.85
        )
    
    def _make_decision(
        self,
        analysis: Dict[str, Any],
        risk: RiskAssessment,
        patterns: Dict[str, List[str]]
    ) -> DecisionType:
        """Make intelligent decision based on analysis."""
        
        # Critical risk - escalate
        if risk.risk_level == "critical":
            return DecisionType.ESCALATE
        
        # Retry loop detected - abort or rollback
        if "retry_loop" in patterns:
            return DecisionType.ABORT
        
        # Cascading failures - escalate
        if "cascading_failure" in patterns:
            return DecisionType.ESCALATE
        
        # Performance degradation - optimize
        if "performance_degradation" in patterns:
            return DecisionType.OPTIMIZE
        
        # High error rate but not critical - retry
        if self.metrics.error_rate > 0.3 and risk.risk_level != "critical":
            return DecisionType.RETRY
        
        # Near completion - push to complete
        if analysis["workflow_progress"] > 0.8:
            return DecisionType.COMPLETE
        
        # Low risk, good progress - continue
        if risk.risk_level == "low" and analysis["health_score"] > 0.7:
            return DecisionType.CONTINUE
        
        # Medium risk - checkpoint for safety
        if risk.risk_level == "medium":
            return DecisionType.CHECKPOINT
        
        # Default - continue with caution
        return DecisionType.CONTINUE
    
    def _determine_next_node(self, decision: DecisionType) -> Optional[str]:
        """Determine next node based on decision."""
        routing_map = {
            DecisionType.CONTINUE: "executor",
            DecisionType.RETRY: "retry_handler",
            DecisionType.ESCALATE: "escalation_handler",
            DecisionType.COMPLETE: "completion_handler",
            DecisionType.ABORT: "cleanup_handler",
            DecisionType.OPTIMIZE: "optimizer",
            DecisionType.CHECKPOINT: "checkpoint_handler",
            DecisionType.ROLLBACK: "rollback_handler"
        }
        return routing_map.get(decision)
    
    def _check_resource_constraints(self, state: CollectiveState) -> List[str]:
        """Check for resource constraints."""
        constraints = []
        resources = state.get("resource_usage", {})
        
        if resources.get("cpu", 0) > 0.8:
            constraints.append("high_cpu")
        if resources.get("memory", 0) > 0.8:
            constraints.append("high_memory")
        if resources.get("network", 0) > 0.8:
            constraints.append("high_network")
        
        return constraints
    
    def _get_decision_factors(self, analysis: Dict[str, Any], risk: RiskAssessment) -> List[str]:
        """Get primary factors influencing the decision."""
        factors = []
        
        if risk.overall_score > 0.7:
            factors.append(f"High risk ({risk.risk_level})")
        
        if self.metrics.error_rate > 0.3:
            factors.append(f"Error rate: {self.metrics.error_rate:.1%}")
        
        if self.metrics.bottleneck_nodes:
            factors.append(f"Bottlenecks: {', '.join(self.metrics.bottleneck_nodes[:3])}")
        
        if analysis["workflow_progress"] > 0.8:
            factors.append(f"Near completion: {analysis['workflow_progress']:.0%}")
        
        if not factors:
            factors.append("Normal operation")
        
        return factors


# Maintain compatibility with existing code
UnifiedAuraSupervisor = SupervisorNode
