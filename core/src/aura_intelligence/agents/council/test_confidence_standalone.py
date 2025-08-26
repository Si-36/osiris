#!/usr/bin/env python3
"""
Standalone Test for Confidence Scoring - Task 7 Implementation

This test validates the confidence scoring implementation without external dependencies.
It includes all necessary components inline to test the core logic.
"""

import torch
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
import structlog

# Configure basic logging
import logging
logging.basicConfig(level=logging.INFO)
logger = structlog.get_logger()


# Inline model definitions for testing
@dataclass
class GPUAllocationRequest:
    """GPU allocation request model."""
    request_id: str
    user_id: str
    project_id: str
    gpu_type: str
    gpu_count: int
    memory_gb: int
    compute_hours: float
    priority: int
    special_requirements: List[str] = field(default_factory=list)
    requires_infiniband: bool = False
    requires_nvlink: bool = False
    deadline: Optional[datetime] = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class LNNCouncilState:
    """LNN Council state model."""
    current_request: Optional[GPUAllocationRequest] = None
    context_cache: Dict[str, Any] = field(default_factory=dict)
    decision_history: List[Dict[str, Any]] = field(default_factory=list)
    confidence_score: float = 0.0
    validation_passed: bool = False
    fallback_triggered: bool = False


# Inline confidence scoring implementation
@dataclass
class ConfidenceMetrics:
    """Metrics for confidence scoring."""
    
    # Neural network confidence
    neural_confidence: float = 0.0
    output_entropy: float = 0.0
    activation_stability: float = 0.0
    
    # Context confidence
    context_quality: float = 0.0
    historical_similarity: float = 0.0
    knowledge_completeness: float = 0.0
    
    # Decision confidence
    constraint_satisfaction: float = 0.0
    resource_availability: float = 0.0
    risk_assessment: float = 0.0
    
    # Overall confidence
    overall_confidence: float = 0.0
    confidence_breakdown: Dict[str, float] = field(default_factory=dict)


@dataclass
class ValidationResult:
    """Result of decision validation."""
    
    is_valid: bool = True
    validation_score: float = 1.0
    violations: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    constraints_checked: List[str] = field(default_factory=list)
    
    def add_violation(self, constraint: str, message: str):
        """Add a constraint violation."""
        self.violations.append(f"{constraint}: {message}")
        self.is_valid = False
    
    def add_warning(self, constraint: str, message: str):
        """Add a constraint warning."""
        self.warnings.append(f"{constraint}: {message}")


class ConfidenceScorer:
    """Advanced confidence scoring for LNN council decisions."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.confidence_threshold = config.get("confidence_threshold", 0.7)
        self.entropy_weight = config.get("entropy_weight", 0.2)
        self.context_weight = config.get("context_weight", 0.3)
        self.constraint_weight = config.get("constraint_weight", 0.3)
        self.neural_weight = config.get("neural_weight", 0.2)
        
        # Calibration parameters
        self.calibration_alpha = config.get("calibration_alpha", 1.0)
        self.calibration_beta = config.get("calibration_beta", 0.0)
    
    def calculate_confidence(
        self,
        neural_output: torch.Tensor,
        state: LNNCouncilState,
        decision: str,
        context_quality: Optional[float] = None
        ) -> ConfidenceMetrics:
            pass
        """Calculate comprehensive confidence score for a decision."""
        
        metrics = ConfidenceMetrics()
        
        # 1. Neural network confidence
        metrics.neural_confidence = self._calculate_neural_confidence(neural_output)
        metrics.output_entropy = self._calculate_output_entropy(neural_output)
        metrics.activation_stability = self._calculate_activation_stability(neural_output)
        
        # 2. Context confidence
        metrics.context_quality = context_quality or self._assess_context_quality(state)
        metrics.historical_similarity = self._calculate_historical_similarity(state)
        metrics.knowledge_completeness = self._assess_knowledge_completeness(state)
        
        # 3. Decision-specific confidence
        metrics.constraint_satisfaction = self._assess_constraint_satisfaction(state, decision)
        metrics.resource_availability = self._assess_resource_availability(state)
        metrics.risk_assessment = self._calculate_risk_score(state, decision)
        
        # 4. Combine into overall confidence
        metrics.overall_confidence = self._combine_confidence_scores(metrics)
        
        # 5. Apply calibration
        metrics.overall_confidence = self._calibrate_confidence(metrics.overall_confidence)
        
        # 6. Create detailed breakdown
        metrics.confidence_breakdown = {
            "neural": metrics.neural_confidence * self.neural_weight,
            "context": metrics.context_quality * self.context_weight,
            "constraints": metrics.constraint_satisfaction * self.constraint_weight,
            "entropy_penalty": -metrics.output_entropy * self.entropy_weight,
            "historical": metrics.historical_similarity * 0.1,
            "risk_penalty": -metrics.risk_assessment * 0.1
        }
        
        return metrics
    
    def _calculate_neural_confidence(self, neural_output: torch.Tensor) -> float:
        """Calculate confidence from neural network output."""
        probs = torch.softmax(neural_output.squeeze(), dim=-1)
        max_prob = torch.max(probs).item()
        
        # Adjust for probability distribution sharpness
        entropy = -torch.sum(probs * torch.log(probs + 1e-8)).item()
        max_entropy = np.log(len(probs))
        normalized_entropy = entropy / max_entropy
        
        confidence = max_prob * (1.0 - normalized_entropy * 0.5)
        return float(np.clip(confidence, 0.0, 1.0))
    
    def _calculate_output_entropy(self, neural_output: torch.Tensor) -> float:
        """Calculate entropy of neural network output."""
        probs = torch.softmax(neural_output.squeeze(), dim=-1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-8)).item()
        max_entropy = np.log(len(probs))
        return entropy / max_entropy
    
    def _calculate_activation_stability(self, neural_output: torch.Tensor) -> float:
        """Calculate stability of neural activations."""
        variance = torch.var(neural_output).item()
        stability = 1.0 / (1.0 + variance)
        return float(np.clip(stability, 0.0, 1.0))
    
    def _assess_context_quality(self, state: LNNCouncilState) -> float:
        """Assess quality of available context."""
        context_cache = state.context_cache
        if not context_cache:
            return 0.3
        
        key_contexts = ["current_utilization", "user_history", "project_context", "system_constraints"]
        quality_factors = []
        
        for context_key in key_contexts:
            if context_key in context_cache:
                context_data = context_cache[context_key]
                if isinstance(context_data, dict) and context_data:
                    quality_factors.append(1.0)
                else:
                    quality_factors.append(0.5)
            else:
                quality_factors.append(0.0)
        
        base_quality = np.mean(quality_factors) if quality_factors else 0.0
        richness_bonus = min(0.2, len(context_cache) * 0.02)
        
        return float(np.clip(base_quality + richness_bonus, 0.0, 1.0))
    
    def _calculate_historical_similarity(self, state: LNNCouncilState) -> float:
        """Calculate similarity to historical decisions."""
        history_length = len(state.decision_history)
        if history_length == 0:
            return 0.3
        elif history_length < 5:
            return 0.6
        elif history_length < 20:
            return 0.8
        else:
            return 0.9
    
    def _assess_knowledge_completeness(self, state: LNNCouncilState) -> float:
        """Assess completeness of knowledge graph context."""
        context_cache = state.context_cache
        kg_indicators = ["entity_context", "relationship_context", "topology_context", "temporal_context"]
        
        kg_completeness = []
        for indicator in kg_indicators:
            if indicator in context_cache:
                kg_completeness.append(1.0)
            else:
                kg_completeness.append(0.0)
        
        return np.mean(kg_completeness) if kg_completeness else 0.5
    
    def _assess_constraint_satisfaction(self, state: LNNCouncilState, decision: str) -> float:
        """Assess how well the decision satisfies constraints."""
        if not state.current_request:
            return 0.0
        
        request = state.current_request
        context = state.context_cache
        satisfaction_score = 1.0
        
        # Resource availability check
        if decision == "approve":
            utilization = context.get("current_utilization", {})
            gpu_usage = utilization.get("gpu_usage", 0.5)
            
            if gpu_usage > 0.9:
                satisfaction_score *= 0.5
            elif gpu_usage > 0.8:
                satisfaction_score *= 0.8
        
        # Budget constraint check
        if decision == "approve":
            project_context = context.get("project_context", {})
            budget_remaining = project_context.get("budget_remaining", 1000.0)
            
            estimated_cost = request.gpu_count * request.compute_hours * 2.5
            
            if estimated_cost > budget_remaining:
                satisfaction_score *= 0.2
            elif estimated_cost > budget_remaining * 0.8:
                satisfaction_score *= 0.7
        
        # Priority alignment check
        if request.priority >= 8 and decision == "deny":
            satisfaction_score *= 0.6
        elif request.priority <= 3 and decision == "approve":
            satisfaction_score *= 0.8
        
        return float(np.clip(satisfaction_score, 0.0, 1.0))
    
    def _assess_resource_availability(self, state: LNNCouncilState) -> float:
        """Assess current resource availability."""
        context = state.context_cache
        utilization = context.get("current_utilization", {})
        
        gpu_usage = utilization.get("gpu_usage", 0.5)
        queue_length = utilization.get("queue_length", 0)
        
        availability = (1.0 - gpu_usage) * 0.7 + max(0, 1.0 - queue_length / 20.0) * 0.3
        return float(np.clip(availability, 0.0, 1.0))
    
    def _calculate_risk_score(self, state: LNNCouncilState, decision: str) -> float:
        """Calculate risk score for the decision."""
        if not state.current_request:
            return 0.5
        
        request = state.current_request
        risk_score = 0.0
        
        if decision == "approve":
            # Resource risk
            resource_risk = min(1.0, request.gpu_count / 8.0)
            risk_score += resource_risk * 0.4
            
            # Duration risk
            duration_risk = min(1.0, request.compute_hours / 168.0)
            risk_score += duration_risk * 0.3
            
            # Special requirements risk
            special_risk = len(request.special_requirements) / 5.0
            risk_score += special_risk * 0.3
        
        return float(np.clip(risk_score, 0.0, 1.0))
    
    def _combine_confidence_scores(self, metrics: ConfidenceMetrics) -> float:
        """Combine individual confidence scores into overall confidence."""
        overall = (
            metrics.neural_confidence * self.neural_weight +
            metrics.context_quality * self.context_weight +
            metrics.constraint_satisfaction * self.constraint_weight -
            metrics.output_entropy * self.entropy_weight
        )
        
        # Apply penalties
        overall *= (1.0 - metrics.risk_assessment * 0.2)
        overall *= (0.8 + metrics.activation_stability * 0.2)
        
        return float(np.clip(overall, 0.0, 1.0))
    
    def _calibrate_confidence(self, raw_confidence: float) -> float:
        """Apply calibration to raw confidence score."""
        calibrated = 1.0 / (1.0 + np.exp(self.calibration_alpha * raw_confidence + self.calibration_beta))
        return float(np.clip(calibrated, 0.0, 1.0))


class DecisionValidator:
    """Validates LNN council decisions against system constraints."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.max_gpu_allocation = config.get("max_gpu_allocation", 8)
        self.max_duration_hours = config.get("max_duration_hours", 168)
        self.budget_check_enabled = config.get("budget_check_enabled", True)
        self.policy_check_enabled = config.get("policy_check_enabled", True)
    
    def validate_decision(
        self,
        decision: str,
        request: GPUAllocationRequest,
        state: LNNCouncilState
        ) -> ValidationResult:
            pass
        """Validate a decision against all constraints."""
        
        result = ValidationResult()
        
        # Only validate approval decisions in detail
        if decision != "approve":
            result.constraints_checked = ["decision_type"]
            return result
        
        # 1. Resource constraints
        self._validate_resource_constraints(request, state, result)
        
        # 2. Budget constraints
        if self.budget_check_enabled:
            self._validate_budget_constraints(request, state, result)
        
        # 3. Policy constraints
        if self.policy_check_enabled:
            self._validate_policy_constraints(request, state, result)
        
        # Calculate overall validation score
        if result.violations:
            result.validation_score = 0.0
        elif result.warnings:
            result.validation_score = 0.7
        else:
            result.validation_score = 1.0
        
        return result
    
    def _validate_resource_constraints(self, request: GPUAllocationRequest, state: LNNCouncilState, result: ValidationResult):
        """Validate resource availability constraints."""
        result.constraints_checked.append("resource_availability")
        
        # Check GPU count limits
        if request.gpu_count > self.max_gpu_allocation:
            result.add_violation("gpu_count", f"Requested {request.gpu_count} GPUs exceeds limit of {self.max_gpu_allocation}")
        
        # Check duration limits
        if request.compute_hours > self.max_duration_hours:
            result.add_violation("duration", f"Requested {request.compute_hours} hours exceeds limit of {self.max_duration_hours}")
        
        # Check system utilization
        context = state.context_cache
        utilization = context.get("current_utilization", {})
        gpu_usage = utilization.get("gpu_usage", 0.5)
        
        if gpu_usage > 0.95:
            result.add_violation("system_utilization", f"System utilization too high: {gpu_usage:.1%}")
        elif gpu_usage > 0.85:
            result.add_warning("system_utilization", f"System utilization high: {gpu_usage:.1%}")
    
    def _validate_budget_constraints(self, request: GPUAllocationRequest, state: LNNCouncilState, result: ValidationResult):
        """Validate budget and cost constraints."""
        result.constraints_checked.append("budget_constraints")
        
        # Estimate cost
        gpu_cost_per_hour = {"A100": 3.0, "H100": 4.0, "V100": 2.0, "RTX4090": 1.5, "RTX3090": 1.0}
        cost_per_hour = gpu_cost_per_hour.get(request.gpu_type, 2.5)
        estimated_cost = request.gpu_count * request.compute_hours * cost_per_hour
        
        # Check project budget
        context = state.context_cache
        project_context = context.get("project_context", {})
        budget_remaining = project_context.get("budget_remaining", 0.0)
        
        if estimated_cost > budget_remaining:
            result.add_violation("budget_exceeded", f"Estimated cost ${estimated_cost:.2f} exceeds remaining budget ${budget_remaining:.2f}")
        elif estimated_cost > budget_remaining * 0.8:
            result.add_warning("budget_concern", f"Estimated cost ${estimated_cost:.2f} uses {estimated_cost/budget_remaining:.1%} of remaining budget")
    
    def _validate_policy_constraints(self, request: GPUAllocationRequest, state: LNNCouncilState, result: ValidationResult):
        """Validate policy compliance constraints."""
        result.constraints_checked.append("policy_compliance")
        
        # Check user allocation limits
        context = state.context_cache
        user_history = context.get("user_history", {})
        current_allocations = user_history.get("current_allocations", 0)
        
        if current_allocations >= 3:
            result.add_violation("user_allocation_limit", f"User has {current_allocations} active allocations (limit: 3)")
        elif current_allocations >= 2:
            result.add_warning("user_allocation_concern", f"User has {current_allocations} active allocations")
        
        # Check special requirements
        if "distributed" in request.special_requirements and request.gpu_count < 2:
            result.add_violation("distributed_requirement", "Distributed training requires at least 2 GPUs")


class ReasoningPathGenerator:
    """Generates human-readable reasoning paths for LNN council decisions."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.include_technical_details = config.get("include_technical_details", False)
        self.max_reasoning_steps = config.get("max_reasoning_steps", 10)
    
    def generate_reasoning_path(
        self,
        decision: str,
        request: GPUAllocationRequest,
        confidence_metrics: ConfidenceMetrics,
        validation_result: ValidationResult,
        state: LNNCouncilState
        ) -> List[str]:
            pass
        """Generate a comprehensive reasoning path for the decision."""
        
        reasoning_path = []
        
        # 1. Request analysis
        reasoning_path.append(
            f"Analyzed request for {request.gpu_count}x {request.gpu_type} "
            f"for {request.compute_hours}h (Priority: {request.priority})"
        )
        
        # 2. Context assessment
        context_quality = confidence_metrics.context_quality
        if context_quality > 0.8:
            reasoning_path.append("High-quality context available from knowledge graphs and memory")
        elif context_quality > 0.5:
            reasoning_path.append("Moderate context available, some information gaps")
        else:
            reasoning_path.append("Limited context available, decision based on request data")
        
        # 3. Neural network analysis
        neural_conf = confidence_metrics.neural_confidence
        if neural_conf > 0.8:
            reasoning_path.append(f"Neural network strongly recommends '{decision}' (confidence: {neural_conf:.2f})")
        elif neural_conf > 0.6:
            reasoning_path.append(f"Neural network moderately supports '{decision}' (confidence: {neural_conf:.2f})")
        else:
            reasoning_path.append(f"Neural network has low confidence in '{decision}' (confidence: {neural_conf:.2f})")
        
        # 4. Constraint validation
        if validation_result.violations:
            reasoning_path.append(f"Found {len(validation_result.violations)} constraint violations:")
            for violation in validation_result.violations[:3]:
                reasoning_path.append(f"  - {violation}")
        elif validation_result.warnings:
            reasoning_path.append(f"Found {len(validation_result.warnings)} warnings:")
            for warning in validation_result.warnings[:2]:
                reasoning_path.append(f"  - {warning}")
        else:
            reasoning_path.append("All constraints satisfied")
        
        # 5. Resource considerations
        resource_availability = confidence_metrics.resource_availability
        if resource_availability > 0.8:
            reasoning_path.append("Sufficient resources available")
        elif resource_availability > 0.5:
            reasoning_path.append("Moderate resource availability")
        else:
            reasoning_path.append("Limited resource availability")
        
        # 6. Final decision rationale
        overall_confidence = confidence_metrics.overall_confidence
        if decision == "approve":
            reasoning_path.append(
                f"APPROVED: Overall confidence {overall_confidence:.2f} meets threshold, "
                f"constraints satisfied, resources available"
            )
        elif decision == "deny":
            if validation_result.violations:
                reasoning_path.append("DENIED: Constraint violations prevent approval")
            else:
                reasoning_path.append(f"DENIED: Low confidence ({overall_confidence:.2f}) or resource constraints")
        else:  # defer
            reasoning_path.append(
                f"DEFERRED: Insufficient confidence ({overall_confidence:.2f}) or "
                f"resource constraints require manual review"
            )
        
        # 7. Technical details (if enabled)
        if self.include_technical_details:
            reasoning_path.append("Technical details:")
            reasoning_path.append(f"  - Neural confidence: {neural_conf:.3f}")
            reasoning_path.append(f"  - Context quality: {context_quality:.3f}")
            reasoning_path.append(f"  - Entropy: {confidence_metrics.output_entropy:.3f}")
        
        return reasoning_path


# Test functions
    def test_confidence_scorer_basic():
        """Test basic confidence scorer functionality."""
        try:
            pass
        config = {
            "confidence_threshold": 0.7,
            "entropy_weight": 0.2,
            "context_weight": 0.3,
            "constraint_weight": 0.3,
            "neural_weight": 0.2
        }
        
        scorer = ConfidenceScorer(config)
        
        # Test neural confidence calculation
        high_conf_output = torch.tensor([5.0, 1.0, 0.5])
        confidence = scorer._calculate_neural_confidence(high_conf_output)
        
        assert 0.0 <= confidence <= 1.0, f"Confidence out of range: {confidence}"
        assert confidence > 0.7, f"High confidence output should yield high confidence: {confidence}"
        
        print(f"✅ Neural confidence calculation: {confidence:.3f}")
        
        # Test entropy calculation
        entropy = scorer._calculate_output_entropy(high_conf_output)
        assert 0.0 <= entropy <= 1.0, f"Entropy out of range: {entropy}"
        assert entropy < 0.5, f"Sharp distribution should have low entropy: {entropy}"
        
        print(f"✅ Entropy calculation: {entropy:.3f}")
        
        return True
        
        except Exception as e:
            pass
        print(f"❌ Confidence scorer basic test failed: {e}")
        return False


    def test_decision_validator_basic():
        """Test basic decision validator functionality."""
        try:
            pass
        config = {
            "max_gpu_allocation": 8,
            "max_duration_hours": 168,
            "budget_check_enabled": True,
            "policy_check_enabled": True
        }
        
        validator = DecisionValidator(config)
        
        # Create test data
        request = GPUAllocationRequest(
            request_id="test_001",
            user_id="test_user",
            project_id="test_project",
            gpu_type="A100",
            gpu_count=2,
            memory_gb=40,
            compute_hours=8.0,
            priority=7
        )
        
        state = LNNCouncilState(current_request=request)
        state.context_cache = {
            "current_utilization": {"gpu_usage": 0.6, "queue_length": 5},
            "user_history": {"current_allocations": 1},
            "project_context": {"budget_remaining": 1000.0}
        }
        
        # Test approve decision validation
        result = validator.validate_decision("approve", request, state)
        
        assert isinstance(result, ValidationResult)
        assert result.is_valid is True
        assert len(result.constraints_checked) > 0
        
        print(f"✅ Decision validation: valid={result.is_valid}, score={result.validation_score:.2f}")
        
        return True
        
        except Exception as e:
            pass
        print(f"❌ Decision validator basic test failed: {e}")
        return False


    def test_full_confidence_calculation():
        """Test full confidence calculation pipeline."""
        try:
            pass
        config = {
            "confidence_threshold": 0.7,
            "entropy_weight": 0.2,
            "context_weight": 0.3,
            "constraint_weight": 0.3,
            "neural_weight": 0.2
        }
        
        scorer = ConfidenceScorer(config)
        
        # Create test data
        request = GPUAllocationRequest(
            request_id="test_001",
            user_id="test_user",
            project_id="test_project",
            gpu_type="A100",
            gpu_count=2,
            memory_gb=40,
            compute_hours=8.0,
            priority=7
        )
        
        state = LNNCouncilState(current_request=request)
        state.context_cache = {
            "current_utilization": {"gpu_usage": 0.6, "queue_length": 5},
            "user_history": {"successful_allocations": 10, "avg_usage": 0.8},
            "project_context": {"budget_remaining": 1000.0, "priority_tier": "high"}
        }
        state.decision_history = [{"decision": "approve"}, {"decision": "approve"}]
        
        # Create realistic neural output
        neural_output = torch.tensor([2.5, 1.0, 0.5])
        
        # Calculate confidence
        metrics = scorer.calculate_confidence(
            neural_output=neural_output,
            state=state,
            decision="approve"
        )
        
        # Verify all metrics are calculated
        assert isinstance(metrics, ConfidenceMetrics)
        assert 0.0 <= metrics.neural_confidence <= 1.0
        assert 0.0 <= metrics.overall_confidence <= 1.0
        assert isinstance(metrics.confidence_breakdown, dict)
        assert len(metrics.confidence_breakdown) > 0
        
        print(f"✅ Full confidence calculation:")
        print(f"   Neural: {metrics.neural_confidence:.3f}")
        print(f"   Context: {metrics.context_quality:.3f}")
        print(f"   Overall: {metrics.overall_confidence:.3f}")
        print(f"   Breakdown: {len(metrics.confidence_breakdown)} components")
        
        return True
        
        except Exception as e:
            pass
        print(f"❌ Full confidence calculation test failed: {e}")
        return False


    def test_reasoning_path_generator():
        """Test reasoning path generator functionality."""
        try:
            pass
        config = {
            "include_technical_details": True,
            "max_reasoning_steps": 10
        }
        
        generator = ReasoningPathGenerator(config)
        
        # Create test data
        request = GPUAllocationRequest(
            request_id="test_001",
            user_id="test_user",
            project_id="test_project",
            gpu_type="A100",
            gpu_count=2,
            memory_gb=40,
            compute_hours=8.0,
            priority=7
        )
        
        confidence_metrics = ConfidenceMetrics()
        confidence_metrics.neural_confidence = 0.85
        confidence_metrics.context_quality = 0.8
        confidence_metrics.overall_confidence = 0.82
        confidence_metrics.resource_availability = 0.9
        
        validation_result = ValidationResult()
        validation_result.is_valid = True
        validation_result.validation_score = 0.95
        
        state = LNNCouncilState(current_request=request)
        
        # Generate reasoning path
        reasoning_path = generator.generate_reasoning_path(
            decision="approve",
            request=request,
            confidence_metrics=confidence_metrics,
            validation_result=validation_result,
            state=state
        )
        
        assert isinstance(reasoning_path, list)
        assert len(reasoning_path) > 0
        
        # Check for key elements
        reasoning_text = " ".join(reasoning_path)
        assert "2x A100" in reasoning_text
        assert "8.0h" in reasoning_text
        assert "APPROVED" in reasoning_text
        
        print(f"✅ Reasoning path generation: {len(reasoning_path)} steps")
        print(f"   Sample: {reasoning_path[0][:80]}...")
        
        return True
        
        except Exception as e:
            pass
        print(f"❌ Reasoning path generator test failed: {e}")
        return False


    def test_integration_scenario():
        """Test integration scenario with all components."""
        try:
            pass
        # Initialize all components
        config = {
            "confidence_threshold": 0.7,
            "entropy_weight": 0.2,
            "context_weight": 0.3,
            "constraint_weight": 0.3,
            "neural_weight": 0.2,
            "max_gpu_allocation": 8,
            "max_duration_hours": 168,
            "budget_check_enabled": True,
            "policy_check_enabled": True,
            "include_technical_details": True,
            "max_reasoning_steps": 12
        }
        
        scorer = ConfidenceScorer(config)
        validator = DecisionValidator(config)
        generator = ReasoningPathGenerator(config)
        
        # Create test scenario
        request = GPUAllocationRequest(
            request_id="integration_001",
            user_id="integration_user",
            project_id="integration_project",
            gpu_type="A100",
            gpu_count=4,
            memory_gb=80,
            compute_hours=12.0,
            priority=8,
            special_requirements=["high_memory"]
        )
        
        state = LNNCouncilState(current_request=request)
        state.context_cache = {
            "current_utilization": {"gpu_usage": 0.7, "queue_length": 8},
            "user_history": {"successful_allocations": 15, "current_allocations": 1},
            "project_context": {"budget_remaining": 2000.0, "security_level": "standard"},
            "system_constraints": {"maintenance_window": None}
        }
        
        neural_output = torch.tensor([3.5, 1.5, 0.8])
        decision = "approve"
        
        # Step 1: Calculate confidence
        confidence_metrics = scorer.calculate_confidence(
            neural_output=neural_output,
            state=state,
            decision=decision
        )
        
        # Step 2: Validate decision
        validation_result = validator.validate_decision(decision, request, state)
        
        # Step 3: Generate reasoning
        reasoning_path = generator.generate_reasoning_path(
            decision=decision,
            request=request,
            confidence_metrics=confidence_metrics,
            validation_result=validation_result,
            state=state
        )
        
        # Verify integration
        assert isinstance(confidence_metrics, ConfidenceMetrics)
        assert isinstance(validation_result, ValidationResult)
        assert isinstance(reasoning_path, list)
        
        assert confidence_metrics.overall_confidence > 0.0
        assert validation_result.validation_score > 0.0
        assert len(reasoning_path) > 5
        
        print(f"✅ Integration scenario:")
        print(f"   Confidence: {confidence_metrics.overall_confidence:.3f}")
        print(f"   Validation: {validation_result.validation_score:.3f}")
        print(f"   Reasoning: {len(reasoning_path)} steps")
        print(f"   Valid: {validation_result.is_valid}")
        
        return True
        
        except Exception as e:
            pass
        print(f"❌ Integration scenario test failed: {e}")
        return False


    def main():
        """Run all confidence scoring tests."""
        print("🧪 Confidence Scoring Standalone Tests - Task 7 Implementation")
        print("=" * 70)
    
        tests = [
        ("Confidence Scorer Basic", test_confidence_scorer_basic),
        ("Decision Validator Basic", test_decision_validator_basic),
        ("Full Confidence Calculation", test_full_confidence_calculation),
        ("Reasoning Path Generator", test_reasoning_path_generator),
        ("Integration Scenario", test_integration_scenario)
        ]
    
        results = []
        for test_name, test_func in tests:
            pass
        print(f"\n🔍 Running: {test_name}")
        try:
            result = test_func()
            results.append(result)
            if result:
                print(f"✅ {test_name}: PASSED")
            else:
                print(f"❌ {test_name}: FAILED")
        except Exception as e:
            print(f"❌ {test_name}: ERROR - {e}")
            results.append(False)
    
        print("\n" + "=" * 70)
        print(f"📊 Test Results: {sum(results)}/{len(results)} passed")
    
        if all(results):
            pass
        print("🎉 ALL CONFIDENCE SCORING TESTS PASSED!")
        print("\n✅ Task 7 Implementation Complete:")
        print("   • Confidence scoring based on neural network outputs ✅")
        print("   • Decision validation against system constraints ✅")
        print("   • Reasoning path generation for explainable decisions ✅")
        print("   • Unit tests for confidence calculation and validation logic ✅")
        print("\n🚀 Ready for Task 8: Implement Fallback Mechanisms")
        return 0
        else:
        print("❌ Some tests failed")
        return 1


        if __name__ == "__main__":
        exit(main())
