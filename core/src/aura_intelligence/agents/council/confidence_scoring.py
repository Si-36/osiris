#!/usr/bin/env python3
"""
Confidence Scoring and Decision Validation for LNN Council Agent

This module implements sophisticated confidence scoring based on neural network outputs,
decision validation against system constraints, and reasoning path generation for
explainable AI decisions.
"""

import torch
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timezone
import structlog

from .models import GPUAllocationRequest, GPUAllocationDecision, LNNCouncilState

logger = structlog.get_logger()


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
    """
    Advanced confidence scoring for LNN council decisions.
    
    Combines multiple confidence signals:
    - Neural network output confidence
    - Context quality and completeness
    - Historical decision similarity
    - Constraint satisfaction
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.confidence_threshold = config.get("confidence_threshold", 0.7)
        self.entropy_weight = config.get("entropy_weight", 0.2)
        self.context_weight = config.get("context_weight", 0.3)
        self.constraint_weight = config.get("constraint_weight", 0.3)
        self.neural_weight = config.get("neural_weight", 0.2)
        
        # Calibration parameters (learned from historical data)
        self.calibration_alpha = config.get("calibration_alpha", 1.0)
        self.calibration_beta = config.get("calibration_beta", 0.0)
        
        logger.info(
            "ConfidenceScorer initialized",
            threshold=self.confidence_threshold,
            weights={
                "neural": self.neural_weight,
                "context": self.context_weight,
                "constraint": self.constraint_weight,
                "entropy": self.entropy_weight
            }
        )
    
    def calculate_confidence(
        self,
        neural_output: torch.Tensor,
        state: LNNCouncilState,
        decision: str,
        context_quality: Optional[float] = None
        ) -> ConfidenceMetrics:
        """
        Calculate comprehensive confidence score for a decision.
        
        Args:
            neural_output: Raw neural network output tensor
            state: Current LNN council state
            decision: The decision made ("approve", "deny", "defer")
            context_quality: Optional context quality score
            
        Returns:
            ConfidenceMetrics with detailed confidence breakdown
        """
        
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
        
        logger.debug(
            "Confidence calculated",
            overall=metrics.overall_confidence,
            neural=metrics.neural_confidence,
            context=metrics.context_quality,
            constraints=metrics.constraint_satisfaction,
            decision=decision
        )
        
        return metrics
    
    def _calculate_neural_confidence(self, neural_output: torch.Tensor) -> float:
        """Calculate confidence from neural network output."""
        
        # Apply softmax to get probabilities
        probs = torch.softmax(neural_output.squeeze(), dim=-1)
        
        # Max probability as base confidence
        max_prob = torch.max(probs).item()
        
        # Adjust for probability distribution sharpness
        # Sharp distributions (low entropy) indicate higher confidence
        entropy = -torch.sum(probs * torch.log(probs + 1e-8)).item()
        max_entropy = np.log(len(probs))
        normalized_entropy = entropy / max_entropy
        
        # Combine max probability with entropy consideration
        confidence = max_prob * (1.0 - normalized_entropy * 0.5)
        
        return float(np.clip(confidence, 0.0, 1.0))
    
    def _calculate_output_entropy(self, neural_output: torch.Tensor) -> float:
        """Calculate entropy of neural network output."""
        
        probs = torch.softmax(neural_output.squeeze(), dim=-1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-8)).item()
        max_entropy = np.log(len(probs))
        
        return entropy / max_entropy  # Normalized entropy
    
    def _calculate_activation_stability(self, neural_output: torch.Tensor) -> float:
        """Calculate stability of neural activations."""
        
        # For now, use output variance as stability measure
        # Lower variance = higher stability
        variance = torch.var(neural_output).item()
        
        # Convert to stability score (higher is better)
        stability = 1.0 / (1.0 + variance)
        
        return float(np.clip(stability, 0.0, 1.0))
    
    def _assess_context_quality(self, state: LNNCouncilState) -> float:
        """Assess quality of available context."""
        
        context_cache = state.context_cache
        if not context_cache:
            return 0.3  # Low confidence without context
        
        quality_factors = []
        
        # Check for key context elements
        key_contexts = [
            "current_utilization",
            "user_history", 
            "project_context",
            "system_constraints"
        ]
        
        for context_key in key_contexts:
            if context_key in context_cache:
                context_data = context_cache[context_key]
                if isinstance(context_data, dict) and context_data:
                    quality_factors.append(1.0)
                else:
                    quality_factors.append(0.5)
            else:
                quality_factors.append(0.0)
        
        # Average quality across contexts
        base_quality = np.mean(quality_factors) if quality_factors else 0.0
        
        # Bonus for additional context richness
        total_context_keys = len(context_cache)
        richness_bonus = min(0.2, total_context_keys * 0.02)
        
        return float(np.clip(base_quality + richness_bonus, 0.0, 1.0))
    
    def _calculate_historical_similarity(self, state: LNNCouncilState) -> float:
        """Calculate similarity to historical decisions."""
        
        decision_history = state.decision_history
        if not decision_history:
            return 0.5  # Neutral when no history
        
        # For now, return a placeholder based on history length
        # In a real implementation, this would compare current request
        # to historical requests using embedding similarity
        
        history_length = len(decision_history)
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
        
        # Check for knowledge graph specific context
        kg_indicators = [
            "entity_context",
            "relationship_context", 
            "topology_context",
            "temporal_context"
        ]
        
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
            
            # Penalize if system is highly utilized
            if gpu_usage > 0.9:
                satisfaction_score *= 0.5
            elif gpu_usage > 0.8:
                satisfaction_score *= 0.8
        
        # Budget constraint check
        if decision == "approve":
            project_context = context.get("project_context", {})
            budget_remaining = project_context.get("budget_remaining", 1000.0)
            
            # Estimate cost
            estimated_cost = request.gpu_count * request.compute_hours * 2.5
            
            if estimated_cost > budget_remaining:
                satisfaction_score *= 0.2  # Major constraint violation
            elif estimated_cost > budget_remaining * 0.8:
                satisfaction_score *= 0.7  # Budget concern
        
        # Priority alignment check
        if request.priority >= 8 and decision == "deny":
            satisfaction_score *= 0.6  # High priority denied
        elif request.priority <= 3 and decision == "approve":
            satisfaction_score *= 0.8  # Low priority approved
        
        return float(np.clip(satisfaction_score, 0.0, 1.0))
    
    def _assess_resource_availability(self, state: LNNCouncilState) -> float:
        """Assess current resource availability."""
        
        context = state.context_cache
        utilization = context.get("current_utilization", {})
        
        gpu_usage = utilization.get("gpu_usage", 0.5)
        queue_length = utilization.get("queue_length", 0)
        
        # Higher availability = lower usage and shorter queue
        availability = (1.0 - gpu_usage) * 0.7 + max(0, 1.0 - queue_length / 20.0) * 0.3
        
        return float(np.clip(availability, 0.0, 1.0))
    
    def _calculate_risk_score(self, state: LNNCouncilState, decision: str) -> float:
        """Calculate risk score for the decision."""
        
        if not state.current_request:
            return 0.5
        
        request = state.current_request
        risk_score = 0.0
        
        # Resource risk
        if decision == "approve":
            # Large allocations are riskier
            resource_risk = min(1.0, request.gpu_count / 8.0)
            risk_score += resource_risk * 0.4
            
            # Long duration allocations are riskier
            duration_risk = min(1.0, request.compute_hours / 168.0)
            risk_score += duration_risk * 0.3
            
            # Special requirements add risk
            special_risk = len(request.special_requirements) / 5.0
            risk_score += special_risk * 0.3
        
        return float(np.clip(risk_score, 0.0, 1.0))
    
    def _combine_confidence_scores(self, metrics: ConfidenceMetrics) -> float:
        """Combine individual confidence scores into overall confidence."""
        
        # Weighted combination
        overall = (
            metrics.neural_confidence * self.neural_weight +
            metrics.context_quality * self.context_weight +
            metrics.constraint_satisfaction * self.constraint_weight -
            metrics.output_entropy * self.entropy_weight
        )
        
        # Apply penalties for high risk or low stability
        overall *= (1.0 - metrics.risk_assessment * 0.2)
        overall *= (0.8 + metrics.activation_stability * 0.2)
        
        return float(np.clip(overall, 0.0, 1.0))
    
    def _calibrate_confidence(self, raw_confidence: float) -> float:
        """Apply calibration to raw confidence score."""
        
        # Platt scaling: P_calibrated = 1 / (1 + exp(alpha * P_raw + beta))
        calibrated = 1.0 / (1.0 + np.exp(self.calibration_alpha * raw_confidence + self.calibration_beta))
        
        return float(np.clip(calibrated, 0.0, 1.0))


class DecisionValidator:
    """
    Validates LNN council decisions against system constraints and policies.
    
    Performs comprehensive validation including:
    - Resource availability checks
    - Budget and cost constraints
    - Policy compliance
    - Security requirements
    - Scheduling constraints
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.max_gpu_allocation = config.get("max_gpu_allocation", 8)
        self.max_duration_hours = config.get("max_duration_hours", 168)
        self.budget_check_enabled = config.get("budget_check_enabled", True)
        self.policy_check_enabled = config.get("policy_check_enabled", True)
        
        logger.info(
            "DecisionValidator initialized",
            max_gpus=self.max_gpu_allocation,
            max_hours=self.max_duration_hours,
            budget_check=self.budget_check_enabled
        )
    
    def validate_decision(
        self,
        decision: str,
        request: GPUAllocationRequest,
        state: LNNCouncilState
        ) -> ValidationResult:
        """
        Validate a decision against all constraints.
        
        Args:
            decision: The decision to validate ("approve", "deny", "defer")
            request: The original GPU allocation request
            state: Current LNN council state
            
        Returns:
            ValidationResult with validation outcome and details
        """
        
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
        
        # 4. Security constraints
        self._validate_security_constraints(request, state, result)
        
        # 5. Scheduling constraints
        self._validate_scheduling_constraints(request, state, result)
        
        # Calculate overall validation score
        if result.violations:
            result.validation_score = 0.0
        elif result.warnings:
            result.validation_score = 0.7  # Warnings reduce score
        else:
            result.validation_score = 1.0
        
        logger.info(
            "Decision validated",
            decision=decision,
            is_valid=result.is_valid,
            score=result.validation_score,
            violations=len(result.violations),
            warnings=len(result.warnings)
        )
        
        return result
    
    def _validate_resource_constraints(
        self,
        request: GPUAllocationRequest,
        state: LNNCouncilState,
        result: ValidationResult
        ):
        """Validate resource availability constraints."""
        
        result.constraints_checked.append("resource_availability")
        
        # Check GPU count limits
        if request.gpu_count > self.max_gpu_allocation:
            result.add_violation(
                "gpu_count",
                f"Requested {request.gpu_count} GPUs exceeds limit of {self.max_gpu_allocation}"
            )
        
        # Check duration limits
        if request.compute_hours > self.max_duration_hours:
            result.add_violation(
                "duration",
                f"Requested {request.compute_hours} hours exceeds limit of {self.max_duration_hours}"
            )
        
        # Check current system utilization
        context = state.context_cache
        utilization = context.get("current_utilization", {})
        gpu_usage = utilization.get("gpu_usage", 0.5)
        
        if gpu_usage > 0.95:
            result.add_violation(
                "system_utilization",
                f"System utilization too high: {gpu_usage:.1%}"
            )
        elif gpu_usage > 0.85:
            result.add_warning(
                "system_utilization",
                f"System utilization high: {gpu_usage:.1%}"
            )
        
        # Check queue length
        queue_length = utilization.get("queue_length", 0)
        if queue_length > 50:
            result.add_warning(
                "queue_length",
                f"Long queue: {queue_length} pending requests"
            )
    
    def _validate_budget_constraints(
        self,
        request: GPUAllocationRequest,
        state: LNNCouncilState,
        result: ValidationResult
        ):
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
            result.add_violation(
                "budget_exceeded",
                f"Estimated cost ${estimated_cost:.2f} exceeds remaining budget ${budget_remaining:.2f}"
            )
        elif estimated_cost > budget_remaining * 0.8:
            result.add_warning(
                "budget_concern",
                f"Estimated cost ${estimated_cost:.2f} uses {estimated_cost/budget_remaining:.1%} of remaining budget"
            )
    
    def _validate_policy_constraints(
        self,
        request: GPUAllocationRequest,
        state: LNNCouncilState,
        result: ValidationResult
        ):
        """Validate policy compliance constraints."""
        
        result.constraints_checked.append("policy_compliance")
        
        # Check user allocation limits (simulated)
        context = state.context_cache
        user_history = context.get("user_history", {})
        current_allocations = user_history.get("current_allocations", 0)
        
        if current_allocations >= 3:
            result.add_violation(
                "user_allocation_limit",
                f"User has {current_allocations} active allocations (limit: 3)"
            )
        elif current_allocations >= 2:
            result.add_warning(
                "user_allocation_concern",
                f"User has {current_allocations} active allocations"
            )
        
        # Check special requirements
        if "distributed" in request.special_requirements and request.gpu_count < 2:
            result.add_violation(
                "distributed_requirement",
                "Distributed training requires at least 2 GPUs"
            )
        
        if request.requires_infiniband and request.gpu_type not in ["A100", "H100"]:
            result.add_warning(
                "infiniband_compatibility",
                f"InfiniBand may not be optimal for {request.gpu_type}"
            )
    
    def _validate_security_constraints(
        self,
        request: GPUAllocationRequest,
        state: LNNCouncilState,
        result: ValidationResult
        ):
        """Validate security requirements."""
        
        result.constraints_checked.append("security_requirements")
        
        # Check for sensitive data handling
        if "sensitive_data" in request.special_requirements:
            # Require specific GPU types for sensitive workloads
            if request.gpu_type not in ["A100", "H100"]:
                result.add_violation(
                    "sensitive_data_gpu",
                    "Sensitive data workloads require A100 or H100 GPUs"
                )
        
        # Check project security clearance (simulated)
        context = state.context_cache
        project_context = context.get("project_context", {})
        security_level = project_context.get("security_level", "standard")
        
        if security_level == "restricted" and request.gpu_count > 4:
            result.add_warning(
                "restricted_project",
                "Large allocations for restricted projects require additional approval"
            )
    
    def _validate_scheduling_constraints(
        self,
        request: GPUAllocationRequest,
        state: LNNCouncilState,
        result: ValidationResult
        ):
        """Validate scheduling and timing constraints."""
        
        result.constraints_checked.append("scheduling_constraints")
        
        # Check maintenance windows
        context = state.context_cache
        system_constraints = context.get("system_constraints", {})
        maintenance_window = system_constraints.get("maintenance_window")
        
        if maintenance_window:
            result.add_warning(
                "maintenance_scheduled",
                f"Maintenance window scheduled: {maintenance_window}"
            )
        
        # Check deadline feasibility
        if request.deadline:
            current_time = datetime.now(timezone.utc)
            time_until_deadline = (request.deadline - current_time).total_seconds() / 3600
            
            if time_until_deadline < request.compute_hours:
                result.add_violation(
                    "deadline_infeasible",
                    f"Deadline in {time_until_deadline:.1f}h but job needs {request.compute_hours}h"
                )
            elif time_until_deadline < request.compute_hours * 1.2:
                result.add_warning(
                    "tight_deadline",
                    f"Tight deadline: {time_until_deadline:.1f}h for {request.compute_hours}h job"
                )


class ReasoningPathGenerator:
    """
    Generates human-readable reasoning paths for LNN council decisions.
    
    Creates explainable AI outputs that detail:
    - Why a decision was made
    - What factors influenced the decision
    - What constraints were considered
    - What confidence factors applied
    """
    
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
        """
        Generate a comprehensive reasoning path for the decision.
        
        Args:
            decision: The decision made
            request: The original request
            confidence_metrics: Confidence scoring results
            validation_result: Validation results
            state: Current state
            
        Returns:
            List of reasoning steps in human-readable format
        """
        
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
            for violation in validation_result.violations[:3]:  # Limit to top 3
                reasoning_path.append(f"  - {violation}")
        elif validation_result.warnings:
            reasoning_path.append(f"Found {len(validation_result.warnings)} warnings:")
            for warning in validation_result.warnings[:2]:  # Limit to top 2
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
        
        # 6. Risk assessment
        risk_score = confidence_metrics.risk_assessment
        if risk_score > 0.7:
            reasoning_path.append("High-risk allocation requiring careful monitoring")
        elif risk_score > 0.4:
            reasoning_path.append("Moderate risk allocation")
        else:
            reasoning_path.append("Low-risk allocation")
        
        # 7. Final decision rationale
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
                reasoning_path.append(
                    f"DENIED: Low confidence ({overall_confidence:.2f}) or resource constraints"
                )
        else:  # defer
            reasoning_path.append(
                f"DEFERRED: Insufficient confidence ({overall_confidence:.2f}) or "
                f"resource constraints require manual review"
            )
        
        # 8. Technical details (if enabled)
        if self.include_technical_details:
            reasoning_path.append("Technical details:")
            reasoning_path.append(f"  - Neural confidence: {neural_conf:.3f}")
            reasoning_path.append(f"  - Context quality: {context_quality:.3f}")
            reasoning_path.append(f"  - Output entropy: {confidence_metrics.output_entropy:.3f}")
            reasoning_path.append(f"  - Constraint satisfaction: {confidence_metrics.constraint_satisfaction:.3f}")
        
        # Limit reasoning path length
        if len(reasoning_path) > self.max_reasoning_steps:
            reasoning_path = reasoning_path[:self.max_reasoning_steps]
            reasoning_path.append("... (additional details truncated)")
        
        return reasoning_path