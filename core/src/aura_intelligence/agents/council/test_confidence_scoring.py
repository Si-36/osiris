#!/usr/bin/env python3
"""
Unit Tests for Confidence Scoring and Decision Validation

Comprehensive test suite for Task 7: Add Confidence Scoring and Decision Validation
Tests all aspects of confidence calculation, decision validation, and reasoning generation.
"""

import pytest
import torch
import numpy as np
from datetime import datetime, timezone, timedelta
from unittest.mock import Mock, patch

from .confidence_scoring import (
    ConfidenceScorer,
    DecisionValidator,
    ReasoningPathGenerator,
    ConfidenceMetrics,
    ValidationResult
)
from .models import GPUAllocationRequest, GPUAllocationDecision, LNNCouncilState


class TestConfidenceScorer:
    """Test suite for ConfidenceScorer class."""
    
    @pytest.fixture
    def config(self):
        """Default configuration for confidence scorer."""
        return {
            "confidence_threshold": 0.7,
            "entropy_weight": 0.2,
            "context_weight": 0.3,
            "constraint_weight": 0.3,
            "neural_weight": 0.2,
            "calibration_alpha": 1.0,
            "calibration_beta": 0.0
        }
    
    @pytest.fixture
    def scorer(self, config):
        """Create ConfidenceScorer instance."""
        return ConfidenceScorer(config)
    
    @pytest.fixture
    def sample_request(self):
        """Create sample GPU allocation request."""
        return GPUAllocationRequest(
            request_id="test_001",
            user_id="test_user",
            project_id="test_project",
            gpu_type="A100",
            gpu_count=2,
            memory_gb=40,
            compute_hours=8.0,
            priority=7,
            special_requirements=["high_memory"],
            created_at=datetime.now(timezone.utc)
        )
    
    @pytest.fixture
    def sample_state(self, sample_request):
        """Create sample LNN council state."""
        state = LNNCouncilState(current_request=sample_request)
        state.context_cache = {
            "current_utilization": {"gpu_usage": 0.6, "queue_length": 5},
            "user_history": {"successful_allocations": 10, "avg_usage": 0.8},
            "project_context": {"budget_remaining": 1000.0, "priority_tier": "high"},
            "system_constraints": {"maintenance_window": None, "capacity_limit": 0.9}
        }
        state.decision_history = [{"decision": "approve"}, {"decision": "approve"}]
        return state
    
    def test_confidence_scorer_initialization(self, config):
        """Test ConfidenceScorer initialization."""
        scorer = ConfidenceScorer(config)
        
        assert scorer.confidence_threshold == 0.7
        assert scorer.entropy_weight == 0.2
        assert scorer.context_weight == 0.3
        assert scorer.constraint_weight == 0.3
        assert scorer.neural_weight == 0.2
    
    def test_neural_confidence_calculation(self, scorer):
        """Test neural network confidence calculation."""
        # High confidence output (sharp distribution)
        high_conf_output = torch.tensor([5.0, 1.0, 0.5])
        confidence = scorer._calculate_neural_confidence(high_conf_output)
        assert confidence > 0.8
        
        # Low confidence output (uniform distribution)
        low_conf_output = torch.tensor([1.0, 1.0, 1.0])
        confidence = scorer._calculate_neural_confidence(low_conf_output)
        assert confidence < 0.6
        
        # Medium confidence output
        med_conf_output = torch.tensor([3.0, 2.0, 1.0])
        confidence = scorer._calculate_neural_confidence(med_conf_output)
        assert 0.6 <= confidence <= 0.8
    
    def test_output_entropy_calculation(self, scorer):
        """Test entropy calculation for neural outputs."""
        # Low entropy (sharp distribution)
        sharp_output = torch.tensor([10.0, 0.0, 0.0])
        entropy = scorer._calculate_output_entropy(sharp_output)
        assert entropy < 0.3
        
        # High entropy (uniform distribution)
        uniform_output = torch.tensor([1.0, 1.0, 1.0])
        entropy = scorer._calculate_output_entropy(uniform_output)
        assert entropy > 0.8
    
    def test_context_quality_assessment(self, scorer, sample_state):
        """Test context quality assessment."""
        # High quality context (all key contexts present)
        quality = scorer._assess_context_quality(sample_state)
        assert quality > 0.7
        
        # Low quality context (empty cache)
        empty_state = LNNCouncilState()
        empty_state.context_cache = {}
        quality = scorer._assess_context_quality(empty_state)
        assert quality < 0.5
        
        # Medium quality context (some contexts missing)
        partial_state = LNNCouncilState()
        partial_state.context_cache = {
            "current_utilization": {"gpu_usage": 0.5},
            "user_history": {"successful_allocations": 5}
        }
        quality = scorer._assess_context_quality(partial_state)
        assert 0.4 <= quality <= 0.8
    
    def test_historical_similarity_calculation(self, scorer, sample_state):
        """Test historical similarity calculation."""
        # State with good history
        similarity = scorer._calculate_historical_similarity(sample_state)
        assert similarity > 0.5
        
        # State with no history
        empty_state = LNNCouncilState()
        empty_state.decision_history = []
        similarity = scorer._calculate_historical_similarity(empty_state)
        assert similarity <= 0.5
    
    def test_constraint_satisfaction_assessment(self, scorer, sample_state):
        """Test constraint satisfaction assessment."""
        # Approve decision with good constraints
        satisfaction = scorer._assess_constraint_satisfaction(sample_state, "approve")
        assert satisfaction > 0.5
        
        # Approve decision with high utilization
        high_util_state = sample_state
        high_util_state.context_cache["current_utilization"]["gpu_usage"] = 0.95
        satisfaction = scorer._assess_constraint_satisfaction(high_util_state, "approve")
        assert satisfaction < 0.8
        
        # Deny decision (should have high satisfaction)
        satisfaction = scorer._assess_constraint_satisfaction(sample_state, "deny")
        assert satisfaction >= 0.8
    
    def test_resource_availability_assessment(self, scorer, sample_state):
        """Test resource availability assessment."""
        # Good availability
        availability = scorer._assess_resource_availability(sample_state)
        assert availability > 0.3
        
        # Poor availability
        poor_state = sample_state
        poor_state.context_cache["current_utilization"] = {
            "gpu_usage": 0.95,
            "queue_length": 25
        }
        availability = scorer._assess_resource_availability(poor_state)
        assert availability < 0.5
    
    def test_risk_score_calculation(self, scorer, sample_state):
        """Test risk score calculation."""
        # Low risk request
        low_risk_request = GPUAllocationRequest(
            request_id="low_risk",
            user_id="test_user",
            project_id="test_project",
            gpu_type="V100",
            gpu_count=1,
            memory_gb=16,
            compute_hours=2.0,
            priority=5,
            special_requirements=[],
            created_at=datetime.now(timezone.utc)
        )
        sample_state.current_request = low_risk_request
        
        risk = scorer._calculate_risk_score(sample_state, "approve")
        assert risk < 0.5
        
        # High risk request
        high_risk_request = GPUAllocationRequest(
            request_id="high_risk",
            user_id="test_user",
            project_id="test_project",
            gpu_type="H100",
            gpu_count=8,
            memory_gb=80,
            compute_hours=168.0,
            priority=3,
            special_requirements=["distributed", "high_memory", "low_latency"],
            created_at=datetime.now(timezone.utc)
        )
        sample_state.current_request = high_risk_request
        
        risk = scorer._calculate_risk_score(sample_state, "approve")
        assert risk > 0.7
    
    def test_confidence_combination(self, scorer):
        """Test confidence score combination."""
        metrics = ConfidenceMetrics()
        metrics.neural_confidence = 0.8
        metrics.context_quality = 0.7
        metrics.constraint_satisfaction = 0.9
        metrics.output_entropy = 0.3
        metrics.risk_assessment = 0.2
        metrics.activation_stability = 0.8
        
        overall = scorer._combine_confidence_scores(metrics)
        assert 0.0 <= overall <= 1.0
        assert overall > 0.5  # Should be reasonably high given good inputs
    
    def test_confidence_calibration(self, scorer):
        """Test confidence calibration."""
        # Test various confidence levels
        test_confidences = [0.1, 0.3, 0.5, 0.7, 0.9]
        
        for raw_conf in test_confidences:
            calibrated = scorer._calibrate_confidence(raw_conf)
            assert 0.0 <= calibrated <= 1.0
    
    def test_full_confidence_calculation(self, scorer, sample_state):
        """Test complete confidence calculation pipeline."""
        # Create realistic neural output
        neural_output = torch.tensor([2.5, 1.0, 0.5])
        
        metrics = scorer.calculate_confidence(
            neural_output=neural_output,
            state=sample_state,
            decision="approve",
            context_quality=0.8
        )
        
        # Verify all metrics are calculated
        assert isinstance(metrics, ConfidenceMetrics)
        assert 0.0 <= metrics.neural_confidence <= 1.0
        assert 0.0 <= metrics.output_entropy <= 1.0
        assert 0.0 <= metrics.activation_stability <= 1.0
        assert 0.0 <= metrics.context_quality <= 1.0
        assert 0.0 <= metrics.historical_similarity <= 1.0
        assert 0.0 <= metrics.knowledge_completeness <= 1.0
        assert 0.0 <= metrics.constraint_satisfaction <= 1.0
        assert 0.0 <= metrics.resource_availability <= 1.0
        assert 0.0 <= metrics.risk_assessment <= 1.0
        assert 0.0 <= metrics.overall_confidence <= 1.0
        
        # Verify confidence breakdown
        assert isinstance(metrics.confidence_breakdown, dict)
        assert len(metrics.confidence_breakdown) > 0


class TestDecisionValidator:
    """Test suite for DecisionValidator class."""
    
    @pytest.fixture
    def config(self):
        """Default configuration for decision validator."""
        return {
            "max_gpu_allocation": 8,
            "max_duration_hours": 168,
            "budget_check_enabled": True,
            "policy_check_enabled": True
        }
    
    @pytest.fixture
    def validator(self, config):
        """Create DecisionValidator instance."""
        return DecisionValidator(config)
    
    @pytest.fixture
    def sample_request(self):
        """Create sample GPU allocation request."""
        return GPUAllocationRequest(
            request_id="test_001",
            user_id="test_user",
            project_id="test_project",
            gpu_type="A100",
            gpu_count=2,
            memory_gb=40,
            compute_hours=8.0,
            priority=7,
            special_requirements=["high_memory"],
            created_at=datetime.now(timezone.utc)
        )
    
    @pytest.fixture
    def sample_state(self, sample_request):
        """Create sample LNN council state."""
        state = LNNCouncilState(current_request=sample_request)
        state.context_cache = {
            "current_utilization": {"gpu_usage": 0.6, "queue_length": 5},
            "user_history": {"successful_allocations": 10, "current_allocations": 1},
            "project_context": {"budget_remaining": 1000.0, "security_level": "standard"},
            "system_constraints": {"maintenance_window": None, "capacity_limit": 0.9}
        }
        return state
    
    def test_validator_initialization(self, config):
        """Test DecisionValidator initialization."""
        validator = DecisionValidator(config)
        
        assert validator.max_gpu_allocation == 8
        assert validator.max_duration_hours == 168
        assert validator.budget_check_enabled is True
        assert validator.policy_check_enabled is True
    
    def test_approve_decision_validation_success(self, validator, sample_request, sample_state):
        """Test successful validation of approve decision."""
        result = validator.validate_decision("approve", sample_request, sample_state)
        
        assert isinstance(result, ValidationResult)
        assert result.is_valid is True
        assert result.validation_score > 0.8
        assert len(result.violations) == 0
        assert len(result.constraints_checked) > 0
    
    def test_deny_decision_validation(self, validator, sample_request, sample_state):
        """Test validation of deny decision (should always pass)."""
        result = validator.validate_decision("deny", sample_request, sample_state)
        
        assert result.is_valid is True
        assert result.validation_score == 1.0
        assert len(result.violations) == 0
    
    def test_defer_decision_validation(self, validator, sample_request, sample_state):
        """Test validation of defer decision (should always pass)."""
        result = validator.validate_decision("defer", sample_request, sample_state)
        
        assert result.is_valid is True
        assert result.validation_score == 1.0
        assert len(result.violations) == 0
    
    def test_resource_constraint_violations(self, validator, sample_state):
        """Test resource constraint violations."""
        # Request exceeding GPU limit
        large_request = GPUAllocationRequest(
            request_id="large_001",
            user_id="test_user",
            project_id="test_project",
            gpu_type="A100",
            gpu_count=16,  # Exceeds limit of 8
            memory_gb=80,
            compute_hours=8.0,
            priority=7,
            created_at=datetime.now(timezone.utc)
        )
        
        result = validator.validate_decision("approve", large_request, sample_state)
        
        assert result.is_valid is False
        assert len(result.violations) > 0
        assert any("gpu_count" in violation for violation in result.violations)
    
    def test_duration_constraint_violations(self, validator, sample_state):
        """Test duration constraint violations."""
        # Request exceeding duration limit
        long_request = GPUAllocationRequest(
            request_id="long_001",
            user_id="test_user",
            project_id="test_project",
            gpu_type="A100",
            gpu_count=2,
            memory_gb=40,
            compute_hours=200.0,  # Exceeds limit of 168
            priority=7,
            created_at=datetime.now(timezone.utc)
        )
        
        result = validator.validate_decision("approve", long_request, sample_state)
        
        assert result.is_valid is False
        assert len(result.violations) > 0
        assert any("duration" in violation for violation in result.violations)
    
    def test_budget_constraint_violations(self, validator, sample_state):
        """Test budget constraint violations."""
        # Request exceeding budget
        expensive_request = GPUAllocationRequest(
            request_id="expensive_001",
            user_id="test_user",
            project_id="test_project",
            gpu_type="H100",
            gpu_count=8,
            memory_gb=80,
            compute_hours=100.0,  # Very expensive
            priority=7,
            created_at=datetime.now(timezone.utc)
        )
        
        # Set low budget
        sample_state.context_cache["project_context"]["budget_remaining"] = 100.0
        
        result = validator.validate_decision("approve", expensive_request, sample_state)
        
        assert result.is_valid is False
        assert len(result.violations) > 0
        assert any("budget" in violation for violation in result.violations)
    
    def test_system_utilization_warnings(self, validator, sample_request, sample_state):
        """Test system utilization warnings."""
        # Set high utilization
        sample_state.context_cache["current_utilization"]["gpu_usage"] = 0.9
        
        result = validator.validate_decision("approve", sample_request, sample_state)
        
        # Should still be valid but with warnings
        assert result.is_valid is True
        assert len(result.warnings) > 0
        assert any("utilization" in warning for warning in result.warnings)
    
    def test_policy_constraint_violations(self, validator, sample_state):
        """Test policy constraint violations."""
        # User with too many allocations
        sample_state.context_cache["user_history"]["current_allocations"] = 4  # Exceeds limit of 3
        
        sample_request = GPUAllocationRequest(
            request_id="policy_001",
            user_id="test_user",
            project_id="test_project",
            gpu_type="A100",
            gpu_count=2,
            memory_gb=40,
            compute_hours=8.0,
            priority=7,
            created_at=datetime.now(timezone.utc)
        )
        
        result = validator.validate_decision("approve", sample_request, sample_state)
        
        assert result.is_valid is False
        assert len(result.violations) > 0
        assert any("allocation_limit" in violation for violation in result.violations)
    
    def test_special_requirements_validation(self, validator, sample_state):
        """Test special requirements validation."""
        # Distributed requirement with insufficient GPUs
        distributed_request = GPUAllocationRequest(
            request_id="distributed_001",
            user_id="test_user",
            project_id="test_project",
            gpu_type="A100",
            gpu_count=1,  # Insufficient for distributed
            memory_gb=40,
            compute_hours=8.0,
            priority=7,
            special_requirements=["distributed"],
            created_at=datetime.now(timezone.utc)
        )
        
        result = validator.validate_decision("approve", distributed_request, sample_state)
        
        assert result.is_valid is False
        assert len(result.violations) > 0
        assert any("distributed" in violation for violation in result.violations)
    
    def test_deadline_constraint_validation(self, validator, sample_state):
        """Test deadline constraint validation."""
        # Request with impossible deadline
        tight_deadline = datetime.now(timezone.utc) + timedelta(hours=2)
        deadline_request = GPUAllocationRequest(
            request_id="deadline_001",
            user_id="test_user",
            project_id="test_project",
            gpu_type="A100",
            gpu_count=2,
            memory_gb=40,
            compute_hours=8.0,  # Needs 8 hours but deadline in 2 hours
            priority=7,
            deadline=tight_deadline,
            created_at=datetime.now(timezone.utc)
        )
        
        result = validator.validate_decision("approve", deadline_request, sample_state)
        
        assert result.is_valid is False
        assert len(result.violations) > 0
        assert any("deadline" in violation for violation in result.violations)


class TestReasoningPathGenerator:
    """Test suite for ReasoningPathGenerator class."""
    
    @pytest.fixture
    def config(self):
        """Default configuration for reasoning path generator."""
        return {
            "include_technical_details": True,
            "max_reasoning_steps": 10
        }
    
    @pytest.fixture
    def generator(self, config):
        """Create ReasoningPathGenerator instance."""
        return ReasoningPathGenerator(config)
    
    @pytest.fixture
    def sample_request(self):
        """Create sample GPU allocation request."""
        return GPUAllocationRequest(
            request_id="test_001",
            user_id="test_user",
            project_id="test_project",
            gpu_type="A100",
            gpu_count=2,
            memory_gb=40,
            compute_hours=8.0,
            priority=7,
            special_requirements=["high_memory"],
            created_at=datetime.now(timezone.utc)
        )
    
    @pytest.fixture
    def sample_confidence_metrics(self):
        """Create sample confidence metrics."""
        metrics = ConfidenceMetrics()
        metrics.neural_confidence = 0.85
        metrics.output_entropy = 0.2
        metrics.activation_stability = 0.9
        metrics.context_quality = 0.8
        metrics.historical_similarity = 0.7
        metrics.knowledge_completeness = 0.75
        metrics.constraint_satisfaction = 0.9
        metrics.resource_availability = 0.8
        metrics.risk_assessment = 0.3
        metrics.overall_confidence = 0.82
        return metrics
    
    @pytest.fixture
    def sample_validation_result(self):
        """Create sample validation result."""
        result = ValidationResult()
        result.is_valid = True
        result.validation_score = 0.95
        result.constraints_checked = ["resource_availability", "budget_constraints", "policy_compliance"]
        return result
    
    @pytest.fixture
    def sample_state(self, sample_request):
        """Create sample LNN council state."""
        state = LNNCouncilState(current_request=sample_request)
        state.context_cache = {
            "current_utilization": {"gpu_usage": 0.6, "queue_length": 5},
            "user_history": {"successful_allocations": 10, "avg_usage": 0.8},
            "project_context": {"budget_remaining": 1000.0, "priority_tier": "high"}
        }
        return state
    
    def test_generator_initialization(self, config):
        """Test ReasoningPathGenerator initialization."""
        generator = ReasoningPathGenerator(config)
        
        assert generator.include_technical_details is True
        assert generator.max_reasoning_steps == 10
    
    def test_approve_decision_reasoning(self, generator, sample_request, sample_confidence_metrics, 
                                     sample_validation_result, sample_state):
        """Test reasoning path generation for approve decision."""
        reasoning_path = generator.generate_reasoning_path(
            decision="approve",
            request=sample_request,
            confidence_metrics=sample_confidence_metrics,
            validation_result=sample_validation_result,
            state=sample_state
        )
        
        assert isinstance(reasoning_path, list)
        assert len(reasoning_path) > 0
        
        # Check for key reasoning elements
        reasoning_text = " ".join(reasoning_path)
        assert "2x A100" in reasoning_text
        assert "8.0h" in reasoning_text
        assert "Priority: 7" in reasoning_text
        assert "APPROVED" in reasoning_text
        assert "confidence" in reasoning_text.lower()
    
    def test_deny_decision_reasoning(self, generator, sample_request, sample_confidence_metrics, 
                                   sample_state):
        """Test reasoning path generation for deny decision."""
        # Create validation result with violations
        validation_result = ValidationResult()
        validation_result.is_valid = False
        validation_result.validation_score = 0.0
        validation_result.violations = ["budget_exceeded: Cost exceeds available budget"]
        
        # Lower confidence for deny decision
        sample_confidence_metrics.overall_confidence = 0.3
        
        reasoning_path = generator.generate_reasoning_path(
            decision="deny",
            request=sample_request,
            confidence_metrics=sample_confidence_metrics,
            validation_result=validation_result,
            state=sample_state
        )
        
        assert isinstance(reasoning_path, list)
        assert len(reasoning_path) > 0
        
        reasoning_text = " ".join(reasoning_path)
        assert "DENIED" in reasoning_text
        assert "violation" in reasoning_text.lower()
    
    def test_defer_decision_reasoning(self, generator, sample_request, sample_confidence_metrics, 
                                    sample_validation_result, sample_state):
        """Test reasoning path generation for defer decision."""
        # Lower confidence for defer decision
        sample_confidence_metrics.overall_confidence = 0.5
        
        reasoning_path = generator.generate_reasoning_path(
            decision="defer",
            request=sample_request,
            confidence_metrics=sample_confidence_metrics,
            validation_result=sample_validation_result,
            state=sample_state
        )
        
        assert isinstance(reasoning_path, list)
        assert len(reasoning_path) > 0
        
        reasoning_text = " ".join(reasoning_path)
        assert "DEFERRED" in reasoning_text
        assert "confidence" in reasoning_text.lower()
    
    def test_technical_details_inclusion(self, sample_request, sample_confidence_metrics, 
                                       sample_validation_result, sample_state):
        """Test inclusion of technical details in reasoning."""
        # Generator with technical details enabled
        config_with_details = {"include_technical_details": True, "max_reasoning_steps": 15}
        generator_with_details = ReasoningPathGenerator(config_with_details)
        
        reasoning_path = generator_with_details.generate_reasoning_path(
            decision="approve",
            request=sample_request,
            confidence_metrics=sample_confidence_metrics,
            validation_result=sample_validation_result,
            state=sample_state
        )
        
        reasoning_text = " ".join(reasoning_path)
        assert "Technical details" in reasoning_text
        assert "Neural confidence" in reasoning_text
        
        # Generator without technical details
        config_no_details = {"include_technical_details": False, "max_reasoning_steps": 10}
        generator_no_details = ReasoningPathGenerator(config_no_details)
        
        reasoning_path_no_details = generator_no_details.generate_reasoning_path(
            decision="approve",
            request=sample_request,
            confidence_metrics=sample_confidence_metrics,
            validation_result=sample_validation_result,
            state=sample_state
        )
        
        reasoning_text_no_details = " ".join(reasoning_path_no_details)
        assert "Technical details" not in reasoning_text_no_details
    
    def test_reasoning_with_warnings(self, generator, sample_request, sample_confidence_metrics, 
                                   sample_state):
        """Test reasoning path generation with validation warnings."""
        validation_result = ValidationResult()
        validation_result.is_valid = True
        validation_result.validation_score = 0.7
        validation_result.warnings = ["system_utilization: High utilization detected"]
        
        reasoning_path = generator.generate_reasoning_path(
            decision="approve",
            request=sample_request,
            confidence_metrics=sample_confidence_metrics,
            validation_result=validation_result,
            state=sample_state
        )
        
        reasoning_text = " ".join(reasoning_path)
        assert "warning" in reasoning_text.lower()
        assert "utilization" in reasoning_text.lower()


class TestConfidenceMetrics:
    """Test suite for ConfidenceMetrics dataclass."""
    
    def test_confidence_metrics_initialization(self):
        """Test ConfidenceMetrics initialization."""
        metrics = ConfidenceMetrics()
        
        assert metrics.neural_confidence == 0.0
        assert metrics.output_entropy == 0.0
        assert metrics.activation_stability == 0.0
        assert metrics.context_quality == 0.0
        assert metrics.historical_similarity == 0.0
        assert metrics.knowledge_completeness == 0.0
        assert metrics.constraint_satisfaction == 0.0
        assert metrics.resource_availability == 0.0
        assert metrics.risk_assessment == 0.0
        assert metrics.overall_confidence == 0.0
        assert isinstance(metrics.confidence_breakdown, dict)
    
    def test_confidence_metrics_with_values(self):
        """Test ConfidenceMetrics with custom values."""
        breakdown = {"neural": 0.8, "context": 0.7}
        
        metrics = ConfidenceMetrics(
            neural_confidence=0.85,
            context_quality=0.75,
            overall_confidence=0.8,
            confidence_breakdown=breakdown
        )
        
        assert metrics.neural_confidence == 0.85
        assert metrics.context_quality == 0.75
        assert metrics.overall_confidence == 0.8
        assert metrics.confidence_breakdown == breakdown


class TestValidationResult:
    """Test suite for ValidationResult dataclass."""
    
    def test_validation_result_initialization(self):
        """Test ValidationResult initialization."""
        result = ValidationResult()
        
        assert result.is_valid is True
        assert result.validation_score == 1.0
        assert isinstance(result.violations, list)
        assert isinstance(result.warnings, list)
        assert isinstance(result.constraints_checked, list)
        assert len(result.violations) == 0
        assert len(result.warnings) == 0
        assert len(result.constraints_checked) == 0
    
    def test_add_violation(self):
        """Test adding violations to ValidationResult."""
        result = ValidationResult()
        
        result.add_violation("budget", "Insufficient funds")
        
        assert result.is_valid is False
        assert len(result.violations) == 1
        assert "budget: Insufficient funds" in result.violations
    
    def test_add_warning(self):
        """Test adding warnings to ValidationResult."""
        result = ValidationResult()
        
        result.add_warning("utilization", "High system load")
        
        assert result.is_valid is True  # Warnings don't invalidate
        assert len(result.warnings) == 1
        assert "utilization: High system load" in result.warnings
    
    def test_multiple_violations_and_warnings(self):
        """Test adding multiple violations and warnings."""
        result = ValidationResult()
        
        result.add_violation("budget", "Insufficient funds")
        result.add_violation("resources", "No GPUs available")
        result.add_warning("queue", "Long wait time")
        result.add_warning("maintenance", "Scheduled downtime")
        
        assert result.is_valid is False
        assert len(result.violations) == 2
        assert len(result.warnings) == 2


# Integration tests
class TestConfidenceScoringIntegration:
    """Integration tests for confidence scoring components."""
    
    @pytest.fixture
    def full_config(self):
        """Complete configuration for all components."""
        return {
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
    
    def test_end_to_end_confidence_scoring_pipeline(self, full_config):
        """Test complete confidence scoring pipeline."""
        # Initialize all components
        scorer = ConfidenceScorer(full_config)
        validator = DecisionValidator(full_config)
        generator = ReasoningPathGenerator(full_config)
        
        # Create test data
        request = GPUAllocationRequest(
            request_id="integration_001",
            user_id="integration_user",
            project_id="integration_project",
            gpu_type="A100",
            gpu_count=4,
            memory_gb=80,
            compute_hours=12.0,
            priority=8,
            special_requirements=["high_memory"],
            created_at=datetime.now(timezone.utc)
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
        
        # Verify consistency
        if validation_result.is_valid and confidence_metrics.overall_confidence > 0.7:
            reasoning_text = " ".join(reasoning_path)
            assert "APPROVED" in reasoning_text
    
    def test_low_confidence_scenario(self, full_config):
        """Test scenario with low confidence leading to defer."""
        scorer = ConfidenceScorer(full_config)
        validator = DecisionValidator(full_config)
        generator = ReasoningPathGenerator(full_config)
        
        # Create low confidence scenario
        request = GPUAllocationRequest(
            request_id="low_conf_001",
            user_id="test_user",
            project_id="test_project",
            gpu_type="V100",
            gpu_count=1,
            memory_gb=16,
            compute_hours=4.0,
            priority=3,
            created_at=datetime.now(timezone.utc)
        )
        
        state = LNNCouncilState(current_request=request)
        state.context_cache = {}  # No context = low confidence
        
        # Uniform neural output = low confidence
        neural_output = torch.tensor([1.0, 1.0, 1.0])
        decision = "defer"
        
        confidence_metrics = scorer.calculate_confidence(
            neural_output=neural_output,
            state=state,
            decision=decision
        )
        
        validation_result = validator.validate_decision(decision, request, state)
        
        reasoning_path = generator.generate_reasoning_path(
            decision=decision,
            request=request,
            confidence_metrics=confidence_metrics,
            validation_result=validation_result,
            state=state
        )
        
        # Verify low confidence handling
        assert confidence_metrics.overall_confidence < 0.6
        assert confidence_metrics.context_quality < 0.5
        
        reasoning_text = " ".join(reasoning_path)
        assert "DEFERRED" in reasoning_text
        assert "confidence" in reasoning_text.lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])