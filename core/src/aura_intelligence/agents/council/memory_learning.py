"""
Memory Learning Engine (2025 Architecture)

Advanced learning from decision outcomes and memory patterns.
Implements latest 2025 research in meta-learning and neural memory.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
from datetime import datetime, timedelta
import structlog

from .config import LNNCouncilConfig
from .models import GPUAllocationRequest, GPUAllocationDecision

logger = structlog.get_logger()


class MemoryLearningEngine:
    """
    Advanced learning engine for memory-augmented decisions.
    
    2025 Features:
    - Meta-learning from decision outcomes
    - Adaptive confidence calibration
    - Pattern-based learning
    - Memory-guided neural updates
    """
    
    def __init__(self, config: LNNCouncilConfig):
        self.config = config
        
        # Learning components
        self.confidence_calibrator = ConfidenceCalibrator(config)
        self.pattern_learner = PatternLearner(config)
        self.outcome_predictor = OutcomePredictor(config)
        
        # Learning statistics
        self.learning_episodes = 0
        self.calibration_accuracy = 0.0
        self.pattern_recognition_score = 0.0
        
        logger.info("Memory Learning Engine initialized")
    
    async def learn_from_decision(
        self,
        request: GPUAllocationRequest,
        decision: GPUAllocationDecision,
        actual_outcome: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Learn from a decision and its outcome.
        
        Args:
            request: Original GPU allocation request
            decision: Decision made by the agent
            actual_outcome: Actual outcome (if available)
            
        Returns:
            Learning insights and updates
        """
        
        learning_results = {}
        
        try:
            # 1. Update confidence calibration
            if actual_outcome:
                calibration_update = await self.confidence_calibrator.update(
                    decision.confidence_score,
                    actual_outcome.get("success", True)
                )
                learning_results["calibration"] = calibration_update
            
            # 2. Learn decision patterns
            pattern_update = await self.pattern_learner.learn_pattern(
                request, decision, actual_outcome
            )
            learning_results["patterns"] = pattern_update
            
            # 3. Update outcome prediction
            if actual_outcome:
                prediction_update = await self.outcome_predictor.update(
                    request, decision, actual_outcome
                )
                learning_results["prediction"] = prediction_update
            
            # 4. Calculate learning quality
            learning_quality = self._assess_learning_quality(learning_results)
            learning_results["quality"] = learning_quality
            
            self.learning_episodes += 1
            
            logger.info(
                "Learning from decision completed",
                learning_quality=learning_quality,
                has_outcome=actual_outcome is not None
            )
            
            return learning_results
            
        except Exception as e:
            logger.warning(f"Learning from decision failed: {e}")
            return {"error": str(e)}
    
    def get_learning_insights(self) -> Dict[str, Any]:
        """Get current learning insights and recommendations."""
        
        return {
            "learning_episodes": self.learning_episodes,
            "calibration_accuracy": self.calibration_accuracy,
            "pattern_recognition_score": self.pattern_recognition_score,
            "confidence_calibration": self.confidence_calibrator.get_calibration_stats(),
            "pattern_insights": self.pattern_learner.get_pattern_insights(),
            "prediction_accuracy": self.outcome_predictor.get_accuracy_stats()
        }
    
    def _assess_learning_quality(self, learning_results: Dict[str, Any]) -> float:
        """Assess the quality of learning from this episode."""
        
        quality_factors = []
        
        # Calibration quality
        if "calibration" in learning_results:
            cal_quality = learning_results["calibration"].get("improvement", 0.0)
            quality_factors.append(cal_quality)
        
        # Pattern learning quality
        if "patterns" in learning_results:
            pattern_quality = learning_results["patterns"].get("pattern_strength", 0.5)
            quality_factors.append(pattern_quality)
        
        # Prediction quality
        if "prediction" in learning_results:
            pred_quality = learning_results["prediction"].get("accuracy_improvement", 0.0)
            quality_factors.append(pred_quality)
        
        return np.mean(quality_factors) if quality_factors else 0.5


class ConfidenceCalibrator:
    """Calibrates confidence scores based on actual outcomes."""
    
    def __init__(self, config: LNNCouncilConfig):
        self.config = config
        self.calibration_data = []
        self.calibration_bins = 10
        self.bin_counts = np.zeros(self.calibration_bins)
        self.bin_accuracies = np.zeros(self.calibration_bins)
    
    async def update(self, predicted_confidence: float, actual_success: bool) -> Dict[str, Any]:
        """Update calibration with new data point."""
        
        # Add to calibration data
        self.calibration_data.append({
            "confidence": predicted_confidence,
            "success": actual_success,
            "timestamp": datetime.now()
        })
        
        # Update bins
        bin_idx = min(int(predicted_confidence * self.calibration_bins), self.calibration_bins - 1)
        self.bin_counts[bin_idx] += 1
        
        if actual_success:
            self.bin_accuracies[bin_idx] += 1
        
        # Calculate calibration error
        calibration_error = self._calculate_calibration_error()
        
        return {
            "calibration_error": calibration_error,
            "improvement": max(0.0, 0.1 - calibration_error),  # Lower error = improvement
            "data_points": len(self.calibration_data)
        }
    
    def _calculate_calibration_error(self) -> float:
        """Calculate Expected Calibration Error (ECE)."""
        
        if len(self.calibration_data) < 10:
            return 0.1  # Default error for insufficient data
        
        total_error = 0.0
        total_samples = sum(self.bin_counts)
        
        for i in range(self.calibration_bins):
            if self.bin_counts[i] > 0:
                bin_confidence = (i + 0.5) / self.calibration_bins
                bin_accuracy = self.bin_accuracies[i] / self.bin_counts[i]
                bin_weight = self.bin_counts[i] / total_samples
                
                total_error += bin_weight * abs(bin_confidence - bin_accuracy)
        
        return total_error
    
    def get_calibration_stats(self) -> Dict[str, Any]:
        """Get calibration statistics."""
        
        return {
            "calibration_error": self._calculate_calibration_error(),
            "data_points": len(self.calibration_data),
            "bin_distribution": self.bin_counts.tolist(),
            "bin_accuracies": (self.bin_accuracies / np.maximum(self.bin_counts, 1)).tolist()
        }


class PatternLearner:
    """Learns patterns from decision sequences."""
    
    def __init__(self, config: LNNCouncilConfig):
        self.config = config
        self.patterns = {}
        self.pattern_outcomes = {}
    
    async def learn_pattern(
        self,
        request: GPUAllocationRequest,
        decision: GPUAllocationDecision,
        outcome: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Learn patterns from request-decision-outcome triplets."""
        
        # Create pattern key
        pattern_key = self._create_pattern_key(request, decision)
        
        # Update pattern statistics
        if pattern_key not in self.patterns:
            self.patterns[pattern_key] = {
                "count": 0,
                "decisions": {"approve": 0, "deny": 0, "defer": 0},
                "avg_confidence": 0.0,
                "outcomes": {"success": 0, "failure": 0}
            }
        
        pattern = self.patterns[pattern_key]
        pattern["count"] += 1
        pattern["decisions"][decision.decision] += 1
        
        # Update confidence
        pattern["avg_confidence"] = (
            pattern["avg_confidence"] * (pattern["count"] - 1) + decision.confidence_score
        ) / pattern["count"]
        
        # Update outcomes if available
        if outcome:
            if outcome.get("success", True):
                pattern["outcomes"]["success"] += 1
            else:
                pattern["outcomes"]["failure"] += 1
        
        # Calculate pattern strength
        pattern_strength = self._calculate_pattern_strength(pattern)
        
        return {
            "pattern_key": pattern_key,
            "pattern_strength": pattern_strength,
            "pattern_count": pattern["count"],
            "success_rate": self._calculate_success_rate(pattern)
        }
    
    def _create_pattern_key(self, request: GPUAllocationRequest, decision: GPUAllocationDecision) -> str:
        """Create a pattern key from request and decision."""
        
        # Discretize continuous values
        gpu_count_bin = "low" if request.gpu_count <= 2 else "medium" if request.gpu_count <= 4 else "high"
        memory_bin = "low" if request.memory_gb <= 20 else "medium" if request.memory_gb <= 40 else "high"
        hours_bin = "short" if request.compute_hours <= 8 else "medium" if request.compute_hours <= 24 else "long"
        priority_bin = "low" if request.priority <= 3 else "medium" if request.priority <= 7 else "high"
        
        return f"{request.gpu_type}_{gpu_count_bin}_{memory_bin}_{hours_bin}_{priority_bin}_{decision.decision}"
    
    def _calculate_pattern_strength(self, pattern: Dict[str, Any]) -> float:
        """Calculate the strength of a pattern."""
        
        count = pattern["count"]
        if count < 3:
            return 0.3  # Weak pattern
        
        # Strength based on frequency and consistency
        frequency_score = min(count / 10.0, 1.0)  # Max at 10 occurrences
        
        # Consistency in decisions
        decision_counts = list(pattern["decisions"].values())
        max_decision = max(decision_counts)
        consistency_score = max_decision / sum(decision_counts)
        
        return (frequency_score + consistency_score) / 2.0
    
    def _calculate_success_rate(self, pattern: Dict[str, Any]) -> float:
        """Calculate success rate for a pattern."""
        
        outcomes = pattern["outcomes"]
        total_outcomes = outcomes["success"] + outcomes["failure"]
        
        if total_outcomes == 0:
            return 0.5  # No outcome data
        
        return outcomes["success"] / total_outcomes
    
    def get_pattern_insights(self) -> Dict[str, Any]:
        """Get insights about learned patterns."""
        
        if not self.patterns:
            return {"total_patterns": 0}
        
        # Find strongest patterns
        strong_patterns = {
            k: v for k, v in self.patterns.items()
            if self._calculate_pattern_strength(v) > 0.7
        }
        
        # Calculate average success rates
        success_rates = [
            self._calculate_success_rate(pattern)
            for pattern in self.patterns.values()
        ]
        
        return {
            "total_patterns": len(self.patterns),
            "strong_patterns": len(strong_patterns),
            "avg_success_rate": np.mean(success_rates) if success_rates else 0.5,
            "pattern_coverage": min(len(self.patterns) / 50.0, 1.0)  # Normalize
        }


class OutcomePredictor:
    """Predicts outcomes based on request and decision patterns."""
    
    def __init__(self, config: LNNCouncilConfig):
        self.config = config
        self.prediction_data = []
        self.feature_weights = np.random.normal(0, 0.1, 10)  # Simple linear model
    
    async def update(
        self,
        request: GPUAllocationRequest,
        decision: GPUAllocationDecision,
        outcome: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Update outcome prediction model."""
        
        # Extract features
        features = self._extract_features(request, decision)
        actual_success = 1.0 if outcome.get("success", True) else 0.0
        
        # Store data point
        self.prediction_data.append({
            "features": features,
            "actual": actual_success,
            "timestamp": datetime.now()
        })
        
        # Update model (simple gradient descent)
        if len(self.prediction_data) > 10:
            self._update_model()
        
        # Calculate current accuracy
        accuracy = self._calculate_accuracy()
        
        return {
            "accuracy": accuracy,
            "accuracy_improvement": max(0.0, accuracy - 0.5),
            "data_points": len(self.prediction_data)
        }
    
    def _extract_features(self, request: GPUAllocationRequest, decision: GPUAllocationDecision) -> np.ndarray:
        """Extract features for outcome prediction."""
        
        features = np.array([
            request.gpu_count / 8.0,
            request.memory_gb / 80.0,
            request.compute_hours / 168.0,
            request.priority / 10.0,
            1.0 if request.gpu_type == "A100" else 0.0,
            1.0 if request.gpu_type == "H100" else 0.0,
            1.0 if decision.decision == "approve" else 0.0,
            decision.confidence_score,
            1.0 if decision.fallback_used else 0.0,
            decision.inference_time_ms / 1000.0  # Normalize
        ])
        
        return features
    
    def _update_model(self):
        """Update the prediction model with recent data."""
        
        if len(self.prediction_data) < 10:
            return
        
        # Simple gradient descent update
        learning_rate = 0.01
        recent_data = self.prediction_data[-20:]  # Use recent 20 points
        
        for data_point in recent_data:
            features = data_point["features"]
            actual = data_point["actual"]
            
            # Predict
            predicted = np.dot(self.feature_weights, features)
            predicted = 1.0 / (1.0 + np.exp(-predicted))  # Sigmoid
            
            # Calculate error
            error = actual - predicted
            
            # Update weights
            gradient = error * predicted * (1 - predicted) * features
            self.feature_weights += learning_rate * gradient
    
    def _calculate_accuracy(self) -> float:
        """Calculate prediction accuracy."""
        
        if len(self.prediction_data) < 5:
            return 0.5
        
        recent_data = self.prediction_data[-10:]  # Recent 10 predictions
        correct = 0
        
        for data_point in recent_data:
            features = data_point["features"]
            actual = data_point["actual"]
            
            # Predict
            predicted = np.dot(self.feature_weights, features)
            predicted = 1.0 / (1.0 + np.exp(-predicted))  # Sigmoid
            
            # Check if prediction is correct (threshold at 0.5)
            predicted_class = 1.0 if predicted > 0.5 else 0.0
            if predicted_class == actual:
                correct += 1
        
        return correct / len(recent_data)
    
    def get_accuracy_stats(self) -> Dict[str, Any]:
        """Get accuracy statistics."""
        
        return {
            "current_accuracy": self._calculate_accuracy(),
            "data_points": len(self.prediction_data),
            "model_weights": self.feature_weights.tolist()
        }