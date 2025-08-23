"""
Adaptation Service for Liquid Neural Networks
Handles real-time parameter adjustments and architecture modifications
"""

import torch
import asyncio
from typing import Dict, Any, Optional, List
import time
import numpy as np
from dataclasses import dataclass
import structlog

logger = structlog.get_logger()


@dataclass
class AdaptationMetrics:
    """Metrics for tracking adaptation performance"""
    total_adaptations: int = 0
    parameter_updates: int = 0
    architecture_changes: int = 0
    successful_adaptations: int = 0
    failed_adaptations: int = 0
    average_improvement: float = 0.0
    last_adaptation_time: float = 0.0


class AdaptationService:
    """
    Service for managing LNN adaptations
    
    Features:
    - Real-time parameter updates
    - Architecture modifications
    - Performance tracking
    - Rollback capabilities
    """
    
    def __init__(self):
        self.metrics = AdaptationMetrics()
        self.adaptation_history: List[Dict[str, Any]] = []
        self.parameter_snapshots: Dict[str, Dict[str, torch.Tensor]] = {}
        self.logger = logger.bind(service="adaptation")
        
    async def adapt_model(self,
                         model: torch.nn.Module,
                         feedback_signal: float,
                         adaptation_strength: float = 0.1,
                         target_metrics: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """
        Adapt model based on feedback
        
        Args:
            model: LNN model to adapt
            feedback_signal: Feedback value (-1 to 1)
            adaptation_strength: How strongly to adapt
            target_metrics: Optional target performance metrics
            
        Returns:
            Adaptation results
        """
        start_time = time.perf_counter()
        
        try:
            # Take snapshot before adaptation
            model_id = id(model)
            self._snapshot_parameters(model_id, model)
            
            # Get current performance
            old_performance = await self._evaluate_performance(model)
            
            # Determine adaptation type
            if abs(feedback_signal) > 0.8 and hasattr(model, 'lnn_core'):
                # Strong feedback - consider architecture change
                result = await self._adapt_architecture(model, feedback_signal, adaptation_strength)
            else:
                # Moderate feedback - parameter update
                result = await self._adapt_parameters(model, feedback_signal, adaptation_strength)
            
            # Evaluate new performance
            new_performance = await self._evaluate_performance(model)
            
            # Check if adaptation improved performance
            improvement = self._calculate_improvement(old_performance, new_performance, target_metrics)
            
            if improvement < 0:
                # Rollback if performance degraded
                self._rollback_parameters(model_id, model)
                result["rollback"] = True
                self.metrics.failed_adaptations += 1
            else:
                result["improvement"] = improvement
                self.metrics.successful_adaptations += 1
                self.metrics.average_improvement = (
                    self.metrics.average_improvement * 0.9 + improvement * 0.1
                )
            
            # Update metrics
            self.metrics.total_adaptations += 1
            self.metrics.last_adaptation_time = time.perf_counter()
            
            # Record history
            self.adaptation_history.append({
                "timestamp": time.time(),
                "feedback": feedback_signal,
                "strength": adaptation_strength,
                "type": result["type"],
                "improvement": improvement,
                "duration_ms": (time.perf_counter() - start_time) * 1000
            })
            
            return {
                "type": result["type"],
                "parameters_changed": result.get("parameters_changed", []),
                "old_performance": old_performance,
                "new_performance": new_performance,
                "improvement": improvement,
                "success": improvement >= 0,
                "duration_ms": (time.perf_counter() - start_time) * 1000
            }
            
        except Exception as e:
            self.logger.error("Adaptation failed", error=str(e))
            self.metrics.failed_adaptations += 1
            return {
                "type": "error",
                "parameters_changed": [],
                "success": False,
                "error": str(e)
            }
    
    async def _adapt_parameters(self,
                               model: torch.nn.Module,
                               feedback: float,
                               strength: float) -> Dict[str, Any]:
        """Adapt model parameters based on feedback"""
        parameters_changed = []
        
        # Adapt time constants
        if hasattr(model, 'lnn_core'):
            for i, layer in enumerate(model.lnn_core):
                if hasattr(layer, 'tau'):
                    # Adjust time constants
                    with torch.no_grad():
                        adjustment = strength * feedback
                        layer.tau *= (1 + adjustment)
                        layer.tau.clamp_(0.1, 10.0)
                    parameters_changed.append(f"layer_{i}_tau")
                
                if hasattr(layer, 'sigma'):
                    # Adjust sensitivity
                    with torch.no_grad():
                        adjustment = -strength * feedback * 0.5  # Inverse relationship
                        layer.sigma *= (1 + adjustment)
                        layer.sigma.clamp_(0.1, 2.0)
                    parameters_changed.append(f"layer_{i}_sigma")
        
        self.metrics.parameter_updates += 1
        
        return {
            "type": "parameter_update",
            "parameters_changed": parameters_changed
        }
    
    async def _adapt_architecture(self,
                                 model: torch.nn.Module,
                                 feedback: float,
                                 strength: float) -> Dict[str, Any]:
        """Adapt model architecture (for adaptive models)"""
        changes = []
        
        if hasattr(model, 'lnn_core') and hasattr(model.lnn_core[0], '_check_adaptation'):
            # Trigger adaptation check in adaptive layers
            for i, layer in enumerate(model.lnn_core):
                if hasattr(layer, '_check_adaptation'):
                    # Create synthetic stress signal
                    stress = abs(feedback) * strength
                    
                    # Temporarily modify growth threshold
                    old_threshold = layer.config.growth_threshold
                    layer.config.growth_threshold = stress
                    
                    # Trigger adaptation
                    dummy_output = torch.randn(1, layer.current_size)
                    info = layer._check_adaptation(dummy_output)
                    
                    # Restore threshold
                    layer.config.growth_threshold = old_threshold
                    
                    if info:
                        changes.append({f"layer_{i}": info})
        
        if changes:
            self.metrics.architecture_changes += 1
        
        return {
            "type": "architecture_change",
            "parameters_changed": ["architecture"],
            "changes": changes
        }
    
    async def _evaluate_performance(self, model: torch.nn.Module) -> Dict[str, float]:
        """Evaluate model performance metrics"""
        # Simple synthetic evaluation
        # In production, this would run actual benchmarks
        
        with torch.no_grad():
            # Test latency
            test_input = torch.randn(1, 128)  # Assuming 128 input size
            start = time.perf_counter()
            
            if hasattr(model, 'forward'):
                _ = model(test_input)
            
            latency = (time.perf_counter() - start) * 1000
            
            # Estimate other metrics
            info = model.get_info() if hasattr(model, 'get_info') else {}
            
            return {
                "latency": latency,
                "parameters": info.get("parameters", 0),
                "sparsity": 0.8,  # Placeholder
                "accuracy": 0.85 + np.random.normal(0, 0.05)  # Simulated
            }
    
    def _calculate_improvement(self,
                              old_perf: Dict[str, float],
                              new_perf: Dict[str, float],
                              targets: Optional[Dict[str, float]] = None) -> float:
        """Calculate overall improvement score"""
        if targets:
            # Calculate improvement relative to targets
            old_distance = sum(
                abs(old_perf.get(k, 0) - v) for k, v in targets.items()
            )
            new_distance = sum(
                abs(new_perf.get(k, 0) - v) for k, v in targets.items()
            )
            return (old_distance - new_distance) / (old_distance + 1e-6)
        else:
            # Simple improvement based on key metrics
            latency_improvement = (old_perf["latency"] - new_perf["latency"]) / old_perf["latency"]
            accuracy_improvement = new_perf.get("accuracy", 0) - old_perf.get("accuracy", 0)
            
            return latency_improvement * 0.3 + accuracy_improvement * 0.7
    
    def _snapshot_parameters(self, model_id: int, model: torch.nn.Module):
        """Take snapshot of model parameters"""
        snapshot = {}
        for name, param in model.named_parameters():
            snapshot[name] = param.data.clone()
        self.parameter_snapshots[model_id] = snapshot
    
    def _rollback_parameters(self, model_id: int, model: torch.nn.Module):
        """Rollback model parameters to snapshot"""
        if model_id in self.parameter_snapshots:
            snapshot = self.parameter_snapshots[model_id]
            with torch.no_grad():
                for name, param in model.named_parameters():
                    if name in snapshot:
                        param.data.copy_(snapshot[name])
            self.logger.info("Rolled back parameters", model_id=model_id)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get adaptation metrics"""
        return {
            "total_adaptations": self.metrics.total_adaptations,
            "parameter_updates": self.metrics.parameter_updates,
            "architecture_changes": self.metrics.architecture_changes,
            "success_rate": (
                self.metrics.successful_adaptations / max(self.metrics.total_adaptations, 1)
            ),
            "average_improvement": self.metrics.average_improvement,
            "last_adaptation": self.metrics.last_adaptation_time
        }
    
    def get_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get adaptation history"""
        return self.adaptation_history[-limit:]