"""
Neural Persistence Integration
=============================
Integrates LNN, MoE, Mamba-2 with causal persistence
"""

import torch
import asyncio
from typing import Dict, Any, Optional, List
from datetime import datetime
import structlog

from ..persistence.causal_state_manager import (
    get_causal_manager,
    StateType,
    CausalContext
)

logger = structlog.get_logger(__name__)

class NeuralPersistenceMixin:
    """Mixin for neural networks to use causal persistence"""
    
    def __init__(self, model_id: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_id = model_id
        self._persistence_manager = None
        self._training_history = []
        self._experiment_branches = {}
        
    async def _ensure_persistence(self):
        """Ensure persistence manager is initialized"""
        if self._persistence_manager is None:
            self._persistence_manager = await get_causal_manager()
    
    async def save_checkpoint(self, 
                            epoch: int,
                            metrics: Dict[str, float],
                            branch_name: Optional[str] = None,
                            causes: Optional[List[str]] = None) -> str:
        """Save model checkpoint with causal tracking"""
        await self._ensure_persistence()
        
        # Extract causal information
        if causes is None:
            causes = self._extract_training_causes(epoch, metrics)
        
        effects = self._predict_training_effects(metrics)
        
        # Create causal context
        causal_context = CausalContext(
            causes=causes,
            effects=effects,
            confidence=metrics.get("validation_accuracy", 0.0),
            energy_cost=metrics.get("training_time", 0.0),
            counterfactuals=self._generate_training_counterfactuals(metrics)
        )
        
        # Prepare checkpoint data
        checkpoint_data = {
            "epoch": epoch,
            "model_state": self.state_dict() if hasattr(self, 'state_dict') else {},
            "optimizer_state": self.optimizer.state_dict() if hasattr(self, 'optimizer') else {},
            "metrics": metrics,
            "timestamp": datetime.now().isoformat(),
            "model_config": self._get_model_config(),
            "training_history": self._training_history[-10:]  # Last 10 epochs
        }
        
        # Save to branch if experimenting
        branch_id = None
        if branch_name:
            branch_id = await self._persistence_manager.create_branch(
                self.model_id,
                branch_name
            )
            logger.info(f"Created experiment branch", 
                       branch_id=branch_id,
                       branch_name=branch_name)
        
        # Save checkpoint
        state_id = await self._persistence_manager.save_state(
            StateType.NEURAL_CHECKPOINT,
            self.model_id,
            checkpoint_data,
            causal_context=causal_context,
            branch_id=branch_id
        )
        
        # Update history
        self._training_history.append({
            "epoch": epoch,
            "state_id": state_id,
            "metrics": metrics,
            "timestamp": datetime.now()
        })
        
        logger.info("Saved neural checkpoint with causality",
                   state_id=state_id,
                   epoch=epoch,
                   branch=branch_name,
                   causes=causes[:2])
        
        return state_id
    
    async def load_checkpoint(self, 
                            checkpoint_id: Optional[str] = None,
                            branch_name: Optional[str] = None,
                            compute_fn: Optional[callable] = None) -> Dict[str, Any]:
        """Load checkpoint with optional computation"""
        await self._ensure_persistence()
        
        # Get branch ID if specified
        branch_id = None
        if branch_name and branch_name in self._experiment_branches:
            branch_id = self._experiment_branches[branch_name]
        
        checkpoint = await self._persistence_manager.load_state(
            StateType.NEURAL_CHECKPOINT,
            self.model_id,
            branch_id=branch_id,
            compute_on_retrieval=compute_fn
        )
        
        if checkpoint:
            # Restore model state
            if hasattr(self, 'load_state_dict') and 'model_state' in checkpoint:
                self.load_state_dict(checkpoint['model_state'])
            
            # Restore optimizer state
            if hasattr(self, 'optimizer') and 'optimizer_state' in checkpoint:
                self.optimizer.load_state_dict(checkpoint['optimizer_state'])
            
            logger.info("Loaded neural checkpoint",
                       epoch=checkpoint.get('epoch'),
                       metrics=checkpoint.get('metrics'))
        
        return checkpoint
    
    async def compare_experiments(self, 
                                branch_names: List[str]) -> Dict[str, Any]:
        """Compare results across experimental branches"""
        await self._ensure_persistence()
        
        comparisons = {}
        
        for branch_name in branch_names:
            if branch_name in self._experiment_branches:
                branch_id = self._experiment_branches[branch_name]
                
                # Load branch checkpoint
                checkpoint = await self._persistence_manager.load_state(
                    StateType.NEURAL_CHECKPOINT,
                    self.model_id,
                    branch_id=branch_id
                )
                
                if checkpoint:
                    comparisons[branch_name] = {
                        "epoch": checkpoint.get("epoch"),
                        "metrics": checkpoint.get("metrics"),
                        "timestamp": checkpoint.get("timestamp")
                    }
        
        # Find best performing branch
        best_branch = max(
            comparisons.items(),
            key=lambda x: x[1]["metrics"].get("validation_accuracy", 0)
        )[0] if comparisons else None
        
        return {
            "branches": comparisons,
            "best_branch": best_branch,
            "recommendation": f"Merge {best_branch}" if best_branch else "Continue main"
        }
    
    async def get_training_causality(self, 
                                   start_epoch: int = 0) -> List[Dict[str, Any]]:
        """Get causal chain of training decisions"""
        await self._ensure_persistence()
        
        causal_chain = []
        
        for entry in self._training_history[start_epoch:]:
            if "state_id" in entry:
                chain = await self._persistence_manager.get_causal_chain(
                    entry["state_id"]
                )
                if chain:
                    causal_chain.extend(chain)
        
        return causal_chain
    
    def _extract_training_causes(self, 
                               epoch: int, 
                               metrics: Dict[str, float]) -> List[str]:
        """Extract what caused this checkpoint"""
        causes = [f"epoch_{epoch}"]
        
        # Learning rate changes
        if hasattr(self, 'scheduler') and hasattr(self.scheduler, 'get_last_lr'):
            lr = self.scheduler.get_last_lr()[0]
            if epoch > 0 and self._training_history:
                prev_lr = self._training_history[-1].get("learning_rate", lr)
                if lr != prev_lr:
                    causes.append(f"lr_changed_to_{lr}")
        
        # Performance triggers
        if metrics.get("validation_loss", float('inf')) < 0.1:
            causes.append("low_validation_loss")
        
        if metrics.get("validation_accuracy", 0) > 0.95:
            causes.append("high_accuracy_achieved")
        
        # Overfitting detection
        if "train_loss" in metrics and "validation_loss" in metrics:
            if metrics["validation_loss"] > metrics["train_loss"] * 1.5:
                causes.append("overfitting_detected")
        
        return causes
    
    def _predict_training_effects(self, metrics: Dict[str, float]) -> List[str]:
        """Predict effects of current training state"""
        effects = []
        
        # Performance effects
        if metrics.get("validation_accuracy", 0) > 0.9:
            effects.append("model_ready_for_deployment")
        
        if metrics.get("validation_loss", float('inf')) < 0.05:
            effects.append("near_optimal_performance")
        
        # Training effects
        if metrics.get("gradient_norm", 0) < 0.001:
            effects.append("training_may_stagnate")
        
        if metrics.get("training_time", 0) > 3600:  # 1 hour
            effects.append("long_training_time")
        
        return effects
    
    def _generate_training_counterfactuals(self, 
                                         metrics: Dict[str, float]) -> Dict[str, Any]:
        """Generate counterfactual training scenarios"""
        counterfactuals = {}
        
        # Different learning rates
        if hasattr(self, 'optimizer'):
            current_lr = self.optimizer.param_groups[0]['lr']
            counterfactuals["higher_lr"] = {
                "learning_rate": current_lr * 2,
                "expected_effect": "faster_convergence_higher_instability"
            }
            counterfactuals["lower_lr"] = {
                "learning_rate": current_lr * 0.5,
                "expected_effect": "slower_convergence_more_stable"
            }
        
        # Different batch sizes
        counterfactuals["larger_batch"] = {
            "batch_size": 128,
            "expected_effect": "smoother_gradients_higher_memory"
        }
        
        return counterfactuals
    
    def _get_model_config(self) -> Dict[str, Any]:
        """Get model configuration"""
        config = {
            "model_type": self.__class__.__name__,
            "model_id": self.model_id
        }
        
        # Add model-specific config
        if hasattr(self, 'config'):
            config.update(self.config)
        elif hasattr(self, 'get_config'):
            config.update(self.get_config())
        
        return config


class PersistentLNN(NeuralPersistenceMixin):
    """LNN with persistence capabilities"""
    
    async def save_liquid_state(self, 
                              time_constants: Dict[str, float],
                              adaptation_state: Dict[str, Any]) -> str:
        """Save LNN-specific state"""
        await self._ensure_persistence()
        
        # LNN-specific causes
        causes = ["liquid_adaptation"]
        if any(tc > 1.0 for tc in time_constants.values()):
            causes.append("slow_dynamics_detected")
        
        causal_context = CausalContext(
            causes=causes,
            effects=["time_constants_updated", "adaptation_complete"],
            confidence=0.9
        )
        
        state_data = {
            "time_constants": time_constants,
            "adaptation_state": adaptation_state,
            "timestamp": datetime.now().isoformat()
        }
        
        return await self._persistence_manager.save_state(
            StateType.COMPONENT_STATE,
            f"{self.model_id}_liquid",
            state_data,
            causal_context=causal_context
        )


class PersistentMoE(NeuralPersistenceMixin):
    """MoE with persistence capabilities"""
    
    async def save_expert_routing(self, 
                                routing_stats: Dict[str, Any],
                                expert_usage: Dict[int, float]) -> str:
        """Save MoE routing decisions with causality"""
        await self._ensure_persistence()
        
        # Analyze expert usage patterns
        causes = ["routing_update"]
        underused = [e for e, usage in expert_usage.items() if usage < 0.1]
        if underused:
            causes.append(f"experts_underused_{underused}")
        
        effects = []
        if max(expert_usage.values()) > 0.5:
            effects.append("expert_specialization_emerging")
        
        causal_context = CausalContext(
            causes=causes,
            effects=effects,
            confidence=0.85
        )
        
        state_data = {
            "routing_stats": routing_stats,
            "expert_usage": expert_usage,
            "timestamp": datetime.now().isoformat()
        }
        
        return await self._persistence_manager.save_state(
            StateType.COMPONENT_STATE,
            f"{self.model_id}_routing",
            state_data,
            causal_context=causal_context
        )


class PersistentMamba(NeuralPersistenceMixin):
    """Mamba-2 with persistence capabilities"""
    
    async def save_ssm_state(self, 
                           hidden_states: torch.Tensor,
                           conv_states: Dict[str, torch.Tensor]) -> str:
        """Save Mamba SSM state with causality"""
        await self._ensure_persistence()
        
        # Analyze state patterns
        causes = ["ssm_update"]
        state_norm = torch.norm(hidden_states).item()
        if state_norm > 100:
            causes.append("high_state_magnitude")
        
        causal_context = CausalContext(
            causes=causes,
            effects=["state_evolution", "memory_updated"],
            confidence=0.9
        )
        
        state_data = {
            "hidden_states_norm": state_norm,
            "hidden_states_shape": list(hidden_states.shape),
            "conv_states_keys": list(conv_states.keys()),
            "timestamp": datetime.now().isoformat()
        }
        
        return await self._persistence_manager.save_state(
            StateType.COMPONENT_STATE,
            f"{self.model_id}_ssm",
            state_data,
            causal_context=causal_context
        )