"""
Neural Decision Engine (2025 Architecture)

Context-aware neural network inference with clean interfaces.
"""

import torch
from typing import Dict, Any
import structlog

from .context_aware_lnn import ContextAwareLNN
from aura_intelligence.config import LNNCouncilConfig
from .models import LNNCouncilState

logger = structlog.get_logger()


class NeuralDecisionEngine:
    """
    Neural decision engine using Context-Aware LNN.
    
    2025 Pattern:
    - Context-aware inference
    - Multi-source integration
    - Clean interfaces
    - Lazy initialization
    """
    
    def __init__(self, config: LNNCouncilConfig):
        self.config = config
        self.context_lnn = None  # Lazy initialization
    
    def _initialize_context_lnn(self) -> ContextAwareLNN:
        """Initialize Context-Aware LNN engine."""
        pass
        if self.context_lnn is None:
            self.context_lnn = ContextAwareLNN(self.config)
            
            if self.config.use_gpu and torch.cuda.is_available():
                self.context_lnn = self.context_lnn.cuda()
            
            logger.info("Context-Aware LNN engine initialized")
        
        return self.context_lnn
    
    def _encode_input(self, state: LNNCouncilState) -> torch.Tensor:
        """Encode state into neural network input."""
        request = state.current_request
        if not request:
            raise ValueError("No request to encode")
        
        # Create feature vector
        features = [
            {"A100": 1.0, "H100": 0.9, "V100": 0.8}.get(request.gpu_type, 0.5),
            request.gpu_count / 8.0,
            request.memory_gb / 80.0,
            request.compute_hours / 168.0,
            request.priority / 10.0
        ]
        
        # Add context features
        context = state.context_cache
        if context:
            utilization = context.get("current_utilization", {})
            features.extend([
                utilization.get("gpu_usage", 0.5),
                utilization.get("queue_length", 0) / 20.0
            ])
        
        # Pad to input size
        while len(features) < self.config.input_size:
            features.append(0.0)
        features = features[:self.config.input_size]
        
        tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
        
        if self.config.use_gpu and torch.cuda.is_available():
            tensor = tensor.cuda()
        
        return tensor
    
        async def make_decision(self, state: LNNCouncilState) -> Dict[str, Any]:
        """Make context-aware neural network decision."""
        # Initialize Context-Aware LNN
        context_lnn = self._initialize_context_lnn()
        
        # Run context-aware inference
        with torch.no_grad():
            output, attention_info = await context_lnn.forward_with_context(
                state, 
                return_attention=True
            )
        
        # Decode output
        decision_logits = output.squeeze()
        confidence_score = torch.sigmoid(decision_logits).max().item()
        
        decision_idx = torch.argmax(decision_logits).item()
        decisions = ["deny", "defer", "approve"]
        decision = decisions[min(decision_idx, len(decisions) - 1)]
        
        # Enhanced result with context information
        result = {
            "neural_decision": decision,
            "confidence_score": confidence_score,
            "decision_logits": decision_logits.tolist(),
            "context_aware": True
        }
        
        # Add attention information if available
        if attention_info:
            result.update({
                "context_sources": attention_info.get("context_sources", 0),
                "context_quality": attention_info.get("context_quality", 0.0),
                "attention_used": attention_info.get("attention_weights") is not None
            })
        
        logger.info(
            "Context-aware decision made",
            decision=decision,
            confidence=confidence_score,
            context_sources=result.get("context_sources", 0)
        )
        
        return result
    
        async def health_check(self) -> Dict[str, Any]:
        """Check neural engine health."""
        pass
        health = {
            "context_lnn_initialized": self.context_lnn is not None,
            "gpu_available": torch.cuda.is_available() if self.config.use_gpu else False,
            "config": {
                "input_size": self.config.input_size,
                "output_size": self.config.output_size,
                "context_aware": True
            }
        }
        
        # Add context provider health if initialized
        if self.context_lnn is not None:
            context_summary = self.context_lnn.get_context_summary(
                # Create a dummy state for health check
                type('DummyState', (), {'context_cache': {}})()
            )
            health["context_providers"] = context_summary
        
        return health