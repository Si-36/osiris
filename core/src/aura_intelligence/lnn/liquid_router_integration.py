"""
🔌 Liquid Neural Router Integration
===================================

Seamlessly integrates enhanced CfC-based LNN into AURA's model router.
Provides dynamic, continuous-time routing decisions.
"""

import asyncio
import torch
import numpy as np
from typing import Dict, Any, Optional, List
import structlog
from dataclasses import dataclass

from .enhanced_liquid_neural import (
    CfCConfig, TorchLiquidBridge, create_liquid_router
)
from ..neural.model_router import AURAModelRouter, ModelProvider

logger = structlog.get_logger(__name__)


@dataclass
class LiquidRoutingConfig:
    """Configuration for liquid neural routing"""
    # LNN settings
    base_neurons: int = 64
    max_neurons: int = 512
    num_tau_bands: int = 4
    
    # Routing thresholds
    complexity_threshold_low: float = 0.3
    complexity_threshold_high: float = 0.7
    
    # Model mapping
    model_complexity_map: Dict[str, tuple] = None
    
    def __post_init__(self):
        if self.model_complexity_map is None:
            self.model_complexity_map = {
                "simple": (0.0, 0.3),      # gpt-3.5-turbo
                "moderate": (0.3, 0.7),    # gpt-4
                "complex": (0.7, 1.0),     # claude-3-opus
            }


class LiquidModelRouter(AURAModelRouter):
    """
    Enhanced model router with liquid neural dynamics.
    Continuously adapts routing based on cognitive load.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # Initialize liquid routing
        self.liquid_config = LiquidRoutingConfig()
        self.liquid_bridge = create_liquid_router(
            base_neurons=self.liquid_config.base_neurons,
            max_neurons=self.liquid_config.max_neurons,
            use_attention=True
        )
        
        # Temporal memory for context
        self.temporal_buffer = []
        self.max_buffer_size = 5
        
        logger.info("Liquid Model Router initialized with CfC dynamics")
        
    async def route_request(self, 
                          prompt: str,
                          params: Optional[Dict[str, Any]] = None,
                          use_liquid: bool = True) -> Dict[str, Any]:
        """
        Route request using liquid dynamics when enabled.
        
        The liquid network analyzes prompt complexity in continuous time
        and selects the optimal model + parameters.
        """
        
        if not use_liquid:
            # Fall back to static routing
            return await super().route_request(prompt, params)
            
        try:
            # 1. Embed prompt
            embedding = await self._embed_prompt(prompt)
            
            # 2. Analyze with liquid dynamics
            complexity_profile = await self.liquid_bridge.adapter.analyze_complexity(embedding)
            
            # 3. Select model based on continuous complexity signal
            selected_model = self._select_model_continuous(complexity_profile)
            
            # 4. Adapt parameters based on dynamics
            adapted_params = self._adapt_parameters(complexity_profile, params or {})
            
            # 5. Route to selected model
            result = await self._execute_request(
                prompt=prompt,
                model=selected_model,
                params=adapted_params
            )
            
            # 6. Update temporal buffer
            self._update_temporal_buffer(complexity_profile)
            
            # Add liquid metrics to result
            result["liquid_metrics"] = {
                "complexity": complexity_profile["cognitive_load"],
                "selected_model": selected_model,
                "neuron_budget": self._compute_neuron_budget(complexity_profile),
                "tau_profile": complexity_profile.get("dominant_tau", "unknown")
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Liquid routing failed: {e}, falling back to static")
            return await super().route_request(prompt, params)
    
    async def _embed_prompt(self, prompt: str) -> np.ndarray:
        """Embed prompt for liquid analysis"""
        # Simple embedding for now - in production use sentence-transformers
        # or extract from model's embedding layer
        tokens = prompt.split()
        
        # Create simple bag-of-words embedding
        vocab_size = 10000
        embedding = np.zeros(128)  # Fixed size for LNN input
        
        for i, token in enumerate(tokens[:128]):
            # Simple hash-based embedding
            idx = hash(token) % 128
            embedding[idx] += 1.0
            
        # Normalize
        if np.linalg.norm(embedding) > 0:
            embedding = embedding / np.linalg.norm(embedding)
            
        return embedding.reshape(1, -1)  # Add batch dimension
    
    def _select_model_continuous(self, complexity: Dict[str, float]) -> str:
        """
        Select model based on continuous complexity signal.
        
        Maps continuous cognitive load to discrete model choices.
        """
        load = complexity["cognitive_load"]
        
        # Map to model based on complexity bands
        if load < self.liquid_config.complexity_threshold_low:
            # Simple prompt - use fast model
            return ModelProvider.OPENAI_GPT35.value
        elif load < self.liquid_config.complexity_threshold_high:
            # Moderate complexity - use balanced model
            return ModelProvider.OPENAI_GPT4.value
        else:
            # High complexity - use most capable model
            return ModelProvider.ANTHROPIC_CLAUDE.value
    
    def _adapt_parameters(self, 
                         complexity: Dict[str, float],
                         base_params: Dict[str, Any]) -> Dict[str, Any]:
        """Adapt generation parameters based on liquid dynamics"""
        params = base_params.copy()
        load = complexity["cognitive_load"]
        
        # Scale temperature inversely with complexity
        # Complex prompts need more deterministic responses
        if "temperature" not in params:
            params["temperature"] = max(0.1, 1.0 - load * 0.7)
        
        # Increase max tokens for complex prompts
        if "max_tokens" not in params:
            base_tokens = 500
            params["max_tokens"] = int(base_tokens * (1 + load))
        
        # Add repetition penalty for high complexity
        if load > 0.7:
            params["frequency_penalty"] = 0.5
            params["presence_penalty"] = 0.5
            
        return params
    
    def _compute_neuron_budget(self, complexity: Dict[str, float]) -> int:
        """Compute how many neurons were used"""
        load = complexity["cognitive_load"]
        return int(self.liquid_config.base_neurons + 
                  load * (self.liquid_config.max_neurons - self.liquid_config.base_neurons))
    
    def _update_temporal_buffer(self, complexity: Dict[str, float]):
        """Maintain temporal context across requests"""
        self.temporal_buffer.append({
            "timestamp": asyncio.get_event_loop().time(),
            "complexity": complexity
        })
        
        # Keep bounded size
        if len(self.temporal_buffer) > self.max_buffer_size:
            self.temporal_buffer.pop(0)
    
    async def stream_with_dynamics(self,
                                  prompt: str,
                                  params: Optional[Dict[str, Any]] = None):
        """
        Streaming generation with liquid dynamics.
        
        The liquid state persists across tokens for coherent generation.
        """
        # Embed prompt
        embedding = await self._embed_prompt(prompt)
        
        # Initialize streaming
        tokens = []
        liquid_states = []
        
        async for token in self._stream_tokens(prompt, params):
            # Update liquid state with each token
            token_embedding = await self._embed_prompt(token)
            
            # Process through liquid network
            output = await self.liquid_bridge.route_with_dynamics(
                torch.from_numpy(token_embedding),
                use_streaming=True
            )
            
            # Store state for context
            liquid_states.append(output)
            tokens.append(token)
            
            yield token
            
        # Return final metrics
        return {
            "tokens": tokens,
            "final_complexity": self._compute_final_complexity(liquid_states)
        }
    
    def _compute_final_complexity(self, states: List[torch.Tensor]) -> float:
        """Compute overall complexity from state sequence"""
        if not states:
            return 0.0
            
        # Average complexity across sequence
        complexities = [torch.norm(s).item() for s in states]
        return np.mean(complexities)
    
    async def explain_routing(self, prompt: str) -> Dict[str, Any]:
        """
        Explain why a particular model was chosen.
        
        Provides interpretable insights from liquid dynamics.
        """
        embedding = await self._embed_prompt(prompt)
        complexity = await self.liquid_bridge.adapter.analyze_complexity(embedding)
        
        selected_model = self._select_model_continuous(complexity)
        
        explanation = {
            "prompt": prompt[:100] + "..." if len(prompt) > 100 else prompt,
            "cognitive_load": complexity["cognitive_load"],
            "complexity_interpretation": self._interpret_complexity(complexity["cognitive_load"]),
            "selected_model": selected_model,
            "reasoning": self._generate_reasoning(complexity, selected_model),
            "neuron_usage": f"{self._compute_neuron_budget(complexity)}/{self.liquid_config.max_neurons}",
            "time_dynamics": self._interpret_tau(complexity.get("dominant_tau", 0))
        }
        
        return explanation
    
    def _interpret_complexity(self, load: float) -> str:
        """Human-readable complexity interpretation"""
        if load < 0.3:
            return "Simple - straightforward question or task"
        elif load < 0.7:
            return "Moderate - requires some reasoning or context"
        else:
            return "Complex - needs deep analysis or creative thinking"
    
    def _generate_reasoning(self, complexity: Dict[str, float], model: str) -> str:
        """Generate explanation for model selection"""
        load = complexity["cognitive_load"]
        
        if load < 0.3:
            return f"Selected {model} for quick response to simple query"
        elif load < 0.7:
            return f"Selected {model} for balanced performance on moderate task"
        else:
            return f"Selected {model} for maximum capability on complex reasoning"
    
    def _interpret_tau(self, tau_band: float) -> str:
        """Interpret dominant time constant"""
        tau_interpretations = [
            "Fast dynamics - reactive processing",
            "Medium dynamics - balanced processing", 
            "Slow dynamics - deliberative processing",
            "Very slow dynamics - deep contemplation"
        ]
        
        idx = min(int(tau_band), len(tau_interpretations) - 1)
        return tau_interpretations[idx]


# Factory function for easy integration
def create_liquid_model_router(config: Optional[Dict[str, Any]] = None) -> LiquidModelRouter:
    """Create a liquid-enhanced model router"""
    
    default_config = {
        "providers": {
            "openai": {"api_key": "..."},
            "anthropic": {"api_key": "..."}
        },
        "enable_caching": True,
        "enable_fallback": True
    }
    
    if config:
        default_config.update(config)
        
    return LiquidModelRouter(default_config)