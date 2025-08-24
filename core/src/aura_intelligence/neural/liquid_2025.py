"""
Liquid Neural Networks 2.0 - MIT August 2025
Self-modifying architecture during runtime
Integrates with your existing LNN Council
"""
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass
import asyncio

# Import your existing LNN system
from aura_intelligence.lnn import LiquidNeuralNetwork, LNNConfig
from ..agents.council.production_lnn_council import ProductionLNNCouncilAgent

@dataclass
class LiquidArchitectureConfig:
    """Configuration for self-modifying liquid architecture"""
    base_neurons: int = 128
    max_neurons: int = 512
    min_neurons: int = 64
    growth_rate: float = 0.1
    pruning_threshold: float = 0.01
    adaptation_window: int = 100
    complexity_threshold: float = 0.8

class AdaptiveNeuronPool(nn.Module):
    """Pool of neurons that can be dynamically allocated"""
    
    def __init__(self, max_size: int, feature_dim: int):
        super().__init__()
        self.max_size = max_size
        self.feature_dim = feature_dim
        
        # Pre-allocate maximum neurons
        self.neuron_weights = nn.Parameter(torch.randn(max_size, feature_dim))
        self.neuron_bias = nn.Parameter(torch.zeros(max_size))
        self.neuron_tau = nn.Parameter(torch.ones(max_size))
        
        # Active neuron mask
        self.register_buffer('active_mask', torch.ones(max_size, dtype=torch.bool))
        self.active_count = max_size
        
        # Usage tracking for pruning
        self.register_buffer('usage_stats', torch.zeros(max_size))
        
    def forward(self, x: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
        """Forward pass through active neurons only"""
        active_weights = self.neuron_weights[self.active_mask]
        active_bias = self.neuron_bias[self.active_mask]
        active_tau = self.neuron_tau[self.active_mask]
        active_hidden = hidden[:, self.active_mask]
        
        # Liquid dynamics
        input_current = torch.matmul(x, active_weights.T) + active_bias
        dx = torch.tanh(input_current + active_hidden)
        new_hidden = active_hidden + (dx - active_hidden) / active_tau
        
        # Update usage statistics
        self.usage_stats[self.active_mask] += torch.mean(torch.abs(new_hidden), dim=0)
        
        return new_hidden
    
    def grow_neurons(self, count: int) -> bool:
        """Add new neurons to the pool"""
        if self.active_count + count > self.max_size:
            return False
        
        # Find inactive neurons to activate
        inactive_indices = torch.where(~self.active_mask)[0][:count]
        if len(inactive_indices) < count:
            return False
        
        # Activate neurons with random initialization
        self.active_mask[inactive_indices] = True
        self.neuron_weights.data[inactive_indices] = torch.randn(count, self.feature_dim) * 0.1
        self.neuron_bias.data[inactive_indices] = torch.zeros(count)
        self.neuron_tau.data[inactive_indices] = torch.ones(count)
        
        self.active_count += count
        return True
    
    def prune_neurons(self, count: int) -> bool:
        """Remove least used neurons"""
        if self.active_count - count < 32:  # Minimum neurons
            return False
        
        # Find least used active neurons
        active_usage = self.usage_stats[self.active_mask]
        _, least_used_indices = torch.topk(active_usage, count, largest=False)
        
        # Convert to global indices
        active_indices = torch.where(self.active_mask)[0]
        prune_indices = active_indices[least_used_indices]
        
        # Deactivate neurons
        self.active_mask[prune_indices] = False
        self.usage_stats[prune_indices] = 0
        
        self.active_count -= count
        return True

class SelfModifyingLiquidNetwork(nn.Module):
    """Self-modifying liquid neural network that adapts architecture during runtime"""
    
    def __init__(self, config: LiquidArchitectureConfig):
        super().__init__()
        self.config = config
        
        # Adaptive neuron pools for each layer
        self.input_pool = AdaptiveNeuronPool(config.max_neurons, config.base_neurons)
        self.hidden_pool = AdaptiveNeuronPool(config.max_neurons, config.max_neurons)
        self.output_pool = AdaptiveNeuronPool(config.max_neurons // 2, config.max_neurons)
        
        # Architecture adaptation tracking
        self.adaptation_history = []
        self.complexity_buffer = []
        self.step_count = 0
        
        # Output projection
        self.output_proj = nn.Linear(config.max_neurons // 2, 4)  # For council votes
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Forward pass with potential architecture modification"""
        batch_size = x.size(0)
        
        # Initialize hidden states
        h1 = torch.zeros(batch_size, self.input_pool.active_count)
        h2 = torch.zeros(batch_size, self.hidden_pool.active_count)
        h3 = torch.zeros(batch_size, self.output_pool.active_count)
        
        # Forward through adaptive layers
        h1 = self.input_pool(x, h1)
        h2 = self.hidden_pool(h1, h2)
        h3 = self.output_pool(h2, h3)
        
        # Output projection
        output = self.output_proj(h3)
        
        # Calculate complexity metrics
        complexity = self._calculate_complexity(h1, h2, h3)
        self.complexity_buffer.append(complexity)
        
        # Trigger adaptation if needed
        adaptation_info = {}
        if len(self.complexity_buffer) >= self.config.adaptation_window:
            adaptation_info = self._adapt_architecture()
        
        self.step_count += 1
        
        return output, {
            'complexity': complexity,
            'active_neurons': {
                'input': self.input_pool.active_count,
                'hidden': self.hidden_pool.active_count,
                'output': self.output_pool.active_count
            },
            'adaptation_info': adaptation_info
        }
    
    def _calculate_complexity(self, h1: torch.Tensor, h2: torch.Tensor, h3: torch.Tensor) -> float:
        """Calculate current processing complexity"""
        # Measure activation diversity and magnitude
        activations = torch.cat([h1.flatten(), h2.flatten(), h3.flatten()])
        
        # Complexity based on entropy and magnitude
        activation_std = torch.std(activations).item()
        activation_mean = torch.mean(torch.abs(activations)).item()
        
        complexity = activation_std * activation_mean
        return min(1.0, complexity)
    
    def _adapt_architecture(self) -> Dict[str, Any]:
        """Adapt network architecture based on complexity history"""
        avg_complexity = np.mean(self.complexity_buffer[-self.config.adaptation_window:])
        self.complexity_buffer.clear()
        
        adaptation_info = {'avg_complexity': avg_complexity, 'actions': []}
        
        # Growth condition: high complexity, low neuron count
        if avg_complexity > self.config.complexity_threshold:
            total_neurons = (self.input_pool.active_count + 
                           self.hidden_pool.active_count + 
                           self.output_pool.active_count)
            
            if total_neurons < self.config.max_neurons * 2:
                # Grow neurons in bottleneck layers
                growth_count = int(self.config.base_neurons * self.config.growth_rate)
                
                if self.hidden_pool.grow_neurons(growth_count):
                    adaptation_info['actions'].append(f'grew_{growth_count}_hidden_neurons')
                
                if self.output_pool.grow_neurons(growth_count // 2):
                    adaptation_info['actions'].append(f'grew_{growth_count//2}_output_neurons')
        
        # Pruning condition: low complexity, high neuron count
        elif avg_complexity < self.config.pruning_threshold:
            prune_count = int(self.config.base_neurons * self.config.growth_rate)
            
            if self.hidden_pool.prune_neurons(prune_count):
                adaptation_info['actions'].append(f'pruned_{prune_count}_hidden_neurons')
            
            if self.output_pool.prune_neurons(prune_count // 2):
                adaptation_info['actions'].append(f'pruned_{prune_count//2}_output_neurons')
        
        self.adaptation_history.append(adaptation_info)
        return adaptation_info

class LiquidCouncilAgent2025(ProductionLNNCouncilAgent):
    """Enhanced council agent with self-modifying liquid architecture"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # Replace static LNN with self-modifying version
        liquid_config = LiquidArchitectureConfig(
            base_neurons=config.get('base_neurons', 128),
            max_neurons=config.get('max_neurons', 512),
            adaptation_window=config.get('adaptation_window', 100)
        )
        
        self.liquid_network = SelfModifyingLiquidNetwork(liquid_config)
        self.adaptation_stats = {'total_adaptations': 0, 'growth_events': 0, 'pruning_events': 0}
    
    async def _lnn_inference_step(self, state) -> Any:
        """Enhanced LNN inference with self-modification"""
        features = state.context.get("prepared_features")
        if features is None:
            raise ValueError("No prepared features found")
        
        # Run through self-modifying liquid network
        with torch.no_grad():
            output, network_info = self.liquid_network(features.unsqueeze(0))
            output_probs = torch.softmax(output, dim=-1)
        
        # Update adaptation statistics
        if network_info['adaptation_info']:
            self.adaptation_stats['total_adaptations'] += 1
            for action in network_info['adaptation_info'].get('actions', []):
                if 'grew' in action:
                    self.adaptation_stats['growth_events'] += 1
                elif 'pruned' in action:
                    self.adaptation_stats['pruning_events'] += 1
        
        state.lnn_output = output_probs
        state.context['network_info'] = network_info
        state.context['adaptation_stats'] = self.adaptation_stats
        
        state.next_step = "generate_vote"
        return state
    
    def get_architecture_stats(self) -> Dict[str, Any]:
        """Get current architecture statistics"""
        return {
            'current_architecture': {
                'input_neurons': self.liquid_network.input_pool.active_count,
                'hidden_neurons': self.liquid_network.hidden_pool.active_count,
                'output_neurons': self.liquid_network.output_pool.active_count,
                'total_neurons': (self.liquid_network.input_pool.active_count + 
                                self.liquid_network.hidden_pool.active_count + 
                                self.liquid_network.output_pool.active_count)
            },
            'adaptation_stats': self.adaptation_stats,
            'recent_adaptations': self.liquid_network.adaptation_history[-10:],
            'complexity_trend': np.mean(self.liquid_network.complexity_buffer) if self.liquid_network.complexity_buffer else 0.0
        }

# Factory function for creating liquid agents
def create_liquid_council_agent(config: Dict[str, Any]) -> LiquidCouncilAgent2025:
    """Create a self-modifying liquid council agent"""
    return LiquidCouncilAgent2025(config)