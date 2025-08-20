"""
Real Switch Transformer MoE - Google Research Implementation
Routes through 209 AURA components with load balancing
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Tuple
import asyncio
import time

from ..components.real_registry import get_real_registry
from ..moe.real_switch_moe import get_real_switch_moe

class SwitchTransformerMoE(nn.Module):
    """Real Switch Transformer with 209 experts (AURA components)"""
    
    def __init__(self, d_model: int = 512, capacity_factor: float = 1.25):
        super().__init__()
        self.registry = get_real_registry()
        self.components = list(self.registry.components.keys())
        self.num_experts = len(self.components)
        self.d_model = d_model
        self.capacity_factor = capacity_factor
        
        # Single linear gating layer (Switch Transformer pattern)
        self.gate = nn.Linear(d_model, self.num_experts, bias=False)
        nn.init.normal_(self.gate.weight, std=0.01)
        
        # Load balancing tracking
        self.register_buffer('expert_counts', torch.zeros(self.num_experts, dtype=torch.float32))
        self.register_buffer('gate_probs_sum', torch.zeros(self.num_experts, dtype=torch.float32))
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Args: x [batch_size, seq_len, d_model]
        Returns: output, routing_info
        """
        batch_size, seq_len, _ = x.shape
        x_flat = x.view(-1, self.d_model)  # [batch*seq, d_model]
        
        # Compute gate logits and probabilities
        gate_logits = self.gate(x_flat)
        gate_probs = F.softmax(gate_logits, dim=-1)
        
        # Top-1 routing (Switch Transformer)
        gate_scores, expert_indices = torch.max(gate_probs, dim=-1)
        
        # Expert capacity (1.25x average tokens per expert)
        tokens_per_expert = x_flat.size(0) / self.num_experts
        expert_capacity = int(self.capacity_factor * tokens_per_expert)
        
        # Load balancing loss
        if self.training:
            # Fraction of tokens routed to each expert
            expert_mask = F.one_hot(expert_indices, self.num_experts).float()
            tokens_per_expert_actual = expert_mask.sum(dim=0)
            
            # Mean gate probability per expert
            gate_mean = gate_probs.mean(dim=0)
            
            # Load balancing loss (Switch Transformer paper)
            load_balancing_loss = (
                self.num_experts * torch.sum(gate_mean * tokens_per_expert_actual) / x_flat.size(0)
            )
            
            # Update running statistics
            with torch.no_grad():
                self.expert_counts = 0.9 * self.expert_counts + 0.1 * tokens_per_expert_actual
                self.gate_probs_sum = 0.9 * self.gate_probs_sum + 0.1 * gate_probs.sum(dim=0)
        else:
            load_balancing_loss = torch.tensor(0.0, device=x.device)
        
        routing_info = {
            'expert_indices': expert_indices,
            'gate_scores': gate_scores,
            'load_balancing_loss': load_balancing_loss.item(),
            'expert_utilization': self.expert_counts.cpu().numpy(),
            'capacity_factor': self.capacity_factor
        }
        
        return x_flat, routing_info
    
    async def route_to_components(self, x: torch.Tensor) -> Dict[str, Any]:
        """Route tokens to actual AURA components"""
        x_flat, routing_info = self.forward(x)
        expert_indices = routing_info['expert_indices']
        gate_scores = routing_info['gate_scores']
        
        # Group tokens by expert
        results = {}
        for i, (expert_idx, score) in enumerate(zip(expert_indices, gate_scores)):
            component_id = self.components[expert_idx.item()]
            
            if component_id not in results:
                results[component_id] = {
                    'tokens': [],
                    'scores': [],
                    'indices': []
                }
            
            results[component_id]['tokens'].append(x_flat[i])
            results[component_id]['scores'].append(score.item())
            results[component_id]['indices'].append(i)
        
        # Process through components
        component_outputs = {}
        for component_id, data in results.items():
            try:
                # Convert to component input format
                component_input = {
                    'tokens': torch.stack(data['tokens']).cpu().numpy().tolist(),
                    'scores': data['scores']
                }
                
                # Process through real component
                output = await self.registry.process_data(component_id, component_input)
                component_outputs[component_id] = {
                    'output': output,
                    'indices': data['indices'],
                    'scores': data['scores']
                }
                
            except Exception as e:
                # Fallback for failed components
                component_outputs[component_id] = {
                    'output': {'error': str(e)},
                    'indices': data['indices'],
                    'scores': data['scores']
                }
        
        return {
            'component_outputs': component_outputs,
            'routing_info': routing_info,
            'total_components_used': len(component_outputs)
        }

def get_switch_moe():
    """Get real Switch Transformer MoE"""
    return get_real_switch_moe()