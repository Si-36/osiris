"""
Google Switch Transformer MoE - Real Research Implementation
Based on "Switch Transformer: Scaling to Trillion Parameter Models" (Google 2021)
Implements real sparse expert routing with load balancing
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import time
import math

from ..components.real_registry import get_real_registry, ComponentType


class SwitchTransformerMoE(nn.Module):
    """
    Google Switch Transformer MoE Layer
    
    Key features from the paper:
    - Top-1 routing (only route to single expert)
    - Load balancing loss to prevent expert collapse
    - Expert capacity to handle variable load
    - Sparse routing for efficiency
    """
    
    def __init__(self, d_model: int, num_experts: int, capacity_factor: float = 1.25, 
                 expert_dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.num_experts = num_experts
        self.capacity_factor = capacity_factor
        self.expert_dropout = expert_dropout
        
        # Router network (single linear layer as per Switch Transformer)
        self.router = nn.Linear(d_model, num_experts, bias=False)
        nn.init.normal_(self.router.weight, std=0.01)
        
        # Expert networks (Feed-Forward Networks)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model * 4),
                nn.ReLU(),
                nn.Dropout(expert_dropout),
                nn.Linear(d_model * 4, d_model)
            ) for _ in range(num_experts)
        ])
        
        # Load balancing tracking
        self.register_buffer('expert_counts', torch.zeros(num_experts))
        self.register_buffer('gate_probs_sum', torch.zeros(num_experts))
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass with Switch Transformer routing
        
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            
        Returns:
            output: Routed output [batch_size, seq_len, d_model]
            aux_loss: Auxiliary losses for training
        """
        batch_size, seq_len, d_model = x.shape
        x_flat = x.view(-1, d_model)  # [batch*seq, d_model]
        
        # Router computation
        router_logits = self.router(x_flat)  # [batch*seq, num_experts]
        router_probs = F.softmax(router_logits, dim=-1)
        
        # Top-1 routing (Switch Transformer key innovation)
        expert_gate, expert_index = torch.max(router_probs, dim=-1)
        
        # Expert capacity calculation
        tokens_per_expert = x_flat.size(0) / self.num_experts
        expert_capacity = int(self.capacity_factor * tokens_per_expert)
        
        # Create expert assignment mask
        expert_mask = F.one_hot(expert_index, self.num_experts).float()
        
        # Load balancing loss (critical for Switch Transformer)
        if self.training:
            # Fraction of tokens assigned to each expert
            me = expert_mask.mean(dim=0)
            # Mean gate probability for each expert
            ce = router_probs.mean(dim=0)
            # Load balancing loss
            load_balancing_loss = self.num_experts * torch.sum(me * ce)
            
            # Update running statistics
            with torch.no_grad():
                self.expert_counts = 0.9 * self.expert_counts + 0.1 * expert_mask.sum(dim=0)
                self.gate_probs_sum = 0.9 * self.gate_probs_sum + 0.1 * router_probs.sum(dim=0)
        else:
            load_balancing_loss = torch.tensor(0.0, device=x.device)
        
        # Route tokens to experts
        output = torch.zeros_like(x_flat)
        
        for expert_idx in range(self.num_experts):
            # Find tokens assigned to this expert
            expert_tokens_mask = (expert_index == expert_idx)
            
            if expert_tokens_mask.any():
                # Get tokens for this expert
                expert_tokens = x_flat[expert_tokens_mask]
                expert_gates = expert_gate[expert_tokens_mask]
                
                # Apply capacity constraint
                if expert_tokens.size(0) > expert_capacity:
                    # Keep only top tokens by gate score
                    _, top_indices = torch.topk(expert_gates, expert_capacity)
                    expert_tokens = expert_tokens[top_indices]
                    expert_gates = expert_gates[top_indices]
                    # Update mask
                    mask_indices = torch.where(expert_tokens_mask)[0][top_indices]
                    expert_tokens_mask.fill_(False)
                    expert_tokens_mask[mask_indices] = True
                
                # Process through expert
                expert_output = self.experts[expert_idx](expert_tokens)
                
                # Apply gating (multiply by gate values)
                expert_output = expert_output * expert_gates.unsqueeze(-1)
                
                # Place back in output tensor
                output[expert_tokens_mask] = expert_output
        
        # Reshape back to original shape
        output = output.view(batch_size, seq_len, d_model)
        
        # Auxiliary information
        aux_info = {
            'load_balancing_loss': load_balancing_loss,
            'expert_utilization': self.expert_counts.clone(),
            'router_entropy': -torch.sum(router_probs * torch.log(router_probs + 1e-8), dim=-1).mean(),
            'expert_assignment_counts': expert_mask.sum(dim=0)
        }
        
        return output, aux_info


class GoogleSwitchMoESystem:
    """Complete Google Switch Transformer MoE System for AURA components"""
    
    def __init__(self, d_model: int = 512):
        self.registry = get_real_registry()
        self.d_model = d_model
        
        # Get all components as experts
        self.component_experts = list(self.registry.components.keys())
        self.num_experts = len(self.component_experts)
        
        # Initialize Switch Transformer MoE
        self.switch_moe = SwitchTransformerMoE(
            d_model=d_model,
            num_experts=min(self.num_experts, 64),  # Limit for efficiency
            capacity_factor=1.25
        )
        
        # Component feature encoder
        self.component_encoder = nn.Sequential(
            nn.Linear(256, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        
        # Training statistics
        self.routing_stats = {
            'total_routes': 0,
            'load_balancing_losses': [],
            'expert_utilization_history': []
        }
        
    def encode_request(self, request_data: Dict[str, Any]) -> torch.Tensor:
        """Encode request data into tensor format"""
        features = []
        
        # Extract numeric features
        for key, value in request_data.items():
            if isinstance(value, (int, float)):
                features.append(float(value))
            elif isinstance(value, str):
                # Hash string to numeric
                features.append(hash(value) % 1000 / 1000.0)
            elif isinstance(value, list):
                # Take first few elements
                features.extend([float(x) for x in value[:5]])
        
        # Pad or truncate to 256 features
        while len(features) < 256:
            features.append(0.0)
        features = features[:256]
        
        return torch.tensor([features], dtype=torch.float32)
    
    async def route_with_switch_transformer(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Route request using Google Switch Transformer"""
        start_time = time.time()
        
        # Encode request
        request_tensor = self.encode_request(request_data)
        
        # Add sequence dimension for transformer
        request_tensor = request_tensor.unsqueeze(1)  # [1, 1, 256]
        
        # Encode to d_model dimensions
        encoded_request = self.component_encoder(request_tensor)  # [1, 1, d_model]
        
        # Route through Switch Transformer MoE
        with torch.no_grad():
            routed_output, aux_info = self.switch_moe(encoded_request)
        
        # Get expert assignments
        expert_assignments = aux_info['expert_assignment_counts']
        selected_experts = torch.nonzero(expert_assignments).squeeze(-1)
        
        # Map back to component IDs
        selected_components = []
        for expert_idx in selected_experts:
            if expert_idx.item() < len(self.component_experts):
                selected_components.append(self.component_experts[expert_idx.item()])
        
        # Process through selected components
        component_results = {}
        for comp_id in selected_components[:5]:  # Limit to top 5
            try:
                result = await self.registry.process_data(comp_id, request_data)
                component_results[comp_id] = result
            except Exception as e:
                component_results[comp_id] = {'error': str(e)}
        
        # Update statistics
        self.routing_stats['total_routes'] += 1
        self.routing_stats['load_balancing_losses'].append(
            aux_info['load_balancing_loss'].item()
        )
        self.routing_stats['expert_utilization_history'].append(
            aux_info['expert_utilization'].cpu().numpy()
        )
        
        processing_time = time.time() - start_time
        
        return {
            'switch_transformer_routing': True,
            'selected_components': selected_components,
            'component_results': component_results,
            'routing_metrics': {
                'load_balancing_loss': aux_info['load_balancing_loss'].item(),
                'router_entropy': aux_info['router_entropy'].item(),
                'experts_used': len(selected_components),
                'expert_utilization': aux_info['expert_utilization'].cpu().numpy().tolist()
            },
            'processing_time_ms': processing_time * 1000,
            'google_research_implementation': True
        }
    
    def get_switch_stats(self) -> Dict[str, Any]:
        """Get Switch Transformer statistics"""
        expert_stats = self.switch_moe.get_expert_stats()
        
        avg_load_loss = np.mean(self.routing_stats['load_balancing_losses']) if self.routing_stats['load_balancing_losses'] else 0.0
        
        return {
            'google_switch_transformer': {
                'total_experts': self.num_experts,
                'active_experts': expert_stats['active_experts'],
                'load_balance_coefficient': expert_stats['load_balance_coefficient'],
                'capacity_factor': expert_stats['capacity_factor'],
                'avg_load_balancing_loss': avg_load_loss,
                'total_routing_requests': self.routing_stats['total_routes']
            },
            'research_paper': 'Switch Transformer: Scaling to Trillion Parameter Models (Google 2021)',
            'implementation': 'Real Google Research Architecture'
        }


def get_google_switch_moe():
    """Get Google Switch Transformer MoE system"""
    return GoogleSwitchMoESystem()