"""
Mixture of Experts - Based on Google's Switch Transformer and PaLM-2 MoE (2025)
Real implementation following DeepSpeed-MoE and FairScale patterns
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed import all_reduce, ReduceOp
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
import math

from ..components.real_registry import get_real_registry

class TopKGate(nn.Module):
    """Top-K gating with load balancing - Google Switch Transformer style"""
    
    def __init__(self, model_dim: int, num_experts: int, top_k: int = 2, 
        capacity_factor: float = 1.25, eval_capacity_factor: float = 2.0):
            pass
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.capacity_factor = capacity_factor
        self.eval_capacity_factor = eval_capacity_factor
        
        # Gating network - single linear layer like Switch Transformer
        self.wg = nn.Linear(model_dim, num_experts, bias=False)
        
        # Load balancing
        self.register_buffer('expert_counts', torch.zeros(num_experts))
        self.register_buffer('gate_scores_sum', torch.zeros(num_experts))
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Args:
            x: [batch_size, seq_len, model_dim]
        Returns:
            gates: [batch_size * seq_len, top_k]
            indices: [batch_size * seq_len, top_k] 
            load_balancing_loss: dict with auxiliary losses
        """
        batch_size, seq_len, model_dim = x.shape
        x = x.view(-1, model_dim)  # [batch_size * seq_len, model_dim]
        
        # Compute gate scores
        gate_logits = self.wg(x)  # [batch_size * seq_len, num_experts]
        gate_scores = F.softmax(gate_logits, dim=-1)
        
        # Top-k selection
        top_k_scores, top_k_indices = torch.topk(gate_scores, self.top_k, dim=-1)
        
        # Normalize top-k scores
        top_k_scores = top_k_scores / (top_k_scores.sum(dim=-1, keepdim=True) + 1e-8)
        
        # Load balancing loss (Switch Transformer)
        if self.training:
            # Fraction of tokens routed to each expert
            expert_mask = F.one_hot(top_k_indices, num_classes=self.num_experts).float()
            expert_counts = expert_mask.sum(dim=(0, 1))  # [num_experts]
            
            # Mean gate score for each expert
            gate_mean = gate_scores.mean(dim=0)  # [num_experts]
            
            # Load balancing loss
            load_balance_loss = self.num_experts * torch.sum(gate_mean * expert_counts) / x.size(0)
            
            # Update running statistics
            self.expert_counts = 0.9 * self.expert_counts + 0.1 * expert_counts
            self.gate_scores_sum = 0.9 * self.gate_scores_sum + 0.1 * gate_scores.sum(dim=0)
        else:
            load_balance_loss = torch.tensor(0.0, device=x.device)
        
        aux_loss = {
            'load_balancing_loss': load_balance_loss,
            'expert_usage': self.expert_counts.clone() if hasattr(self, 'expert_counts') else torch.zeros(self.num_experts)
        }
        
        return top_k_scores, top_k_indices, aux_loss

class MoELayer(nn.Module):
    """MoE Layer following DeepSpeed-MoE and FairScale implementations"""
    
    def __init__(self, model_dim: int, expert_dim: int, num_experts: int, top_k: int = 2):
        super().__init__()
        self.model_dim = model_dim
        self.expert_dim = expert_dim
        self.num_experts = num_experts
        self.top_k = top_k
        
        # Gate
        self.gate = TopKGate(model_dim, num_experts, top_k)
        
        # Experts - using grouped linear layers for efficiency
        self.experts_w1 = nn.Parameter(torch.randn(num_experts, model_dim, expert_dim))
        self.experts_w2 = nn.Parameter(torch.randn(num_experts, expert_dim, model_dim))
        self.experts_b1 = nn.Parameter(torch.zeros(num_experts, expert_dim))
        self.experts_b2 = nn.Parameter(torch.zeros(num_experts, model_dim))
        
        # Initialize like Transformer
        nn.init.xavier_uniform_(self.experts_w1)
        nn.init.xavier_uniform_(self.experts_w2)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Args:
            x: [batch_size, seq_len, model_dim]
        Returns:
            output: [batch_size, seq_len, model_dim]
            aux_loss: auxiliary losses for training
        """
        batch_size, seq_len, model_dim = x.shape
        x_flat = x.view(-1, model_dim)  # [batch_size * seq_len, model_dim]
        
        # Gating
        gate_scores, expert_indices, aux_loss = self.gate(x)
        
        # Dispatch to experts
        output = torch.zeros_like(x_flat)
        
        for i in range(self.top_k):
            # Get tokens for i-th expert choice
            expert_idx = expert_indices[:, i]  # [batch_size * seq_len]
            gate_score = gate_scores[:, i:i+1]  # [batch_size * seq_len, 1]
            
            # Group by expert
            for expert_id in range(self.num_experts):
                expert_mask = (expert_idx == expert_id)
                if not expert_mask.any():
                    continue
                
                # Get tokens for this expert
                expert_input = x_flat[expert_mask]  # [num_tokens, model_dim]
                expert_gate = gate_score[expert_mask]  # [num_tokens, 1]
                
                # Expert computation: FFN with SwiGLU activation (PaLM style)
                h1 = torch.matmul(expert_input, self.experts_w1[expert_id]) + self.experts_b1[expert_id]
                h1_gate, h1_up = h1.chunk(2, dim=-1) if h1.size(-1) % 2 == 0 else (h1, h1)
                h1_activated = F.silu(h1_gate) * h1_up if h1.size(-1) % 2 == 0 else F.silu(h1)
                
                expert_output = torch.matmul(h1_activated, self.experts_w2[expert_id]) + self.experts_b2[expert_id]
                expert_output = expert_output * expert_gate
                
                # Add to output
                output[expert_mask] += expert_output
        
        output = output.view(batch_size, seq_len, model_dim)
        return output, aux_loss

class ProductionMoE(nn.Module):
    """Production MoE following latest 2025 patterns"""
    
    def __init__(self, model_dim: int = 512, expert_dim: int = 2048, num_experts: int = 8, 
        num_layers: int = 4, top_k: int = 2):
            pass
        super().__init__()
        self.model_dim = model_dim
        self.num_experts = num_experts
        self.num_layers = num_layers
        
        # Input projection
        self.input_proj = nn.Linear(64, model_dim)  # From task features
        
        # MoE layers
        self.moe_layers = nn.ModuleList([
            MoELayer(model_dim, expert_dim, num_experts, top_k)
            for _ in range(num_layers)
        ])
        
        # Layer norms
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(model_dim) for _ in range(num_layers)
        ])
        
        # Output head
        self.output_head = nn.Linear(model_dim, num_experts)  # Expert selection scores
        
    def forward(self, task_features: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Args:
            task_features: [batch_size, feature_dim]
        Returns:
            expert_scores: [batch_size, num_experts]
            aux_losses: auxiliary losses for training
        """
        x = self.input_proj(task_features).unsqueeze(1)  # [batch_size, 1, model_dim]
        
        total_aux_loss = {}
        
        # Pass through MoE layers
        for i, (moe_layer, layer_norm) in enumerate(zip(self.moe_layers, self.layer_norms)):
            residual = x
            x = layer_norm(x)
            x, aux_loss = moe_layer(x)
            x = x + residual  # Residual connection
            
            # Accumulate auxiliary losses
            for key, value in aux_loss.items():
                if key not in total_aux_loss:
                    total_aux_loss[key] = value
                else:
                    total_aux_loss[key] += value
        
        # Output projection
        expert_scores = self.output_head(x.squeeze(1))  # [batch_size, num_experts]
        
        return expert_scores, total_aux_loss

class AURAMixtureOfExperts:
    """Production MoE system for AURA's 209 components"""
    
    def __init__(self, top_k: int = 3):
        self.registry = get_real_registry()
        self.components = list(self.registry.components.keys())
        self.num_experts = len(self.components)
        self.top_k = top_k
        
        # Production MoE network
        self.moe_network = ProductionMoE(
            model_dim=512,
            expert_dim=2048, 
            num_experts=min(64, self.num_experts),  # Limit for efficiency
            num_layers=4,
            top_k=top_k
        )
        
        # Expert mapping (map MoE experts to actual components)
        self.expert_to_components = self._create_expert_mapping()
        
        # Performance tracking
        self.expert_stats = {}
        self.routing_history = []
        
    def _create_expert_mapping(self) -> Dict[int, List[str]]:
        """Map MoE experts to groups of actual components"""
        pass
        mapping = {}
        components_per_expert = max(1, len(self.components) // min(64, self.num_experts))
        
        for i in range(min(64, self.num_experts)):
            start_idx = i * components_per_expert
            end_idx = min(start_idx + components_per_expert, len(self.components))
            mapping[i] = self.components[start_idx:end_idx]
        
        return mapping
    
    def _extract_task_features(self, task_data: Dict[str, Any]) -> torch.Tensor:
        """Extract features using latest feature engineering techniques"""
        features = []
        
        # Task metadata
        task_type = task_data.get('type', 'unknown')
        complexity = task_data.get('complexity', 0.5)
        priority = task_data.get('priority', 0.5)
        
        # One-hot encoding for task type
        task_types = ['neural', 'memory', 'agent', 'tda', 'orchestration', 'observability']
        type_encoding = [1.0 if t in task_type.lower() else 0.0 for t in task_types]
        features.extend(type_encoding)
        
        # Numerical features
        features.extend([
            complexity,
            priority,
            len(str(task_data)) / 1000.0,  # Data size
            hash(str(task_data)) % 1000 / 1000.0,  # Content hash
        ])
        
        # Statistical features
        if isinstance(task_data.get('data'), (list, np.ndarray)):
            data_array = np.array(task_data['data'])
            features.extend([
                np.mean(data_array) if data_array.size > 0 else 0.0,
                np.std(data_array) if data_array.size > 0 else 0.0,
                np.min(data_array) if data_array.size > 0 else 0.0,
                np.max(data_array) if data_array.size > 0 else 0.0,
            ])
        else:
            features.extend([0.0, 0.0, 0.0, 0.0])
        
        # Pad to 64 dimensions
        while len(features) < 64:
            features.append(0.0)
        
        return torch.tensor(features[:64], dtype=torch.float32).unsqueeze(0)
    
        async def route_task(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
            pass
        """Route task using production MoE"""
        start_time = time.time()
        
        # Extract features
        task_features = self._extract_task_features(task_data)
        
        # Get expert scores from MoE
        with torch.no_grad():
            expert_scores, aux_losses = self.moe_network(task_features)
            expert_probs = F.softmax(expert_scores, dim=-1)
        
        # Select top-k experts
        top_k_probs, top_k_indices = torch.topk(expert_probs, self.top_k, dim=-1)
        selected_expert_ids = top_k_indices.squeeze(0).tolist()
        selected_probs = top_k_probs.squeeze(0).tolist()
        
        # Map to actual components
        selected_components = []
        for expert_id, prob in zip(selected_expert_ids, selected_probs):
            components = self.expert_to_components.get(expert_id, [])
            if components:
                # Select best component from group based on recent performance
                best_component = self._select_best_component(components)
                selected_components.append((best_component, prob))
        
        # Execute on selected components
        results = {}
        for component_id, weight in selected_components:
            try:
                result = await self.registry.process_data(component_id, task_data)
                results[component_id] = {
                    'result': result,
                    'weight': weight,
                    'success': True
                }
                self._update_component_stats(component_id, True, time.time() - start_time)
            except Exception as e:
                results[component_id] = {
                    'error': str(e),
                    'weight': weight,
                    'success': False
                }
                self._update_component_stats(component_id, False, time.time() - start_time)
        
        # Weighted aggregation
        successful_results = [(k, v) for k, v in results.items() if v['success']]
        
        if successful_results:
            # Weighted average of results
            total_weight = sum(r[1]['weight'] for r in successful_results)
            aggregated_confidence = sum(
                r[1]['result'].get('confidence', 0.5) * r[1]['weight'] 
                for r in successful_results
            ) / total_weight if total_weight > 0 else 0.5
            
            final_result = {
                'success': True,
                'aggregated_confidence': aggregated_confidence,
                'selected_components': [r[0] for r in successful_results],
                'component_weights': {r[0]: r[1]['weight'] for r in successful_results},
                'moe_routing': True,
                'expert_scores': expert_probs.squeeze(0).tolist(),
                'aux_losses': {k: v.item() if torch.is_tensor(v) else v for k, v in aux_losses.items()}
            }
        else:
            final_result = {
                'success': False,
                'error': 'All selected components failed',
                'selected_components': [r[0] for r in selected_components],
                'moe_routing': True
            }
        
        # Record routing
        self.routing_history.append({
            'task_type': task_data.get('type', 'unknown'),
            'selected_experts': selected_expert_ids,
            'selected_components': [c[0] for c in selected_components],
            'success': final_result['success'],
            'processing_time': time.time() - start_time
        })
        
        return final_result
    
    def _select_best_component(self, components: List[str]) -> str:
        """Select best component from group based on performance"""
        if len(components) == 1:
            return components[0]
        
        # Select based on success rate and recent performance
        best_component = components[0]
        best_score = 0.0
        
        for component_id in components:
            stats = self.expert_stats.get(component_id, {'success_rate': 0.5, 'avg_time': 1.0})
            # Score = success_rate / avg_processing_time
            score = stats['success_rate'] / max(0.001, stats['avg_time'])
            if score > best_score:
                best_score = score
                best_component = component_id
        
        return best_component
    
    def _update_component_stats(self, component_id: str, success: bool, processing_time: float):
        """Update component performance statistics"""
        if component_id not in self.expert_stats:
            self.expert_stats[component_id] = {
                'total_requests': 0,
                'successful_requests': 0,
                'total_time': 0.0,
                'success_rate': 0.5,
                'avg_time': 1.0
            }
        
        stats = self.expert_stats[component_id]
        stats['total_requests'] += 1
        stats['total_time'] += processing_time
        
        if success:
            stats['successful_requests'] += 1
        
        stats['success_rate'] = stats['successful_requests'] / stats['total_requests']
        stats['avg_time'] = stats['total_time'] / stats['total_requests']
    
    def get_routing_stats(self) -> Dict[str, Any]:
        """Get comprehensive routing statistics"""
        pass
        if not self.routing_history:
            return {'no_history': True}
        
        recent_history = self.routing_history[-100:]
        success_rate = sum(1 for h in recent_history if h['success']) / len(recent_history)
        avg_time = np.mean([h['processing_time'] for h in recent_history])
        
        # Expert utilization
        expert_usage = {}
        for history in recent_history:
            for expert_id in history['selected_experts']:
                expert_usage[expert_id] = expert_usage.get(expert_id, 0) + 1
        
        return {
            'total_routings': len(self.routing_history),
            'recent_success_rate': success_rate,
            'avg_processing_time': avg_time,
            'expert_utilization': expert_usage,
            'moe_architecture': {
                'model_dim': self.moe_network.model_dim,
                'num_moe_experts': self.moe_network.num_experts,
                'num_actual_components': len(self.components),
                'top_k': self.top_k
            },
            'component_performance': dict(list(self.expert_stats.items())[:10])  # Top 10
        }

# Global instance
_aura_moe = None

    def get_aura_moe():
        global _aura_moe
        if _aura_moe is None:
            pass
        _aura_moe = AURAMixtureOfExperts()
        return _aura_moe
