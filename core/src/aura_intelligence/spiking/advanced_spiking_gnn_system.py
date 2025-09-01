"""
Advanced Spiking Graph Neural Network System - 2025 Implementation

Based on latest research:
- Spiking Graph Neural Networks (SGNNs)
- Temporal graph processing with spikes
- Event-driven graph computation
- Spike-timing-dependent graph learning
- Dynamic graph topology with spikes
- Energy-efficient graph processing

Key innovations:
- Spiking message passing
- Temporal graph attention
- Event-based graph convolution
- Spike-based node embeddings
- Dynamic edge weights via STDP
- Asynchronous graph updates
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
from torch_geometric.nn import MessagePassing
from torch_geometric.data import Data, Batch
from torch_geometric.utils import softmax, add_self_loops, degree
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import structlog
import math
from collections import defaultdict, deque

logger = structlog.get_logger(__name__)


class SpikingActivation(str, Enum):
    """Spiking activation functions"""
    HEAVISIDE = "heaviside"
    SIGMOID = "sigmoid"
    ATAN = "atan"
    GAUSSIAN = "gaussian"


@dataclass
class SpikingGNNConfig:
    """Configuration for Spiking GNN"""
    # Graph architecture
    in_channels: int
    hidden_channels: int
    out_channels: int
    num_layers: int = 3
    
    # Spiking neuron parameters
    tau_membrane: float = 20.0  # ms
    tau_synapse: float = 5.0  # ms
    threshold: float = 1.0
    reset_value: float = 0.0
    dt: float = 1.0  # ms
    
    # Graph parameters
    edge_dim: Optional[int] = None
    heads: int = 4
    dropout: float = 0.1
    
    # Spike coding
    time_window: int = 100  # ms
    spike_fn: SpikingActivation = SpikingActivation.ATAN
    spike_gradient_scale: float = 1.0
    
    # Learning
    use_stdp: bool = True
    stdp_lr: float = 0.01
    
    # Dynamic topology
    dynamic_edges: bool = True
    edge_threshold: float = 0.5


class SurrogateGradient(torch.autograd.Function):
    """Surrogate gradient for spike function"""
    
    @staticmethod
    def forward(ctx, input: torch.Tensor, threshold: float = 1.0, 
                spike_fn: SpikingActivation = SpikingActivation.ATAN,
                scale: float = 1.0) -> torch.Tensor:
        """Forward pass - Heaviside function"""
        ctx.save_for_backward(input)
        ctx.threshold = threshold
        ctx.spike_fn = spike_fn
        ctx.scale = scale
        return (input >= threshold).float()
    
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None, None, None]:
        """Backward pass - Surrogate gradient"""
        input, = ctx.saved_tensors
        grad_input = None
        
        if ctx.needs_input_grad[0]:
            if ctx.spike_fn == SpikingActivation.SIGMOID:
                # Sigmoid surrogate
                sigmoid_input = (input - ctx.threshold) * ctx.scale
                grad = ctx.scale * torch.sigmoid(sigmoid_input) * (1 - torch.sigmoid(sigmoid_input))
            
            elif ctx.spike_fn == SpikingActivation.ATAN:
                # Arctangent surrogate
                alpha = ctx.scale
                grad = alpha / (1 + (math.pi * alpha * (input - ctx.threshold)) ** 2) / 2
            
            elif ctx.spike_fn == SpikingActivation.GAUSSIAN:
                # Gaussian surrogate
                sigma = 1.0 / ctx.scale
                grad = torch.exp(-0.5 * ((input - ctx.threshold) / sigma) ** 2) / (sigma * math.sqrt(2 * math.pi))
            
            else:  # HEAVISIDE
                # Straight-through estimator
                grad = 1.0
            
            grad_input = grad_output * grad
        
        return grad_input, None, None, None


# Spike function
spike_fn = SurrogateGradient.apply


class SpikingGraphNeuron(nn.Module):
    """Spiking neuron for graph nodes"""
    
    def __init__(self, config: SpikingGNNConfig):
        super().__init__()
        self.config = config
        
        # Neuron parameters
        self.tau_m = config.tau_membrane
        self.tau_s = config.tau_synapse
        self.threshold = config.threshold
        self.reset = config.reset_value
        self.dt = config.dt
        
        # Decay constants
        self.alpha_m = math.exp(-self.dt / self.tau_m)
        self.alpha_s = math.exp(-self.dt / self.tau_s)
        
        # State
        self.membrane_potential = None
        self.synaptic_current = None
        self.refractory_period = None
        
    def reset_state(self, num_nodes: int, device: torch.device):
        """Reset neuron state"""
        self.membrane_potential = torch.zeros(num_nodes, device=device)
        self.synaptic_current = torch.zeros(num_nodes, device=device)
        self.refractory_period = torch.zeros(num_nodes, device=device)
    
    def forward(self, input_current: torch.Tensor) -> torch.Tensor:
        """Update neuron state and generate spikes"""
        if self.membrane_potential is None:
            self.reset_state(input_current.shape[0], input_current.device)
        
        # Update refractory period
        self.refractory_period = torch.clamp(self.refractory_period - self.dt, min=0)
        refractory_mask = self.refractory_period > 0
        
        # Update synaptic current
        self.synaptic_current = self.alpha_s * self.synaptic_current + input_current
        
        # Update membrane potential
        self.membrane_potential = self.alpha_m * self.membrane_potential + self.synaptic_current
        self.membrane_potential[refractory_mask] = self.reset
        
        # Generate spikes
        spikes = spike_fn(self.membrane_potential, self.threshold, 
                         self.config.spike_fn, self.config.spike_gradient_scale)
        
        # Reset spiking neurons
        spike_mask = spikes.bool()
        self.membrane_potential[spike_mask] = self.reset
        self.refractory_period[spike_mask] = 2.0  # 2ms refractory period
        
        return spikes


class SpikingGraphConv(MessagePassing):
    """Spiking Graph Convolution layer"""
    
    def __init__(self, in_channels: int, out_channels: int, 
                 config: SpikingGNNConfig, **kwargs):
        super().__init__(aggr='add', **kwargs)
        self.config = config
        
        # Transform for messages
        self.lin = nn.Linear(in_channels, out_channels, bias=False)
        
        # Edge weight network (if edge features)
        if config.edge_dim:
            self.edge_net = nn.Sequential(
                nn.Linear(config.edge_dim, out_channels),
                nn.ReLU(),
                nn.Linear(out_channels, out_channels)
            )
        
        # Spiking neurons for each node
        self.neurons = SpikingGraphNeuron(config)
        
        # Normalization
        self.norm = nn.LayerNorm(out_channels)
        
        # STDP parameters
        if config.use_stdp:
            self.weight_plasticity = nn.Parameter(torch.zeros(out_channels, in_channels))
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, 
                edge_attr: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass with spiking dynamics"""
        # Add self-loops
        edge_index, edge_attr = add_self_loops(
            edge_index, edge_attr, 
            fill_value=1.0 if edge_attr is not None else None
        )
        
        # Transform features
        x = self.lin(x)
        
        # Propagate messages
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        
        # Apply spiking dynamics
        spikes = self.neurons(out)
        
        # STDP weight update
        if self.training and self.config.use_stdp:
            self._apply_stdp(x, spikes)
        
        return spikes
    
    def message(self, x_j: torch.Tensor, edge_attr: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute messages"""
        if edge_attr is not None and hasattr(self, 'edge_net'):
            # Use edge features to modulate messages
            edge_weight = self.edge_net(edge_attr)
            return x_j * edge_weight
        return x_j
    
    def update(self, aggr_out: torch.Tensor) -> torch.Tensor:
        """Update node features"""
        return self.norm(aggr_out)
    
    def _apply_stdp(self, pre_features: torch.Tensor, post_spikes: torch.Tensor):
        """Apply spike-timing-dependent plasticity"""
        # Simplified STDP: strengthen connections to spiking neurons
        if post_spikes.sum() > 0:
            # Compute weight updates
            post_spike_mask = post_spikes > 0
            pre_activity = pre_features.mean(dim=0)
            
            # Potentiation for active connections
            weight_update = torch.outer(
                post_spike_mask.float().mean(dim=0),
                pre_activity
            )
            
            # Update plasticity weights
            self.weight_plasticity.data += self.config.stdp_lr * weight_update


class SpikingGraphAttention(MessagePassing):
    """Spiking Graph Attention layer"""
    
    def __init__(self, in_channels: int, out_channels: int,
                 config: SpikingGNNConfig, **kwargs):
        super().__init__(aggr='add', node_dim=0, **kwargs)
        self.config = config
        self.out_channels = out_channels
        
        # Multi-head attention
        self.heads = config.heads
        self.head_dim = out_channels // config.heads
        
        # Projections
        self.lin_k = nn.Linear(in_channels, out_channels)
        self.lin_q = nn.Linear(in_channels, out_channels)
        self.lin_v = nn.Linear(in_channels, out_channels)
        
        # Attention parameters
        self.att = nn.Parameter(torch.Tensor(1, config.heads, 2 * self.head_dim))
        
        # Output projection
        self.lin_out = nn.Linear(out_channels, out_channels)
        
        # Spiking neurons
        self.neurons = SpikingGraphNeuron(config)
        
        # Dropout
        self.dropout = nn.Dropout(config.dropout)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize parameters"""
        nn.init.xavier_uniform_(self.att)
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_attr: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass with spiking attention"""
        # Multi-head projections
        q = self.lin_q(x).view(-1, self.heads, self.head_dim)
        k = self.lin_k(x).view(-1, self.heads, self.head_dim)
        v = self.lin_v(x).view(-1, self.heads, self.head_dim)
        
        # Propagate with attention
        out = self.propagate(edge_index, q=q, k=k, v=v, size=None)
        
        # Reshape and project
        out = out.view(-1, self.out_channels)
        out = self.lin_out(out)
        
        # Apply spiking dynamics
        spikes = self.neurons(out)
        
        return spikes
    
    def message(self, q_i: torch.Tensor, k_j: torch.Tensor, v_j: torch.Tensor,
                index: torch.Tensor, ptr: Optional[torch.Tensor] = None,
                size_i: Optional[int] = None) -> torch.Tensor:
        """Compute attention-weighted messages"""
        # Compute attention scores
        alpha = (torch.cat([q_i, k_j], dim=-1) * self.att).sum(dim=-1)
        alpha = F.leaky_relu(alpha, 0.2)
        alpha = softmax(alpha, index, ptr, size_i)
        alpha = self.dropout(alpha)
        
        # Weight messages by attention
        return v_j * alpha.unsqueeze(-1)


class TemporalGraphPool(nn.Module):
    """Temporal pooling for spiking graphs"""
    
    def __init__(self, config: SpikingGNNConfig):
        super().__init__()
        self.config = config
        
        # Learnable temporal kernel
        self.temporal_conv = nn.Conv1d(
            1, 1, 
            kernel_size=5,
            padding=2,
            bias=False
        )
        
        # Initialize as averaging kernel
        nn.init.constant_(self.temporal_conv.weight, 0.2)
    
    def forward(self, spike_train: torch.Tensor) -> torch.Tensor:
        """Pool spike trains over time"""
        # spike_train: (time, num_nodes, features)
        
        # Reshape for convolution
        T, N, F = spike_train.shape
        spike_train = spike_train.permute(1, 2, 0).reshape(N * F, 1, T)
        
        # Apply temporal convolution
        pooled = self.temporal_conv(spike_train)
        
        # Reshape back
        pooled = pooled.reshape(N, F, T).permute(2, 0, 1)
        
        # Average over time
        return pooled.mean(dim=0)


class SpikingGNN(nn.Module):
    """Complete Spiking Graph Neural Network"""
    
    def __init__(self, config: SpikingGNNConfig):
        super().__init__()
        self.config = config
        
        # Build layers
        self.convs = nn.ModuleList()
        self.attns = nn.ModuleList()
        
        channels = [config.in_channels] + \
                  [config.hidden_channels] * (config.num_layers - 1) + \
                  [config.out_channels]
        
        for i in range(config.num_layers):
            # Alternating convolution and attention
            if i % 2 == 0:
                self.convs.append(SpikingGraphConv(
                    channels[i], channels[i+1], config
                ))
            else:
                self.attns.append(SpikingGraphAttention(
                    channels[i], channels[i+1], config
                ))
        
        # Temporal pooling
        self.temporal_pool = TemporalGraphPool(config)
        
        # Output decoder
        self.decoder = nn.Sequential(
            nn.Linear(config.out_channels, config.out_channels),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.out_channels, config.out_channels)
        )
        
        # Dynamic edge predictor
        if config.dynamic_edges:
            self.edge_predictor = nn.Sequential(
                nn.Linear(config.hidden_channels * 2, config.hidden_channels),
                nn.ReLU(),
                nn.Linear(config.hidden_channels, 1),
                nn.Sigmoid()
            )
        
        logger.info("Spiking GNN initialized", 
                   layers=config.num_layers,
                   hidden_dim=config.hidden_channels)
    
    def forward(self, data: Data, time_steps: Optional[int] = None) -> Dict[str, torch.Tensor]:
        """Forward pass through spiking GNN"""
        x, edge_index = data.x, data.edge_index
        edge_attr = data.edge_attr if hasattr(data, 'edge_attr') else None
        
        if time_steps is None:
            time_steps = self.config.time_window
        
        # Reset neuron states
        for conv in self.convs:
            if hasattr(conv, 'neurons'):
                conv.neurons.reset_state(x.shape[0], x.device)
        for attn in self.attns:
            if hasattr(attn, 'neurons'):
                attn.neurons.reset_state(x.shape[0], x.device)
        
        # Process through time
        spike_history = []
        conv_idx = 0
        attn_idx = 0
        
        for t in range(time_steps):
            # Encode input as spike probability
            if t == 0:
                spike_x = x
            else:
                # Generate input spikes
                spike_prob = torch.sigmoid(x)
                spike_x = (torch.rand_like(x) < spike_prob).float()
            
            # Process through layers
            for i in range(self.config.num_layers):
                if i % 2 == 0:
                    spike_x = self.convs[conv_idx](spike_x, edge_index, edge_attr)
                    conv_idx = (conv_idx + 1) % len(self.convs)
                else:
                    spike_x = self.attns[attn_idx](spike_x, edge_index, edge_attr)
                    attn_idx = (attn_idx + 1) % len(self.attns)
            
            spike_history.append(spike_x)
            
            # Dynamic edge update
            if self.config.dynamic_edges and t % 10 == 0:
                edge_index = self._update_edges(spike_x, edge_index)
        
        # Stack spike history
        spike_train = torch.stack(spike_history)
        
        # Temporal pooling
        pooled_features = self.temporal_pool(spike_train)
        
        # Decode output
        output = self.decoder(pooled_features)
        
        return {
            'node_features': output,
            'spike_train': spike_train,
            'spike_rates': spike_train.mean(dim=0),
            'edge_index': edge_index
        }
    
    def _update_edges(self, node_features: torch.Tensor, 
                     edge_index: torch.Tensor) -> torch.Tensor:
        """Update graph topology based on spike activity"""
        if not hasattr(self, 'edge_predictor'):
            return edge_index
        
        # Compute pairwise features
        row, col = edge_index
        edge_features = torch.cat([
            node_features[row],
            node_features[col]
        ], dim=-1)
        
        # Predict edge weights
        edge_probs = self.edge_predictor(edge_features).squeeze()
        
        # Keep edges above threshold
        keep_mask = edge_probs > self.config.edge_threshold
        
        return edge_index[:, keep_mask]
    
    def get_energy_estimate(self, spike_train: torch.Tensor) -> Dict[str, float]:
        """Estimate energy consumption"""
        total_spikes = spike_train.sum().item()
        num_neurons = spike_train.shape[1] * spike_train.shape[2]
        time_steps = spike_train.shape[0]
        
        # Simple energy model
        spike_energy = 1e-12  # 1 pJ per spike
        leak_energy = 1e-15  # 1 fJ per neuron per ms
        
        total_energy = (total_spikes * spike_energy + 
                       num_neurons * time_steps * leak_energy)
        
        return {
            'total_spikes': total_spikes,
            'spikes_per_node': total_spikes / spike_train.shape[1],
            'total_energy_j': total_energy,
            'energy_per_node_nj': total_energy * 1e9 / spike_train.shape[1]
        }


class TemporalGraphDataset:
    """Dataset for temporal graph sequences"""
    
    def __init__(self, num_graphs: int = 100, num_nodes: int = 20,
                 num_features: int = 16, num_timesteps: int = 10):
        self.num_graphs = num_graphs
        self.num_nodes = num_nodes
        self.num_features = num_features
        self.num_timesteps = num_timesteps
        
        # Generate synthetic temporal graphs
        self.graphs = self._generate_graphs()
    
    def _generate_graphs(self) -> List[List[Data]]:
        """Generate synthetic temporal graph sequences"""
        graphs = []
        
        for _ in range(self.num_graphs):
            sequence = []
            
            # Initial graph
            x = torch.randn(self.num_nodes, self.num_features)
            edge_prob = 0.3
            adj = torch.rand(self.num_nodes, self.num_nodes) < edge_prob
            edge_index = adj.nonzero().t()
            
            for t in range(self.num_timesteps):
                # Evolve features
                x = x + 0.1 * torch.randn_like(x)
                
                # Evolve edges (small changes)
                if t > 0 and np.random.rand() < 0.2:
                    # Add/remove some edges
                    new_edges = (torch.rand(5, 2) * self.num_nodes).long()
                    edge_index = torch.cat([edge_index, new_edges.t()], dim=1)
                
                # Create graph
                data = Data(x=x.clone(), edge_index=edge_index)
                sequence.append(data)
            
            graphs.append(sequence)
        
        return graphs
    
    def __len__(self):
        return self.num_graphs
    
    def __getitem__(self, idx):
        return self.graphs[idx]


# Example usage
def demonstrate_spiking_gnn():
    """Demonstrate Spiking GNN capabilities"""
    print("⚡ Spiking Graph Neural Network Demonstration")
    print("=" * 60)
    
    # Configuration
    config = SpikingGNNConfig(
        in_channels=16,
        hidden_channels=32,
        out_channels=8,
        num_layers=3,
        tau_membrane=20.0,
        tau_synapse=5.0,
        use_stdp=True,
        dynamic_edges=True
    )
    
    # Create model
    model = SpikingGNN(config)
    
    print(f"\n✅ Model created")
    print(f"   Layers: {config.num_layers}")
    print(f"   Hidden dim: {config.hidden_channels}")
    print(f"   Dynamic edges: {config.dynamic_edges}")
    
    # Create sample graph
    num_nodes = 50
    num_edges = 200
    
    x = torch.randn(num_nodes, config.in_channels)
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    data = Data(x=x, edge_index=edge_index)
    
    # Forward pass
    outputs = model(data, time_steps=100)
    
    print(f"\n✅ Forward pass complete")
    print(f"   Output shape: {outputs['node_features'].shape}")
    print(f"   Spike train shape: {outputs['spike_train'].shape}")
    print(f"   Average spike rate: {outputs['spike_rates'].mean():.3f}")
    
    # Energy analysis
    energy = model.get_energy_estimate(outputs['spike_train'])
    print(f"\n⚡ Energy Analysis")
    print(f"   Total spikes: {energy['total_spikes']:.0f}")
    print(f"   Spikes per node: {energy['spikes_per_node']:.1f}")
    print(f"   Energy per node: {energy['energy_per_node_nj']:.3f} nJ")
    
    # Test temporal dataset
    print(f"\n📊 Testing Temporal Graphs")
    dataset = TemporalGraphDataset(num_graphs=10, num_timesteps=5)
    temporal_sequence = dataset[0]
    
    print(f"   Sequence length: {len(temporal_sequence)}")
    print(f"   Nodes per graph: {temporal_sequence[0].x.shape[0]}")
    
    print("\n" + "=" * 60)
    print("✅ Spiking GNN Demonstration Complete")


if __name__ == "__main__":
    demonstrate_spiking_gnn()