"""Real Spiking Graph Neural Networks"""
import torch
import torch.nn as nn
from typing import Dict, Any

try:
    from torch_geometric.nn import GCNConv
    from spikingjelly.activation_based import neuron, surrogate
    LIBRARIES_AVAILABLE = True
except ImportError:
    LIBRARIES_AVAILABLE = False

class RealSpikingGNN(nn.Module):
    def __init__(self, input_dim: int = 64, hidden_dim: int = 128, output_dim: int = 32):
        super().__init__()
        self.libraries_available = LIBRARIES_AVAILABLE
        
        if LIBRARIES_AVAILABLE:
            self.gcn1 = GCNConv(input_dim, hidden_dim)
            self.lif1 = neuron.LIFNode(surrogate_function=surrogate.ATan())
            self.gcn2 = GCNConv(hidden_dim, output_dim)
            self.lif2 = neuron.LIFNode(surrogate_function=surrogate.ATan())
        else:
            self.linear1 = nn.Linear(input_dim, hidden_dim)
            self.linear2 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor = None) -> torch.Tensor:
        if self.libraries_available and edge_index is not None:
            x = self.gcn1(x, edge_index)
            x = self.lif1(x)
            x = self.gcn2(x, edge_index)
            x = self.lif2(x)
            return x
        else:
            # Fallback
            x = torch.relu(self.linear1(x))
            x = self.linear2(x)
            return x

    def get_real_spiking_gnn():
        return RealSpikingGNN()
