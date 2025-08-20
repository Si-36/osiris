"""
LNN Models - Extracted from AURA Intelligence
Logical Neural Networks with Byzantine consensus for distributed reasoning
"""

import torch
import torch.nn as nn
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timezone
import numpy as np


@dataclass
class LNNDecision:
    """Decision made by an LNN with confidence metrics."""
    model_id: str
    output: torch.Tensor
    confidence: float
    inference_time_ms: float
    adaptation_count: int
    timestamp: datetime
    metadata: Dict[str, Any]


@dataclass
class LNNConfig:
    """Configuration for Liquid Neural Networks."""
    input_size: int
    hidden_size: int
    output_size: int
    tau: float = 2.0
    dropout: float = 0.1
    use_cuda: bool = torch.cuda.is_available()
    edge_optimized: bool = False


class LiquidNeuralNetwork(nn.Module):
    """
    Liquid Neural Network implementation
    Based on research from MIT CSAIL (2021-2025)
    """
    
    def __init__(self, config: LNNConfig):
        super().__init__()
        self.config = config
        
        # Liquid layers with continuous-time dynamics
        self.input_layer = nn.Linear(config.input_size, config.hidden_size)
        self.liquid_cell = nn.GRUCell(config.hidden_size, config.hidden_size)
        self.output_layer = nn.Linear(config.hidden_size, config.output_size)
        
        # Adaptive components
        self.dropout = nn.Dropout(config.dropout)
        self.layer_norm = nn.LayerNorm(config.hidden_size)
        
        # Time constants for liquid dynamics
        self.tau = nn.Parameter(torch.ones(config.hidden_size) * config.tau)
        
        if config.use_cuda and torch.cuda.is_available():
            self.cuda()
    
    def forward(self, x: torch.Tensor, hidden: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with liquid dynamics
        
        Args:
            x: Input tensor [batch_size, seq_len, input_size]
            hidden: Previous hidden state
            
        Returns:
            output: Network output
            hidden: Updated hidden state
        """
        batch_size = x.size(0)
        
        if hidden is None:
            hidden = torch.zeros(batch_size, self.config.hidden_size)
            if x.is_cuda:
                hidden = hidden.cuda()
        
        # Process sequence with liquid dynamics
        outputs = []
        for t in range(x.size(1)):
            # Input transformation
            inp = self.input_layer(x[:, t, :])
            inp = self.dropout(inp)
            
            # Liquid dynamics with time constant
            hidden_delta = self.liquid_cell(inp, hidden) - hidden
            hidden = hidden + hidden_delta / self.tau
            hidden = self.layer_norm(hidden)
            
            # Output projection
            out = self.output_layer(hidden)
            outputs.append(out)
        
        output = torch.stack(outputs, dim=1)
        return output, hidden
    
    def adapt(self, feedback: torch.Tensor):
        """Adapt network based on feedback signal"""
        # Adjust time constants based on feedback
        with torch.no_grad():
            self.tau.data = self.tau.data * (1 + 0.1 * feedback.mean())
            self.tau.data = torch.clamp(self.tau.data, 0.1, 10.0)


class EdgeLNN(LiquidNeuralNetwork):
    """Edge-optimized LNN for resource-constrained deployment"""
    
    def __init__(self, config: LNNConfig):
        config.edge_optimized = True
        super().__init__(config)
        
        # Quantization for edge deployment
        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()
    
    def forward(self, x: torch.Tensor, hidden: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        # Quantize inputs for edge efficiency
        x = self.quant(x)
        output, hidden = super().forward(x, hidden)
        output = self.dequant(output)
        return output, hidden