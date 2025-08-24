"""
🧠 REAL Liquid Neural Networks - Based on MIT Research
=====================================================
Implementation of actual Liquid Neural Networks with:
- Continuous-time dynamics (ODEs)
- Neural Circuit Policies (NCPs) 
- Closed-form Continuous-time (CfC) models
- Liquid Time-Constant (LTC) networks

References:
- Hasani et al. 2021: "Liquid Time-constant Networks"
- Lechner et al. 2020: "Neural Circuit Policies"
- Hasani et al. 2022: "Closed-form Continuous-time Neural Networks"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, Tuple, Optional, List, Union
from dataclasses import dataclass
import logging
from enum import Enum
import math

# Try to import NCP library if available
try:
    from ncps.torch import CfC, LTC
    from ncps.wirings import AutoNCP, FullyConnected, Random, NCP
    NCPS_AVAILABLE = True
except ImportError:
    NCPS_AVAILABLE = False
    logging.warning("NCPs library not available. Install with: pip install ncps")


class WiringType(Enum):
    """Types of neural wiring architectures"""
    FULLY_CONNECTED = "fully_connected"
    AUTO_NCP = "auto_ncp"  # Automatic Neural Circuit Policy
    RANDOM = "random"
    SPARSE = "sparse"
    SMALL_WORLD = "small_world"


@dataclass
class LiquidConfig:
    """Configuration for Liquid Neural Networks"""
    input_size: int
    hidden_size: int
    output_size: int
    ode_unfolds: int = 6  # Number of ODE solver steps
    epsilon: float = 1e-8
    use_mixed: bool = True  # Mix continuous and discrete time
    wiring: WiringType = WiringType.AUTO_NCP
    sparsity: float = 0.8  # For sparse connections
    time_constant_min: float = 1.0
    time_constant_max: float = 100.0


class RealLiquidTimeConstant(nn.Module):
    """
    Real Liquid Time-Constant Network
    Based on Hasani et al. 2021: "Liquid Time-constant Networks"
    """
    
    def __init__(self, config: LiquidConfig):
        super().__init__()
        self.config = config
        
        # Time constants (learnable)
        self.tau = nn.Parameter(
            torch.FloatTensor(config.hidden_size).uniform_(
                config.time_constant_min, 
                config.time_constant_max
            )
        )
        
        # Input weights
        self.W_in = nn.Linear(config.input_size, config.hidden_size)
        
        # Recurrent weights (with wiring mask)
        self.W_rec = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.register_buffer('wiring_mask', self._create_wiring_mask())
        
        # Output weights
        self.W_out = nn.Linear(config.hidden_size, config.output_size)
        
        # Leak and input scaling
        self.alpha = nn.Parameter(torch.ones(config.hidden_size) * 0.9)
        self.beta = nn.Parameter(torch.ones(config.hidden_size))
        
        # Activation function parameters
        self.mu = nn.Parameter(torch.zeros(config.hidden_size))
        self.sigma = nn.Parameter(torch.ones(config.hidden_size))
        
    def _create_wiring_mask(self) -> torch.Tensor:
        """Create connectivity mask based on wiring type"""
        h = self.config.hidden_size
        
        if self.config.wiring == WiringType.FULLY_CONNECTED:
            return torch.ones(h, h)
        
        elif self.config.wiring == WiringType.SPARSE:
            # Random sparse connectivity
            mask = torch.rand(h, h) > self.config.sparsity
            # Ensure each neuron has at least one connection
            for i in range(h):
                if mask[i].sum() == 0:
                    mask[i, torch.randint(0, h, (1,))] = 1
                if mask[:, i].sum() == 0:
                    mask[torch.randint(0, h, (1,)), i] = 1
            return mask.float()
        
        elif self.config.wiring == WiringType.SMALL_WORLD:
            # Small-world network: local connections + random long-range
            mask = torch.zeros(h, h)
            k = 4  # Local connectivity
            p = 0.1  # Rewiring probability
            
            # Ring lattice
            for i in range(h):
                for j in range(1, k//2 + 1):
                    mask[i, (i+j) % h] = 1
                    mask[i, (i-j) % h] = 1
            
            # Random rewiring
            for i in range(h):
                for j in range(h):
                    if mask[i, j] == 1 and torch.rand(1) < p:
                        mask[i, j] = 0
                        new_j = torch.randint(0, h, (1,))
                        mask[i, new_j] = 1
            
            return mask
        
        else:
            return torch.ones(h, h)
    
    def ode_step(self, h: torch.Tensor, x: torch.Tensor, delta_t: float) -> torch.Tensor:
        """Single ODE integration step"""
        # Input contribution
        i_in = self.W_in(x)
        
        # Recurrent contribution with wiring mask
        w_rec = self.W_rec.weight * self.wiring_mask
        i_rec = F.linear(h, w_rec)
        
        # Nonlinear activation (sigmoid with learnable parameters)
        f = torch.sigmoid((i_rec - self.mu) / (self.sigma + self.config.epsilon))
        
        # ODE: dh/dt = -alpha * h + beta * f(W_rec * h + W_in * x)
        dhdt = -self.alpha * h + self.beta * f * (i_in + i_rec)
        
        # Euler integration with time constants
        h_new = h + delta_t * dhdt / self.tau
        
        return h_new
    
    def forward(self, x: torch.Tensor, h: Optional[torch.Tensor] = None, 
                timespans: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with continuous-time dynamics
        
        Args:
            x: Input tensor (batch, seq_len, input_size)
            h: Initial hidden state (batch, hidden_size)
            timespans: Time intervals between samples (batch, seq_len)
            
        Returns:
            output: (batch, seq_len, output_size)
            h_final: (batch, hidden_size)
        """
        batch_size, seq_len = x.shape[0], x.shape[1]
        
        if h is None:
            h = torch.zeros(batch_size, self.config.hidden_size, device=x.device)
        
        if timespans is None:
            timespans = torch.ones(batch_size, seq_len, device=x.device)
        
        outputs = []
        
        for t in range(seq_len):
            x_t = x[:, t]
            delta_t = timespans[:, t].unsqueeze(1)
            
            # Multiple ODE steps for better integration
            for _ in range(self.config.ode_unfolds):
                h = self.ode_step(h, x_t, delta_t / self.config.ode_unfolds)
            
            # Output
            y_t = self.W_out(h)
            outputs.append(y_t)
        
        output = torch.stack(outputs, dim=1)
        return output, h


class RealClosedFormContinuous(nn.Module):
    """
    Real Closed-form Continuous-time (CfC) Network
    Based on Hasani et al. 2022: "Closed-form Continuous-time Neural Networks"
    """
    
    def __init__(self, config: LiquidConfig):
        super().__init__()
        self.config = config
        
        # Backbone network
        self.backbone = nn.LSTM(
            config.input_size, 
            config.hidden_size, 
            batch_first=True
        )
        
        # Time-continuous head with closed-form solution
        self.tau_sys = nn.Parameter(torch.rand(config.hidden_size) * 10 + 1)
        self.A = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.B = nn.Linear(config.input_size, config.hidden_size)
        self.C = nn.Linear(config.hidden_size, config.output_size)
        
        # Nonlinearity parameters
        self.sigma = nn.Parameter(torch.ones(config.hidden_size) * 0.5)
        
    def cfc_layer(self, x: torch.Tensor, h: torch.Tensor, ts: torch.Tensor) -> torch.Tensor:
        """
        Closed-form continuous layer
        Solves: dh/dt = -h/tau + A*f(h) + B*x
        """
        # Normalized time constants
        tau = self.tau_sys.unsqueeze(0)
        
        # System matrix with time scaling
        A_eff = self.A.weight / tau
        
        # Nonlinearity
        f_h = torch.tanh(h / self.sigma)
        
        # Closed-form solution using matrix exponential approximation
        # For small timesteps: exp(A*t) ≈ I + A*t + (A*t)²/2
        I = torch.eye(self.config.hidden_size, device=x.device)
        At = A_eff * ts.unsqueeze(-1).unsqueeze(-1)
        exp_At = I + At + 0.5 * torch.bmm(At, At)
        
        # Evolution
        h_new = torch.bmm(exp_At, h.unsqueeze(-1)).squeeze(-1)
        h_new = h_new + ts.unsqueeze(-1) * (self.B(x) + self.A(f_h)) / tau
        
        return h_new
    
    def forward(self, x: torch.Tensor, timespans: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass with closed-form continuous dynamics"""
        batch_size, seq_len = x.shape[0], x.shape[1]
        
        if timespans is None:
            timespans = torch.ones(batch_size, seq_len, device=x.device)
        
        # Get backbone features
        backbone_out, (h_n, c_n) = self.backbone(x)
        
        # Apply CfC layer
        outputs = []
        h = torch.zeros(batch_size, self.config.hidden_size, device=x.device)
        
        for t in range(seq_len):
            h = self.cfc_layer(x[:, t], h, timespans[:, t])
            h = h + backbone_out[:, t]  # Residual connection
            y = self.C(h)
            outputs.append(y)
        
        return torch.stack(outputs, dim=1)


class RealNeuralCircuitPolicy(nn.Module):
    """
    Real Neural Circuit Policy (NCP)
    Based on Lechner et al. 2020: "Neural Circuit Policies Enabling Auditable Autonomy"
    """
    
    def __init__(self, config: LiquidConfig):
        super().__init__()
        self.config = config
        
        if NCPS_AVAILABLE and config.wiring == WiringType.AUTO_NCP:
            # Use official NCP implementation
            self.wiring = AutoNCP(config.hidden_size, config.output_size)
            self.rnn = LTC(config.input_size, self.wiring, batch_first=True)
        else:
            # Custom implementation
            self._build_custom_ncp()
    
    def _build_custom_ncp(self):
        """Build custom NCP when library not available"""
        # Sensory neurons
        self.sensory = nn.Linear(self.config.input_size, self.config.hidden_size // 3)
        
        # Inter neurons (hidden layer)
        self.inter = nn.Linear(self.config.hidden_size // 3, self.config.hidden_size // 3)
        
        # Command neurons (pre-motor)
        self.command = nn.Linear(self.config.hidden_size // 3, self.config.hidden_size // 3)
        
        # Motor neurons (output)
        self.motor = nn.Linear(self.config.hidden_size // 3, self.config.output_size)
        
        # Recurrent connections
        self.inter_recurrent = nn.Linear(self.config.hidden_size // 3, self.config.hidden_size // 3)
        self.command_recurrent = nn.Linear(self.config.hidden_size // 3, self.config.hidden_size // 3)
        
        # Time constants for each layer
        h_third = self.config.hidden_size // 3
        self.tau_sensory = nn.Parameter(torch.ones(h_third) * 5)
        self.tau_inter = nn.Parameter(torch.ones(h_third) * 10)
        self.tau_command = nn.Parameter(torch.ones(h_third) * 20)
    
    def forward(self, x: torch.Tensor, h: Optional[Dict[str, torch.Tensor]] = None) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Forward pass through neural circuit"""
        batch_size, seq_len = x.shape[0], x.shape[1]
        
        if NCPS_AVAILABLE and hasattr(self, 'rnn'):
            # Use library implementation
            if h is None:
                h = torch.zeros(batch_size, self.rnn.state_size, device=x.device)
            output, h_new = self.rnn(x, h)
            return output, {'hidden': h_new}
        
        # Custom implementation
        if h is None:
            h_third = self.config.hidden_size // 3
            h = {
                'sensory': torch.zeros(batch_size, h_third, device=x.device),
                'inter': torch.zeros(batch_size, h_third, device=x.device),
                'command': torch.zeros(batch_size, h_third, device=x.device)
            }
        
        outputs = []
        
        for t in range(seq_len):
            x_t = x[:, t]
            
            # Sensory layer
            s_input = self.sensory(x_t)
            h['sensory'] = (1 - 1/self.tau_sensory) * h['sensory'] + (1/self.tau_sensory) * torch.tanh(s_input)
            
            # Inter layer
            i_input = self.inter(h['sensory']) + self.inter_recurrent(h['inter'])
            h['inter'] = (1 - 1/self.tau_inter) * h['inter'] + (1/self.tau_inter) * torch.tanh(i_input)
            
            # Command layer
            c_input = self.command(h['inter']) + self.command_recurrent(h['command'])
            h['command'] = (1 - 1/self.tau_command) * h['command'] + (1/self.tau_command) * torch.tanh(c_input)
            
            # Motor output
            output = self.motor(h['command'])
            outputs.append(output)
        
        return torch.stack(outputs, dim=1), h


class LiquidEnsemble(nn.Module):
    """
    Ensemble of different Liquid Neural Network variants
    Combines LTC, CfC, and NCP for robust predictions
    """
    
    def __init__(self, config: LiquidConfig):
        super().__init__()
        self.config = config
        
        # Create ensemble members
        self.ltc = RealLiquidTimeConstant(config)
        self.cfc = RealClosedFormContinuous(config)
        self.ncp = RealNeuralCircuitPolicy(config)
        
        # Ensemble combination weights
        self.ensemble_weights = nn.Parameter(torch.ones(3) / 3)
        
        # Meta-learner for dynamic weighting
        self.meta_learner = nn.Sequential(
            nn.Linear(config.output_size * 3, 32),
            nn.ReLU(),
            nn.Linear(32, 3),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, x: torch.Tensor, timespans: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Forward pass through ensemble"""
        # Get predictions from each model
        ltc_out, _ = self.ltc(x, timespans=timespans)
        cfc_out = self.cfc(x, timespans=timespans)
        ncp_out, _ = self.ncp(x)
        
        # Stack predictions
        all_preds = torch.stack([ltc_out, cfc_out, ncp_out], dim=-1)
        
        # Dynamic weighting based on predictions
        concat_preds = torch.cat([ltc_out, cfc_out, ncp_out], dim=-1)
        dynamic_weights = self.meta_learner(concat_preds)
        
        # Combine with static and dynamic weights
        static_weight = F.softmax(self.ensemble_weights, dim=0)
        final_weights = 0.5 * static_weight + 0.5 * dynamic_weights.mean(dim=1).mean(dim=0)
        
        # Weighted combination
        ensemble_out = (all_preds * final_weights.view(1, 1, 1, -1)).sum(dim=-1)
        
        return {
            'prediction': ensemble_out,
            'ltc': ltc_out,
            'cfc': cfc_out,
            'ncp': ncp_out,
            'weights': final_weights,
            'uncertainty': all_preds.std(dim=-1)
        }


# Specialized variants for different use cases
class EdgeLiquidNN(RealLiquidTimeConstant):
    """Optimized for edge devices with quantization and pruning"""
    
    def __init__(self, config: LiquidConfig):
        super().__init__(config)
        self.quantized = False
    
    def quantize(self):
        """Quantize weights for edge deployment"""
        self.quantized = True
        # Implement quantization logic
        self.qconfig = torch.quantization.get_default_qconfig('fbgemm')
        torch.quantization.prepare(self, inplace=True)
    
    def prune_connections(self, sparsity: float = 0.9):
        """Prune weak connections"""
        with torch.no_grad():
            threshold = torch.quantile(torch.abs(self.W_rec.weight), sparsity)
            mask = torch.abs(self.W_rec.weight) > threshold
            self.wiring_mask *= mask.float()


class DistributedLiquidNN(nn.Module):
    """Distributed Liquid NN for large-scale systems"""
    
    def __init__(self, config: LiquidConfig, num_partitions: int = 4):
        super().__init__()
        self.config = config
        self.num_partitions = num_partitions
        
        # Create partitioned networks
        partition_size = config.hidden_size // num_partitions
        self.partitions = nn.ModuleList([
            RealLiquidTimeConstant(LiquidConfig(
                input_size=config.input_size,
                hidden_size=partition_size,
                output_size=partition_size,
                **{k: v for k, v in config.__dict__.items() 
                   if k not in ['input_size', 'hidden_size', 'output_size']}
            ))
            for _ in range(num_partitions)
        ])
        
        # Cross-partition communication
        self.cross_partition = nn.Linear(partition_size * num_partitions, config.output_size)
    
    def forward(self, x: torch.Tensor, timespans: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass with distributed computation"""
        partition_outputs = []
        
        for partition in self.partitions:
            out, _ = partition(x, timespans=timespans)
            partition_outputs.append(out)
        
        # Aggregate partition outputs
        combined = torch.cat(partition_outputs, dim=-1)
        return self.cross_partition(combined)


# Factory for creating variants
def create_liquid_nn(variant: str, config: LiquidConfig) -> nn.Module:
    """Factory to create different LNN variants"""
    variants = {
        'ltc': RealLiquidTimeConstant,
        'cfc': RealClosedFormContinuous,
        'ncp': RealNeuralCircuitPolicy,
        'ensemble': LiquidEnsemble,
        'edge': EdgeLiquidNN,
        'distributed': DistributedLiquidNN
    }
    
    if variant not in variants:
        raise ValueError(f"Unknown variant: {variant}. Available: {list(variants.keys())}")
    
    return variants[variant](config)


# Export all variants
__all__ = [
    'LiquidConfig',
    'RealLiquidTimeConstant',
    'RealClosedFormContinuous', 
    'RealNeuralCircuitPolicy',
    'LiquidEnsemble',
    'EdgeLiquidNN',
    'DistributedLiquidNN',
    'create_liquid_nn'
]