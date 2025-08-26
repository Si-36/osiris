"""
Core Liquid Neural Network components.

Implements the fundamental building blocks of LNNs including neurons,
layers, and full networks with continuous-time dynamics.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Tuple, List, Callable, Union
from dataclasses import dataclass, field
import numpy as np
from enum import Enum
import structlog

logger = structlog.get_logger()


class ActivationType(Enum):
    """Supported activation functions for liquid neurons."""
    SIGMOID = "sigmoid"
    TANH = "tanh"
    RELU = "relu"
    GELU = "gelu"
    SILU = "silu"
    LIQUID = "liquid"  # Custom liquid activation


@dataclass
class TimeConstants:
    """Time constants for liquid neuron dynamics."""
    tau_min: float = 0.1
    tau_max: float = 10.0
    tau_init: str = "uniform"  # uniform, log_uniform, learned
    adaptive: bool = True
    
    def initialize(self, size: int) -> torch.Tensor:
        """Initialize time constants."""
        if self.tau_init == "uniform":
            return torch.empty(size).uniform_(self.tau_min, self.tau_max)
        elif self.tau_init == "log_uniform":
            log_min = np.log(self.tau_min)
            log_max = np.log(self.tau_max)
            return torch.exp(torch.empty(size).uniform_(log_min, log_max))
        elif self.tau_init == "learned":
            return torch.ones(size) * (self.tau_min + self.tau_max) / 2
        else:
            raise ValueError(f"Unknown initialization: {self.tau_init}")


    @dataclass
class WiringConfig:
    """Configuration for sparse wiring in liquid networks."""
    sparsity: float = 0.8  # 80% sparse connections
    wiring_type: str = "random"  # random, small_world, scale_free
    self_connections: bool = True
    learnable_wiring: bool = True
    prune_threshold: float = 0.01
    
    def create_mask(self, in_features: int, out_features: int) -> torch.Tensor:
        """Create sparse connectivity mask."""
        if self.wiring_type == "random":
            mask = torch.rand(out_features, in_features) > self.sparsity
        elif self.wiring_type == "small_world":
        mask = self._small_world_mask(in_features, out_features)
        elif self.wiring_type == "scale_free":
        mask = self._scale_free_mask(in_features, out_features)
        else:
        raise ValueError(f"Unknown wiring type: {self.wiring_type}")
        
        # Ensure self-connections if specified
        if self.self_connections and in_features == out_features:
            mask.diagonal().fill_(1)
        
        return mask.float()
    
    def _small_world_mask(self, in_features: int, out_features: int) -> torch.Tensor:
        """Create small-world connectivity pattern."""
        mask = torch.zeros(out_features, in_features)
        
        # Local connections
        for i in range(out_features):
            for j in range(max(0, i-2), min(in_features, i+3)):
                if j < in_features:
                    mask[i, j] = 1
        
        # Random long-range connections
        n_random = int((1 - self.sparsity) * in_features * out_features * 0.1)
        for _ in range(n_random):
            i = torch.randint(0, out_features, (1,)).item()
            j = torch.randint(0, in_features, (1,)).item()
            mask[i, j] = 1
        
        return mask
    
    def _scale_free_mask(self, in_features: int, out_features: int) -> torch.Tensor:
        """Create scale-free connectivity pattern."""
        mask = torch.zeros(out_features, in_features)
        degrees = torch.ones(max(in_features, out_features))
        
        # Preferential attachment
        n_edges = int((1 - self.sparsity) * in_features * out_features)
        for _ in range(n_edges):
        # Sample based on degree
        probs = degrees / degrees.sum()
        i = torch.multinomial(probs[:out_features], 1).item()
        j = torch.multinomial(probs[:in_features], 1).item()
            
        mask[i, j] = 1
        degrees[i] += 1
        if j < len(degrees):
            degrees[j] += 1
        
        return mask


    @dataclass
class LiquidConfig:
    """Configuration for liquid neural networks."""
    # Neuron dynamics
    time_constants: TimeConstants = field(default_factory=TimeConstants)
    activation: ActivationType = ActivationType.TANH
    use_bias: bool = True
    
    # Network structure
    wiring: WiringConfig = field(default_factory=WiringConfig)
    hidden_sizes: List[int] = field(default_factory=lambda: [128, 64])
    
    # ODE solver
    solver_type: str = "rk4"  # rk4, euler, adaptive
    dt: float = 0.01
    adaptive_dt: bool = True
    
    # Training
    liquid_reg: float = 0.01  # Regularization for liquid dynamics
    sparsity_reg: float = 0.001  # Sparsity regularization
    stability_reg: float = 0.001  # Stability regularization
    
    # Performance
    use_cuda: bool = True
    mixed_precision: bool = True
    compile_mode: Optional[str] = "reduce-overhead"  # torch.compile mode


class LiquidActivation(nn.Module):
    """Custom liquid activation function."""
    
    def __init__(self, beta: float = 1.0):
        super().__init__()
        self.beta = nn.Parameter(torch.tensor(beta))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Liquid activation: adaptive between sigmoid and tanh."""
        return torch.sigmoid(self.beta * x) * torch.tanh(x)


class LiquidNeuron(nn.Module):
    """
    Single liquid neuron with continuous-time dynamics.
    
    Implements: dx/dt = -x/tau + W @ f(x) + b + I(t)
    """
    
    def __init__(
        self,
        input_size: int,
        output_size: int,
        config: LiquidConfig
    ):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.config = config
        
        # Time constants
        self.tau = nn.Parameter(
            config.time_constants.initialize(output_size)
        )
        
        # Weights with sparse connectivity
        self.weight_mask = config.wiring.create_mask(input_size, output_size)
        self.register_buffer("mask", self.weight_mask)
        
        self.weight = nn.Parameter(
            torch.randn(output_size, input_size) * np.sqrt(2.0 / input_size)
        )
        
        # Bias
        if config.use_bias:
            self.bias = nn.Parameter(torch.zeros(output_size))
        else:
            self.register_parameter("bias", None)
        
        # Activation
        if config.activation == ActivationType.LIQUID:
            self.activation = LiquidActivation()
        else:
            self.activation = self._get_activation(config.activation)
        
        # State
        self.register_buffer("state", torch.zeros(output_size))
        
    def _get_activation(self, activation_type: ActivationType) -> Callable:
        """Get activation function."""
        activations = {
        ActivationType.SIGMOID: torch.sigmoid,
        ActivationType.TANH: torch.tanh,
        ActivationType.RELU: F.relu,
        ActivationType.GELU: F.gelu,
        ActivationType.SILU: F.silu
        }
        return activations[activation_type]
    
        def forward(
        self,
        input_current: torch.Tensor,
        state: Optional[torch.Tensor] = None,
        dt: Optional[float] = None
        ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass computing liquid dynamics.
        
        Args:
            input_current: Input at current time (batch_size, input_size)
            state: Current neuron state (batch_size, output_size)
            dt: Time step (uses config.dt if not provided)
            
        Returns:
            new_state: Updated neuron state
            output: Neuron output after activation
        """
        if state is None:
            state = self.state.unsqueeze(0).expand(input_current.shape[0], -1)
        
        if dt is None:
            dt = self.config.dt
        
        # Apply sparse connectivity
        masked_weight = self.weight * self.mask
        
        # Compute dynamics: dx/dt = -x/tau + W @ f(x) + b + I(t)
        decay = -state / self.tau
        recurrent = F.linear(self.activation(state), masked_weight.t())
        
        if self.bias is not None:
            # Ensure bias dimensions match recurrent tensor
            if self.bias.shape[0] == recurrent.shape[-1]:
                recurrent = recurrent + self.bias
            else:
                # Reshape bias to match recurrent dimensions
                bias_expanded = self.bias[:recurrent.shape[-1]] if self.bias.shape[0] > recurrent.shape[-1] else self.bias
                recurrent = recurrent + bias_expanded
        
        # Add input current - ensure proper tensor dimensions
        if input_current.shape[-1] == self.output_size:
            # Direct input
            dynamics = decay + recurrent + input_current
        else:
            # Project input to match output size
            if not hasattr(self, 'input_projection'):
                self.input_projection = nn.Linear(
                    input_current.shape[-1], 
                    self.output_size,
                    bias=False
                ).to(input_current.device)
            
            # Ensure input projection parameters are on same device and dtype as input
            if self.input_projection.weight.device != input_current.device:
                self.input_projection = self.input_projection.to(input_current.device, input_current.dtype)
                
            projected_input = self.input_projection(input_current)
            
            # Ensure all tensors have matching dimensions before addition
            if decay.shape != recurrent.shape:
                min_size = min(decay.shape[-1], recurrent.shape[-1])
                decay = decay[..., :min_size]
                recurrent = recurrent[..., :min_size]
            
            if projected_input.shape != decay.shape:
                min_size = min(projected_input.shape[-1], decay.shape[-1])
                projected_input = projected_input[..., :min_size]
                decay = decay[..., :min_size]
                recurrent = recurrent[..., :min_size]
                
            dynamics = decay + recurrent + projected_input
        
        # Update state using ODE solver
        if self.config.solver_type == "euler":
            new_state = state + dt * dynamics
        elif self.config.solver_type == "rk4":
            new_state = self._rk4_step(state, input_current, dt)
        else:
            raise ValueError(f"Unknown solver: {self.config.solver_type}")
        
        # Output is activated state
        output = self.activation(new_state)
        
        return new_state, output
    
    def _rk4_step(
        self,
        state: torch.Tensor,
        input_current: torch.Tensor,
        dt: float
        ) -> torch.Tensor:
        """Runge-Kutta 4th order integration step."""
    def dynamics(s, i):
            decay = -s / self.tau
            masked_weight = self.weight * self.mask
            recurrent = F.linear(self.activation(s), masked_weight.t())
            if self.bias is not None:
                # Ensure bias dimensions match recurrent tensor
            if self.bias.shape[0] == recurrent.shape[-1]:
                recurrent = recurrent + self.bias
            else:
            # Reshape bias to match recurrent dimensions
            bias_expanded = self.bias[:recurrent.shape[-1]] if self.bias.shape[0] > recurrent.shape[-1] else self.bias
            recurrent = recurrent + bias_expanded
            # Ensure dimension compatibility
            if i.shape[-1] != decay.shape[-1]:
                if not hasattr(self, 'input_projection'):
                    self.input_projection = nn.Linear(
            i.shape[-1],
            decay.shape[-1],
            bias=False
            ).to(i.device)
            i = self.input_projection(i)
            return decay + recurrent + i
        
            k1 = dynamics(state, input_current)
            k2 = dynamics(state + 0.5 * dt * k1, input_current)
            k3 = dynamics(state + 0.5 * dt * k2, input_current)
            k4 = dynamics(state + dt * k3, input_current)
        
            return state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
    
    def reset_state(self, batch_size: Optional[int] = None):
                """Reset neuron state."""
            if batch_size is None:
                self.state.zero_()
            else:
            return torch.zeros(batch_size, self.output_size, device=self.state.device)


class LiquidLayer(nn.Module):
    """Layer of liquid neurons with shared dynamics."""
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        config: LiquidConfig,
        return_sequences: bool = False
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.config = config
        self.return_sequences = return_sequences
        
        # Liquid neurons
        self.neurons = LiquidNeuron(input_size, hidden_size, config)
        
        # Layer normalization for stability
        self.layer_norm = nn.LayerNorm(hidden_size)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.1)
        
    def forward(
        self,
        inputs: torch.Tensor,
        initial_state: Optional[torch.Tensor] = None,
        return_states: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Process sequence through liquid layer.
        
        Args:
            inputs: Input tensor (batch, seq_len, input_size) or (batch, input_size)
            initial_state: Initial hidden state
            return_states: Whether to return all intermediate states
            
        Returns:
            output: Layer output
            states: Hidden states (if return_states=True)
        """
        if inputs.dim() == 2:
            # Single time step
            state, output = self.neurons(inputs, initial_state)
            output = self.layer_norm(output)
            output = self.dropout(output)
            
            if return_states:
                return output, state
            return output
        
        # Sequential input
        batch_size, seq_len, _ = inputs.shape
        
        if initial_state is None:
            state = self.neurons.reset_state(batch_size)
        else:
            state = initial_state
        
        outputs = []
        states = []
        
        for t in range(seq_len):
            state, output = self.neurons(inputs[:, t], state)
            output = self.layer_norm(output)
            output = self.dropout(output)
            
            outputs.append(output)
            if return_states:
                states.append(state)
        
        outputs = torch.stack(outputs, dim=1)
        
        if not self.return_sequences:
            outputs = outputs[:, -1]  # Return only last output
        
        if return_states:
            states = torch.stack(states, dim=1)
            return outputs, states
        
        return outputs


class LiquidNeuralNetwork(nn.Module):
    """
    Complete Liquid Neural Network with multiple layers.
    
    Supports various architectures and training modes.
    """
    
    def __init__(
        self,
        input_size: int,
        output_size: int,
        config: Optional[LiquidConfig] = None
    ):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.config = config or LiquidConfig()
        
        # Build layers
        self.layers = nn.ModuleList()
        
        # Input projection
        self.input_proj = nn.Linear(input_size, self.config.hidden_sizes[0])
        
        # Liquid layers - properly handle layer connections
        prev_size = self.config.hidden_sizes[0]
        
        # First liquid layer processes the projected input
        first_layer = LiquidLayer(
            prev_size,  # Input from input_proj
            prev_size,  # Keep same size for first layer
            self.config,
            return_sequences=True
        )
        self.layers.append(first_layer)
        
        # Additional layers if specified
        for i in range(1, len(self.config.hidden_sizes)):
            hidden_size = self.config.hidden_sizes[i]
            layer = LiquidLayer(
                prev_size,
                hidden_size,
                self.config,
                return_sequences=True
            )
            self.layers.append(layer)
            prev_size = hidden_size
        
        # Output projection
        self.output_proj = nn.Linear(prev_size, output_size)
        
        # Initialize weights
        self._initialize_weights()
        
        # Compile if specified
        if self.config.compile_mode and hasattr(torch, 'compile'):
            self = torch.compile(self, mode=self.config.compile_mode)
        
        logger.info(
            "Created LiquidNeuralNetwork",
            input_size=input_size,
            output_size=output_size,
            hidden_sizes=self.config.hidden_sizes,
            total_params=sum(p.numel() for p in self.parameters())
        )
    
    def _initialize_weights(self):
        """Initialize network weights."""
        pass
        for module in self.modules():
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    
        def forward(
        self,
        inputs: torch.Tensor,
        return_dynamics: bool = False
        ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        """
        Forward pass through liquid network.
        
        Args:
            inputs: Input tensor (batch, seq_len, input_size) or (batch, input_size)
            return_dynamics: Whether to return intermediate dynamics
            
        Returns:
            output: Network output
            dynamics: Dictionary of intermediate values (if requested)
        """
        # Handle both 2D and 3D inputs
        if inputs.dim() == 2:
            # Add sequence dimension: (batch, input_size) -> (batch, 1, input_size)
            inputs = inputs.unsqueeze(1)
        
        batch_size = inputs.shape[0]
        
        # Project input
        x = self.input_proj(inputs)
        
        dynamics = {"layers": []} if return_dynamics else None
        
        # Process through liquid layers
        for i, layer in enumerate(self.layers):
            if return_dynamics:
                x, states = layer(x, return_states=True)
                dynamics["layers"].append({
                    "output": x,
                    "states": states,
                    "tau": layer.neurons.tau.detach()
                })
            else:
                x = layer(x)
        
        # Output projection
        output = self.output_proj(x)
        
        if return_dynamics:
            dynamics["output"] = output
            return output, dynamics
        
        return output
    
    def get_sparsity(self) -> Dict[str, float]:
        """Get sparsity statistics of the network."""
        pass
        sparsity_stats = {}
        
        for i, layer in enumerate(self.layers):
        mask = layer.neurons.mask
        sparsity = 1.0 - (mask.sum() / mask.numel()).item()
        sparsity_stats[f"layer_{i}"] = sparsity
        
        sparsity_stats["average"] = np.mean(list(sparsity_stats.values()))
        
        return sparsity_stats
    
    def prune_connections(self, threshold: float = 0.01):
            """Prune weak connections below threshold."""
        pruned_count = 0
        
        for layer in self.layers:
            neurons = layer.neurons
            weights = neurons.weight.data.abs()
            
            # Find connections below threshold
            weak_connections = weights < threshold
            
            # Update mask
            neurons.mask[weak_connections] = 0
            
            # Zero out weights
            neurons.weight.data[weak_connections] = 0
            
            pruned_count += weak_connections.sum().item()
        
        logger.info(f"Pruned {pruned_count} connections")
        
        return pruned_count
    
    def analyze_dynamics(self, inputs: torch.Tensor) -> Dict[str, Any]:
        """Analyze network dynamics on given inputs."""
        with torch.no_grad():
            _, dynamics = self.forward(inputs, return_dynamics=True)
        
        analysis = {
        "layer_dynamics": [],
        "stability_metrics": {},
        "information_flow": {}
        }
        
        for i, layer_data in enumerate(dynamics["layers"]):
        states = layer_data["states"]
        tau = layer_data["tau"]
            
        # Compute statistics
        layer_analysis = {
        "mean_activation": states.mean().item(),
        "std_activation": states.std().item(),
        "mean_tau": tau.mean().item(),
        "tau_range": (tau.min().item(), tau.max().item()),
        "state_norm": states.norm(dim=-1).mean().item()
        }
            
        # Check for stability
        if states.max().item() > 100 or torch.isnan(states).any():
            layer_analysis["stability"] = "unstable"
        else:
        layer_analysis["stability"] = "stable"
            
        analysis["layer_dynamics"].append(layer_analysis)
        
        # Overall stability
        all_stable = all(
        l["stability"] == "stable"
        for l in analysis["layer_dynamics"]
        )
        analysis["stability_metrics"]["overall"] = "stable" if all_stable else "unstable"
        
        return analysis
    
    def reset_states(self):
            """Reset all neuron states in the network."""
        pass
        for layer in self.layers:
            layer.neurons.reset_state()
    
    def to_onnx(self, dummy_input: torch.Tensor, path: str):
        """Export network to ONNX format."""
        torch.onnx.export(
        self,
        dummy_input,
        path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
        'input': {0: 'batch_size', 1: 'sequence_length'},
        'output': {0: 'batch_size', 1: 'sequence_length'}
        }
        )
        logger.info(f"Exported model to {path}")
    # ============================================================================
    # MAIN LNN CORE CLASS (Compatibility alias)
    # ============================================================================

class LNNCore(LiquidNeuralNetwork):
    """
    Main LNN Core class - alias for LiquidNeuralNetwork for compatibility.
    
    This is the main entry point for Liquid Neural Networks in AURA Intelligence.
    """
    
    def __init__(self, 
        input_size: int = 10,
                 output_size: int = 10, 
                 config: Optional[LiquidConfig] = None):
        """Initialize LNN Core with default configuration."""
        pass
        if config is None:
            config = LiquidConfig()
        
        super().__init__(input_size, output_size, config)
        logger.info(f"🧠 LNN Core initialized with {len(self.config.hidden_sizes)} layers")
    
    async def process(self, data: torch.Tensor) -> torch.Tensor:
        """Async processing method for compatibility."""
        return self.forward(data)
    
    def get_info(self) -> Dict[str, Any]:
        """Get LNN Core information."""
        pass
        return {
            "type": "LNNCore",
            "layers": len(self.config.hidden_sizes),
            "neurons_per_layer": self.config.hidden_sizes[0] if self.config.hidden_sizes else 128,
            "total_parameters": sum(p.numel() for p in self.parameters()),
            "device": str(next(self.parameters()).device),
            "dtype": str(next(self.parameters()).dtype)
        }


# Export the main class
__all__ = [
        'LNNCore',
        'LiquidNeuralNetwork',
        'LiquidLayer',
        'LiquidNeuron',
        'LiquidConfig',
        'TimeConstants',
        'WiringConfig',
        'ActivationType'
]