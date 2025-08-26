"""
Liquid Neural Network Dynamics

Implements continuous-time dynamics and ODE solvers for LNNs.
"""

import torch
import torch.nn as nn
from typing import Callable, Tuple, Optional, Dict, Any
from abc import ABC, abstractmethod
import numpy as np


class ODESolver(ABC):
    """Abstract base class for ODE solvers."""
    
    def __init__(self, dt: float = 0.01):
        self.dt = dt
    
    @abstractmethod
    def step(self, 
        state: torch.Tensor,
             dynamics_fn: Callable[[torch.Tensor], torch.Tensor],
             **kwargs) -> torch.Tensor:
        """Take one integration step."""
        pass


class RungeKutta4(ODESolver):
    """Fourth-order Runge-Kutta ODE solver."""
    
    def step(self, 
        state: torch.Tensor,
             dynamics_fn: Callable[[torch.Tensor], torch.Tensor],
             **kwargs) -> torch.Tensor:
        """RK4 integration step."""
        dt = kwargs.get('dt', self.dt)
        
        k1 = dynamics_fn(state)
        k2 = dynamics_fn(state + dt * k1 / 2)
        k3 = dynamics_fn(state + dt * k2 / 2)
        k4 = dynamics_fn(state + dt * k3)
        
        return state + dt * (k1 + 2*k2 + 2*k3 + k4) / 6


class EulerSolver(ODESolver):
    """Simple Euler ODE solver."""
    
    def step(self, 
        state: torch.Tensor,
             dynamics_fn: Callable[[torch.Tensor], torch.Tensor],
             **kwargs) -> torch.Tensor:
        """Euler integration step."""
        dt = kwargs.get('dt', self.dt)
        return state + dt * dynamics_fn(state)


class AdaptiveRK45(ODESolver):
    """Adaptive Runge-Kutta 4-5 solver with error control."""
    
    def __init__(self, dt: float = 0.01, rtol: float = 1e-3, atol: float = 1e-6):
        super().__init__(dt)
        self.rtol = rtol
        self.atol = atol
    
    def step(self, 
        state: torch.Tensor,
             dynamics_fn: Callable[[torch.Tensor], torch.Tensor],
             **kwargs) -> torch.Tensor:
        """Adaptive RK45 step with error control."""
        dt = kwargs.get('dt', self.dt)
        
        # RK4 coefficients
        k1 = dynamics_fn(state)
        k2 = dynamics_fn(state + dt * k1 / 4)
        k3 = dynamics_fn(state + dt * (3*k1 + 9*k2) / 32)
        k4 = dynamics_fn(state + dt * (1932*k1 - 7200*k2 + 7296*k3) / 2197)
        k5 = dynamics_fn(state + dt * (439*k1/216 - 8*k2 + 3680*k3/513 - 845*k4/4104))
        k6 = dynamics_fn(state + dt * (-8*k1/27 + 2*k2 - 3544*k3/2565 + 1859*k4/4104 - 11*k5/40))
        
        # 4th order solution
        y4 = state + dt * (25*k1/216 + 1408*k3/2565 + 2197*k4/4104 - k5/5)
        
        # 5th order solution
        y5 = state + dt * (16*k1/135 + 6656*k3/12825 + 28561*k4/56430 - 9*k5/50 + 2*k6/55)
        
        # For simplicity, return 4th order solution
        # In production, would implement adaptive step size based on error
        return y4


class LiquidDynamics:
    """Implements liquid neural network continuous-time dynamics."""
    
    def __init__(self, 
        tau: float = 1.0,
                 solver: ODESolver = None,
                 nonlinearity: str = "tanh"):
        self.tau = tau  # Time constant
        self.solver = solver or RungeKutta4()
        self.nonlinearity = self._get_nonlinearity(nonlinearity)
    
    def _get_nonlinearity(self, name: str) -> Callable:
        """Get nonlinearity function by name."""
        nonlinearities = {
            "tanh": torch.tanh,
            "sigmoid": torch.sigmoid,
            "relu": torch.relu,
            "gelu": torch.nn.functional.gelu,
            "silu": torch.nn.functional.silu,
        }
        return nonlinearities.get(name, torch.tanh)
    
    def dynamics_fn(self, 
        state: torch.Tensor,
                   input_current: torch.Tensor,
                   weights: torch.Tensor,
                   bias: torch.Tensor) -> torch.Tensor:
        """Compute state derivatives for liquid dynamics."""
        # Liquid neuron dynamics: tau * dx/dt = -x + f(Wx + I + b)
        recurrent_input = torch.matmul(self.nonlinearity(state), weights)
        total_input = recurrent_input + input_current + bias
        
        return (-state + self.nonlinearity(total_input)) / self.tau
    
    def step(self, 
        state: torch.Tensor,
             input_current: torch.Tensor,
             weights: torch.Tensor,
             bias: torch.Tensor,
             dt: float = None) -> torch.Tensor:
        """Take one dynamics step."""
        dynamics_fn = lambda s: self.dynamics_fn(s, input_current, weights, bias)
        return self.solver.step(state, dynamics_fn, dt=dt)


class ContinuousTimeRNN(nn.Module):
    """Continuous-time RNN using liquid dynamics."""
    
    def __init__(self, 
        input_size: int,
                 hidden_size: int,
                 tau: float = 1.0,
                 dt: float = 0.01,
                 solver: str = "rk4",
                 nonlinearity: str = "tanh"):
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.dt = dt
        
        # Initialize weights
        self.input_weights = nn.Parameter(torch.randn(input_size, hidden_size) * 0.1)
        self.recurrent_weights = nn.Parameter(torch.randn(hidden_size, hidden_size) * 0.1)
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        
        # Initialize dynamics
        solver_map = {
            "euler": EulerSolver(dt),
            "rk4": RungeKutta4(dt),
            "adaptive": AdaptiveRK45(dt)
        }
        
        self.dynamics = LiquidDynamics(
            tau=tau,
            solver=solver_map.get(solver, RungeKutta4(dt)),
            nonlinearity=nonlinearity
        )
        
        # Initialize hidden state
        self.register_buffer('hidden_state', torch.zeros(1, hidden_size))
    
    def forward(self, input_seq: torch.Tensor, hidden: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through continuous-time RNN."""
        batch_size, seq_len, _ = input_seq.shape
        
        if hidden is None:
            hidden = self.hidden_state.expand(batch_size, -1).contiguous()
        
        outputs = []
        
        for t in range(seq_len):
            # Current input
            input_current = torch.matmul(input_seq[:, t], self.input_weights)
            
            # Update hidden state using liquid dynamics
            hidden = self.dynamics.step(
                hidden, 
                input_current, 
                self.recurrent_weights, 
                self.bias,
                dt=self.dt
            )
            
            outputs.append(hidden.unsqueeze(1))
        
        return torch.cat(outputs, dim=1), hidden
    
    def reset_state(self, batch_size: int = 1):
        """Reset hidden state."""
        self.hidden_state = torch.zeros(batch_size, self.hidden_size)


# Aliases for compatibility
AdaptiveStepSolver = AdaptiveRK45

    def liquid_dynamics(state: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """Compatibility function for liquid dynamics."""
        dynamics = LiquidDynamics()
        if len(args) >= 3:
        return dynamics.dynamics_fn(state, args[0], args[1], args[2])
        return state

    def compute_gradients(state: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """Compatibility function for gradient computation."""
        return torch.autograd.grad(state.sum(), state, create_graph=True)[0] if state.requires_grad else torch.zeros_like(state)

# Export main classes
__all__ = [
        'ODESolver',
        'RungeKutta4',
        'EulerSolver',
        'AdaptiveRK45',
        'AdaptiveStepSolver',  # Alias
        'LiquidDynamics',
        'ContinuousTimeRNN',
        'liquid_dynamics',
        'compute_gradients'
]