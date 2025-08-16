"""
LNN Council Configuration (2025 Standards)

Type-safe configuration with validation and defaults.
"""

from dataclasses import dataclass, field
from typing import List, Optional
from enum import Enum

from ...lnn.core import ActivationType, LiquidConfig, TimeConstants, WiringConfig
from ..base import AgentConfig


class SolverType(Enum):
    """ODE solver types for LNN."""
    EULER = "euler"
    RK4 = "rk4"
    ADAPTIVE = "adaptive"


@dataclass
class LNNCouncilConfig:
    """
    LNN Council Agent Configuration.
    
    2025 Standards:
    - Immutable dataclass
    - Type hints everywhere
    - Validation methods
    - Sensible defaults
    """
    
    # Core settings
    name: str = "lnn_council_agent"
    version: str = "1.0.0"
    
    # Neural network
    input_size: int = 128  # Reduced for efficiency
    output_size: int = 32
    hidden_sizes: List[int] = field(default_factory=lambda: [64, 32])
    
    # LNN specifics
    activation_type: ActivationType = ActivationType.LIQUID
    solver_type: SolverType = SolverType.RK4
    dt: float = 0.01
    
    # Decision parameters
    confidence_threshold: float = 0.7
    max_inference_time: float = 2.0
    enable_fallback: bool = True
    
    # Performance
    use_gpu: bool = False  # Default to CPU for reliability
    batch_size: int = 16
    
    def validate(self) -> None:
        """Validate configuration."""
        if not self.name:
            raise ValueError("Agent name required")
        
        if self.input_size <= 0 or self.output_size <= 0:
            raise ValueError("Sizes must be positive")
        
        if not (0.0 <= self.confidence_threshold <= 1.0):
            raise ValueError("Confidence threshold must be 0-1")
    
    def to_liquid_config(self) -> LiquidConfig:
        """Convert to LiquidConfig for neural network."""
        return LiquidConfig(
            time_constants=TimeConstants(tau_min=0.1, tau_max=10.0),
            activation=self.activation_type,
            wiring=WiringConfig(sparsity=0.8),
            hidden_sizes=self.hidden_sizes,
            solver_type=self.solver_type.value,
            dt=self.dt,
            use_cuda=self.use_gpu
        )
    
    def to_agent_config(self) -> AgentConfig:
        """Convert to base AgentConfig."""
        return AgentConfig(
            name=self.name,
            model="lnn-council",
            timeout_seconds=int(self.max_inference_time),
            enable_memory=True
        )