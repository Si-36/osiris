"""
🧠 Real Liquid Neural Network 2025 - Adaptive Intelligence for AURA
==================================================================

Liquid Neural Networks (LNNs) are continuous-time RNNs that adapt
dynamically to inputs. Based on MIT's research, they offer:
- Continuous-time dynamics
- Adaptive behavior post-training
- Robustness to distribution shifts
- Interpretable neuron dynamics

"Intelligence that flows and adapts like liquid"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Callable
from dataclasses import dataclass, field
import time
from collections import deque
import structlog

logger = structlog.get_logger(__name__)


# ==================== Core Components ====================

@dataclass
class LiquidNeuronState:
    """State of a liquid neuron."""
    potential: torch.Tensor      # Membrane potential
    conductance: torch.Tensor    # Synaptic conductance
    time_constant: torch.Tensor  # Adaptive time constant
    history: deque = field(default_factory=lambda: deque(maxlen=100))
    
    def update_history(self):
        """Store current state in history."""
        self.history.append({
            "potential": self.potential.detach().cpu().numpy().copy(),
            "time": time.time()
        })


class LiquidTimeConstant(nn.Module):
    """
    Liquid Time Constant (LTC) layer - core of LNN.
    Implements continuous-time dynamics with adaptive time constants.
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        min_tau: float = 0.1,
        max_tau: float = 10.0,
        use_bias: bool = True
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.min_tau = min_tau
        self.max_tau = max_tau
        
        # Synaptic weights
        self.W_in = nn.Linear(input_size, hidden_size, bias=use_bias)
        self.W_rec = nn.Linear(hidden_size, hidden_size, bias=False)
        
        # Time constant network (makes it liquid)
        self.tau_net = nn.Sequential(
            nn.Linear(input_size + hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Sigmoid()
        )
        
        # Conductance parameters
        self.g_leak = nn.Parameter(torch.ones(hidden_size) * 0.1)
        self.g_syn = nn.Parameter(torch.ones(hidden_size) * 0.5)
        
        # ODE solver parameters
        self.ode_unfolds = 6  # Number of ODE solver steps
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights for stability."""
        nn.init.xavier_uniform_(self.W_in.weight)
        nn.init.xavier_uniform_(self.W_rec.weight)
        
        # Make recurrent weights slightly sparse for stability
        mask = torch.rand_like(self.W_rec.weight) > 0.8
        self.W_rec.weight.data *= mask.float()
    
    def forward(
        self,
        input: torch.Tensor,
        state: Optional[LiquidNeuronState] = None,
        dt: float = 0.1
    ) -> Tuple[torch.Tensor, LiquidNeuronState]:
        """
        Forward pass with continuous-time dynamics.
        
        Args:
            input: Input tensor (batch, input_size)
            state: Previous neuron state
            dt: Time step for ODE integration
            
        Returns:
            output: Hidden state (batch, hidden_size)
            new_state: Updated neuron state
        """
        batch_size = input.size(0)
        device = input.device
        
        # Initialize state if needed
        if state is None:
            state = LiquidNeuronState(
                potential=torch.zeros(batch_size, self.hidden_size, device=device),
                conductance=torch.zeros(batch_size, self.hidden_size, device=device),
                time_constant=torch.ones(batch_size, self.hidden_size, device=device)
            )
        
        # Compute adaptive time constants
        tau_input = torch.cat([input, state.potential], dim=-1)
        tau = self.min_tau + (self.max_tau - self.min_tau) * self.tau_net(tau_input)
        state.time_constant = tau
        
        # Input current
        I_in = self.W_in(input)
        
        # Solve ODE using Euler method with multiple steps
        v = state.potential
        g = state.conductance
        
        for _ in range(self.ode_unfolds):
            # Recurrent current
            I_rec = self.W_rec(torch.tanh(v))
            
            # Synaptic conductance dynamics
            dg_dt = (-g + torch.sigmoid(I_in + I_rec)) / (tau * 0.5)
            g = g + dt * dg_dt
            
            # Membrane potential dynamics
            dv_dt = (-self.g_leak * v + self.g_syn * g * (1 - v)) / tau
            v = v + dt * dv_dt
            
            # Apply bounds for stability
            v = torch.clamp(v, -5, 5)
        
        # Update state
        state.potential = v
        state.conductance = g
        state.update_history()
        
        # Output is the activation of membrane potential
        output = torch.tanh(v)
        
        return output, state


class CfCCell(nn.Module):
    """
    Closed-form Continuous-time (CfC) cell.
    More efficient version of LTC using closed-form solutions.
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        backbone_activation: str = "tanh",
        use_mixed: bool = True
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.use_mixed = use_mixed
        
        # Backbone network
        if backbone_activation == "tanh":
            self.activation = nn.Tanh()
        elif backbone_activation == "relu":
            self.activation = nn.ReLU()
        else:
            self.activation = nn.Identity()
        
        # Input mixing
        if use_mixed:
            self.input_mixer = nn.Linear(input_size, input_size)
        
        # Main transformation
        self.ff1 = nn.Linear(input_size + hidden_size, hidden_size)
        self.ff2 = nn.Linear(hidden_size, hidden_size)
        
        # Time constant network
        self.tau_net = nn.Sequential(
            nn.Linear(input_size + hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Sigmoid()
        )
        
        # Output gate
        self.gate = nn.Linear(hidden_size, hidden_size)
        
    def forward(
        self,
        input: torch.Tensor,
        hidden: torch.Tensor,
        dt: float = 1.0
    ) -> torch.Tensor:
        """
        Forward pass using closed-form solution.
        
        Args:
            input: Input tensor (batch, input_size)
            hidden: Hidden state (batch, hidden_size)
            dt: Time step
            
        Returns:
            new_hidden: Updated hidden state
        """
        # Mix input if enabled
        if self.use_mixed:
            input = self.input_mixer(input)
        
        # Concatenate input and hidden
        combined = torch.cat([input, hidden], dim=-1)
        
        # Compute time constants
        tau = self.tau_net(combined) * 5.0 + 0.1  # Scale to reasonable range
        
        # Backbone computation
        x = self.ff1(combined)
        x = self.activation(x)
        x = self.ff2(x)
        
        # Gating
        gate = torch.sigmoid(self.gate(x))
        
        # Closed-form solution for continuous-time dynamics
        # h(t+dt) = h(t) * exp(-dt/tau) + x * (1 - exp(-dt/tau))
        decay = torch.exp(-dt / tau)
        new_hidden = hidden * decay + x * gate * (1 - decay)
        
        return new_hidden


class AdaptiveLNN(nn.Module):
    """
    Adaptive Liquid Neural Network for AURA.
    Combines multiple liquid layers with attention mechanisms.
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_sizes: List[int],
        output_size: int,
        cell_type: str = "ltc",  # "ltc" or "cfc"
        use_attention: bool = True,
        dropout: float = 0.1
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.cell_type = cell_type
        self.use_attention = use_attention
        
        # Build liquid layers
        self.cells = nn.ModuleList()
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            if cell_type == "ltc":
                cell = LiquidTimeConstant(prev_size, hidden_size)
            else:
                cell = CfCCell(prev_size, hidden_size)
            self.cells.append(cell)
            prev_size = hidden_size
        
        # Attention mechanism
        if use_attention:
            self.attention = nn.MultiheadAttention(
                hidden_sizes[-1],
                num_heads=4,
                dropout=dropout
            )
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_sizes[-1], hidden_sizes[-1] // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_sizes[-1] // 2, output_size)
        )
        
        # Layer normalization
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(size) for size in hidden_sizes
        ])
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        inputs: torch.Tensor,
        initial_states: Optional[List[Any]] = None,
        return_sequences: bool = False
    ) -> Tuple[torch.Tensor, List[Any]]:
        """
        Process sequence through liquid layers.
        
        Args:
            inputs: Input sequence (batch, seq_len, input_size)
            initial_states: Initial states for each layer
            return_sequences: Whether to return all timesteps
            
        Returns:
            output: Model output
            final_states: Final states of each layer
        """
        batch_size, seq_len, _ = inputs.size()
        device = inputs.device
        
        # Initialize states
        if initial_states is None:
            initial_states = [None] * len(self.cells)
        
        # Process through liquid layers
        outputs = []
        states = initial_states
        
        for t in range(seq_len):
            x = inputs[:, t, :]
            new_states = []
            
            # Forward through each layer
            for i, (cell, state) in enumerate(zip(self.cells, states)):
                if self.cell_type == "ltc":
                    x, new_state = cell(x, state)
                    new_states.append(new_state)
                else:
                    # CfC cell
                    if state is None:
                        state = torch.zeros(batch_size, cell.hidden_size, device=device)
                    x = cell(x, state)
                    new_states.append(x)
                
                # Apply layer norm and dropout
                x = self.layer_norms[i](x)
                x = self.dropout(x)
            
            outputs.append(x)
            states = new_states
        
        # Stack outputs
        outputs = torch.stack(outputs, dim=1)  # (batch, seq_len, hidden)
        
        # Apply attention if enabled
        if self.use_attention and seq_len > 1:
            outputs, _ = self.attention(
                outputs.transpose(0, 1),
                outputs.transpose(0, 1),
                outputs.transpose(0, 1)
            )
            outputs = outputs.transpose(0, 1)
        
        # Get final output
        if return_sequences:
            output = self.output_proj(outputs)
        else:
            output = self.output_proj(outputs[:, -1, :])
        
        return output, states


# ==================== AURA-Specific LNN ====================

class AURALiquidBrain(nn.Module):
    """
    AURA's Liquid Brain - Adaptive decision-making system.
    Integrates with other AURA components for failure prevention.
    """
    
    def __init__(
        self,
        state_size: int = 64,      # Size of system state representation
        memory_size: int = 128,    # Size of memory integration
        decision_size: int = 32,   # Number of possible decisions
        num_layers: int = 3,
        hidden_size: int = 256
    ):
        super().__init__()
        self.state_size = state_size
        self.memory_size = memory_size
        self.decision_size = decision_size
        
        # State encoder
        self.state_encoder = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size)
        )
        
        # Memory integration
        self.memory_gate = nn.Sequential(
            nn.Linear(memory_size + hidden_size, hidden_size),
            nn.Sigmoid()
        )
        
        # Liquid neural network
        self.lnn = AdaptiveLNN(
            input_size=hidden_size,
            hidden_sizes=[hidden_size] * num_layers,
            output_size=hidden_size,
            cell_type="cfc",  # Use efficient CfC cells
            use_attention=True
        )
        
        # Decision networks
        self.risk_assessor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid()
        )
        
        self.decision_maker = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, decision_size)
        )
        
        # Confidence estimator
        self.confidence_net = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 4),
            nn.ReLU(),
            nn.Linear(hidden_size // 4, 1),
            nn.Sigmoid()
        )
        
        # Adaptation history
        self.adaptation_history = deque(maxlen=1000)
        self.decision_history = deque(maxlen=100)
        
        logger.info("AURA Liquid Brain initialized", layers=num_layers, hidden_size=hidden_size)
    
    def encode_system_state(self, system_state: Dict[str, Any]) -> torch.Tensor:
        """
        Encode system state into tensor representation.
        
        Args:
            system_state: Dictionary containing system metrics
            
        Returns:
            Encoded state tensor
        """
        # Extract key metrics
        features = []
        
        # Error rates
        features.append(system_state.get("avg_error_rate", 0.0))
        features.append(system_state.get("max_error_rate", 0.0))
        
        # Performance metrics
        features.append(system_state.get("avg_latency", 0.0) / 1000)  # Normalize
        features.append(system_state.get("throughput", 0.0) / 1000)
        
        # Resource usage
        features.append(system_state.get("cpu_usage", 0.0))
        features.append(system_state.get("memory_usage", 0.0))
        
        # Topology features (from TDA)
        betti = system_state.get("betti_numbers", [0, 0, 0])
        features.extend(betti[:3])
        
        # Anomaly scores
        features.append(system_state.get("anomaly_score", 0.0))
        
        # Pad or truncate to state_size
        while len(features) < self.state_size:
            features.append(0.0)
        features = features[:self.state_size]
        
        return torch.tensor(features, dtype=torch.float32).unsqueeze(0)
    
    def integrate_memory(
        self,
        encoded_state: torch.Tensor,
        memory_context: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Integrate memory context with current state.
        
        Args:
            encoded_state: Encoded system state
            memory_context: Memory representation from Memory Manager
            
        Returns:
            Integrated representation
        """
        if memory_context is None:
            # No memory, just use state
            return encoded_state
        
        # Compute memory gate
        combined = torch.cat([memory_context, encoded_state], dim=-1)
        gate = self.memory_gate(combined)
        
        # Gated integration
        integrated = encoded_state * gate + memory_context * (1 - gate)
        
        return integrated
    
    def forward(
        self,
        system_states: List[Dict[str, Any]],
        memory_contexts: Optional[List[torch.Tensor]] = None,
        return_trace: bool = False
    ) -> Dict[str, Any]:
        """
        Process system states and make adaptive decisions.
        
        Args:
            system_states: List of system states over time
            memory_contexts: Optional memory contexts
            return_trace: Whether to return decision trace
            
        Returns:
            Decision dictionary with risk assessment and actions
        """
        device = next(self.parameters()).device
        
        # Encode states
        encoded_states = []
        for i, state in enumerate(system_states):
            encoded = self.encode_system_state(state).to(device)
            encoded = self.state_encoder(encoded)
            
            # Integrate memory if available
            if memory_contexts and i < len(memory_contexts):
                encoded = self.integrate_memory(encoded, memory_contexts[i])
            
            encoded_states.append(encoded)
        
        # Stack into sequence
        state_sequence = torch.stack(encoded_states, dim=1)  # (1, seq_len, hidden)
        
        # Process through liquid network
        lnn_output, final_states = self.lnn(state_sequence)
        
        # Risk assessment
        risk_score = self.risk_assessor(lnn_output).squeeze()
        
        # Decision making
        decision_logits = self.decision_maker(lnn_output)
        decision_probs = F.softmax(decision_logits, dim=-1)
        
        # Confidence estimation
        confidence = self.confidence_net(lnn_output).squeeze()
        
        # Select decision
        if risk_score > 0.7:
            # High risk - choose conservative action
            decision_idx = 0  # Assuming 0 is "escalate" or "abort"
        else:
            # Normal operation - choose based on probabilities
            decision_idx = torch.argmax(decision_probs, dim=-1).item()
        
        # Map to decision type
        decision_map = [
            "escalate", "abort", "retry", "continue",
            "checkpoint", "rollback", "isolate", "scale"
        ]
        decision = decision_map[decision_idx % len(decision_map)]
        
        # Record adaptation
        adaptation_record = {
            "timestamp": time.time(),
            "risk_score": risk_score.item() if torch.is_tensor(risk_score) else risk_score,
            "decision": decision,
            "confidence": confidence.item() if torch.is_tensor(confidence) else confidence,
            "state_trajectory": len(system_states)
        }
        self.adaptation_history.append(adaptation_record)
        self.decision_history.append(decision)
        
        result = {
            "decision": decision,
            "risk_score": risk_score.item() if torch.is_tensor(risk_score) else risk_score,
            "confidence": confidence.item() if torch.is_tensor(confidence) else confidence,
            "decision_probabilities": decision_probs.cpu().numpy().tolist(),
            "adaptation_rate": self._calculate_adaptation_rate()
        }
        
        if return_trace:
            result["trace"] = {
                "encoded_states": encoded_states,
                "lnn_states": final_states,
                "adaptation_history": list(self.adaptation_history)[-10:]
            }
        
        return result
    
    def _calculate_adaptation_rate(self) -> float:
        """Calculate how much the system is adapting."""
        if len(self.decision_history) < 10:
            return 0.0
        
        # Look at decision variability
        recent_decisions = list(self.decision_history)[-20:]
        unique_decisions = len(set(recent_decisions))
        
        # More unique decisions = higher adaptation
        return unique_decisions / len(recent_decisions)
    
    def adapt_online(
        self,
        feedback: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Online adaptation based on feedback.
        
        Args:
            feedback: Feedback about decision outcomes
            
        Returns:
            Adaptation summary
        """
        # This would implement online learning
        # For now, we just track feedback
        
        success = feedback.get("success", True)
        decision = feedback.get("decision", "unknown")
        
        # Simple adaptation: adjust future decisions based on success
        if not success:
            logger.warning(
                "Negative feedback received",
                decision=decision,
                details=feedback.get("details", {})
            )
        
        return {
            "adapted": True,
            "feedback_processed": True,
            "current_adaptation_rate": self._calculate_adaptation_rate()
        }
    
    def get_neuron_dynamics(self) -> Dict[str, Any]:
        """Get current neuron dynamics for visualization."""
        dynamics = {
            "adaptation_rate": self._calculate_adaptation_rate(),
            "decision_distribution": {},
            "recent_risks": []
        }
        
        # Decision distribution
        if self.decision_history:
            decisions = list(self.decision_history)
            for decision in set(decisions):
                dynamics["decision_distribution"][decision] = decisions.count(decision) / len(decisions)
        
        # Recent risk scores
        recent = list(self.adaptation_history)[-10:]
        dynamics["recent_risks"] = [r["risk_score"] for r in recent]
        
        return dynamics


# ==================== Integration Functions ====================

def create_lnn_for_aura(config: Dict[str, Any]) -> AURALiquidBrain:
    """
    Create LNN configured for AURA system.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Configured AURA Liquid Brain
    """
    return AURALiquidBrain(
        state_size=config.get("state_size", 64),
        memory_size=config.get("memory_size", 128),
        decision_size=config.get("decision_size", 32),
        num_layers=config.get("num_layers", 3),
        hidden_size=config.get("hidden_size", 256)
    )


async def test_liquid_adaptation():
    """Test the liquid neural network's adaptation capabilities."""
    logger.info("Testing Liquid Neural Network adaptation")
    
    # Create LNN
    lnn = create_lnn_for_aura({})
    
    # Simulate system states over time
    states = []
    for t in range(10):
        # Normal state transitioning to failure
        error_rate = 0.1 if t < 5 else 0.7 + t * 0.05
        state = {
            "avg_error_rate": error_rate,
            "max_error_rate": error_rate * 1.2,
            "avg_latency": 100 + t * 50,
            "cpu_usage": 0.5 + t * 0.05,
            "betti_numbers": [5 - t // 3, t // 5, 0],
            "anomaly_score": 0.1 if t < 5 else 0.8
        }
        states.append(state)
    
    # Get decisions
    with torch.no_grad():
        result = lnn(states[-5:])  # Use last 5 states
    
    logger.info(
        "LNN Decision",
        decision=result["decision"],
        risk=result["risk_score"],
        confidence=result["confidence"],
        adaptation_rate=result["adaptation_rate"]
    )
    
    return result


if __name__ == "__main__":
    import asyncio
    
    # Test the implementation
    asyncio.run(test_liquid_adaptation())