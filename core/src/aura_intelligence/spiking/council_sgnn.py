"""Spiking GNN Council - Energy-efficient multi-agent coordination"""
import asyncio, time
import numpy as np
from typing import Dict, Any, Optional, List
from collections import defaultdict

class SpikingCouncil:
    """Energy-efficient spiking GNN for component coordination"""
    
    def __init__(self, num_components: int = 209, feature_dim: int = 32):
        self.num_components = num_components
        self.feature_dim = feature_dim
        self.spike_threshold = 0.5
        self.decay_factor = 0.9
        
        # Component state matrix (N x D)
        self.node_features = np.zeros((num_components, feature_dim))
        self.membrane_potentials = np.zeros(num_components)
        self.spike_history = defaultdict(list)
        
        # Simple adjacency matrix (can be loaded from Neo4j)
        self.adjacency = self._init_adjacency()
        
        # Metrics
        self.power_consumption = 0.0
        self.sparsity_ratio = 0.0
        
    def _init_adjacency(self) -> np.ndarray:
        """Initialize component adjacency matrix"""
        # Simple ring topology + random connections
        adj = np.zeros((self.num_components, self.num_components))
        
        # Ring connections
        for i in range(self.num_components):
            adj[i, (i + 1) % self.num_components] = 1.0
            adj[i, (i - 1) % self.num_components] = 1.0
        
        # Random long-range connections (10% density)
        np.random.seed(42)
        random_mask = np.random.random((self.num_components, self.num_components)) < 0.1
        adj[random_mask] = 1.0
        
        # Make symmetric
        adj = (adj + adj.T) / 2
        np.fill_diagonal(adj, 0)
        
        return adj
    
    def encode_to_spikes(self, component_messages: Dict[str, Dict[str, Any]]) -> np.ndarray:
        """Convert component messages to spike trains"""
        spike_rates = np.zeros(self.num_components)
        
        for i, (comp_id, message) in enumerate(component_messages.items()):
            if i >= self.num_components: break
            
            # Extract features from message
            confidence = float(message.get("confidence", 0.5))
            priority = float(message.get("priority", 0.5))
            tda_risk = float(message.get("tda_anomaly", 0.0))
            
            # Poisson encoding: rate proportional to confidence and priority
            rate = (confidence + priority) / 2.0
            # Reduce rate if high TDA risk
            rate *= (1.0 - tda_risk)
            
            spike_rates[i] = rate
        
        return spike_rates
    
    def forward_pass(self, spike_rates: np.ndarray, steps: int = 10) -> Dict[str, Any]:
        """Forward pass through spiking GNN"""
        start_time = time.perf_counter()
        
        spikes_generated = np.zeros((steps, self.num_components))
        active_neurons = 0
        
        for step in range(steps):
            # Generate spikes based on rates
            spikes = np.random.random(self.num_components) < spike_rates
            spikes_generated[step] = spikes.astype(float)
            
            # Update membrane potentials
            # Decay
            self.membrane_potentials *= self.decay_factor
            
            # Input current from spikes
            self.membrane_potentials += spikes.astype(float) * 0.1
            
            # Lateral connections (simplified GNN message passing)
            lateral_input = np.dot(self.adjacency, spikes.astype(float)) * 0.05
            self.membrane_potentials += lateral_input
            
            # Count active neurons
            active_neurons += np.sum(spikes)
        
        # Calculate metrics
        total_spikes = np.sum(spikes_generated)
        self.sparsity_ratio = 1.0 - (active_neurons / (steps * self.num_components))
        
        # Estimate power consumption (lower is better)
        self.power_consumption = active_neurons * 0.1  # mW estimate
        
        # Aggregate output signal
        output_signal = np.mean(spikes_generated, axis=0)
        consensus_strength = np.std(output_signal)  # Higher std = less consensus
        
        processing_time = (time.perf_counter() - start_time) * 1000  # ms
        
        return {
            "consensus_signal": output_signal[:32].tolist(),  # Limit to 32-dim
            "consensus_strength": float(consensus_strength),
            "sparsity_ratio": float(self.sparsity_ratio),
            "power_mw": float(self.power_consumption),
            "latency_ms": float(processing_time),
            "total_spikes": int(total_spikes)
        }
    
    async def process_component_messages(self, messages: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Main processing function"""
        if not messages:
            return {"consensus_signal": [0.0] * 32, "status": "no_input"}
        
        # Encode messages to spikes
        spike_rates = self.encode_to_spikes(messages)
        
        # Forward pass
        result = self.forward_pass(spike_rates)
        
        # Add status
        result["status"] = "processed"
        result["components_processed"] = len(messages)
        
        return result
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get spiking council metrics"""
        return {
            "power_consumption_mw": float(self.power_consumption),
            "sparsity_ratio": float(self.sparsity_ratio),
            "num_components": self.num_components,
            "feature_dim": self.feature_dim,
            "spike_threshold": self.spike_threshold
        }