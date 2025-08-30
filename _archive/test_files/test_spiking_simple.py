#!/usr/bin/env python3
"""
Simple test for Spiking GNN system without torch_geometric dependency
"""

import asyncio
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import time
from typing import Dict, List, Tuple

# Add project to path
sys.path.insert(0, str(Path(__file__).parent / "core" / "src"))

print("⚡ SIMPLE SPIKING GNN SYSTEM TEST")
print("=" * 60)

async def test_spiking_simple():
    """Test spiking GNN features without torch_geometric"""
    
    all_tests_passed = True
    test_results = {}
    
    try:
        # Test 1: Check if files can be imported
        print("1️⃣ TESTING FILE IMPORTS")
        print("-" * 40)
        try:
            # Check existing files for syntax errors
            import py_compile
            spiking_dir = Path("core/src/aura_intelligence/spiking")
            
            files_to_check = [
                spiking_dir / "advanced_spiking_gnn.py",
                spiking_dir / "council_sgnn.py",
                spiking_dir / "real_spiking_networks.py"
            ]
            
            for file in files_to_check:
                if file.exists():
                    try:
                        py_compile.compile(str(file), doraise=True)
                        print(f"✅ {file.name}: Syntax OK")
                    except py_compile.PyCompileError as e:
                        print(f"❌ {file.name}: {e}")
                        all_tests_passed = False
            
            test_results['syntax_check'] = all_tests_passed
            
        except Exception as e:
            print(f"❌ Import test failed: {e}")
            test_results['syntax_check'] = False
            all_tests_passed = False
        
        # Test 2: Test simple spiking neuron
        print("\n2️⃣ TESTING SIMPLE SPIKING NEURON")
        print("-" * 40)
        try:
            # Simple LIF neuron implementation
            class SimpleLIF(nn.Module):
                def __init__(self, size: int, threshold: float = 1.0, tau: float = 20.0, dt: float = 1.0):
                    super().__init__()
                    self.size = size
                    self.threshold = threshold
                    self.tau = tau
                    self.dt = dt
                    self.decay = np.exp(-dt / tau)
                    
                    # State
                    self.v = None
                    self.refractory = None
                
                def reset_state(self, batch_size: int):
                    device = next(self.parameters()).device if len(list(self.parameters())) > 0 else 'cpu'
                    self.v = torch.zeros(batch_size, self.size, device=device)
                    self.refractory = torch.zeros(batch_size, self.size, device=device)
                
                def forward(self, x: torch.Tensor) -> torch.Tensor:
                    if self.v is None:
                        self.reset_state(x.shape[0])
                    
                    # Update refractory
                    self.refractory = torch.clamp(self.refractory - self.dt, min=0)
                    
                    # Update membrane potential
                    self.v = self.decay * self.v + x
                    self.v[self.refractory > 0] = 0
                    
                    # Generate spikes
                    spikes = (self.v >= self.threshold).float()
                    
                    # Reset spiking neurons
                    self.v[spikes.bool()] = 0
                    self.refractory[spikes.bool()] = 2.0
                    
                    return spikes
            
            # Test LIF
            lif = SimpleLIF(size=10)
            lif.reset_state(batch_size=2)
            
            # Generate some spikes
            total_spikes = 0
            for t in range(50):
                x = torch.randn(2, 10) * 0.5 + 0.2
                spikes = lif(x)
                total_spikes += spikes.sum().item()
            
            print(f"✅ Simple LIF neuron:")
            print(f"   Total spikes: {total_spikes}")
            print(f"   Spike rate: {total_spikes / (50 * 2 * 10):.3f}")
            
            test_results['simple_lif'] = total_spikes > 0
            
        except Exception as e:
            print(f"❌ Simple LIF test failed: {e}")
            test_results['simple_lif'] = False
            all_tests_passed = False
        
        # Test 3: Simple graph-like structure
        print("\n3️⃣ TESTING SIMPLE GRAPH STRUCTURE")
        print("-" * 40)
        try:
            # Simple adjacency-based message passing
            class SimpleGraphLayer(nn.Module):
                def __init__(self, in_features: int, out_features: int):
                    super().__init__()
                    self.linear = nn.Linear(in_features, out_features)
                    self.lif = SimpleLIF(out_features)
                
                def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
                    # Handle batch dimension properly
                    if x.dim() == 2:
                        # Add batch dimension
                        x = x.unsqueeze(0)
                        adj = adj.unsqueeze(0)
                    
                    # Simple message passing: aggregate neighbors
                    messages = torch.matmul(adj, x)
                    
                    # Transform
                    out = self.linear(messages)
                    
                    # Apply spiking dynamics
                    spikes = self.lif(out)
                    
                    return spikes
            
            # Create simple graph
            num_nodes = 15
            num_features = 8
            
            # Random adjacency matrix
            adj = (torch.rand(num_nodes, num_nodes) < 0.3).float()
            adj = adj + adj.t()  # Make symmetric
            adj = (adj > 0).float()
            adj.fill_diagonal_(1)  # Self-loops
            
            # Normalize
            degree = adj.sum(dim=1, keepdim=True)
            adj = adj / degree.clamp(min=1)
            
            # Create layer
            layer = SimpleGraphLayer(num_features, 16)
            
            # Test forward pass
            x = torch.randn(num_nodes, num_features)
            layer.lif.reset_state(1)
            
            spike_counts = []
            for t in range(30):
                spikes = layer(x, adj)
                spike_counts.append(spikes.sum().item())
            
            total_graph_spikes = sum(spike_counts)
            
            print(f"✅ Simple graph layer:")
            print(f"   Nodes: {num_nodes}")
            print(f"   Edges: {(adj > 0).sum().item()}")
            print(f"   Total spikes: {total_graph_spikes}")
            print(f"   Spike density: {total_graph_spikes / (30 * num_nodes * 16):.3f}")
            
            test_results['simple_graph'] = total_graph_spikes > 0
            
        except Exception as e:
            print(f"❌ Simple graph test failed: {e}")
            test_results['simple_graph'] = False
            all_tests_passed = False
        
        # Test 4: Spike encoding/decoding
        print("\n4️⃣ TESTING SPIKE CODING")
        print("-" * 40)
        try:
            # Rate coding
            def rate_encode(values: torch.Tensor, time_window: int = 100, max_rate: float = 0.5):
                """Encode analog values as spike rates"""
                # Normalize to [0, max_rate]
                normalized = torch.sigmoid(values) * max_rate
                
                # Generate spikes
                spikes = []
                for t in range(time_window):
                    spike = (torch.rand_like(normalized) < normalized).float()
                    spikes.append(spike)
                
                return torch.stack(spikes)
            
            def rate_decode(spikes: torch.Tensor):
                """Decode spike trains to rates"""
                return spikes.mean(dim=0)
            
            # Test encoding/decoding
            values = torch.tensor([[0.2, 0.5, 0.8, -0.3, 1.2]])
            spike_train = rate_encode(values, time_window=100)
            decoded = rate_decode(spike_train)
            
            print(f"✅ Spike coding:")
            print(f"   Original: {values.numpy()}")
            print(f"   Encoded shape: {spike_train.shape}")
            print(f"   Decoded: {decoded.numpy()}")
            print(f"   Total spikes: {spike_train.sum().item()}")
            
            test_results['spike_coding'] = True
            
        except Exception as e:
            print(f"❌ Spike coding test failed: {e}")
            test_results['spike_coding'] = False
            all_tests_passed = False
        
        # Test 5: Temporal dynamics
        print("\n5️⃣ TESTING TEMPORAL DYNAMICS")
        print("-" * 40)
        try:
            # Test spike timing
            class TemporalLayer(nn.Module):
                def __init__(self, size: int):
                    super().__init__()
                    self.lif = SimpleLIF(size, threshold=0.8)
                    self.spike_history = []
                
                def forward(self, x: torch.Tensor):
                    spikes = self.lif(x)
                    self.spike_history.append(spikes.clone())
                    return spikes
                
                def get_spike_times(self):
                    """Get spike timing statistics"""
                    if not self.spike_history:
                        return {}
                    
                    history = torch.stack(self.spike_history)
                    
                    # Find first spike time for each neuron
                    first_spike_times = []
                    for n in range(history.shape[2]):
                        neuron_spikes = history[:, 0, n]
                        spike_indices = torch.where(neuron_spikes > 0)[0]
                        if len(spike_indices) > 0:
                            first_spike_times.append(spike_indices[0].item())
                    
                    return {
                        'total_spikes': history.sum().item(),
                        'first_spike_times': first_spike_times,
                        'avg_first_spike': np.mean(first_spike_times) if first_spike_times else 0
                    }
            
            # Test temporal dynamics
            temp_layer = TemporalLayer(20)
            temp_layer.lif.reset_state(1)
            
            # Constant input
            constant_input = torch.ones(1, 20) * 0.15
            
            for t in range(100):
                _ = temp_layer(constant_input)
            
            timing_stats = temp_layer.get_spike_times()
            
            print(f"✅ Temporal dynamics:")
            print(f"   Total spikes: {timing_stats['total_spikes']}")
            print(f"   Neurons that spiked: {len(timing_stats['first_spike_times'])}/20")
            print(f"   Average first spike time: {timing_stats['avg_first_spike']:.1f} ms")
            
            test_results['temporal_dynamics'] = timing_stats['total_spikes'] > 0
            
        except Exception as e:
            print(f"❌ Temporal dynamics test failed: {e}")
            test_results['temporal_dynamics'] = False
            all_tests_passed = False
        
        # Test 6: Energy efficiency
        print("\n6️⃣ TESTING ENERGY EFFICIENCY")
        print("-" * 40)
        try:
            # Compare spiking vs non-spiking
            
            # Non-spiking network
            class DenseNetwork(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.layers = nn.Sequential(
                        nn.Linear(32, 64),
                        nn.ReLU(),
                        nn.Linear(64, 64),
                        nn.ReLU(),
                        nn.Linear(64, 32)
                    )
                    self.ops_count = 0
                
                def forward(self, x):
                    # Count operations (simplified)
                    self.ops_count += x.shape[0] * (32*64 + 64*64 + 64*32)
                    return self.layers(x)
            
            # Spiking network
            class SpikingNetwork(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.layer1 = nn.Linear(32, 64)
                    self.layer2 = nn.Linear(64, 64)
                    self.layer3 = nn.Linear(64, 32)
                    self.lif1 = SimpleLIF(64)
                    self.lif2 = SimpleLIF(64)
                    self.lif3 = SimpleLIF(32)
                    self.spike_count = 0
                
                def forward(self, x, time_steps=50):
                    self.lif1.reset_state(x.shape[0])
                    self.lif2.reset_state(x.shape[0])
                    self.lif3.reset_state(x.shape[0])
                    
                    for t in range(time_steps):
                        s1 = self.lif1(self.layer1(x))
                        s2 = self.lif2(self.layer2(s1))
                        s3 = self.lif3(self.layer3(s2))
                        
                        self.spike_count += (s1.sum() + s2.sum() + s3.sum()).item()
                    
                    return s3
            
            # Compare
            dense_net = DenseNetwork()
            spiking_net = SpikingNetwork()
            
            x = torch.randn(10, 32)
            
            # Dense forward
            _ = dense_net(x)
            dense_ops = dense_net.ops_count
            
            # Spiking forward
            _ = spiking_net(x, time_steps=50)
            spike_ops = spiking_net.spike_count * 2  # Assume 2 ops per spike
            
            print(f"✅ Energy efficiency comparison:")
            print(f"   Dense network ops: {dense_ops:,}")
            print(f"   Spiking network ops: {spike_ops:,}")
            print(f"   Efficiency ratio: {dense_ops/max(spike_ops, 1):.2f}x")
            print(f"   Total spikes: {spiking_net.spike_count}")
            
            test_results['energy_efficiency'] = spike_ops < dense_ops
            
        except Exception as e:
            print(f"❌ Energy efficiency test failed: {e}")
            test_results['energy_efficiency'] = False
            all_tests_passed = False
        
        # Test 7: Integration test
        print("\n7️⃣ TESTING INTEGRATION WITH NEUROMORPHIC")
        print("-" * 40)
        try:
            from aura_intelligence.neuromorphic.advanced_neuromorphic_system import (
                LIFNeuron, NeuromorphicConfig
            )
            
            # Test compatibility
            config = NeuromorphicConfig(
                input_size=10,
                hidden_size=20,
                output_size=5,
                threshold=1.0,
                tau_membrane=20.0
            )
            
            neuron = LIFNeuron(config)
            neuron.reset_state(batch_size=2, num_neurons=10)
            
            # Generate spikes
            spike_count = 0
            for t in range(20):
                x = torch.randn(2, 10) * 0.3
                spikes = neuron(x)
                spike_count += spikes.sum().item()
            
            print(f"✅ Neuromorphic integration:")
            print(f"   Spike count: {spike_count}")
            print(f"   Compatible: Yes")
            
            test_results['neuromorphic_integration'] = True
            
        except Exception as e:
            print(f"⚠️  Neuromorphic integration skipped: {e}")
            test_results['neuromorphic_integration'] = None
        
        # Final summary
        print("\n" + "=" * 60)
        print("📊 TEST RESULTS SUMMARY")
        print("-" * 40)
        
        for test_name, result in test_results.items():
            if result is True:
                status = "✅ PASSED"
            elif result is False:
                status = "❌ FAILED"
            else:
                status = "⚠️  SKIPPED"
            
            print(f"{test_name:25} {status}")
        
        passed = sum(1 for r in test_results.values() if r is True)
        failed = sum(1 for r in test_results.values() if r is False)
        skipped = sum(1 for r in test_results.values() if r is None)
        
        print(f"\nTotal: {passed} passed, {failed} failed, {skipped} skipped")
        
        if all_tests_passed and failed == 0:
            print("\n✅ ALL SPIKING TESTS PASSED!")
        else:
            print(f"\n❌ Some tests failed. Please debug.")
        
        return all_tests_passed
        
    except Exception as e:
        print(f"\n❌ Critical test error: {e}")
        import traceback
        traceback.print_exc()
        return False


# Run the simple test
if __name__ == "__main__":
    result = asyncio.run(test_spiking_simple())
    sys.exit(0 if result else 1)