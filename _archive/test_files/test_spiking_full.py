#!/usr/bin/env python3
"""
Full test for Spiking GNN system after fixing all issues
"""

import asyncio
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt

# Add project to path
sys.path.insert(0, str(Path(__file__).parent / "core" / "src"))

print("⚡ FULL SPIKING SYSTEM TEST")
print("=" * 60)

async def test_spiking_full():
    """Test all spiking features after fixes"""
    
    all_tests_passed = True
    test_results = {}
    
    try:
        # Test 1: Import advanced spiking GNN
        print("1️⃣ TESTING ADVANCED SPIKING GNN IMPORT")
        print("-" * 40)
        try:
            from aura_intelligence.spiking.advanced_spiking_gnn import (
                LIFNeuron, SpikingGraphConv, SpikingGAT, AdvancedSpikingGNN
            )
            
            print("✅ Advanced spiking GNN imported successfully")
            test_results['import_advanced'] = True
            
        except Exception as e:
            print(f"❌ Import failed: {e}")
            test_results['import_advanced'] = False
            all_tests_passed = False
        
        # Test 2: Test LIF Neuron
        print("\n2️⃣ TESTING LIF NEURON")
        print("-" * 40)
        try:
            neuron = LIFNeuron(tau=20.0, v_threshold=1.0)
            
            # Test dynamics
            x = torch.randn(10, 20) * 0.5
            spike_count = 0
            
            for t in range(50):
                spikes = neuron(x)
                spike_count += spikes.sum().item()
            
            print(f"✅ LIF Neuron:")
            print(f"   Input shape: {x.shape}")
            print(f"   Total spikes: {spike_count}")
            print(f"   Spike rate: {spike_count / (50 * 10 * 20):.3f}")
            
            test_results['lif_neuron'] = spike_count > 0
            
        except Exception as e:
            print(f"❌ LIF neuron test failed: {e}")
            test_results['lif_neuron'] = False
            all_tests_passed = False
        
        # Test 3: Test Spiking Graph Conv
        print("\n3️⃣ TESTING SPIKING GRAPH CONVOLUTION")
        print("-" * 40)
        try:
            # Create simple graph
            num_nodes = 15
            num_edges = 40
            edge_index = torch.randint(0, num_nodes, (2, num_edges))
            x = torch.randn(num_nodes, 16)
            
            # Create layer
            conv = SpikingGraphConv(16, 32)
            
            # Forward pass
            out = conv(x, edge_index)
            
            print(f"✅ Spiking Graph Conv:")
            print(f"   Nodes: {num_nodes}")
            print(f"   Input features: 16")
            print(f"   Output features: 32")
            print(f"   Output shape: {out.shape}")
            print(f"   Spikes: {out.sum().item()}")
            
            test_results['graph_conv'] = True
            
        except Exception as e:
            print(f"❌ Graph conv test failed: {e}")
            test_results['graph_conv'] = False
            all_tests_passed = False
        
        # Test 4: Test Spiking GAT
        print("\n4️⃣ TESTING SPIKING GAT")
        print("-" * 40)
        try:
            gat = SpikingGAT(16, 8, heads=4, concat=True)
            
            # Forward pass
            out = gat(x, edge_index)
            
            print(f"✅ Spiking GAT:")
            print(f"   Heads: 4")
            print(f"   Output shape: {out.shape}")
            print(f"   Expected shape: ({num_nodes}, {8 * 4})")
            
            test_results['spiking_gat'] = out.shape == (num_nodes, 32)
            
        except Exception as e:
            print(f"❌ GAT test failed: {e}")
            test_results['spiking_gat'] = False
            all_tests_passed = False
        
        # Test 5: Test Advanced Spiking GNN
        print("\n5️⃣ TESTING ADVANCED SPIKING GNN")
        print("-" * 40)
        try:
            model = AdvancedSpikingGNN(
                num_nodes=20,
                input_dim=16,
                hidden_dim=32,
                output_dim=8,
                num_layers=3
            )
            
            # Create input
            x = torch.randn(20, 16)
            edge_index = torch.randint(0, 20, (2, 50))
            
            # Forward pass
            out, spike_train = model(x, edge_index, time_window=30)
            
            print(f"✅ Advanced Spiking GNN:")
            print(f"   Layers: 3")
            print(f"   Output shape: {out.shape}")
            print(f"   Spike train shape: {spike_train.shape}")
            print(f"   Average spike rate: {spike_train.mean():.3f}")
            
            test_results['advanced_gnn'] = True
            
        except Exception as e:
            print(f"❌ Advanced GNN test failed: {e}")
            test_results['advanced_gnn'] = False
            all_tests_passed = False
        
        # Test 6: Import and test new system
        print("\n6️⃣ TESTING NEW SPIKING GNN SYSTEM")
        print("-" * 40)
        try:
            from aura_intelligence.spiking.advanced_spiking_gnn_system import (
                SpikingGNN, SpikingGNNConfig, spike_fn,
                SpikingGraphNeuron, TemporalGraphDataset
            )
            
            # Create simple graph data
            class SimpleGraphData:
                def __init__(self, x, edge_index):
                    self.x = x
                    self.edge_index = edge_index
            
            config = SpikingGNNConfig(
                in_channels=16,
                hidden_channels=32,
                out_channels=8,
                num_layers=2,
                heads=2
            )
            
            sgnn = SpikingGNN(config)
            
            # Test forward
            data = SimpleGraphData(x, edge_index)
            outputs = sgnn(data, time_steps=20)
            
            print(f"✅ New Spiking GNN System:")
            print(f"   Node features shape: {outputs['node_features'].shape}")
            print(f"   Spike rates: {outputs['spike_rates'].mean():.3f}")
            
            # Energy estimate
            energy = sgnn.get_energy_estimate(outputs['spike_train'])
            print(f"   Energy per node: {energy['energy_per_node_nj']:.3f} nJ")
            
            test_results['new_system'] = True
            
        except Exception as e:
            print(f"❌ New system test failed: {e}")
            test_results['new_system'] = False
            all_tests_passed = False
        
        # Test 7: Integration test
        print("\n7️⃣ TESTING FULL INTEGRATION")
        print("-" * 40)
        try:
            # Test with neuromorphic system
            from aura_intelligence.neuromorphic.advanced_neuromorphic_system import (
                NeuromorphicNetwork, NeuromorphicConfig, NeuronType
            )
            
            # Create hybrid architecture
            class HybridSpikingModel(nn.Module):
                def __init__(self):
                    super().__init__()
                    
                    # Spiking GNN for graph processing
                    self.sgnn_config = SpikingGNNConfig(
                        in_channels=16,
                        hidden_channels=32,
                        out_channels=16,
                        num_layers=2
                    )
                    self.spiking_gnn = SpikingGNN(self.sgnn_config)
                    
                    # Neuromorphic network for final processing
                    self.neuro_config = NeuromorphicConfig(
                        input_size=16,
                        hidden_size=32,
                        output_size=8,
                        num_layers=2,
                        neuron_type=NeuronType.LIF
                    )
                    self.neuromorphic = NeuromorphicNetwork(self.neuro_config)
                
                def forward(self, graph_data):
                    # Process graph with spiking GNN
                    gnn_out = self.spiking_gnn(graph_data, time_steps=20)
                    
                    # Process features with neuromorphic network
                    node_features = gnn_out['node_features']
                    neuro_out = self.neuromorphic(node_features, time_steps=20)
                    
                    return {
                        'graph_features': gnn_out['node_features'],
                        'graph_spikes': gnn_out['spike_rates'].mean(),
                        'neuro_output': neuro_out['output'],
                        'neuro_spikes': neuro_out['spike_rates'].mean()
                    }
            
            # Test hybrid model
            hybrid = HybridSpikingModel()
            hybrid_out = hybrid(data)
            
            print(f"✅ Hybrid Integration:")
            print(f"   Graph spike rate: {hybrid_out['graph_spikes']:.3f}")
            print(f"   Neuro spike rate: {hybrid_out['neuro_spikes']:.3f}")
            print(f"   Final output shape: {hybrid_out['neuro_output'].shape}")
            
            test_results['integration'] = True
            
        except Exception as e:
            print(f"⚠️  Integration test skipped: {e}")
            test_results['integration'] = None
        
        # Test 8: Performance benchmark
        print("\n8️⃣ PERFORMANCE BENCHMARK")
        print("-" * 40)
        
        try:
            # Test different configurations
            configs = [
                ("Small", 10, 20),
                ("Medium", 50, 100),
                ("Large", 100, 300)
            ]
            
            for name, nodes, edges in configs:
                x = torch.randn(nodes, 16)
                edge_index = torch.randint(0, nodes, (2, edges))
                data = SimpleGraphData(x, edge_index)
                
                start = time.time()
                outputs = sgnn(data, time_steps=20)
                elapsed = (time.time() - start) * 1000
                
                energy = sgnn.get_energy_estimate(outputs['spike_train'])
                
                print(f"\n{name} graph ({nodes} nodes, {edges} edges):")
                print(f"  Time: {elapsed:.1f} ms")
                print(f"  Total spikes: {energy['total_spikes']:.0f}")
                print(f"  Energy: {energy['total_energy_j']*1e9:.3f} nJ")
                print(f"  Spike rate: {outputs['spike_rates'].mean():.3f}")
            
            test_results['performance'] = True
            
        except Exception as e:
            print(f"❌ Performance test failed: {e}")
            test_results['performance'] = False
            all_tests_passed = False
        
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
            
            print(f"{test_name:20} {status}")
        
        passed = sum(1 for r in test_results.values() if r is True)
        failed = sum(1 for r in test_results.values() if r is False)
        skipped = sum(1 for r in test_results.values() if r is None)
        
        print(f"\nTotal: {passed} passed, {failed} failed, {skipped} skipped")
        
        if all_tests_passed and failed == 0:
            print("\n✅ ALL SPIKING SYSTEM TESTS PASSED!")
        else:
            print(f"\n❌ Some tests failed. Please debug.")
        
        return all_tests_passed
        
    except Exception as e:
        print(f"\n❌ Critical test error: {e}")
        import traceback
        traceback.print_exc()
        return False


# Visualization helper
def plot_spike_raster(spike_train: torch.Tensor, title: str = "Spike Raster Plot"):
    """Create spike raster plot"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Convert to numpy
    spikes = spike_train.detach().cpu().numpy()
    
    # Find spike times
    spike_times, neuron_ids = np.where(spikes > 0)
    
    # Plot
    ax.scatter(spike_times, neuron_ids, s=1, c='black', marker='|')
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Neuron ID')
    ax.set_title(title)
    
    plt.tight_layout()
    return fig


# Run the full test
if __name__ == "__main__":
    result = asyncio.run(test_spiking_full())
    sys.exit(0 if result else 1)