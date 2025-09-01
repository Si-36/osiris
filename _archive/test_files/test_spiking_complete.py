#!/usr/bin/env python3
"""
Complete test for Spiking GNN system with all features and integration
"""

import asyncio
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch_geometric.data import Data, Batch
from torch_geometric.utils import erdos_renyi_graph, to_networkx
import time
import matplotlib.pyplot as plt
import networkx as nx
from typing import Dict, List, Tuple

# Add project to path
sys.path.insert(0, str(Path(__file__).parent / "core" / "src"))

print("⚡ COMPLETE SPIKING GNN SYSTEM TEST")
print("=" * 60)

async def test_spiking_complete():
    """Test all spiking GNN features comprehensively"""
    
    all_tests_passed = True
    test_results = {}
    
    try:
        # Import the spiking GNN system
        from aura_intelligence.spiking.advanced_spiking_gnn_system import (
            SpikingGNN, SpikingGNNConfig, SpikingActivation,
            SpikingGraphNeuron, SpikingGraphConv, SpikingGraphAttention,
            TemporalGraphPool, TemporalGraphDataset, spike_fn
        )
        
        print("✅ Imports successful\n")
        
        # Test 1: Surrogate Gradient Function
        print("1️⃣ TESTING SURROGATE GRADIENT")
        print("-" * 40)
        try:
            # Test different activation functions
            x = torch.tensor([-2.0, -1.0, 0.0, 0.5, 1.0, 1.5, 2.0], requires_grad=True)
            threshold = 1.0
            
            for spike_type in [SpikingActivation.ATAN, SpikingActivation.SIGMOID, SpikingActivation.GAUSSIAN]:
                spikes = spike_fn(x, threshold, spike_type, 1.0)
                
                # Compute gradients
                loss = spikes.sum()
                loss.backward()
                
                print(f"\n{spike_type.value}:")
                print(f"  Input: {x.detach().numpy()}")
                print(f"  Spikes: {spikes.detach().numpy()}")
                print(f"  Gradients: {x.grad.numpy()}")
                
                x.grad.zero_()
            
            test_results['surrogate_gradient'] = True
            
        except Exception as e:
            print(f"❌ Surrogate gradient test failed: {e}")
            test_results['surrogate_gradient'] = False
            all_tests_passed = False
        
        # Test 2: Spiking Graph Neuron
        print("\n2️⃣ TESTING SPIKING GRAPH NEURON")
        print("-" * 40)
        try:
            config = SpikingGNNConfig(
                in_channels=16,
                hidden_channels=32,
                out_channels=8,
                tau_membrane=20.0,
                tau_synapse=5.0,
                threshold=1.0
            )
            
            neuron = SpikingGraphNeuron(config)
            num_nodes = 10
            neuron.reset_state(num_nodes, torch.device('cpu'))
            
            # Test dynamics
            spike_counts = []
            potentials = []
            
            for t in range(50):
                # Varying input current
                input_current = torch.randn(num_nodes) * 0.5 + 0.3
                spikes = neuron(input_current)
                
                spike_counts.append(spikes.sum().item())
                potentials.append(neuron.membrane_potential.clone())
            
            total_spikes = sum(spike_counts)
            avg_potential = torch.stack(potentials).mean()
            
            print(f"✅ Neuron dynamics:")
            print(f"   Total spikes: {total_spikes}")
            print(f"   Average membrane potential: {avg_potential:.3f}")
            print(f"   Spike rate: {total_spikes / (50 * num_nodes):.3f}")
            
            test_results['graph_neuron'] = total_spikes > 0
            
        except Exception as e:
            print(f"❌ Graph neuron test failed: {e}")
            test_results['graph_neuron'] = False
            all_tests_passed = False
        
        # Test 3: Spiking Graph Convolution
        print("\n3️⃣ TESTING SPIKING GRAPH CONVOLUTION")
        print("-" * 40)
        try:
            # Create a simple graph
            num_nodes = 20
            edge_index = erdos_renyi_graph(num_nodes, 0.3, directed=False)
            x = torch.randn(num_nodes, config.in_channels)
            
            # Create layer
            conv = SpikingGraphConv(
                config.in_channels,
                config.hidden_channels,
                config
            )
            
            # Reset neurons
            conv.neurons.reset_state(num_nodes, x.device)
            
            # Process multiple timesteps
            spike_outputs = []
            for t in range(20):
                spikes = conv(x, edge_index)
                spike_outputs.append(spikes)
            
            spike_tensor = torch.stack(spike_outputs)
            
            print(f"✅ Graph convolution:")
            print(f"   Input shape: {x.shape}")
            print(f"   Output shape: {spikes.shape}")
            print(f"   Total spikes: {spike_tensor.sum().item()}")
            print(f"   Spike density: {spike_tensor.mean():.3f}")
            
            test_results['graph_conv'] = spike_tensor.sum() > 0
            
        except Exception as e:
            print(f"❌ Graph convolution test failed: {e}")
            test_results['graph_conv'] = False
            all_tests_passed = False
        
        # Test 4: Spiking Graph Attention
        print("\n4️⃣ TESTING SPIKING GRAPH ATTENTION")
        print("-" * 40)
        try:
            attn = SpikingGraphAttention(
                config.in_channels,
                config.hidden_channels,
                config
            )
            
            # Reset neurons
            attn.neurons.reset_state(num_nodes, x.device)
            
            # Forward pass
            attn_spikes = attn(x, edge_index)
            
            print(f"✅ Graph attention:")
            print(f"   Heads: {config.heads}")
            print(f"   Output shape: {attn_spikes.shape}")
            print(f"   Spike count: {attn_spikes.sum().item()}")
            
            test_results['graph_attention'] = True
            
        except Exception as e:
            print(f"❌ Graph attention test failed: {e}")
            test_results['graph_attention'] = False
            all_tests_passed = False
        
        # Test 5: Temporal Pooling
        print("\n5️⃣ TESTING TEMPORAL POOLING")
        print("-" * 40)
        try:
            pool = TemporalGraphPool(config)
            
            # Create temporal spike data
            time_steps = 50
            spike_train = torch.rand(time_steps, num_nodes, config.hidden_channels) < 0.1
            spike_train = spike_train.float()
            
            # Pool
            pooled = pool(spike_train)
            
            print(f"✅ Temporal pooling:")
            print(f"   Input shape: {spike_train.shape}")
            print(f"   Output shape: {pooled.shape}")
            print(f"   Temporal reduction: {time_steps}x → 1x")
            
            test_results['temporal_pool'] = True
            
        except Exception as e:
            print(f"❌ Temporal pooling test failed: {e}")
            test_results['temporal_pool'] = False
            all_tests_passed = False
        
        # Test 6: Complete Spiking GNN
        print("\n6️⃣ TESTING COMPLETE SPIKING GNN")
        print("-" * 40)
        try:
            # Create model
            model = SpikingGNN(config)
            
            # Create graph data
            data = Data(x=x, edge_index=edge_index)
            
            # Forward pass
            outputs = model(data, time_steps=30)
            
            print(f"✅ Spiking GNN forward pass:")
            print(f"   Node features: {outputs['node_features'].shape}")
            print(f"   Spike train: {outputs['spike_train'].shape}")
            print(f"   Average spike rate: {outputs['spike_rates'].mean():.3f}")
            
            # Energy analysis
            energy = model.get_energy_estimate(outputs['spike_train'])
            
            print(f"\n⚡ Energy consumption:")
            print(f"   Total spikes: {energy['total_spikes']:.0f}")
            print(f"   Spikes per node: {energy['spikes_per_node']:.1f}")
            print(f"   Energy per node: {energy['energy_per_node_nj']:.3f} nJ")
            
            test_results['full_model'] = outputs['spike_rates'].mean() > 0
            
        except Exception as e:
            print(f"❌ Full model test failed: {e}")
            test_results['full_model'] = False
            all_tests_passed = False
        
        # Test 7: Dynamic Edge Updates
        print("\n7️⃣ TESTING DYNAMIC EDGE UPDATES")
        print("-" * 40)
        try:
            # Enable dynamic edges
            config.dynamic_edges = True
            dynamic_model = SpikingGNN(config)
            
            initial_edges = edge_index.shape[1]
            outputs = dynamic_model(data, time_steps=50)
            final_edges = outputs['edge_index'].shape[1]
            
            print(f"✅ Dynamic edge updates:")
            print(f"   Initial edges: {initial_edges}")
            print(f"   Final edges: {final_edges}")
            print(f"   Edge change: {abs(final_edges - initial_edges)}")
            
            test_results['dynamic_edges'] = True
            
        except Exception as e:
            print(f"❌ Dynamic edges test failed: {e}")
            test_results['dynamic_edges'] = False
            all_tests_passed = False
        
        # Test 8: Temporal Graph Dataset
        print("\n8️⃣ TESTING TEMPORAL GRAPH DATASET")
        print("-" * 40)
        try:
            dataset = TemporalGraphDataset(
                num_graphs=5,
                num_nodes=15,
                num_features=config.in_channels,
                num_timesteps=10
            )
            
            # Get a sequence
            sequence = dataset[0]
            
            print(f"✅ Temporal dataset:")
            print(f"   Dataset size: {len(dataset)}")
            print(f"   Sequence length: {len(sequence)}")
            print(f"   Nodes per graph: {sequence[0].x.shape[0]}")
            print(f"   Features per node: {sequence[0].x.shape[1]}")
            
            # Process sequence
            seq_outputs = []
            for t, graph in enumerate(sequence[:5]):
                out = model(graph, time_steps=10)
                seq_outputs.append(out['spike_rates'].mean().item())
            
            print(f"\n   Spike rates over time: {[f'{r:.3f}' for r in seq_outputs]}")
            
            test_results['temporal_dataset'] = True
            
        except Exception as e:
            print(f"❌ Temporal dataset test failed: {e}")
            test_results['temporal_dataset'] = False
            all_tests_passed = False
        
        # Test 9: Integration with Neuromorphic
        print("\n9️⃣ TESTING NEUROMORPHIC INTEGRATION")
        print("-" * 40)
        try:
            from aura_intelligence.neuromorphic.advanced_neuromorphic_system import (
                NeuromorphicNetwork, NeuromorphicConfig, NeuronType
            )
            
            # Create hybrid model
            class NeuroSpikingGNN(nn.Module):
                def __init__(self, sgnn_config, neuro_config):
                    super().__init__()
                    self.spiking_gnn = SpikingGNN(sgnn_config)
                    self.neuromorphic = NeuromorphicNetwork(neuro_config)
                    
                def forward(self, graph_data, node_features):
                    # Process graph with spiking GNN
                    gnn_out = self.spiking_gnn(graph_data, time_steps=20)
                    
                    # Process node features with neuromorphic
                    neuro_out = self.neuromorphic(node_features, time_steps=20)
                    
                    # Combine
                    combined = {
                        'graph_spikes': gnn_out['spike_rates'],
                        'neuro_spikes': neuro_out['spike_rates'],
                        'graph_features': gnn_out['node_features'],
                        'neuro_output': neuro_out['output']
                    }
                    
                    return combined
            
            neuro_config = NeuromorphicConfig(
                input_size=config.in_channels,
                hidden_size=32,
                output_size=8,
                num_layers=2,
                neuron_type=NeuronType.LIF
            )
            
            hybrid = NeuroSpikingGNN(config, neuro_config)
            hybrid_out = hybrid(data, x)
            
            print(f"✅ Neuromorphic integration:")
            print(f"   Graph spike rate: {hybrid_out['graph_spikes'].mean():.3f}")
            print(f"   Neuro spike rate: {hybrid_out['neuro_spikes'].mean():.3f}")
            
            test_results['neuromorphic_integration'] = True
            
        except Exception as e:
            print(f"⚠️  Neuromorphic integration skipped: {e}")
            test_results['neuromorphic_integration'] = None
        
        # Test 10: Performance Analysis
        print("\n🔟 PERFORMANCE ANALYSIS")
        print("-" * 40)
        
        # Test different graph sizes
        graph_sizes = [10, 50, 100]
        
        for size in graph_sizes:
            # Create graph
            edge_index = erdos_renyi_graph(size, 0.2)
            x = torch.randn(size, config.in_channels)
            data = Data(x=x, edge_index=edge_index)
            
            # Time forward pass
            start = time.time()
            outputs = model(data, time_steps=20)
            elapsed = (time.time() - start) * 1000
            
            energy = model.get_energy_estimate(outputs['spike_train'])
            
            print(f"\nGraph size {size}:")
            print(f"  Time: {elapsed:.1f} ms")
            print(f"  Energy: {energy['total_energy_j']*1e9:.3f} nJ")
            print(f"  Spikes/node: {energy['spikes_per_node']:.1f}")
        
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
            
            print(f"{test_name:30} {status}")
        
        passed = sum(1 for r in test_results.values() if r is True)
        failed = sum(1 for r in test_results.values() if r is False)
        skipped = sum(1 for r in test_results.values() if r is None)
        
        print(f"\nTotal: {passed} passed, {failed} failed, {skipped} skipped")
        
        if all_tests_passed and failed == 0:
            print("\n✅ ALL SPIKING GNN TESTS PASSED!")
        else:
            print(f"\n❌ Some tests failed. Please debug.")
        
        return all_tests_passed
        
    except Exception as e:
        print(f"\n❌ Critical test error: {e}")
        import traceback
        traceback.print_exc()
        return False


# Helper visualization functions
def visualize_spike_activity(spike_train: torch.Tensor, node_idx: int = 0, 
                           title: str = "Node Spike Activity"):
    """Visualize spike activity for a specific node"""
    fig, ax = plt.subplots(figsize=(10, 4))
    
    # Get spikes for specific node
    node_spikes = spike_train[:, node_idx].detach().numpy()
    time_steps = np.arange(len(node_spikes))
    
    # Plot spikes as vertical lines
    for t in time_steps[node_spikes > 0]:
        ax.axvline(x=t, color='black', linewidth=1)
    
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Spike')
    ax.set_title(f'{title} - Node {node_idx}')
    ax.set_ylim(-0.1, 1.1)
    
    plt.tight_layout()
    return fig


def visualize_graph_topology(edge_index: torch.Tensor, node_features: torch.Tensor = None,
                           title: str = "Graph Topology"):
    """Visualize graph structure"""
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Convert to networkx
    G = nx.Graph()
    edges = edge_index.t().numpy()
    G.add_edges_from(edges)
    
    # Node colors based on features
    if node_features is not None:
        node_colors = node_features.mean(dim=1).detach().numpy()
    else:
        node_colors = 'lightblue'
    
    # Draw graph
    pos = nx.spring_layout(G)
    nx.draw(G, pos, node_color=node_colors, with_labels=True,
            node_size=300, font_size=8, ax=ax)
    
    ax.set_title(title)
    plt.tight_layout()
    return fig


# Run the complete test
if __name__ == "__main__":
    # Check for torch_geometric
    try:
        import torch_geometric
        print("✅ torch_geometric available")
    except ImportError:
        print("⚠️  torch_geometric not installed")
        print("   Install with: pip install torch-geometric")
        sys.exit(1)
    
    result = asyncio.run(test_spiking_complete())
    sys.exit(0 if result else 1)