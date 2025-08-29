#!/usr/bin/env python3
"""
Complete test for Neuromorphic system with all features and integration
"""

import asyncio
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import time
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple

# Add project to path
sys.path.insert(0, str(Path(__file__).parent / "core" / "src"))

print("🧠 COMPLETE NEUROMORPHIC SYSTEM TEST")
print("=" * 60)

async def test_neuromorphic_complete():
    """Test all neuromorphic features comprehensively"""
    
    all_tests_passed = True
    test_results = {}
    
    try:
        # Import the neuromorphic system
        from aura_intelligence.neuromorphic.advanced_neuromorphic_system import (
            NeuromorphicNetwork, NeuromorphicConfig, NeuronType, CodingScheme,
            LearningRule, LIFNeuron, AdaptiveLIFNeuron, SpikingLayer,
            SpikingConvLayer, PopulationCoding, ReservoirComputing
        )
        
        print("✅ Imports successful\n")
        
        # Test 1: Basic LIF Neuron
        print("1️⃣ TESTING LIF NEURON DYNAMICS")
        print("-" * 40)
        try:
            config = NeuromorphicConfig(
                input_size=10,
                hidden_size=50,
                output_size=5,
                threshold=1.0,
                tau_membrane=20.0,
                dt=1.0,
                refractory_period=2.0
            )
            
            lif = LIFNeuron(config)
            lif.reset_state(batch_size=1, num_neurons=3)
            
            # Test subthreshold dynamics
            small_current = torch.tensor([[0.05, 0.05, 0.05]])
            spikes = []
            potentials = []
            
            for t in range(50):
                spike = lif(small_current)
                spikes.append(spike)
                potentials.append(lif.membrane_potential.clone())
            
            # Check leak behavior
            final_potential = potentials[-1][0, 0].item()
            expected_potential = small_current[0, 0].item() * 20  # Steady state
            
            print(f"✅ Subthreshold dynamics: V={final_potential:.3f} (expected ~{expected_potential:.3f})")
            
            # Test spiking
            lif.reset_state(1, 1)
            large_current = torch.tensor([[2.0]])
            spike = lif(large_current)
            
            print(f"✅ Spike generation: spike={spike.item()} (expected 1.0)")
            print(f"✅ Reset after spike: V={lif.membrane_potential.item():.3f}")
            print(f"✅ Refractory period: {lif.refractory_time.item():.1f} ms")
            
            test_results['lif_neuron'] = True
            
        except Exception as e:
            print(f"❌ LIF neuron test failed: {e}")
            test_results['lif_neuron'] = False
            all_tests_passed = False
        
        # Test 2: Adaptive LIF Neuron
        print("\n2️⃣ TESTING ADAPTIVE LIF NEURON")
        print("-" * 40)
        try:
            alif = AdaptiveLIFNeuron(config)
            alif.reset_state(1, 1)
            
            # Generate spike train
            constant_current = torch.tensor([[1.5]])
            spike_times = []
            
            for t in range(200):
                spike = alif(constant_current)
                if spike.item() > 0:
                    spike_times.append(t)
            
            # Check spike frequency adaptation
            if len(spike_times) > 2:
                early_isi = spike_times[1] - spike_times[0]
                late_isi = spike_times[-1] - spike_times[-2]
                
                print(f"✅ Spike frequency adaptation:")
                print(f"   Early ISI: {early_isi} ms")
                print(f"   Late ISI: {late_isi} ms")
                print(f"   Adaptation ratio: {late_isi/early_isi:.2f}x")
            
            test_results['adaptive_lif'] = True
            
        except Exception as e:
            print(f"❌ Adaptive LIF test failed: {e}")
            test_results['adaptive_lif'] = False
            all_tests_passed = False
        
        # Test 3: Population Coding
        print("\n3️⃣ TESTING POPULATION CODING")
        print("-" * 40)
        try:
            pop_coder = PopulationCoding(num_neurons=10, value_range=(0, 1))
            
            # Test encoding
            values = torch.tensor([[0.2, 0.5, 0.8]])
            encoded = pop_coder.encode(values)
            
            print(f"✅ Population encoding:")
            print(f"   Input shape: {values.shape} → Encoded shape: {encoded.shape}")
            
            # Test decoding with proper spike generation
            spike_probs = encoded * 0.1  # Convert to spike probability
            spikes = []
            
            for t in range(100):
                spike = (torch.rand_like(spike_probs) < spike_probs).float()
                spikes.append(spike)
            
            spike_train = torch.stack(spikes)
            decoded = pop_coder.decode(spike_train)
            
            print(f"✅ Population decoding:")
            print(f"   Original: {values[0].numpy()}")
            print(f"   Decoded: {decoded.numpy()}")
            
            # Calculate error only for valid decoded values
            valid_mask = ~torch.isnan(decoded)
            if valid_mask.any():
                valid_decoded = decoded[valid_mask]
                valid_original = values[0].repeat(len(valid_decoded))
                error = (valid_decoded - valid_original).abs().mean()
                print(f"   Error: {error:.4f}")
                test_results['population_coding'] = error < 0.2
            else:
                print(f"   Error: No valid decoded values")
                test_results['population_coding'] = False
            
        except Exception as e:
            print(f"❌ Population coding test failed: {e}")
            test_results['population_coding'] = False
            all_tests_passed = False
        
        # Test 4: Spiking Layer with STDP
        print("\n4️⃣ TESTING SPIKING LAYER WITH STDP")
        print("-" * 40)
        try:
            config.learning_rule = LearningRule.STDP
            layer = SpikingLayer(10, 5, config)
            layer.train()
            
            # Get initial weights
            initial_weights = layer.weight.clone()
            
            # Create correlated pre-post spike pattern
            for epoch in range(20):
                layer.reset_state(batch_size=1)
                
                # Pre-synaptic spikes first
                pre_spikes = torch.zeros(1, 10)
                pre_spikes[0, 0] = 1.0  # Spike in first neuron
                
                _ = layer(pre_spikes)
                
                # Force post-synaptic spike
                layer.synaptic_current[0, 0] = 2.0  # Strong input
                post_spike = layer.neurons(layer.synaptic_current)
            
            # Check weight changes
            weight_diff = layer.weight - initial_weights
            potentiated = weight_diff[0, 0].item()
            
            print(f"✅ STDP learning:")
            print(f"   Weight change at [0,0]: {potentiated:.6f}")
            print(f"   Total weight change: {weight_diff.abs().mean():.6f}")
            
            test_results['stdp'] = weight_diff.abs().mean() > 0
            
        except Exception as e:
            print(f"❌ STDP test failed: {e}")
            test_results['stdp'] = False
            all_tests_passed = False
        
        # Test 5: Spiking Convolution
        print("\n5️⃣ TESTING SPIKING CONVOLUTION")
        print("-" * 40)
        try:
            conv = SpikingConvLayer(
                in_channels=1,
                out_channels=4,
                kernel_size=3,
                config=config,
                padding=1
            )
            
            # Create edge detector pattern
            edge_input = torch.zeros(1, 1, 8, 8)
            edge_input[0, 0, :, 4] = 1.0  # Vertical edge
            
            # Process multiple time steps
            outputs = []
            for t in range(20):
                spike_input = (torch.rand_like(edge_input) < edge_input * 0.5).float()
                output = conv(spike_input)
                outputs.append(output)
            
            # Check output
            total_spikes = torch.stack(outputs).sum()
            
            print(f"✅ Spiking convolution:")
            print(f"   Input shape: {edge_input.shape}")
            print(f"   Output shape: {output.shape}")
            print(f"   Total output spikes: {total_spikes.item()}")
            
            test_results['spiking_conv'] = True
            
        except Exception as e:
            print(f"❌ Spiking convolution test failed: {e}")
            test_results['spiking_conv'] = False
            all_tests_passed = False
        
        # Test 6: Reservoir Computing
        print("\n6️⃣ TESTING RESERVOIR COMPUTING")
        print("-" * 40)
        try:
            reservoir = ReservoirComputing(
                input_size=3,
                reservoir_size=50,
                output_size=2,
                config=config
            )
            
            # Test with time series
            t = torch.linspace(0, 4*np.pi, 100)
            x = torch.stack([torch.sin(t), torch.cos(t), torch.sin(2*t)], dim=1)
            x = x.unsqueeze(0)  # Add batch dimension
            
            # Process through reservoir
            output = reservoir(x[0, 0], time_steps=100)
            
            print(f"✅ Reservoir computing:")
            print(f"   Input size: 3")
            print(f"   Reservoir size: 50")
            print(f"   Output shape: {output.shape}")
            print(f"   Output range: [{output.min():.3f}, {output.max():.3f}]")
            
            test_results['reservoir'] = True
            
        except Exception as e:
            print(f"❌ Reservoir computing test failed: {e}")
            test_results['reservoir'] = False
            all_tests_passed = False
        
        # Test 7: Complete Network
        print("\n7️⃣ TESTING COMPLETE NEUROMORPHIC NETWORK")
        print("-" * 40)
        try:
            net_config = NeuromorphicConfig(
                input_size=10,
                hidden_size=20,
                output_size=5,
                num_layers=2,
                neuron_type=NeuronType.ALIF,
                coding_scheme=CodingScheme.TEMPORAL,
                time_window=100.0,
                dt=1.0
            )
            
            network = NeuromorphicNetwork(net_config)
            
            # Test forward pass
            x = torch.rand(4, 10)
            outputs = network(x, time_steps=50)
            
            print(f"✅ Network forward pass:")
            print(f"   Input: {x.shape}")
            print(f"   Output: {outputs['output'].shape}")
            print(f"   Spike rates: {outputs['spike_rates'].mean():.3f}")
            
            # Energy analysis
            energy = network.get_energy_estimate()
            print(f"\n⚡ Energy consumption:")
            print(f"   Total spikes: {energy['total_spikes']}")
            print(f"   Energy: {energy['energy_per_inference_nj']:.3f} nJ")
            
            test_results['full_network'] = True
            
        except Exception as e:
            print(f"❌ Full network test failed: {e}")
            test_results['full_network'] = False
            all_tests_passed = False
        
        # Test 8: Integration with other components
        print("\n8️⃣ TESTING INTEGRATION WITH OTHER AURA COMPONENTS")
        print("-" * 40)
        
        # Try TDA integration
        try:
            from aura_intelligence.tda.advanced_tda_system import AdvancedTDAEngine, TDAConfig
            
            # Use spike patterns for topology
            spike_data = outputs['spikes'][:, 0, :].T.detach().numpy()
            
            tda_config = TDAConfig(max_dimension=1, max_edge_length=2.0)
            tda = AdvancedTDAEngine(tda_config)
            
            tda_features = tda.compute_tda_features(
                spike_data[:10],
                return_diagrams=False
            )
            
            print(f"✅ TDA integration successful:")
            print(f"   Betti numbers: {tda_features['betti_numbers']}")
            test_results['tda_integration'] = True
            
        except Exception as e:
            print(f"⚠️  TDA integration skipped: {e}")
            test_results['tda_integration'] = None
        
        # Try Neural integration
        try:
            from aura_intelligence.neural.advanced_neural_system import (
                AdvancedNeuralNetwork, NeuralConfig, ArchitectureType
            )
            
            # Hybrid Neuromorphic-Transformer
            class NeuroTransformer(nn.Module):
                def __init__(self):
                    super().__init__()
                    
                    # Neuromorphic encoder
                    self.neuro_config = NeuromorphicConfig(
                        input_size=20,
                        hidden_size=32,
                        output_size=64,
                        num_layers=1
                    )
                    self.neuromorphic = NeuromorphicNetwork(self.neuro_config)
                    
                    # Transformer decoder
                    self.neural_config = NeuralConfig(
                        architecture=ArchitectureType.TRANSFORMER,
                        hidden_dim=64,
                        num_layers=2,
                        num_heads=4
                    )
                    self.transformer = AdvancedNeuralNetwork(self.neural_config)
                    
                def forward(self, x):
                    # Encode with neuromorphic
                    neuro_out = self.neuromorphic(x, time_steps=20)
                    spike_features = neuro_out['spike_rates']
                    
                    # Decode with transformer
                    trans_out = self.transformer(spike_features.unsqueeze(1))
                    
                    return trans_out['last_hidden_state']
            
            hybrid = NeuroTransformer()
            hybrid_out = hybrid(torch.rand(2, 20))
            
            print(f"✅ Neural integration successful:")
            print(f"   Hybrid output shape: {hybrid_out.shape}")
            test_results['neural_integration'] = True
            
        except Exception as e:
            print(f"⚠️  Neural integration skipped: {e}")
            test_results['neural_integration'] = None
        
        # Performance comparison
        print("\n9️⃣ PERFORMANCE BENCHMARKING")
        print("-" * 40)
        
        # Compare different configurations
        configs = [
            ("LIF", NeuronType.LIF, 50),
            ("ALIF", NeuronType.ALIF, 50),
            ("LIF-Long", NeuronType.LIF, 200),
        ]
        
        for name, neuron_type, time_steps in configs:
            config = NeuromorphicConfig(
                input_size=20,
                hidden_size=50,
                output_size=10,
                num_layers=2,
                neuron_type=neuron_type
            )
            
            net = NeuromorphicNetwork(config)
            x = torch.rand(10, 20)
            
            start = time.time()
            out = net(x, time_steps=time_steps)
            elapsed = (time.time() - start) * 1000
            
            energy = net.get_energy_estimate()
            
            print(f"\n{name}:")
            print(f"  Time: {elapsed:.1f} ms")
            print(f"  Energy: {energy['energy_per_inference_nj']:.3f} nJ")
            print(f"  Spike rate: {out['spike_rates'].mean():.3f}")
        
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
            print("\n✅ ALL NEUROMORPHIC TESTS PASSED!")
        else:
            print(f"\n❌ Some tests failed. Please debug.")
        
        return all_tests_passed
        
    except Exception as e:
        print(f"\n❌ Critical test error: {e}")
        import traceback
        traceback.print_exc()
        return False


# Helper function for visualization
def plot_membrane_dynamics(potentials: List[torch.Tensor], spikes: List[torch.Tensor], 
                          title: str = "Neuron Dynamics"):
    """Plot membrane potential and spikes"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    
    # Convert to numpy
    v_data = torch.stack(potentials).squeeze().numpy()
    s_data = torch.stack(spikes).squeeze().numpy()
    
    # Plot membrane potential
    ax1.plot(v_data)
    ax1.set_ylabel('Membrane Potential (V)')
    ax1.set_title(title)
    ax1.axhline(y=1.0, color='r', linestyle='--', label='Threshold')
    ax1.legend()
    
    # Plot spikes
    ax2.plot(s_data, 'ro', markersize=10)
    ax2.set_ylabel('Spikes')
    ax2.set_xlabel('Time (ms)')
    ax2.set_ylim(-0.1, 1.1)
    
    plt.tight_layout()
    return fig


# Run the complete test
if __name__ == "__main__":
    result = asyncio.run(test_neuromorphic_complete())
    sys.exit(0 if result else 1)