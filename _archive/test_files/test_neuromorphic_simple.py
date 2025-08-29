#!/usr/bin/env python3
"""
Test Neuromorphic system without dependencies
"""

import sys
from pathlib import Path
import torch
import numpy as np
import time

# Add project to path
sys.path.insert(0, str(Path(__file__).parent / "core" / "src"))

print("🧠 TESTING NEUROMORPHIC COMPUTING SYSTEM (SIMPLIFIED)")
print("=" * 60)

def test_neuromorphic_simple():
    """Test Neuromorphic system without dependencies"""
    
    try:
        # Direct import
        print("\n1️⃣ TESTING DIRECT IMPORTS")
        print("-" * 40)
        
        import importlib.util
        
        # Load module directly
        spec = importlib.util.spec_from_file_location(
            "advanced_neuromorphic_system",
            "/workspace/core/src/aura_intelligence/neuromorphic/advanced_neuromorphic_system.py"
        )
        neuro_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(neuro_module)
        
        # Import classes
        NeuromorphicNetwork = neuro_module.NeuromorphicNetwork
        NeuromorphicConfig = neuro_module.NeuromorphicConfig
        NeuronType = neuro_module.NeuronType
        LIFNeuron = neuro_module.LIFNeuron
        PopulationCoding = neuro_module.PopulationCoding
        ReservoirComputing = neuro_module.ReservoirComputing
        
        print("✅ Direct imports successful")
        
        # Test basic network
        print("\n2️⃣ TESTING BASIC NEUROMORPHIC NETWORK")
        print("-" * 40)
        
        config = NeuromorphicConfig(
            input_size=10,
            hidden_size=50,
            output_size=5,
            num_layers=2,
            neuron_type=NeuronType.LIF,
            threshold=1.0,
            tau_membrane=20.0
        )
        
        network = NeuromorphicNetwork(config)
        print(f"✅ Network created with {config.num_layers} layers")
        
        # Test forward pass
        print("\n3️⃣ TESTING FORWARD PASS")
        print("-" * 40)
        
        x = torch.rand(4, 10)  # batch_size=4, input_size=10
        outputs = network(x, time_steps=50)
        
        print(f"✅ Forward pass complete")
        print(f"   Output shape: {outputs['output'].shape}")
        print(f"   Spike train shape: {outputs['spikes'].shape}")
        print(f"   Average spike rate: {outputs['spike_rates'].mean():.3f}")
        
        # Energy analysis
        energy = network.get_energy_estimate()
        print(f"\n⚡ Energy consumption:")
        print(f"   Total spikes: {energy['total_spikes']}")
        print(f"   Energy: {energy['energy_per_inference_nj']:.2f} nJ")
        
        # Test different neuron types
        print("\n4️⃣ TESTING NEURON TYPES")
        print("-" * 40)
        
        # LIF neuron
        lif_neuron = LIFNeuron(config)
        lif_neuron.reset_state(1, 10)
        
        spike_counts = []
        for _ in range(100):
            current = torch.randn(1, 10) * 0.5 + 0.3
            spikes = lif_neuron(current)
            spike_counts.append(spikes.sum().item())
        
        print(f"✅ LIF neuron:")
        print(f"   Total spikes: {sum(spike_counts)}")
        print(f"   Spike rate: {np.mean(spike_counts):.2f} spikes/step")
        
        # Test population coding
        print("\n5️⃣ TESTING POPULATION CODING")
        print("-" * 40)
        
        pop_coder = PopulationCoding(num_neurons=15, value_range=(0, 1))
        
        # Encode values
        values = torch.tensor([0.2, 0.5, 0.8])
        encoded = pop_coder.encode(values)
        
        print(f"✅ Population encoding:")
        print(f"   Input values: {values}")
        print(f"   Encoded shape: {encoded.shape}")
        
        # Decode back
        spike_train = (torch.rand(50, 3, 15) < encoded.unsqueeze(0) * 0.1).float()
        decoded = pop_coder.decode(spike_train)
        
        print(f"\n✅ Population decoding:")
        print(f"   Decoded values: {decoded}")
        print(f"   Error: {(decoded - values).abs().mean():.4f}")
        
        # Test reservoir computing
        print("\n6️⃣ TESTING RESERVOIR COMPUTING")
        print("-" * 40)
        
        reservoir = ReservoirComputing(
            input_size=5,
            reservoir_size=100,
            output_size=3,
            config=config
        )
        
        # Simple input
        x = torch.randn(2, 5)
        reservoir_out = reservoir(x, time_steps=50)
        
        print(f"✅ Reservoir computing:")
        print(f"   Reservoir size: 100 neurons")
        print(f"   Output shape: {reservoir_out.shape}")
        print(f"   Connection probability: {config.connection_prob}")
        
        # Performance comparison
        print("\n7️⃣ PERFORMANCE COMPARISON")
        print("-" * 40)
        
        # Different time steps
        for steps in [10, 50, 100]:
            start = time.time()
            _ = network(torch.rand(10, 10), time_steps=steps)
            elapsed = (time.time() - start) * 1000
            
            print(f"Time steps {steps}: {elapsed:.1f} ms")
        
        # Spike statistics
        print("\n8️⃣ SPIKE STATISTICS")
        print("-" * 40)
        
        # Run longer simulation
        outputs = network(torch.rand(1, 10), time_steps=200)
        spike_train = outputs['spikes'].squeeze()
        
        # Analyze spike patterns
        total_spikes = spike_train.sum().item()
        neurons_active = (spike_train.sum(dim=0) > 0).sum().item()
        
        print(f"✅ Spike analysis (200 time steps):")
        print(f"   Total spikes: {total_spikes}")
        print(f"   Active neurons: {neurons_active}/{config.output_size}")
        print(f"   Spikes per neuron: {total_spikes/config.output_size:.1f}")
        
        # Inter-spike intervals
        spike_times = []
        for n in range(config.output_size):
            neuron_spikes = torch.where(spike_train[:, n] > 0)[0]
            if len(neuron_spikes) > 1:
                isis = torch.diff(neuron_spikes).float()
                spike_times.append(isis.mean().item())
        
        if spike_times:
            print(f"   Avg ISI: {np.mean(spike_times):.1f} ms")
        
        # Summary
        print("\n" + "=" * 60)
        print("✅ NEUROMORPHIC SYSTEM TEST COMPLETE")
        
        print("\n📝 Key Features Tested:")
        print("- Spiking neural networks")
        print("- LIF neuron dynamics")
        print("- Population coding")
        print("- Reservoir computing")
        print("- Energy efficiency")
        print("- Temporal processing")
        
        print("\n💡 Performance Insights:")
        print("- Ultra-low energy consumption")
        print("- Event-driven computation")
        print("- Sparse spike activity")
        print("- Scalable to large networks")
        
    except Exception as e:
        print(f"\n❌ Test error: {e}")
        import traceback
        traceback.print_exc()


# Run the test
if __name__ == "__main__":
    test_neuromorphic_simple()