#!/usr/bin/env python3
"""
🧠 TEST FIXED NEURAL NETWORK
============================

Test if the neural network tensor issue is fixed.
"""

import sys
from pathlib import Path
import torch

# Add paths
core_path = Path(__file__).parent / "core" / "src"
sys.path.insert(0, str(core_path))

def test_neural_network():
    """Test the fixed neural network"""
    
    print("🧠 TESTING FIXED NEURAL NETWORK")
    print("=" * 40)
    
    try:
        from aura_intelligence.lnn.core import LiquidNeuralNetwork
        
        print("✅ Import successful")
        
        # Create neural network
        lnn = LiquidNeuralNetwork(input_size=10, output_size=10)
        print(f"✅ Neural network created: {lnn.total_params} parameters")
        
        # Test forward pass
        test_input = torch.randn(1, 10)
        print(f"📊 Input shape: {test_input.shape}")
        
        with torch.no_grad():
            output = lnn.forward(test_input)
        
        print(f"📊 Output shape: {output.shape}")
        print(f"📊 Output sample: {output[0][:3].tolist()}")
        
        if output.shape == (1, 10):
            print("✅ NEURAL NETWORK FIXED AND WORKING!")
            return True
        else:
            print(f"❌ Wrong output shape: expected (1, 10), got {output.shape}")
            return False
            
    except Exception as e:
        print(f"❌ Neural network test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_neural_network()
    if success:
        print("\n🎉 Neural network is now working!")
    else:
        print("\n🔧 Neural network still needs fixes")