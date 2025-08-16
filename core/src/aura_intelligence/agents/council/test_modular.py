#!/usr/bin/env python3
"""
Quick test for modular LNN Council Agent (2025 Architecture)
"""

import asyncio
import sys
import os

# Add path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))

from core_agent import LNNCouncilAgent
from config import LNNCouncilConfig
from models import GPUAllocationRequest


async def test_modular_agent():
    """Test the modular agent implementation."""
    print("🧪 Testing Modular LNN Council Agent (2025)")
    
    # Create config
    config = LNNCouncilConfig(
        name="modular_test_agent",
        input_size=64,
        output_size=16,
        hidden_sizes=[32, 16],
        use_gpu=False
    )
    
    print(f"✅ Config created: {config.name}")
    
    # Create agent
    agent = LNNCouncilAgent(config)
    print(f"✅ Agent created: {agent.name}")
    
    # Create request
    request = GPUAllocationRequest(
        user_id="test_user",
        project_id="test_project",
        gpu_type="A100",
        gpu_count=2,
        memory_gb=40,
        compute_hours=8.0,
        priority=7
    )
    
    print(f"✅ Request created: {request.request_id}")
    
    # Test workflow
    try:
        decision = await agent.process(request)
        
        print(f"✅ Decision made: {decision.decision}")
        print(f"   Confidence: {decision.confidence_score:.3f}")
        print(f"   Inference Time: {decision.inference_time_ms:.1f}ms")
        print(f"   Fallback Used: {decision.fallback_used}")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = asyncio.run(test_modular_agent())
    print(f"\n{'🎉 SUCCESS' if success else '❌ FAILED'}")
    sys.exit(0 if success else 1)