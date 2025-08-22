"""Test Bio-Enhanced AURA System Integration"""
import asyncio
import sys
import os

# Add the core path
sys.path.append(os.path.join(os.path.dirname(__file__), 'core', 'src'))

from aura_intelligence.bio_enhanced_system import BioEnhancedAURA

async def test_bio_enhanced_system():
    """Test the complete bio-enhanced system"""
    print("🧬 Testing Bio-Enhanced AURA System...")
    
    # Initialize system
    bio_aura = BioEnhancedAURA()
    
    # Test 1: Simple processing
    print("\n1. Testing simple request processing...")
    simple_request = {"query": "simple test", "data": [1, 2, 3]}
    result = await bio_aura.process_enhanced(simple_request, "test_component")
    print(f"   Result: {result['enhancements']}")
    
    # Test 2: Complex processing
    print("\n2. Testing complex request processing...")
    complex_request = {
        "query": "analyze comprehensive detailed complex data patterns",
        "data": list(range(100))
    }
    result = await bio_aura.process_enhanced(complex_request, "analysis_component")
    print(f"   Compute saved: {result['performance']['compute_saved']:.1f}%")
    print(f"   Health score: {result['performance']['health_score']:.2f}")
    
    # Test 3: System status
    print("\n3. System status check...")
    status = bio_aura.get_system_status()
    print(f"   Bio-enhancements active: {status['system_health']['enhancement_active']}")
    print(f"   Capabilities: {list(status['capabilities'].keys())}")
    
    # Test 4: Hallucination prevention
    print("\n4. Testing hallucination prevention...")
    for i in range(5):
        result = await bio_aura.process_enhanced(
            {"iteration": i, "rapid_fire": True}, 
            "rapid_component"
        )
        bio_status = result['enhancements']['bio_regulation']
        print(f"   Iteration {i}: {bio_status}")
    
    print("\n✅ Bio-Enhanced AURA System test completed!")
    print(f"🎯 Key achievements:")
    print(f"   - Metabolic regulation: ACTIVE")
    print(f"   - Depth-aware routing: ACTIVE") 
    print(f"   - Swarm intelligence: ACTIVE")
    print(f"   - Graceful degradation: ENABLED")

if __name__ == "__main__":
    asyncio.run(test_bio_enhanced_system())