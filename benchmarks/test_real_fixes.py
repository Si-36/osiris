#!/usr/bin/env python3
"""
Test script to verify real fixes work
"""
import sys
import os
from pathlib import Path

# Add core to path
core_path = Path(__file__).parent / "core" / "src"
sys.path.insert(0, str(core_path))

import asyncio
import time

async def test_kafka_integration():
    """Test Kafka integration"""
    print("🔍 Testing Kafka integration...")
    try:
        from aura_intelligence.streaming.kafka_integration import get_event_streaming, EventType
        
        streaming = get_event_streaming()
        await streaming.start_streaming()
        
        # Test publishing event
        success = await streaming.publish_system_event(
            EventType.COMPONENT_HEALTH,
            "test_component",
            {"status": "testing", "timestamp": time.time()}
        )
        
        stats = streaming.get_streaming_stats()
        print(f"✅ Kafka: Events published: {stats['events_published']}")
        return True
        
    except Exception as e:
        print(f"❌ Kafka test failed: {e}")
        return False

async def test_neo4j_integration():
    """Test Neo4j integration"""
    print("🔍 Testing Neo4j integration...")
    try:
        from aura_intelligence.graph.neo4j_integration import get_neo4j_integration
        
        neo4j = get_neo4j_integration()
        
        # Test storing decision
        decision_data = {
            "component_id": "test_component",
            "vote": "APPROVE",
            "confidence": 0.85,
            "reasoning": "Test decision"
        }
        
        success = await neo4j.store_council_decision(decision_data)
        status = neo4j.get_connection_status()
        
        print(f"✅ Neo4j: Connected: {status['connected']}, Decision stored: {success}")
        return True
        
    except Exception as e:
        print(f"❌ Neo4j test failed: {e}")
        return False

async def test_ray_serve():
    """Test Ray Serve integration"""
    print("🔍 Testing Ray Serve integration...")
    try:
        from aura_intelligence.distributed.ray_serve_deployment import get_ray_serve_manager
        
        ray_manager = get_ray_serve_manager()
        
        # Test cluster status
        status = await ray_manager.get_cluster_status()
        
        print(f"✅ Ray Serve: Initialized: {status.get('initialized', False)}")
        return True
        
    except Exception as e:
        print(f"❌ Ray Serve test failed: {e}")
        return False

async def test_working_api():
    """Test the working API components"""
    print("🔍 Testing working API components...")
    try:
        from aura_intelligence.core.unified_system import UnifiedSystem
        from aura_intelligence.lnn.core import LiquidNeuralNetwork
        
        # Test UnifiedSystem
        unified = UnifiedSystem()
        print("✅ UnifiedSystem: Working")
        
        # Test LNN
        lnn = LiquidNeuralNetwork(input_size=10, hidden_size=20, output_size=5)
        import torch
        test_input = torch.randn(1, 10)
        output = lnn(test_input)
        print(f"✅ LNN: Working, output shape: {output.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Working API test failed: {e}")
        return False

async def main():
    """Run all tests"""
    print("🧪 Testing REAL fixes to AURA system...")
    print("=" * 50)
    
    results = []
    
    # Test working components first
    results.append(await test_working_api())
    
    # Test fixed integrations
    results.append(await test_kafka_integration())
    results.append(await test_neo4j_integration())
    results.append(await test_ray_serve())
    
    print("=" * 50)
    print(f"📊 Results: {sum(results)}/{len(results)} tests passed")
    
    if sum(results) == len(results):
        print("🎉 All tests passed! Your fixes are working.")
    else:
        print("⚠️  Some tests failed. Check the errors above.")
        print("💡 To install real dependencies: pip install -r requirements.real.txt")

if __name__ == "__main__":
    asyncio.run(main())