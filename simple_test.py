#!/usr/bin/env python3
"""
Simple test of core AURA components
"""
import sys
from pathlib import Path

# Add core to path
core_path = Path(__file__).parent / "core" / "src"
sys.path.insert(0, str(core_path))

def test_components():
    print("🧪 Testing AURA Core Components...")
    
    # Test component registry
    try:
        from aura_intelligence.components.real_registry import get_real_registry
        registry = get_real_registry()
        print(f"✅ Registry: {len(registry.components)} components loaded")
        
        # Test processing
        result = registry.process_data("neural_001_lnn_processor", {"values": [1,2,3,4,5]})
        print(f"✅ Processing: {result.get('success', 'completed')}")
        
    except Exception as e:
        print(f"❌ Registry test failed: {e}")
    
    # Test Kafka integration
    try:
        from aura_intelligence.streaming.kafka_integration import get_event_streaming, EventType
        streaming = get_event_streaming()
        success = streaming.publish_system_event(
            EventType.COMPONENT_HEALTH,
            "test_component", 
            {"status": "testing"}
        )
        print(f"✅ Kafka: Event published: {success}")
        
    except Exception as e:
        print(f"❌ Kafka test failed: {e}")
    
    # Test Neo4j integration
    try:
        from aura_intelligence.graph.neo4j_integration import get_neo4j_integration
        neo4j = get_neo4j_integration()
        status = neo4j.get_connection_status()
        print(f"✅ Neo4j: Connected: {status['connected']}")
        
    except Exception as e:
        print(f"❌ Neo4j test failed: {e}")

if __name__ == "__main__":
    test_components()