"""
Basic Memory System Test - Validates Core Functionality
"""

import asyncio
import time
import numpy as np

async def test_basic_memory():
    """Test basic memory operations"""
    print("\n=== Testing AURA Memory System ===\n")
    
    # Import memory system
    from core.src.aura_intelligence.memory.core.memory_api import (
        AURAMemorySystem,
        MemoryType,
        RetrievalMode,
        MemoryQuery
    )
    
    # Create memory system
    print("1. Creating memory system...")
    memory = AURAMemorySystem()
    
    # Test workflow data
    workflow = {
        "workflow_id": "test_1",
        "agents": [
            {"id": "agent_1"},
            {"id": "agent_2"}, 
            {"id": "agent_3"}
        ],
        "dependencies": [
            {"source": "agent_1", "target": "agent_2"},
            {"source": "agent_2", "target": "agent_3"},
            {"source": "agent_3", "target": "agent_1"}  # Create cycle
        ]
    }
    
    # Store memory
    print("2. Storing topological memory...")
    memory_id = await memory.store(
        content={"test": "data"},
        memory_type=MemoryType.TOPOLOGICAL,
        workflow_data=workflow
    )
    print(f"   Stored: {memory_id}")
    
    # Retrieve by topology
    print("3. Retrieving by topology...")
    query = MemoryQuery(
        mode=RetrievalMode.SHAPE_MATCH,
        topology_constraints={"min_loops": 1}
    )
    
    results = await memory.retrieve(query)
    print(f"   Found {len(results.memories)} memories")
    print(f"   Retrieval time: {results.retrieval_time_ms:.2f}ms")
    
    # Test failure prediction
    print("4. Testing failure prediction...")
    prediction = await memory.predict_workflow_failure(workflow)
    print(f"   Failure probability: {prediction['failure_probability']:.2%}")
    
    print("\n✅ Basic test complete!")


if __name__ == "__main__":
    asyncio.run(test_basic_memory())