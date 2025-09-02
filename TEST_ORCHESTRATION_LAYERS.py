#!/usr/bin/env python3
"""
🎼 Test Hierarchical Orchestration
===================================

Tests the restored 3-layer orchestration system.
"""

import sys
import asyncio
sys.path.insert(0, 'core/src')

async def test_orchestration():
    """Test hierarchical orchestration layers"""
    print("Testing Hierarchical Orchestration System...")
    print("="*50)
    
    from aura_intelligence.orchestration.hierarchical_orchestrator import (
        HierarchicalOrchestrator,
        OrchestrationLayer,
        LayerRequest,
        create_hierarchical_orchestrator,
    )
    
    # Test 1: Layer Definitions
    print("\n1. Testing Orchestration Layers...")
    layers = [
        OrchestrationLayer.STRATEGIC,
        OrchestrationLayer.TACTICAL,
        OrchestrationLayer.OPERATIONAL,
    ]
    
    for layer in layers:
        print(f"   ✅ {layer.value} layer available")
    
    # Test 2: Create Orchestrator
    print("\n2. Creating Hierarchical Orchestrator...")
    try:
        orchestrator = await create_hierarchical_orchestrator()
        print(f"   ✅ Orchestrator created: {type(orchestrator).__name__}")
    except Exception as e:
        print(f"   ❌ Failed to create: {e}")
        return
    
    # Test 3: Start Orchestrator
    print("\n3. Starting Orchestrator...")
    try:
        await orchestrator.start()
        print(f"   ✅ Orchestrator started")
    except Exception as e:
        print(f"   ❌ Failed to start: {e}")
    
    # Test 4: Submit Strategic Request
    print("\n4. Testing Strategic Layer...")
    try:
        request = LayerRequest(
            request_id="test-001",
            layer=OrchestrationLayer.STRATEGIC,
            operation="plan_strategy",
            data={"goal": "optimize_system"},
            priority=1
        )
        
        result = await orchestrator.submit_request(request)
        print(f"   ✅ Strategic request processed: {result}")
    except Exception as e:
        print(f"   ❌ Strategic failed: {e}")
    
    # Test 5: Submit Tactical Request
    print("\n5. Testing Tactical Layer...")
    try:
        request = LayerRequest(
            request_id="test-002",
            layer=OrchestrationLayer.TACTICAL,
            operation="coordinate_agents",
            data={"agents": ["agent1", "agent2"]},
            priority=2
        )
        
        result = await orchestrator.submit_request(request)
        print(f"   ✅ Tactical request processed: {result}")
    except Exception as e:
        print(f"   ❌ Tactical failed: {e}")
    
    # Test 6: Submit Operational Request
    print("\n6. Testing Operational Layer...")
    try:
        request = LayerRequest(
            request_id="test-003",
            layer=OrchestrationLayer.OPERATIONAL,
            operation="execute_task",
            data={"task": "process_data"},
            priority=3
        )
        
        result = await orchestrator.submit_request(request)
        print(f"   ✅ Operational request processed: {result}")
    except Exception as e:
        print(f"   ❌ Operational failed: {e}")
    
    # Test 7: Test Escalation
    print("\n7. Testing Layer Escalation...")
    try:
        # Submit request that needs escalation
        request = LayerRequest(
            request_id="test-004",
            layer=OrchestrationLayer.OPERATIONAL,
            operation="complex_decision",
            data={"complexity": "high"},
            priority=1
        )
        
        result = await orchestrator.submit_request(request)
        print(f"   ✅ Escalation handled: {result}")
    except Exception as e:
        print(f"   ❌ Escalation failed: {e}")
    
    # Test 8: Get System Status
    print("\n8. Getting System Status...")
    try:
        status = await orchestrator.get_system_status()
        print(f"   ✅ System Status:")
        for layer, info in status.items():
            print(f"      - {layer}: {info}")
    except Exception as e:
        print(f"   ❌ Status failed: {e}")
    
    # Test 9: Stop Orchestrator
    print("\n9. Stopping Orchestrator...")
    try:
        await orchestrator.stop()
        print(f"   ✅ Orchestrator stopped")
    except Exception as e:
        print(f"   ❌ Failed to stop: {e}")
    
    print("\n" + "="*50)
    print("Orchestration Test Complete!")

if __name__ == "__main__":
    asyncio.run(test_orchestration())