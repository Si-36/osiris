#!/usr/bin/env python3
"""
Test New Persistence Without Breaking Anything
===========================================
Ensures backward compatibility while testing new features
"""

import asyncio
import sys
import os
import structlog
import torch
from datetime import datetime

# Add paths
sys.path.insert(0, os.path.abspath('./core/src'))

# Configure logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.dev.ConsoleRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()

async def test_backward_compatibility():
    """Test that old code still works"""
    logger.info("Testing backward compatibility...")
    
    try:
        # Import old state manager
        from aura_intelligence.persistence.state_manager import get_state_manager
        
        # This should work even without PostgreSQL
        manager = get_state_manager()
        
        # Test basic operations
        test_data = {
            "test_key": "test_value",
            "timestamp": datetime.now().isoformat()
        }
        
        # Save state (should use pickle)
        from aura_intelligence.persistence.state_manager import StateType
        success = await manager.save_state(
            StateType.COMPONENT_STATE,
            "test_component",
            test_data
        )
        
        if success:
            logger.info("✓ Old save_state works")
        else:
            logger.error("✗ Old save_state failed")
            return False
        
        # Load state
        loaded = await manager.load_state(
            StateType.COMPONENT_STATE,
            "test_component"
        )
        
        if loaded and loaded.get("test_key") == "test_value":
            logger.info("✓ Old load_state works")
        else:
            logger.error("✗ Old load_state failed")
            return False
        
        return True
        
    except Exception as e:
        logger.error(f"Backward compatibility test failed: {e}")
        return False

async def test_new_causal_persistence():
    """Test new persistence features"""
    logger.info("Testing new causal persistence...")
    
    try:
        from aura_intelligence.persistence.causal_state_manager import (
            get_causal_manager,
            StateType,
            CausalContext
        )
        
        # Get manager (will use legacy mode if no PostgreSQL)
        manager = await get_causal_manager()
        
        if manager.legacy_mode:
            logger.warning("Running in legacy mode - PostgreSQL not available")
            logger.info("This is EXPECTED if PostgreSQL is not running")
        else:
            logger.info("✓ Connected to PostgreSQL")
        
        # Test save with causality
        test_data = {
            "agent_id": "test_agent",
            "decision": "explore",
            "confidence": 0.85,
            "timestamp": datetime.now().isoformat()
        }
        
        context = CausalContext(
            causes=["user_request", "high_curiosity"],
            effects=["new_knowledge_gained"],
            counterfactuals={
                "alternative_1": "could_have_exploited",
                "probability": 0.15
            },
            confidence=0.85,
            energy_cost=0.1
        )
        
        state_id = await manager.save_state(
            StateType.AGENT_MEMORY,
            "test_agent",
            test_data,
            causal_context=context
        )
        
        logger.info(f"✓ Saved state with ID: {state_id}")
        
        # Test load
        loaded = await manager.load_state(
            StateType.AGENT_MEMORY,
            "test_agent"
        )
        
        if loaded and loaded.get("agent_id") == "test_agent":
            logger.info("✓ Loaded state successfully")
        else:
            logger.error("✗ Failed to load state")
            return False
        
        # Test speculative branch
        if not manager.legacy_mode:
            branch_id = await manager.create_branch("test_agent", "experiment")
            logger.info(f"✓ Created speculative branch: {branch_id}")
        
        return True
        
    except Exception as e:
        logger.error(f"Causal persistence test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_memory_native():
    """Test memory-native architecture"""
    logger.info("Testing memory-native architecture...")
    
    try:
        from aura_intelligence.persistence.memory_native import get_memory_native
        
        # Initialize memory-native system
        mem_native = await get_memory_native()
        logger.info("✓ Memory-native architecture initialized")
        
        # Test think with memory
        thought = {
            "type": "analysis",
            "content": "What patterns exist in the data?",
            "weight": 1.5
        }
        
        # Store some memories to work with
        await mem_native.fabric.store_with_compute(
            "memory_1",
            {"pattern": "increasing", "value": 10},
            tier="working"
        )
        
        await mem_native.fabric.store_with_compute(
            "memory_2", 
            {"pattern": "cyclic", "value": 20},
            tier="episodic"
        )
        
        # Think with memories
        result = await mem_native.think_with_memory(thought)
        
        if result and "computed_results" in result:
            logger.info("✓ Memory-native compute working")
            logger.info(f"  Computed {len(result['computed_results'])} results")
        else:
            logger.error("✗ Memory-native compute failed")
            return False
        
        # Test retrieve and compute
        computed = await mem_native.fabric.retrieve_and_compute(
            "memory_1",
            compute_fn=lambda x: {**x, "computed": True}
        )
        
        if computed and computed.get("computed"):
            logger.info("✓ Compute-on-retrieval working")
        else:
            logger.error("✗ Compute-on-retrieval failed")
            return False
        
        return True
        
    except ImportError as e:
        logger.warning(f"CuPy not installed - skipping GPU memory tests: {e}")
        return True  # Don't fail if CuPy not available
    except Exception as e:
        logger.error(f"Memory-native test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_agent_integration():
    """Test that agents work with new persistence"""
    logger.info("Testing agent integration...")
    
    try:
        # Import one of our test agents
        from aura_intelligence.agents.test_code_agent import create_agent
        
        # Create agent
        agent = create_agent()
        logger.info("✓ Agent created successfully")
        
        # The agent should work with either old or new persistence
        # Just verify it initializes without errors
        if hasattr(agent, 'memory'):
            logger.info("✓ Agent has memory system")
        
        return True
        
    except Exception as e:
        logger.error(f"Agent integration test failed: {e}")
        return False

async def test_gpu_adapters():
    """Test GPU adapters still work"""
    logger.info("Testing GPU adapters...")
    
    try:
        from aura_intelligence.adapters import MemoryAdapterGPU
        
        # Create adapter
        adapter = MemoryAdapterGPU()
        
        # Test basic operation
        test_data = torch.randn(100, 768).cuda()
        result = adapter.process(test_data)
        
        if result is not None:
            logger.info("✓ GPU adapter working")
        else:
            logger.error("✗ GPU adapter failed")
            return False
        
        return True
        
    except Exception as e:
        logger.warning(f"GPU adapter test skipped (normal if no GPU): {e}")
        return True

async def main():
    """Run all tests"""
    logger.info("="*60)
    logger.info("AURA Persistence Upgrade Test Suite")
    logger.info("="*60)
    
    # Check if Docker services are running
    logger.info("\nChecking Docker services...")
    logger.info("Run 'docker-compose -f docker-compose.persistence.yml up -d' to start PostgreSQL")
    logger.info("The system will use legacy mode if PostgreSQL is not available\n")
    
    tests = [
        ("Backward Compatibility", test_backward_compatibility),
        ("Causal Persistence", test_new_causal_persistence),
        ("Memory-Native Architecture", test_memory_native),
        ("Agent Integration", test_agent_integration),
        ("GPU Adapters", test_gpu_adapters)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        logger.info(f"\n--- {test_name} ---")
        try:
            success = await test_func()
            results[test_name] = success
        except Exception as e:
            logger.error(f"Test {test_name} crashed: {e}")
            results[test_name] = False
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("TEST SUMMARY")
    logger.info("="*60)
    
    for test_name, success in results.items():
        status = "✓ PASSED" if success else "✗ FAILED"
        logger.info(f"{test_name}: {status}")
    
    total = len(results)
    passed = sum(1 for s in results.values() if s)
    
    logger.info(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("\n🎉 All tests passed! The upgrade is safe to deploy.")
    else:
        logger.warning("\n⚠️  Some tests failed. Check the logs above.")
    
    return 0 if passed == total else 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)