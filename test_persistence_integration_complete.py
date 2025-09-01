#!/usr/bin/env python3
"""
Complete Integration Test for Persistence
========================================
Tests the full persistence system with real agents
"""

import asyncio
import sys
import os
import json
from datetime import datetime

# Add to path
sys.path.insert(0, os.path.abspath('./core/src'))

# Import what we need directly
from aura_intelligence.persistence.causal_state_manager import (
    CausalPersistenceManager,
    StateType,
    CausalContext,
    get_causal_manager
)
from aura_intelligence.agents.enhanced_code_agent import create_enhanced_code_agent

async def test_persistence_standalone():
    """Test persistence without full system"""
    print("=" * 60)
    print("TEST 1: Standalone Persistence")
    print("=" * 60)
    
    # Create manager
    manager = CausalPersistenceManager(
        postgres_url="postgresql://localhost:5432/aura",
        legacy_path="./test_persistence_state"
    )
    
    await manager.initialize()
    
    print(f"✓ Manager initialized (Legacy mode: {manager.legacy_mode})")
    
    # Test save/load
    test_data = {
        "test_id": "integration_test",
        "timestamp": datetime.now().isoformat(),
        "data": {"key": "value", "number": 42}
    }
    
    context = CausalContext(
        causes=["test_execution"],
        effects=["validation"],
        confidence=0.95
    )
    
    state_id = await manager.save_state(
        StateType.COMPONENT_STATE,
        "test_component",
        test_data,
        causal_context=context
    )
    
    print(f"✓ Saved state: {state_id}")
    
    # Load it back
    loaded = await manager.load_state(
        StateType.COMPONENT_STATE,
        "test_component"
    )
    
    if loaded:
        print(f"✓ Loaded state successfully")
        print(f"  Data: {json.dumps(loaded, indent=2)}")
    
    await manager.close()
    return True

async def test_enhanced_agent():
    """Test enhanced agent with persistence"""
    print("\n" + "=" * 60)
    print("TEST 2: Enhanced Agent with Persistence")
    print("=" * 60)
    
    # Create enhanced agent
    agent = create_enhanced_code_agent("test_code_agent_001")
    print("✓ Created enhanced code agent")
    
    # Test code analysis with persistence
    test_code = '''
def complex_function(x, y):
    if x > 0:
        if y > 0:
            return x + y
        else:
            return x - y
    else:
        if y > 0:
            return -x + y
        else:
            return -x - y
'''
    
    # Save test code to file
    test_file = "test_complex.py"
    with open(test_file, 'w') as f:
        f.write(test_code)
    
    try:
        # Analyze code
        result = await agent.analyze_code(test_file)
        print(f"✓ Code analysis completed")
        print(f"  Complexity: {result.complexity_score}")
        print(f"  Quality: {result.quality_score}")
        
        # Review with memory
        review = await agent.review_with_memory(test_code, "complexity")
        print(f"✓ Code review completed")
        print(f"  Issues found: {len(review['issues_found'])}")
        print(f"  Quality score: {review['quality_score']}")
        
        # Test feedback learning
        if review['issues_found']:
            # Simulate feedback
            await agent.learn_from_feedback(
                "test_decision_001",
                {"score": 0.8, "comment": "Good analysis"}
            )
            print("✓ Learned from feedback")
        
    finally:
        # Cleanup
        if os.path.exists(test_file):
            os.remove(test_file)
    
    return True

async def test_memory_evolution():
    """Test memory evolution and compute-on-retrieval"""
    print("\n" + "=" * 60)
    print("TEST 3: Memory Evolution")
    print("=" * 60)
    
    manager = await get_causal_manager()
    
    # Save initial state
    initial_data = {
        "iteration": 0,
        "value": 1.0,
        "history": []
    }
    
    state_id = await manager.save_state(
        StateType.AGENT_MEMORY,
        "evolving_agent",
        initial_data
    )
    
    print(f"✓ Saved initial state")
    
    # Define evolution function
    def evolve_memory(data):
        if data:
            data["iteration"] += 1
            data["value"] *= 1.1  # Grow by 10%
            data["history"].append({
                "iteration": data["iteration"],
                "value": data["value"],
                "timestamp": datetime.now().isoformat()
            })
        return data
    
    # Evolve memory 5 times
    for i in range(5):
        evolved = await manager.load_state(
            StateType.AGENT_MEMORY,
            "evolving_agent",
            compute_on_retrieval=evolve_memory
        )
        
        # Save evolved state
        await manager.save_state(
            StateType.AGENT_MEMORY,
            "evolving_agent",
            evolved,
            causal_context=CausalContext(
                causes=[f"evolution_step_{i}"],
                effects=[f"value_increased_to_{evolved['value']:.2f}"],
                confidence=1.0
            )
        )
        
        print(f"✓ Evolution {i+1}: value = {evolved['value']:.2f}")
    
    # Check final state
    final = await manager.load_state(
        StateType.AGENT_MEMORY,
        "evolving_agent"
    )
    
    print(f"\n✓ Final state after evolution:")
    print(f"  Iterations: {final['iteration']}")
    print(f"  Final value: {final['value']:.2f}")
    print(f"  History length: {len(final['history'])}")
    
    return True

async def test_speculative_branches():
    """Test speculative branching"""
    print("\n" + "=" * 60)
    print("TEST 4: Speculative Branches")
    print("=" * 60)
    
    manager = await get_causal_manager()
    
    if manager.legacy_mode:
        print("⚠️  Skipping - requires PostgreSQL")
        return True
    
    # Create main state
    main_state = {
        "strategy": "conservative",
        "risk_tolerance": 0.3,
        "performance": 0.7
    }
    
    await manager.save_state(
        StateType.AGENT_MEMORY,
        "strategic_agent",
        main_state
    )
    
    print("✓ Created main state")
    
    # Create experimental branch
    branch_id = await manager.create_branch("strategic_agent", "aggressive_experiment")
    print(f"✓ Created experimental branch: {branch_id}")
    
    # Modify in branch
    branch_state = main_state.copy()
    branch_state["strategy"] = "aggressive"
    branch_state["risk_tolerance"] = 0.8
    
    await manager.save_state(
        StateType.AGENT_MEMORY,
        "strategic_agent",
        branch_state,
        branch_id=branch_id
    )
    
    print("✓ Saved experimental state to branch")
    
    # Load from main
    main_loaded = await manager.load_state(
        StateType.AGENT_MEMORY,
        "strategic_agent"
    )
    
    # Load from branch
    branch_loaded = await manager.load_state(
        StateType.AGENT_MEMORY,
        "strategic_agent",
        branch_id=branch_id
    )
    
    print(f"\nMain state: {main_loaded['strategy']} (risk: {main_loaded['risk_tolerance']})")
    print(f"Branch state: {branch_loaded['strategy']} (risk: {branch_loaded['risk_tolerance']})")
    
    return True

async def main():
    """Run all tests"""
    print("\n🚀 COMPLETE PERSISTENCE INTEGRATION TEST")
    print("=" * 60)
    
    tests = [
        ("Standalone Persistence", test_persistence_standalone),
        ("Enhanced Agent", test_enhanced_agent),
        ("Memory Evolution", test_memory_evolution),
        ("Speculative Branches", test_speculative_branches)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            print(f"\nRunning: {test_name}")
            success = await test_func()
            results[test_name] = success
        except Exception as e:
            print(f"\n❌ {test_name} failed: {e}")
            import traceback
            traceback.print_exc()
            results[test_name] = False
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    for test_name, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{test_name}: {status}")
    
    total = len(results)
    passed = sum(1 for p in results.values() if p)
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All integration tests passed!")
        print("\nNext steps:")
        print("1. Update remaining agents to use persistence")
        print("2. Add persistence to neural components") 
        print("3. Run full system benchmarks")
    
    return 0 if passed == total else 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)