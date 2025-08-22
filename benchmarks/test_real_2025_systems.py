#!/usr/bin/env python3
"""
COMPREHENSIVE TEST OF REAL 2025 SYSTEMS
Tests all the newly implemented real systems vs the old mocks
"""

import asyncio
import time
import sys
from pathlib import Path

# Add to path
sys.path.insert(0, str(Path(__file__).parent / "core" / "src"))

async def test_real_distributed_system():
    """Test real Ray distributed system"""
    print("🚀 Testing Real Ray Distributed System...")
    
    try:
        from aura_intelligence.distributed.real_ray_system import get_real_ray_system
        
        system = get_real_ray_system()
        
        test_data = {
            'values': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            'points': [[1, 2], [3, 4], [5, 6], [7, 8]]
        }
        
        result = await system.process_distributed(test_data)
        
        print(f"  ✅ Distributed processing: {result['actors_used']} actors")
        print(f"  ⏱️  Processing time: {result['total_processing_time_ms']:.2f}ms")
        print(f"  🖥️  Ray available: {result['ray_available']}")
        
        # Validate real processing
        real_results = [r for r in result['distributed_results'].values() 
                       if r.get('real_processing', False)]
        print(f"  🔥 Real processing results: {len(real_results)}")
        
        return len(real_results) > 0
        
    except Exception as e:
        print(f"  ❌ Ray system failed: {e}")
        return False

async def test_real_constitutional_ai():
    """Test real Constitutional AI 3.0 system"""
    print("🧠 Testing Real Constitutional AI 3.0...")
    
    try:
        from aura_intelligence.dpo.real_constitutional_ai import get_real_constitutional_ai
        
        system = get_real_constitutional_ai()
        
        # Train the model
        training_result = system.train(epochs=3)
        
        print(f"  ✅ Training completed: {training_result['training_completed']}")
        print(f"  📊 Constitutional AI version: {training_result['constitutional_ai_version']}")
        print(f"  🎯 DPO enabled: {training_result['dpo_enabled']}")
        
        # Test preference evaluation
        evaluation = system.evaluate_constitutional_preference(
            prompt="How should AI behave?",
            response_a="AI should be helpful, harmless, and follow constitutional principles.",
            response_b="AI should do whatever it wants without any constraints."
        )
        
        print(f"  🎯 Preferred response: {evaluation['preferred_response']}")
        print(f"  🛡️  Constitutional override: {evaluation.get('override_reason', 'None')}")
        print(f"  ✅ Both responses safe: {evaluation['both_responses_safe']}")
        
        return training_result['training_completed'] and evaluation['constitutional_ai_version'] == '3.0'
        
    except Exception as e:
        print(f"  ❌ Constitutional AI failed: {e}")
        return False

async def test_real_hybrid_memory():
    """Test real hybrid memory system"""
    print("💾 Testing Real Hybrid Memory System...")
    
    try:
        from aura_intelligence.memory_tiers.real_hybrid_memory import get_real_hybrid_memory
        
        memory = get_real_hybrid_memory()
        
        # Test storage across tiers
        test_data = {
            'small_item': 'small data',
            'medium_item': 'medium data' * 100,
            'large_item': 'large data' * 1000
        }
        
        # Store items
        storage_results = []
        for key, data in test_data.items():
            result = await memory.store(key, data)
            storage_results.append(result)
            print(f"  📦 Stored {key} in tier: {result['tier']}")
        
        # Retrieve items
        retrieval_results = []
        for key in test_data.keys():
            result = await memory.retrieve(key)
            retrieval_results.append(result)
            if result['found']:
                print(f"  📥 Retrieved {key} from tier: {result['tier']}")
        
        # Get memory stats
        stats = memory.get_memory_stats()
        print(f"  📊 Memory tiers active: {len(stats['tier_usage'])}")
        print(f"  🎯 Hit rates available: {'hit_rates' in stats}")
        
        memory.shutdown()
        
        all_stored = all(r['stored'] for r in storage_results)
        all_found = all(r['found'] for r in retrieval_results)
        
        return all_stored and all_found
        
    except Exception as e:
        print(f"  ❌ Hybrid memory failed: {e}")
        return False

async def test_real_pearl_inference():
    """Test real PEARL inference engine"""
    print("⚡ Testing Real PEARL Inference Engine...")
    
    try:
        from aura_intelligence.advanced_processing.real_pearl_inference import get_real_pearl_engine, PEARLConfig
        
        config = PEARLConfig(
            draft_length_adaptive=True,
            pre_verify_enabled=True,
            post_verify_enabled=True,
            energy_efficiency_mode=True
        )
        
        engine = get_real_pearl_engine(config)
        
        # Test inference
        input_tokens = [1, 2, 3, 4, 5]  # Simple token sequence
        
        result = await engine.pearl_inference(input_tokens)
        
        print(f"  ✅ Tokens generated: {len(result['tokens'])}")
        print(f"  📏 Draft length: {result['draft_length']}")
        print(f"  🎯 Acceptance rate: {result['acceptance_rate']:.3f}")
        print(f"  ⚡ Speedup: {result['speedup']:.2f}x")
        print(f"  🔋 Energy efficiency: {result['energy_efficiency']:.2f}x")
        print(f"  🚀 PEARL version: {result['pearl_version']}")
        
        # Get performance stats
        stats = engine.get_performance_stats()
        print(f"  📊 Total inferences: {stats['total_inferences']}")
        
        return (len(result['tokens']) > 0 and 
                result['pearl_version'] == '2025' and
                result['energy_efficiency'] > 1.0)
        
    except Exception as e:
        print(f"  ❌ PEARL inference failed: {e}")
        return False

async def test_system_integration():
    """Test integration between real systems"""
    print("🔗 Testing Real System Integration...")
    
    try:
        # Import all systems
        from aura_intelligence.distributed.real_ray_system import get_real_ray_system
        from aura_intelligence.memory_tiers.real_hybrid_memory import get_real_hybrid_memory
        from aura_intelligence.advanced_processing.real_pearl_inference import get_real_pearl_engine
        
        # Initialize systems
        ray_system = get_real_ray_system()
        memory_system = get_real_hybrid_memory()
        pearl_engine = get_real_pearl_engine()
        
        # Test data flow between systems
        test_data = {'values': [0.1, 0.2, 0.3, 0.4, 0.5]}
        
        # 1. Process with Ray
        ray_result = await ray_system.process_distributed(test_data)
        
        # 2. Store result in hybrid memory
        memory_result = await memory_system.store('ray_result', ray_result)
        
        # 3. Use PEARL for inference
        pearl_result = await pearl_engine.pearl_inference([1, 2, 3, 4, 5])
        
        # 4. Store PEARL result
        pearl_memory_result = await memory_system.store('pearl_result', pearl_result)
        
        # 5. Retrieve both results
        retrieved_ray = await memory_system.retrieve('ray_result')
        retrieved_pearl = await memory_system.retrieve('pearl_result')
        
        print(f"  ✅ Ray processing: {ray_result['ray_available']}")
        print(f"  ✅ Memory storage: {memory_result['stored']}")
        print(f"  ✅ PEARL inference: {len(pearl_result['tokens'])} tokens")
        print(f"  ✅ Memory retrieval: {retrieved_ray['found'] and retrieved_pearl['found']}")
        
        memory_system.shutdown()
        
        return (ray_result['actors_used'] > 0 and
                memory_result['stored'] and
                len(pearl_result['tokens']) > 0 and
                retrieved_ray['found'] and
                retrieved_pearl['found'])
        
    except Exception as e:
        print(f"  ❌ System integration failed: {e}")
        return False

async def main():
    """Run comprehensive test suite"""
    print("🚀 COMPREHENSIVE REAL 2025 SYSTEMS TEST")
    print("=" * 60)
    
    tests = [
        ("Real Ray Distributed System", test_real_distributed_system),
        ("Real Constitutional AI 3.0", test_real_constitutional_ai),
        ("Real Hybrid Memory System", test_real_hybrid_memory),
        ("Real PEARL Inference Engine", test_real_pearl_inference),
        ("Real System Integration", test_system_integration)
    ]
    
    results = {}
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n🧪 {test_name}")
        print("-" * 40)
        
        try:
            start_time = time.perf_counter()
            result = await test_func()
            test_time = (time.perf_counter() - start_time) * 1000
            
            results[test_name] = {
                'passed': result,
                'time_ms': test_time
            }
            
            if result:
                passed += 1
                print(f"✅ PASSED ({test_time:.2f}ms)")
            else:
                print(f"❌ FAILED ({test_time:.2f}ms)")
                
        except Exception as e:
            results[test_name] = {
                'passed': False,
                'error': str(e),
                'time_ms': 0
            }
            print(f"💥 CRASHED: {e}")
    
    print("\n" + "=" * 60)
    print(f"📊 FINAL RESULTS: {passed}/{total} TESTS PASSED")
    print(f"🏆 SUCCESS RATE: {passed/total*100:.1f}%")
    
    if passed == total:
        print("🎉 ALL REAL SYSTEMS WORKING - NO MORE MOCKS!")
    elif passed >= total * 0.8:
        print("✅ MOSTLY REAL - Minor issues to fix")
    elif passed >= total * 0.5:
        print("⚠️  PARTIALLY REAL - Major work needed")
    else:
        print("❌ STILL MOSTLY FAKE - System needs complete overhaul")
    
    # Detailed results
    print("\n📋 Detailed Results:")
    for test_name, result in results.items():
        status = "✅ PASS" if result['passed'] else "❌ FAIL"
        time_str = f"{result['time_ms']:.2f}ms" if 'time_ms' in result else "N/A"
        print(f"  {status} {test_name} ({time_str})")
        if 'error' in result:
            print(f"    Error: {result['error']}")
    
    return passed, total

if __name__ == "__main__":
    asyncio.run(main())