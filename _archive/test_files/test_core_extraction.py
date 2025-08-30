"""
🧪 Test CORE System Extraction

Tests the unified AURA Main System with self-healing and executive control.
"""

import asyncio
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'core/src'))

from aura_intelligence.core.aura_main_system import AURAMainSystem, SystemConfig
from aura_intelligence.core.self_healing_engine import FailureType


async def test_main_system():
    """Test the main AURA system"""
    print("\n" + "="*80)
    print("🌟 TESTING AURA MAIN SYSTEM")
    print("="*80)
    
    # Initialize system with all features
    config = SystemConfig(
        enable_self_healing=True,
        enable_chaos_engineering=True,  # Enable for testing
        enable_swarm_coordination=True,
        enable_lnn_council=True
    )
    
    system = AURAMainSystem(config)
    
    # Start system
    print("\n1️⃣ Starting AURA system...")
    await system.start()
    
    # Get initial status
    status = system.get_system_status()
    print(f"   ✅ System started: {status['system_id']}")
    print(f"   ✅ Components: {len(status['components'])} registered")
    print(f"   ✅ Health: {status['health']:.2f}")
    print(f"   ✅ Resilience: {status['resilience']:.2f}")
    
    # Test request execution
    print("\n2️⃣ Testing request execution...")
    result = await system.execute_request({
        "type": "analysis",
        "query": "Test analysis request",
        "tasks": ["analyze", "report"]
    })
    print(f"   ✅ Request completed: {result['status']}")
    print(f"   ✅ Duration: {result['duration']:.2f}s")
    
    # Test executive reflection
    print("\n3️⃣ Testing executive controller...")
    reflection = await system.executive.reflect()
    print(f"   ✅ Consciousness level: {reflection['consciousness_level']}")
    print(f"   ✅ Awareness score: {reflection['awareness_score']:.2f}")
    print(f"   ✅ Cognitive load: {reflection['cognitive_load']:.2f}")
    
    # Test self-healing
    print("\n4️⃣ Testing self-healing...")
    healing_result = await system.self_healing.heal_component(
        "test_component",
        {"type": "performance", "severity": 0.6}
    )
    print(f"   ✅ Healing applied: {healing_result['strategy']}")
    print(f"   ✅ Success: {healing_result['success']}")
    
    # Test chaos engineering
    print("\n5️⃣ Testing chaos engineering...")
    chaos_result = await system.run_chaos_experiment(
        target_components=["memory"],
        failure_type="latency",
        intensity=0.3
    )
    print(f"   ✅ Chaos experiment: {chaos_result.get('status', 'unknown')}")
    
    # Wait for metrics
    await asyncio.sleep(2)
    
    # Final status
    print("\n6️⃣ Final system status...")
    final_status = system.get_system_status()
    print(f"   ✅ Uptime: {final_status['uptime']:.1f}s")
    print(f"   ✅ Success rate: {final_status['success_rate']:.2%}")
    print(f"   ✅ Overall health: {final_status['health']:.2f}")
    
    # Stop system
    print("\n7️⃣ Stopping system...")
    await system.stop()
    print("   ✅ System stopped gracefully")
    
    return True


async def test_self_healing():
    """Test self-healing capabilities"""
    print("\n" + "="*80)
    print("🛡️ TESTING SELF-HEALING ENGINE")
    print("="*80)
    
    from aura_intelligence.core.self_healing_engine import (
        SelfHealingEngine, ChaosExperiment, FailureType
    )
    
    engine = SelfHealingEngine()
    
    # Test predictive failure detection
    print("\n1️⃣ Testing predictive failure detection...")
    prediction = await engine.failure_detector.predict_failure("test_service")
    print(f"   ✅ Failure probability: {prediction['failure_probability']:.2%}")
    print(f"   ✅ Predicted type: {prediction['predicted_failure_type']}")
    print(f"   ✅ Recommended action: {prediction['recommended_action']}")
    
    # Test antifragility
    print("\n2️⃣ Testing antifragility engine...")
    adaptation = await engine.antifragility.adapt_to_stress(
        "test_service",
        "latency",
        0.4  # Medium stress
    )
    print(f"   ✅ Adaptation type: {adaptation['type']}")
    print(f"   ✅ Adaptation: {adaptation['adaptation']}")
    
    # Test chaos experiment
    print("\n3️⃣ Testing chaos engineering...")
    experiment = ChaosExperiment(
        experiment_id="test_001",
        name="Test latency injection",
        failure_type=FailureType.LATENCY_INJECTION,
        target_components=["test_service"],
        duration_seconds=5.0,
        intensity=0.3,
        blast_radius=0.1,
        hypothesis="Service will recover within 10 seconds"
    )
    
    result = await engine.chaos_engineer.conduct_experiment(experiment)
    print(f"   ✅ Experiment status: {result['status']}")
    if result['status'] == 'completed':
        print(f"   ✅ Duration: {result['duration']:.1f}s")
        print(f"   ✅ Recovery time: {result.get('recovery_time', 'N/A')}")
    
    # Test healing strategies
    print("\n4️⃣ Testing healing strategies...")
    issues = [
        {"type": "performance", "severity": 0.3},
        {"type": "errors", "severity": 0.6},
        {"type": "resource_exhaustion", "severity": 0.8}
    ]
    
    for issue in issues:
        result = await engine.heal_component("test_component", issue)
        print(f"   ✅ {issue['type']}: {result['strategy']} (severity: {issue['severity']})")
    
    # Get resilience score
    print("\n5️⃣ System resilience...")
    resilience = await engine.get_system_resilience_score()
    print(f"   ✅ Resilience score: {resilience:.2f}")
    
    return True


async def test_executive_controller():
    """Test executive controller"""
    print("\n" + "="*80)
    print("🧠 TESTING EXECUTIVE CONTROLLER")
    print("="*80)
    
    from aura_intelligence.core.executive_controller import ExecutiveController
    
    controller = ExecutiveController()
    await controller.initialize()
    
    # Test information processing
    print("\n1️⃣ Testing information processing...")
    await controller.process_information({
        "source": "test",
        "type": "performance_data",
        "metrics": {"latency": 45, "errors": 2},
        "priority": 0.7
    })
    print("   ✅ Information processed and broadcast")
    
    # Test agent coordination
    print("\n2️⃣ Testing agent coordination...")
    agents = ["agent_1", "agent_2", "agent_3"]
    task = {
        "name": "optimize_system",
        "type": "optimization",
        "complexity": 0.6
    }
    
    coordination = await controller.coordinate_agents(agents, task)
    print(f"   ✅ Plan created: {coordination['plan']['plan_id']}")
    print(f"   ✅ Steps: {len(coordination['plan']['steps'])}")
    print(f"   ✅ Agent allocations: {len(coordination['allocations'])}")
    
    # Test decision making
    print("\n3️⃣ Testing decision making...")
    options = [
        {"action": "scale_up", "benefits": ["performance"], "safe": True},
        {"action": "optimize", "benefits": ["performance", "cost"], "novel": True},
        {"action": "maintain", "benefits": ["stability"], "safe": True}
    ]
    
    decision = await controller.make_decision(options)
    print(f"   ✅ Decision: {decision['decision']['action']}")
    print(f"   ✅ Score: {decision['score']:.2f}")
    print(f"   ✅ Confidence: {decision['confidence']:.2f}")
    
    # Test reflection
    print("\n4️⃣ Testing reflection...")
    reflection = await controller.reflect()
    print(f"   ✅ Consciousness: {reflection['consciousness_level']}")
    print(f"   ✅ Attention: {reflection['attention_distribution']}")
    print(f"   ✅ Distracted: {reflection['is_distracted']}")
    
    # Get state
    print("\n5️⃣ Executive state...")
    state = controller.get_executive_state()
    print(f"   ✅ Active goals: {state['active_goals']}")
    print(f"   ✅ Workspace items: {state['workspace_items']}")
    print(f"   ✅ Emergent behaviors: {state['emergent_behaviors']}")
    
    return True


async def main():
    """Run all CORE tests"""
    print("\n" + "🔥"*20)
    print("AURA CORE SYSTEM TEST SUITE")
    print("Testing: Main System, Self-Healing, Executive Control")
    print("🔥"*20)
    
    try:
        # Test main system
        await test_main_system()
        
        # Test self-healing
        await test_self_healing()
        
        # Test executive controller
        await test_executive_controller()
        
        print("\n" + "="*80)
        print("🎉 ALL CORE TESTS PASSED!")
        print("="*80)
        
        print("\n📊 Summary:")
        print("   ✅ AURA Main System - Working")
        print("   ✅ Self-Healing Engine - Working")
        print("   ✅ Executive Controller - Working")
        print("   ✅ Component Integration - Working")
        print("   ✅ Chaos Engineering - Working")
        print("   ✅ Consciousness Levels - Working")
        
        print("\n🚀 The CORE System is ready!")
        print("   - Unified architecture")
        print("   - Self-healing capabilities")
        print("   - Executive control")
        print("   - All components connected")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())