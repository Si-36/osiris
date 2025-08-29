#!/usr/bin/env python3
"""
Test chaos engineering experiments with real resilience testing
"""

import asyncio
import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent / "core" / "src"))

print("🌪️ TESTING CHAOS ENGINEERING COMPONENTS")
print("=" * 60)

async def test_chaos():
    """Test chaos engineering experiments"""
    
    # Test 1: Import modules
    print("\n📦 Testing imports...")
    try:
        from aura_intelligence.chaos.experiments import (
            ChaosExperiment,
            LatencyChaosExperiment,
            ErrorRateChaosExperiment,
            ResourceStressChaosExperiment,
            NetworkPartitionChaosExperiment,
            ChaosOrchestrator,
            ExperimentConfig,
            ChaosType,
            ImpactLevel
        )
        print("✅ All imports successful")
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return
    
    # Test 2: Create experiment configurations
    print("\n🔧 Testing Experiment Configuration...")
    try:
        # Low-impact latency test
        latency_config = ExperimentConfig(
            name="test_latency",
            chaos_type=ChaosType.LATENCY,
            impact_level=ImpactLevel.LOW,
            duration_seconds=5,  # Short duration for testing
            parameters={"delay_ms": 50},
            dry_run=True,  # Dry run for safety
            success_criteria={"max_recovery_seconds": 5}
        )
        print(f"✅ Latency config: {latency_config.name} ({latency_config.impact_level.name})")
        
        # Error injection test
        error_config = ExperimentConfig(
            name="test_errors",
            chaos_type=ChaosType.ERROR,
            impact_level=ImpactLevel.LOW,
            duration_seconds=3,
            parameters={"error_rate": 0.05},
            dry_run=True
        )
        print(f"✅ Error config: {error_config.name} (5% error rate)")
        
        # CPU stress test
        cpu_config = ExperimentConfig(
            name="test_cpu_stress",
            chaos_type=ChaosType.RESOURCE,
            impact_level=ImpactLevel.LOW,
            duration_seconds=3,
            parameters={"resource": "cpu", "stress_level": 0.2},
            dry_run=False  # Actually run this one
        )
        print(f"✅ CPU stress config: {cpu_config.name} (20% stress)")
    except Exception as e:
        print(f"❌ Configuration failed: {e}")
        return
    
    # Test 3: Create experiments
    print("\n🧪 Testing Experiment Creation...")
    try:
        latency_exp = LatencyChaosExperiment(latency_config)
        print("✅ Latency experiment created")
        
        error_exp = ErrorRateChaosExperiment(error_config)
        print("✅ Error rate experiment created")
        
        cpu_exp = ResourceStressChaosExperiment(cpu_config)
        print("✅ Resource stress experiment created")
    except Exception as e:
        print(f"❌ Experiment creation failed: {e}")
        return
    
    # Test 4: Run a dry-run experiment
    print("\n🏃 Testing Dry-Run Experiment...")
    try:
        result = await latency_exp.run()
        print(f"✅ Dry run completed: {result.experiment_name}")
        print(f"   - Success: {result.success}")
        print(f"   - Duration: {(result.end_time - result.start_time).total_seconds():.2f}s")
        print(f"   - Observations: {result.observations}")
    except Exception as e:
        print(f"❌ Dry run failed: {e}")
    
    # Test 5: Run actual CPU stress test
    print("\n💪 Testing Real CPU Stress Experiment...")
    try:
        print("   Starting CPU stress (20% for 3 seconds)...")
        result = await cpu_exp.run()
        print(f"✅ CPU stress completed")
        print(f"   - Success: {result.success}")
        print(f"   - CPU before: {result.metrics_before.get('cpu_percent', 0):.1f}%")
        print(f"   - CPU during: {result.metrics_during.get('cpu_percent', 0):.1f}%")
        print(f"   - CPU after: {result.metrics_after.get('cpu_percent', 0):.1f}%")
        print(f"   - Recovery time: {result.recovery_time_seconds:.2f}s")
    except Exception as e:
        print(f"❌ CPU stress failed: {e}")
    
    # Test 6: Test orchestrator
    print("\n🎭 Testing Chaos Orchestrator...")
    try:
        orchestrator = ChaosOrchestrator()
        
        # Register experiments
        orchestrator.register_experiment(latency_exp)
        orchestrator.register_experiment(error_exp)
        print(f"✅ Registered {len(orchestrator.experiments)} experiments")
        
        # Run a scenario
        print("   Running chaos scenario...")
        results = await orchestrator.run_scenario(
            ["test_latency", "test_errors"],
            parallel=False
        )
        print(f"✅ Scenario completed with {len(results)} experiments")
        
        # Generate report
        report = orchestrator.generate_report()
        print(f"✅ Report generated:")
        print(f"   - Total experiments: {report['total_experiments']}")
        print(f"   - Success rate: {report['success_rate']*100:.1f}%")
        print(f"   - Avg recovery: {report['average_recovery_seconds']:.2f}s")
    except Exception as e:
        print(f"❌ Orchestrator test failed: {e}")
    
    # Test 7: Test network partition experiment
    print("\n🌐 Testing Network Partition Experiment...")
    try:
        partition_config = ExperimentConfig(
            name="test_partition",
            chaos_type=ChaosType.NETWORK,
            impact_level=ImpactLevel.MEDIUM,
            duration_seconds=2,
            parameters={"targets": ["service1:8080", "service2:8081"]},
            dry_run=True
        )
        
        partition_exp = NetworkPartitionChaosExperiment(partition_config)
        result = await partition_exp.run()
        print(f"✅ Network partition test completed (dry run)")
    except Exception as e:
        print(f"❌ Network partition test failed: {e}")
    
    # Test 8: Test resilience patterns
    print("\n🛡️ Testing Resilience Patterns...")
    try:
        # Test auto-rollback
        rollback_config = ExperimentConfig(
            name="test_rollback",
            chaos_type=ChaosType.ERROR,
            impact_level=ImpactLevel.HIGH,
            duration_seconds=1,
            auto_rollback=True,
            dry_run=True
        )
        
        rollback_exp = ErrorRateChaosExperiment(rollback_config)
        result = await rollback_exp.run()
        print(f"✅ Auto-rollback test completed")
        
        # Test success criteria evaluation
        if not result.errors:
            print("✅ Success criteria evaluation working")
    except Exception as e:
        print(f"❌ Resilience pattern test failed: {e}")
    
    print("\n" + "=" * 60)
    print("CHAOS ENGINEERING TEST COMPLETE")
    
# Run the test
if __name__ == "__main__":
    asyncio.run(test_chaos())