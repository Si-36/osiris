#!/usr/bin/env python3
"""
Real benchmark test connecting to actual AURA components
"""

import asyncio
import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent / "core" / "src"))

print("🚀 RUNNING REAL BENCHMARK TESTS")
print("=" * 60)

async def test_benchmarks():
    """Test benchmarks with real components"""
    
    # Import benchmark module
    try:
        from aura_intelligence.benchmarks.workflow_benchmarks import WorkflowBenchmark, run_all_benchmarks
        print("✅ Benchmark module imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import benchmark module: {e}")
        return
    
    # Test individual benchmark functions
    benchmark = WorkflowBenchmark(output_dir="test_benchmark_results")
    
    # Test 1: TDA Workflow Benchmark
    print("\n📊 Testing TDA Workflow Benchmark...")
    try:
        tda_summary = await benchmark.benchmark_tda_workflow(iterations=5)
        print(f"✅ TDA Benchmark: {tda_summary.mean_duration_ms:.2f}ms avg, {tda_summary.success_rate*100:.1f}% success")
    except Exception as e:
        print(f"❌ TDA Benchmark failed: {e}")
    
    # Test 2: LNN Inference Benchmark
    print("\n📊 Testing LNN Inference Benchmark...")
    try:
        lnn_summary = await benchmark.benchmark_lnn_inference(iterations=5)
        print(f"✅ LNN Benchmark: {lnn_summary.mean_duration_ms:.2f}ms avg, {lnn_summary.success_rate*100:.1f}% success")
    except Exception as e:
        print(f"❌ LNN Benchmark failed: {e}")
    
    # Test 3: Memory Operations
    print("\n📊 Testing Memory Operations Benchmark...")
    try:
        memory_summary = await benchmark.benchmark_memory_operations(iterations=5)
        print(f"✅ Memory Benchmark: {memory_summary.mean_duration_ms:.2f}ms avg, {memory_summary.mean_throughput:.2f} ops/sec")
    except Exception as e:
        print(f"❌ Memory Benchmark failed: {e}")
    
    # Test 4: Swarm Coordination
    print("\n📊 Testing Swarm Coordination Benchmark...")
    try:
        swarm_summary = await benchmark.benchmark_swarm_coordination(agents=5, iterations=3)
        print(f"✅ Swarm Benchmark: {swarm_summary.mean_duration_ms:.2f}ms avg, {swarm_summary.mean_throughput:.2f} agents/sec")
    except Exception as e:
        print(f"❌ Swarm Benchmark failed: {e}")
    
    # Test 5: Full Pipeline
    print("\n📊 Testing Full Pipeline Benchmark...")
    try:
        pipeline_summary = await benchmark.benchmark_full_pipeline(iterations=3)
        print(f"✅ Pipeline Benchmark: {pipeline_summary.mean_duration_ms:.2f}ms avg")
    except Exception as e:
        print(f"❌ Pipeline Benchmark failed: {e}")
    
    # Test component connections
    print("\n🔗 Testing Component Connections...")
    
    # Try importing actual components
    components_status = {}
    
    try:
        from aura_intelligence.core.topology import TopologyEngine
        components_status['TDA'] = "✅ Available"
    except ImportError:
        components_status['TDA'] = "❌ Not Available"
    
    try:
        from aura_intelligence.lnn.core import LiquidNeuralNetwork
        components_status['LNN'] = "✅ Available"
    except ImportError:
        components_status['LNN'] = "❌ Not Available"
    
    try:
        from aura_intelligence.swarm_intelligence.collective import SwarmIntelligence
        components_status['Swarm'] = "✅ Available"
    except ImportError:
        components_status['Swarm'] = "❌ Not Available"
    
    try:
        from aura_intelligence.memory.hybrid_manager import HybridMemoryManager
        components_status['Memory'] = "✅ Available"
    except ImportError:
        components_status['Memory'] = "❌ Not Available"
    
    print("\nComponent Status:")
    for component, status in components_status.items():
        print(f"  {component}: {status}")
    
    # Run performance comparison
    print("\n📈 Running Performance Comparison...")
    try:
        # Create a mini benchmark run
        all_summaries = []
        if await benchmark.benchmark_memory_operations(iterations=3):
            all_summaries.append(await benchmark.benchmark_memory_operations(iterations=3))
        if await benchmark.benchmark_swarm_coordination(agents=3, iterations=2):
            all_summaries.append(await benchmark.benchmark_swarm_coordination(agents=3, iterations=2))
        
        if all_summaries:
            benchmark.plot_results(all_summaries)
            print("✅ Performance plots generated")
    except Exception as e:
        print(f"❌ Performance comparison failed: {e}")
    
    print("\n" + "=" * 60)
    print("BENCHMARK TEST COMPLETE")
    
# Run the test
if __name__ == "__main__":
    asyncio.run(test_benchmarks())