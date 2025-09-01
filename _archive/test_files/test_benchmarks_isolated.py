#!/usr/bin/env python3
"""
Test benchmarks in isolation without broken dependencies
"""

import asyncio
import sys
import os
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent / "core" / "src"))

# Mock the broken imports
class MockModule:
    def __getattr__(self, name):
        return lambda *args, **kwargs: None

# Replace broken imports
sys.modules['aura_intelligence.core.topology'] = MockModule()
sys.modules['aura_intelligence.lnn.core'] = MockModule()
sys.modules['aura_intelligence.swarm_intelligence.collective'] = MockModule()
sys.modules['aura_intelligence.core.consciousness'] = MockModule()
sys.modules['aura_intelligence.memory.hybrid_manager'] = MockModule()

print("🚀 RUNNING ISOLATED BENCHMARK TESTS")
print("=" * 60)

async def test_benchmarks():
    """Test benchmarks in isolation"""
    
    # Import benchmark module
    try:
        from aura_intelligence.benchmarks.workflow_benchmarks import WorkflowBenchmark
        print("✅ Benchmark module imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import benchmark module: {e}")
        return
    
    # Create benchmark instance
    benchmark = WorkflowBenchmark(output_dir="test_benchmark_results")
    print("✅ Benchmark instance created")
    
    # Test 1: Memory Operations (should work without dependencies)
    print("\n📊 Testing Memory Operations Benchmark...")
    try:
        memory_summary = await benchmark.benchmark_memory_operations(iterations=10)
        print(f"✅ Memory Benchmark Results:")
        print(f"   - Mean Duration: {memory_summary.mean_duration_ms:.2f}ms")
        print(f"   - Std Duration: {memory_summary.std_duration_ms:.2f}ms")
        print(f"   - Min/Max: {memory_summary.min_duration_ms:.2f}ms / {memory_summary.max_duration_ms:.2f}ms")
        print(f"   - Mean Throughput: {memory_summary.mean_throughput:.2f} ops/sec")
        print(f"   - Success Rate: {memory_summary.success_rate*100:.1f}%")
        
        # Save results
        benchmark.save_results(memory_summary)
    except Exception as e:
        print(f"❌ Memory Benchmark failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 2: Swarm Coordination
    print("\n📊 Testing Swarm Coordination Benchmark...")
    try:
        swarm_summary = await benchmark.benchmark_swarm_coordination(agents=5, iterations=10)
        print(f"✅ Swarm Benchmark Results:")
        print(f"   - Mean Duration: {swarm_summary.mean_duration_ms:.2f}ms")
        print(f"   - Mean Throughput: {swarm_summary.mean_throughput:.2f} agents/sec")
        print(f"   - Success Rate: {swarm_summary.success_rate*100:.1f}%")
        
        # Save results
        benchmark.save_results(swarm_summary)
    except Exception as e:
        print(f"❌ Swarm Benchmark failed: {e}")
    
    # Test 3: Full Pipeline
    print("\n📊 Testing Full Pipeline Benchmark...")
    try:
        pipeline_summary = await benchmark.benchmark_full_pipeline(iterations=5)
        print(f"✅ Pipeline Benchmark Results:")
        print(f"   - Mean Duration: {pipeline_summary.mean_duration_ms:.2f}ms")
        print(f"   - Success Rate: {pipeline_summary.success_rate*100:.1f}%")
        
        # Save results
        benchmark.save_results(pipeline_summary)
    except Exception as e:
        print(f"❌ Pipeline Benchmark failed: {e}")
    
    # Test 4: Mock component benchmarks
    print("\n📊 Testing Mock Component Benchmarks...")
    try:
        # TDA benchmark (will use mock)
        tda_summary = await benchmark.benchmark_tda_workflow(iterations=10)
        print(f"✅ TDA Benchmark (Mock) Results:")
        print(f"   - Mean Duration: {tda_summary.mean_duration_ms:.2f}ms")
        
        # LNN benchmark (will use mock)
        lnn_summary = await benchmark.benchmark_lnn_inference(iterations=10)
        print(f"✅ LNN Benchmark (Mock) Results:")
        print(f"   - Mean Duration: {lnn_summary.mean_duration_ms:.2f}ms")
    except Exception as e:
        print(f"❌ Mock benchmarks failed: {e}")
    
    # Test 5: Generate comparison plots
    print("\n📈 Generating Performance Comparison...")
    try:
        all_summaries = [memory_summary, swarm_summary, pipeline_summary, tda_summary, lnn_summary]
        benchmark.plot_results(all_summaries)
        print("✅ Performance plots generated")
    except Exception as e:
        print(f"❌ Plot generation failed: {e}")
    
    print("\n" + "=" * 60)
    print("BENCHMARK TEST COMPLETE")
    print(f"Results saved to: test_benchmark_results/")
    
    # List generated files
    if os.path.exists("test_benchmark_results"):
        files = list(Path("test_benchmark_results").glob("*"))
        if files:
            print("\nGenerated files:")
            for f in files:
                print(f"  - {f.name}")
    
# Run the test
if __name__ == "__main__":
    # Ensure output directory exists
    Path("test_benchmark_results").mkdir(exist_ok=True)
    
    asyncio.run(test_benchmarks())