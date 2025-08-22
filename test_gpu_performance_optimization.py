#!/usr/bin/env python3
"""
GPU Performance Optimization Test - Phase 1
Test the optimized BERT component with pre-loaded models
Target: Reduce BERT processing from 421ms to <50ms
"""

import asyncio
import time
import sys
import statistics
from pathlib import Path

# Add to path
sys.path.insert(0, str(Path(__file__).parent / "core" / "src"))

async def test_gpu_performance_optimization():
    """Test GPU performance improvements with model pre-loading"""
    print("🚀 GPU PERFORMANCE OPTIMIZATION TEST")
    print("Target: Reduce BERT processing from 421ms to <50ms")
    print("=" * 60)
    
    # Import components
    from aura_intelligence.components.real_components import (
        RealAttentionComponent,
        model_manager,
        gpu_manager
    )
    
    # Step 1: Initialize global model manager (pre-load models)
    print("\n📥 Step 1: Pre-loading models...")
    start_init = time.perf_counter()
    
    await model_manager.initialize()
    
    init_time = (time.perf_counter() - start_init) * 1000
    print(f"✅ Model pre-loading completed: {init_time:.1f}ms")
    
    # Show model stats
    stats = model_manager.get_model_stats()
    print(f"📊 Pre-loaded models: {stats['preloaded_models']}")
    print(f"🎮 GPU models: {stats['gpu_models']}")
    print(f"📋 Available models: {stats['available_models']}")
    
    # Step 2: Test BERT component performance
    print("\n⚡ Step 2: Testing BERT Performance...")
    
    bert_component = RealAttentionComponent("optimized_bert_test")
    
    # Test data samples
    test_samples = [
        {"text": "Optimized BERT inference performance test"},
        {"text": "GPU acceleration with pre-loaded transformer models"},
        {"text": "Production-ready neural processing with sub-50ms latency"},
        {"text": "Real-time AI inference optimization validation"},
        {"text": "Advanced neural architecture performance benchmarking"}
    ]
    
    # Warmup run (not counted in benchmarks)
    print("🔥 Warmup inference...")
    warmup_result = await bert_component.process({"text": "warmup run"})
    print(f"   Warmup time: {warmup_result.get('processing_time_ms', 0):.2f}ms")
    
    # Performance benchmark runs
    print("\n📈 Performance Benchmark (5 runs)...")
    processing_times = []
    
    for i, sample in enumerate(test_samples, 1):
        start_time = time.perf_counter()
        result = await bert_component.process(sample)
        end_time = time.perf_counter()
        
        # Get both measured times
        total_time = (end_time - start_time) * 1000
        gpu_time = result.get('processing_time_ms', 0)
        
        processing_times.append(gpu_time)
        
        print(f"   Run {i}: {gpu_time:.2f}ms GPU, {total_time:.2f}ms total")
        print(f"         GPU: {result.get('gpu_accelerated', False)}, "
              f"Preloaded: {result.get('preloaded_model', False)}")
    
    # Calculate performance statistics
    avg_time = statistics.mean(processing_times)
    min_time = min(processing_times)
    max_time = max(processing_times)
    std_dev = statistics.stdev(processing_times) if len(processing_times) > 1 else 0
    
    print(f"\n📊 PERFORMANCE STATISTICS:")
    print(f"   Average time: {avg_time:.2f}ms")
    print(f"   Min time: {min_time:.2f}ms")
    print(f"   Max time: {max_time:.2f}ms")
    print(f"   Std deviation: {std_dev:.2f}ms")
    
    # Step 3: Performance validation
    print("\n🎯 PERFORMANCE VALIDATION:")
    
    target_time = 50.0  # 50ms target
    baseline_time = 421.0  # Original baseline
    
    if avg_time <= target_time:
        improvement = ((baseline_time - avg_time) / baseline_time) * 100
        speedup = baseline_time / avg_time
        print(f"   ✅ TARGET ACHIEVED: {avg_time:.1f}ms ≤ {target_time}ms")
        print(f"   🚀 Performance improvement: {improvement:.1f}% faster")
        print(f"   ⚡ Speedup: {speedup:.1f}x acceleration")
        success = True
    else:
        improvement = ((baseline_time - avg_time) / baseline_time) * 100
        print(f"   ⚠️ TARGET MISSED: {avg_time:.1f}ms > {target_time}ms")
        print(f"   📈 Partial improvement: {improvement:.1f}% faster than baseline")
        success = False
    
    # Step 4: GPU utilization check
    print("\n🎮 GPU UTILIZATION CHECK:")
    gpu_info = gpu_manager.get_memory_info()
    
    if gpu_info['gpu_available']:
        print(f"   ✅ GPU available: {gpu_info['current_device']}")
        print(f"   📊 Memory allocated: {gpu_info['memory_allocated'] / (1024*1024):.1f} MB")
        print(f"   🔄 Memory cached: {gpu_info['memory_cached'] / (1024*1024):.1f} MB")
    else:
        print(f"   ❌ GPU not available - using CPU fallback")
    
    # Step 5: Concurrent performance test
    print("\n🔄 CONCURRENT PERFORMANCE TEST:")
    concurrent_requests = 5
    
    start_concurrent = time.perf_counter()
    
    # Create multiple components and run them concurrently
    concurrent_components = [
        RealAttentionComponent(f"concurrent_bert_{i}") 
        for i in range(concurrent_requests)
    ]
    
    concurrent_tasks = [
        asyncio.create_task(comp.process({"text": f"Concurrent test {i}"}))
        for i, comp in enumerate(concurrent_components)
    ]
    
    concurrent_results = await asyncio.gather(*concurrent_tasks)
    concurrent_time = (time.perf_counter() - start_concurrent) * 1000
    
    # Analyze concurrent results
    concurrent_processing_times = [
        result.get('processing_time_ms', 0) for result in concurrent_results
    ]
    avg_concurrent_time = statistics.mean(concurrent_processing_times)
    
    print(f"   ✅ Concurrent requests: {concurrent_requests}")
    print(f"   ⏱️ Total concurrent time: {concurrent_time:.2f}ms")
    print(f"   📊 Average per-request time: {avg_concurrent_time:.2f}ms")
    print(f"   🚀 Effective throughput: {(concurrent_requests / concurrent_time) * 1000:.1f} req/sec")
    
    # Final Results Summary
    print("\n" + "=" * 60)
    print("🏁 GPU PERFORMANCE OPTIMIZATION RESULTS")
    print("=" * 60)
    
    results = {
        "optimization_successful": success,
        "target_time_ms": target_time,
        "achieved_time_ms": avg_time,
        "baseline_time_ms": baseline_time,
        "improvement_percent": ((baseline_time - avg_time) / baseline_time) * 100,
        "speedup_factor": baseline_time / avg_time,
        "gpu_accelerated": gpu_info['gpu_available'],
        "model_preloaded": True,
        "concurrent_throughput_rps": (concurrent_requests / concurrent_time) * 1000
    }
    
    status_icon = "✅" if success else "⚠️"
    status_text = "SUCCESS" if success else "PARTIAL SUCCESS"
    
    print(f"{status_icon} Status: {status_text}")
    print(f"📊 Performance: {avg_time:.1f}ms (target: {target_time}ms)")
    print(f"🚀 Improvement: {results['improvement_percent']:.1f}% faster")
    print(f"⚡ Speedup: {results['speedup_factor']:.1f}x")
    print(f"🎮 GPU Acceleration: {'Enabled' if gpu_info['gpu_available'] else 'Disabled'}")
    print(f"📥 Model Pre-loading: Enabled")
    print(f"🔄 Throughput: {results['concurrent_throughput_rps']:.1f} req/sec")
    
    if success:
        print(f"\n🎉 OPTIMIZATION COMPLETE: BERT processing optimized!")
        print(f"💪 Ready for production workloads with sub-50ms latency")
    else:
        print(f"\n🔧 OPTIMIZATION IN PROGRESS: Further tuning needed")
        print(f"📝 Next steps: GPU memory optimization, model quantization")
    
    return results

if __name__ == "__main__":
    try:
        optimization_results = asyncio.run(test_gpu_performance_optimization())
        
        # Exit with appropriate code
        if optimization_results["optimization_successful"]:
            print(f"\n🎯 GPU Optimization PASSED: Target achieved!")
            sys.exit(0)
        else:
            print(f"\n⚠️ GPU Optimization PARTIAL: Improvement made but target missed")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n⚠️ GPU optimization test interrupted")
        sys.exit(130)
    except Exception as e:
        print(f"\n💥 GPU optimization test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)