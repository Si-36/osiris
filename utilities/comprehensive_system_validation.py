#!/usr/bin/env python3
"""
AURA Comprehensive System Validation - All Phases Integration Test
Final validation of the complete optimized AURA Intelligence system
"""

import asyncio
import time
import sys
from pathlib import Path

# Add to path
sys.path.insert(0, str(Path(__file__).parent / "core" / "src"))

async def comprehensive_system_test():
    """Comprehensive test of all 5 optimization phases working together"""
    print("🚀 AURA COMPREHENSIVE SYSTEM VALIDATION")
    print("Testing all 5 optimization phases in integrated environment")
    print("=" * 70)
    
    results = {
        "phase1_gpu_acceleration": False,
        "phase2_redis_batching": False,
        "phase3_dashboard": False,
        "phase4_monitoring": False,
        "phase5_neural_optimization": False,
        "integration_score": 0.0,
        "overall_performance": {}
    }
    
    # Phase 1: GPU Acceleration Test
    print("\n🎮 PHASE 1: GPU Acceleration Validation")
    try:
        from aura_intelligence.components.real_components import (
            RealAttentionComponent,
            RealLNNComponent,
            gpu_manager
        )
        
        bert_component = RealAttentionComponent("validation_bert")
        test_data = {"text": "AURA comprehensive system validation test"}
        
        start_time = time.perf_counter()
        result = await bert_component.process(test_data)
        processing_time = (time.perf_counter() - start_time) * 1000
        
        gpu_info = gpu_manager.get_memory_info()
        gpu_available = gpu_info.get("gpu_available", False)
        
        if result.get("gpu_accelerated") and processing_time < 100:  # Sub-100ms target
            results["phase1_gpu_acceleration"] = True
            print(f"  ✅ GPU acceleration: {processing_time:.2f}ms")
            print(f"  ✅ GPU device: {gpu_info.get('current_device', 'cpu')}")
        else:
            print(f"  ⚠️ GPU acceleration: {processing_time:.2f}ms (target: <100ms)")
            
    except Exception as e:
        print(f"  ❌ Phase 1 failed: {e}")
    
    # Phase 2: Redis Pooling + Async Batching Test
    print("\n📦 PHASE 2: Redis Pooling + Async Batching Validation")
    try:
        from aura_intelligence.components.real_components import (
            redis_pool,
            batch_processor
        )
        
        # Test Redis pool
        pool_initialized = await redis_pool.initialize()
        pool_stats = redis_pool.get_pool_stats()
        
        # Test batch processing
        lnn_component = RealLNNComponent("validation_lnn")
        batch_requests = [
            {"values": [1, 2, 3, 4, 5]},
            {"values": [6, 7, 8, 9, 10]},
            {"values": [11, 12, 13, 14, 15]}
        ]
        
        start_time = time.perf_counter()
        batch_results = await lnn_component.process_batch(batch_requests)
        batch_time = (time.perf_counter() - start_time) * 1000
        
        batch_stats = batch_processor.get_performance_stats()
        
        if pool_stats.get("status") == "active" and len(batch_results) == 3 and batch_time < 50:
            results["phase2_redis_batching"] = True
            print(f"  ✅ Redis pool: {pool_stats['status']}")
            print(f"  ✅ Batch processing: {batch_time:.2f}ms for 3 requests")
            print(f"  ✅ Batches processed: {batch_stats['batches_processed']}")
        else:
            print(f"  ⚠️ Redis: {pool_stats.get('status')}, Batch: {batch_time:.2f}ms")
            
    except Exception as e:
        print(f"  ❌ Phase 2 failed: {e}")
    
    # Phase 3: Dashboard Test
    print("\n📊 PHASE 3: Real-time Dashboard Validation")
    try:
        from real_time_dashboard import RealTimeDashboard
        
        dashboard = RealTimeDashboard()
        
        # Test metrics collection
        metrics = await dashboard.get_current_metrics()
        
        # Test health check
        health = await dashboard.get_system_health()
        
        # Test HTML generation
        html = dashboard.get_dashboard_html()
        
        if ("timestamp" in metrics and 
            "overall_status" in health and 
            "AURA Intelligence Dashboard" in html):
            results["phase3_dashboard"] = True
            print(f"  ✅ Metrics collection: {len(metrics)} categories")
            print(f"  ✅ Health status: {health['overall_status']}")
            print(f"  ✅ Dashboard HTML: {len(html)} characters")
        else:
            print(f"  ⚠️ Dashboard validation incomplete")
            
    except Exception as e:
        print(f"  ❌ Phase 3 failed: {e}")
    
    # Phase 4: Production Monitoring Test
    print("\n🔧 PHASE 4: Production Monitoring Validation")
    try:
        from production_monitoring_system import get_monitoring_system
        
        monitoring = get_monitoring_system()
        await monitoring.initialize()
        
        # Create test alert
        alert = await monitoring.create_alert(
            "info",
            "validation",
            "Comprehensive system validation in progress"
        )
        
        # Get monitoring status
        status = monitoring.get_monitoring_status()
        
        if (status["monitoring_active"] and 
            status["tracing_enabled"] and 
            status["self_healing_enabled"]):
            results["phase4_monitoring"] = True
            print(f"  ✅ Monitoring active: {status['monitoring_active']}")
            print(f"  ✅ Self-healing enabled: {status['self_healing_enabled']}")
            print(f"  ✅ Total alerts: {status['total_alerts']}")
        else:
            print(f"  ⚠️ Monitoring validation incomplete")
            
    except Exception as e:
        print(f"  ❌ Phase 4 failed: {e}")
    
    # Phase 5: Advanced Neural Optimization Test
    print("\n🧠 PHASE 5: Advanced Neural Optimization Validation")
    try:
        from advanced_neural_optimization import get_neural_optimizer
        
        optimizer = get_neural_optimizer()
        await optimizer.initialize()
        
        # Test architecture optimization
        config = await optimizer.optimize_neural_architecture("transformer", "latency")
        
        # Test compression
        compression_result = await optimizer.compress_neural_model("transformer", 0.7)
        
        # Get optimization status
        opt_status = optimizer.get_optimization_status()
        
        if (opt_status["optimization_active"] and 
            config.inference_time_ms > 0 and 
            compression_result["compressed_size_mb"] > 0):
            results["phase5_neural_optimization"] = True
            print(f"  ✅ Architecture optimization: {config.inference_time_ms:.2f}ms")
            print(f"  ✅ Model compression: {compression_result['compression_ratio']:.2f}x")
            print(f"  ✅ GPU management: {opt_status['gpu_management']['total_gpus']} GPUs")
        else:
            print(f"  ⚠️ Neural optimization validation incomplete")
            
    except Exception as e:
        print(f"  ❌ Phase 5 failed: {e}")
    
    # Integration Performance Test
    print("\n⚡ INTEGRATION PERFORMANCE TEST")
    try:
        # Test complete pipeline with all optimizations
        bert_component = RealAttentionComponent("integration_bert")
        
        # Test caching performance
        test_query = {"text": "Integration test query for complete pipeline"}
        
        # First run (cache miss)
        start_time = time.perf_counter()
        result1 = await bert_component.process_with_cache(test_query)
        first_run_time = (time.perf_counter() - start_time) * 1000
        
        # Second run (cache hit)
        start_time = time.perf_counter()
        result2 = await bert_component.process_with_cache(test_query)
        second_run_time = (time.perf_counter() - start_time) * 1000
        
        # Calculate performance metrics
        cache_speedup = first_run_time / max(second_run_time, 0.001)
        gpu_acceleration = result1.get("gpu_accelerated", False)
        cache_hit = result2.get("cache_hit", False)
        
        results["overall_performance"] = {
            "first_run_ms": first_run_time,
            "cached_run_ms": second_run_time,
            "cache_speedup": cache_speedup,
            "gpu_accelerated": gpu_acceleration,
            "cache_working": cache_hit
        }
        
        print(f"  ✅ First run (cache miss): {first_run_time:.2f}ms")
        print(f"  ✅ Second run (cache hit): {second_run_time:.2f}ms")
        print(f"  ✅ Cache speedup: {cache_speedup:.1f}x")
        print(f"  ✅ GPU acceleration: {gpu_acceleration}")
        
    except Exception as e:
        print(f"  ❌ Integration test failed: {e}")
    
    # Calculate overall integration score
    phase_scores = [
        results["phase1_gpu_acceleration"],
        results["phase2_redis_batching"],
        results["phase3_dashboard"],
        results["phase4_monitoring"],
        results["phase5_neural_optimization"]
    ]
    
    results["integration_score"] = sum(phase_scores) / len(phase_scores)
    
    # Final Results Summary
    print("\n" + "=" * 70)
    print("🎯 COMPREHENSIVE VALIDATION RESULTS")
    print("=" * 70)
    
    phase_status = [
        ("Phase 1: GPU Acceleration", results["phase1_gpu_acceleration"]),
        ("Phase 2: Redis + Batching", results["phase2_redis_batching"]),
        ("Phase 3: Real-time Dashboard", results["phase3_dashboard"]),
        ("Phase 4: Production Monitoring", results["phase4_monitoring"]),
        ("Phase 5: Neural Optimization", results["phase5_neural_optimization"])
    ]
    
    for phase_name, status in phase_status:
        status_icon = "✅" if status else "❌"
        print(f"{status_icon} {phase_name}")
    
    print(f"\n📊 Integration Score: {results['integration_score']:.1%}")
    
    if results["integration_score"] >= 0.8:
        print("🎉 EXCELLENT: All systems operational!")
    elif results["integration_score"] >= 0.6:
        print("✅ GOOD: Most systems operational")
    else:
        print("⚠️ NEEDS ATTENTION: Some systems require fixes")
    
    # Performance Summary
    perf = results["overall_performance"]
    if perf:
        print(f"\n⚡ PERFORMANCE HIGHLIGHTS:")
        print(f"  • Cache Performance: {perf.get('cache_speedup', 0):.1f}x speedup")
        print(f"  • GPU Acceleration: {'Enabled' if perf.get('gpu_accelerated') else 'Disabled'}")
        print(f"  • Sub-millisecond cached responses: {perf.get('cached_run_ms', 0) < 1}")
        print(f"  • Production-ready latency: {perf.get('first_run_ms', 0) < 100}")
    
    print(f"\n🚀 AURA INTELLIGENCE SYSTEM STATUS: {'PRODUCTION READY' if results['integration_score'] >= 0.8 else 'DEVELOPMENT'}")
    
    return results

if __name__ == "__main__":
    try:
        validation_results = asyncio.run(comprehensive_system_test())
        
        # Exit with appropriate code
        success_rate = validation_results["integration_score"]
        if success_rate >= 0.8:
            print(f"\n🎯 Validation PASSED: {success_rate:.1%} success rate")
            sys.exit(0)
        else:
            print(f"\n⚠️ Validation PARTIAL: {success_rate:.1%} success rate")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n⚠️ Comprehensive validation interrupted")
        sys.exit(130)
    except Exception as e:
        print(f"\n💥 Comprehensive validation failed: {e}")
        sys.exit(1)