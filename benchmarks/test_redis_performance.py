#!/usr/bin/env python3
"""
🚀 REDIS HIGH-PERFORMANCE BATCH TESTING
Test the new ultra-high-performance Redis adapter with async batching
"""

import asyncio
import time
import json
from typing import Dict, Any, List
from pathlib import Path
import statistics

# Add to path
import sys
sys.path.insert(0, str(Path(__file__).parent / "core" / "src"))

from aura_intelligence.adapters.redis_high_performance import (
    HighPerformanceRedisAdapter,
    create_ultra_high_performance_config,
    get_ultra_high_performance_adapter
)

class RedisPerformanceTester:
    def __init__(self):
        self.adapter = None
        self.test_results = {}
        
    async def initialize(self):
        """Initialize high-performance Redis adapter"""
        print("🚀 Initializing Ultra-High-Performance Redis Adapter...")
        try:
            self.adapter = await get_ultra_high_performance_adapter()
            health = await self.adapter.health_check()
            print(f"✅ Redis adapter initialized - Status: {health['status']}")
            print(f"   Ping time: {health.get('ping_time_ms', 0):.2f}ms")
            print(f"   Redis version: {health.get('redis_version', 'unknown')}")
            return True
        except Exception as e:
            print(f"❌ Failed to initialize Redis adapter: {e}")
            return False
    
    async def test_single_operations(self, num_operations: int = 1000) -> Dict[str, Any]:
        """Test individual operations (baseline)"""
        print(f"\n📊 Testing {num_operations} individual operations...")
        
        start_time = time.perf_counter()
        operations_completed = 0
        errors = 0
        
        for i in range(num_operations):
            try:
                key = f"test_single_{i}"
                data = {"operation_id": i, "timestamp": time.time(), "data": f"test_data_{i}"}
                
                # Single SET operation
                await self.adapter.batch_set(key, data, ttl=300)
                
                # Single GET operation
                result = await self.adapter.batch_get(key)
                
                if result:
                    operations_completed += 1
                    
            except Exception as e:
                errors += 1
                if errors < 5:  # Only print first few errors
                    print(f"   Error in operation {i}: {e}")
        
        total_time = time.perf_counter() - start_time
        
        results = {
            'test_type': 'single_operations',
            'operations_requested': num_operations * 2,  # SET + GET
            'operations_completed': operations_completed * 2,
            'errors': errors,
            'total_time_seconds': total_time,
            'operations_per_second': (operations_completed * 2) / total_time if total_time > 0 else 0,
            'avg_operation_time_ms': (total_time / (operations_completed * 2) * 1000) if operations_completed > 0 else 0
        }
        
        print(f"   ✅ Completed: {operations_completed * 2} operations")
        print(f"   ⚡ Rate: {results['operations_per_second']:.0f} ops/sec")
        print(f"   ⏱️  Avg time: {results['avg_operation_time_ms']:.2f}ms per operation")
        
        return results
    
    async def test_concurrent_operations(self, num_operations: int = 1000, concurrency: int = 50) -> Dict[str, Any]:
        """Test concurrent operations with batching"""
        print(f"\n🔄 Testing {num_operations} concurrent operations (concurrency: {concurrency})...")
        
        async def single_operation(op_id: int):
            try:
                key = f"test_concurrent_{op_id}"
                data = {"operation_id": op_id, "timestamp": time.time(), "concurrent": True}
                
                # Concurrent SET
                await self.adapter.batch_set(key, data, ttl=300)
                
                # Concurrent GET
                result = await self.adapter.batch_get(key)
                
                return result is not None
            except Exception:
                return False
        
        start_time = time.perf_counter()
        
        # Create tasks in batches to control concurrency
        all_tasks = []
        for i in range(0, num_operations, concurrency):
            batch_tasks = [
                single_operation(j) 
                for j in range(i, min(i + concurrency, num_operations))
            ]
            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
            all_tasks.extend(batch_results)
        
        total_time = time.perf_counter() - start_time
        
        successful_ops = sum(1 for result in all_tasks if result is True)
        errors = sum(1 for result in all_tasks if isinstance(result, Exception) or result is False)
        
        results = {
            'test_type': 'concurrent_operations',
            'operations_requested': num_operations * 2,  # SET + GET
            'operations_completed': successful_ops * 2,
            'errors': errors,
            'concurrency_level': concurrency,
            'total_time_seconds': total_time,
            'operations_per_second': (successful_ops * 2) / total_time if total_time > 0 else 0,
            'avg_operation_time_ms': (total_time / (successful_ops * 2) * 1000) if successful_ops > 0 else 0
        }
        
        print(f"   ✅ Completed: {successful_ops * 2} operations")
        print(f"   ⚡ Rate: {results['operations_per_second']:.0f} ops/sec")
        print(f"   ⏱️  Avg time: {results['avg_operation_time_ms']:.2f}ms per operation")
        print(f"   🔥 Speedup vs sequential: {results['operations_per_second'] / 100:.1f}x (estimated)")
        
        return results
    
    async def test_bulk_operations(self, num_patterns: int = 500) -> Dict[str, Any]:
        """Test bulk pattern storage"""
        print(f"\n📦 Testing bulk storage of {num_patterns} patterns...")
        
        # Generate test patterns
        patterns = {}
        for i in range(num_patterns):
            key = f"pattern_bulk_{i}"
            patterns[key] = {
                "pattern_id": i,
                "timestamp": time.time(),
                "data": [j * 0.1 for j in range(20)],  # Some numeric data
                "metadata": {"type": "test_pattern", "batch": "bulk_test"}
            }
        
        start_time = time.perf_counter()
        
        try:
            results = await self.adapter.store_patterns_bulk(patterns, ttl=300)
            successful_stores = sum(1 for r in results if r)
            
            # Test retrieval
            retrieval_start = time.perf_counter()
            retrieved = 0
            for key in list(patterns.keys())[:100]:  # Test first 100
                result = await self.adapter.get_pattern(key)
                if result:
                    retrieved += 1
            retrieval_time = time.perf_counter() - retrieval_start
            
            total_time = time.perf_counter() - start_time
            
            results_dict = {
                'test_type': 'bulk_operations',
                'patterns_requested': num_patterns,
                'patterns_stored': successful_stores,
                'patterns_retrieved': retrieved,
                'total_time_seconds': total_time,
                'store_time_seconds': total_time - retrieval_time,
                'retrieval_time_seconds': retrieval_time,
                'store_rate_per_second': successful_stores / (total_time - retrieval_time) if (total_time - retrieval_time) > 0 else 0,
                'retrieval_rate_per_second': retrieved / retrieval_time if retrieval_time > 0 else 0
            }
            
            print(f"   ✅ Stored: {successful_stores}/{num_patterns} patterns")
            print(f"   ✅ Retrieved: {retrieved}/100 test patterns")
            print(f"   ⚡ Store rate: {results_dict['store_rate_per_second']:.0f} patterns/sec")
            print(f"   ⚡ Retrieval rate: {results_dict['retrieval_rate_per_second']:.0f} patterns/sec")
            
            return results_dict
            
        except Exception as e:
            print(f"   ❌ Bulk operation failed: {e}")
            return {'test_type': 'bulk_operations', 'error': str(e)}
    
    async def test_batch_performance_metrics(self) -> Dict[str, Any]:
        """Test batch performance and get metrics"""
        print(f"\n📈 Testing batch performance metrics...")
        
        # Force a batch flush to clear pending operations
        await self.adapter.force_batch_flush()
        
        # Reset metrics for clean test
        await self.adapter.reset_metrics()
        
        # Generate load to test batching
        operations = []
        for i in range(200):
            key = f"metrics_test_{i}"
            data = {"metric_test": i, "timestamp": time.time()}
            operations.append(self.adapter.batch_set(key, data))
            operations.append(self.adapter.batch_get(key))
        
        start_time = time.perf_counter()
        results = await asyncio.gather(*operations, return_exceptions=True)
        test_time = time.perf_counter() - start_time
        
        # Get performance metrics
        metrics = await self.adapter.get_performance_metrics()
        
        successful_ops = sum(1 for r in results if not isinstance(r, Exception))
        
        print(f"   ✅ Operations completed: {successful_ops}/{len(operations)}")
        print(f"   ⚡ Test rate: {successful_ops / test_time:.0f} ops/sec")
        print(f"   📊 Cache hit rate: {metrics['cache_hit_rate'] * 100:.1f}%")
        print(f"   📦 Average batch size: {metrics['avg_batch_size']:.1f}")
        print(f"   ⏱️  Average response time: {metrics['avg_response_time'] * 1000:.2f}ms")
        print(f"   🔄 Total batches processed: {metrics['total_batches_processed']}")
        print(f"   📈 Operations per second: {metrics['operations_per_second']:.0f}")
        
        return {
            'test_type': 'batch_metrics',
            'operations_completed': successful_ops,
            'test_rate_ops_per_second': successful_ops / test_time,
            'metrics': metrics
        }
    
    async def run_comprehensive_test(self) -> Dict[str, Any]:
        """Run comprehensive performance test suite"""
        print("🧪 Redis High-Performance Batch Testing")
        print("=" * 50)
        
        if not await self.initialize():
            return {'error': 'Failed to initialize Redis adapter'}
        
        # Test suite
        test_results = {}
        
        try:
            # 1. Single operations baseline
            test_results['single_ops'] = await self.test_single_operations(500)
            
            # 2. Concurrent operations
            test_results['concurrent_ops'] = await self.test_concurrent_operations(1000, 100)
            
            # 3. Bulk operations
            test_results['bulk_ops'] = await self.test_bulk_operations(200)
            
            # 4. Batch metrics
            test_results['batch_metrics'] = await self.test_batch_performance_metrics()
            
            # 5. Final health check
            test_results['final_health'] = await self.adapter.health_check()
            
        except Exception as e:
            print(f"❌ Test suite error: {e}")
            test_results['error'] = str(e)
        
        return test_results
    
    async def close(self):
        """Clean up adapter"""
        if self.adapter:
            await self.adapter.close()

async def main():
    """Run Redis performance tests"""
    tester = RedisPerformanceTester()
    
    try:
        results = await tester.run_comprehensive_test()
        
        # Print summary
        print("\n" + "=" * 50)
        print("🏆 REDIS PERFORMANCE TEST SUMMARY")
        print("=" * 50)
        
        if 'error' not in results:
            if 'single_ops' in results:
                single = results['single_ops']
                print(f"📊 Single Operations: {single['operations_per_second']:.0f} ops/sec")
            
            if 'concurrent_ops' in results:
                concurrent = results['concurrent_ops']
                print(f"🔄 Concurrent Operations: {concurrent['operations_per_second']:.0f} ops/sec")
                
                # Calculate speedup
                if 'single_ops' in results:
                    speedup = concurrent['operations_per_second'] / single['operations_per_second']
                    print(f"⚡ Concurrency Speedup: {speedup:.1f}x")
            
            if 'bulk_ops' in results:
                bulk = results['bulk_ops']
                print(f"📦 Bulk Store Rate: {bulk.get('store_rate_per_second', 0):.0f} patterns/sec")
                print(f"📥 Bulk Retrieval Rate: {bulk.get('retrieval_rate_per_second', 0):.0f} patterns/sec")
            
            if 'batch_metrics' in results and 'metrics' in results['batch_metrics']:
                metrics = results['batch_metrics']['metrics']
                print(f"🎯 Cache Hit Rate: {metrics['cache_hit_rate'] * 100:.1f}%")
                print(f"📈 Avg Batch Size: {metrics['avg_batch_size']:.1f}")
                print(f"⏱️  Avg Response Time: {metrics['avg_response_time'] * 1000:.2f}ms")
            
            if 'final_health' in results:
                health = results['final_health']
                print(f"🔋 Final Health: {health['status']}")
                if 'ping_time_ms' in health:
                    print(f"🏓 Final Ping: {health['ping_time_ms']:.2f}ms")
        
        # Save results
        results_file = Path("redis_performance_results.json")
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\n💾 Detailed results saved to: {results_file}")
        
        # Performance verdict
        if 'concurrent_ops' in results:
            ops_per_second = results['concurrent_ops']['operations_per_second']
            if ops_per_second > 10000:
                print("\n🎉 EXCELLENT - Ultra-high performance achieved!")
            elif ops_per_second > 5000:
                print("\n✅ GOOD - High performance achieved!")
            elif ops_per_second > 1000:
                print("\n⚠️  FAIR - Moderate performance")
            else:
                print("\n❌ POOR - Performance needs improvement")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
    finally:
        await tester.close()

if __name__ == "__main__":
    asyncio.run(main())