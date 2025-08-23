#!/usr/bin/env python3
"""
🚀 ASYNC BATCH PROCESSING TEST
Test the new async batch processor with real component performance
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

from aura_intelligence.components.async_batch_processor import (
    AsyncBatchProcessor,
    BatchProcessorConfig,
    BatchType,
    get_global_batch_processor,
    process_with_batching
)

class BatchComponentTester:
    def __init__(self):
        self.processor = None
        
    async def initialize(self):
        """Initialize batch processor"""
        print("🚀 Initializing Async Batch Processor...")
        try:
            self.processor = await get_global_batch_processor()
            print("✅ Batch processor initialized")
            return True
        except Exception as e:
            print(f"❌ Failed to initialize batch processor: {e}")
            return False
    
    async def test_sequential_processing(self, num_operations: int = 1000) -> Dict[str, Any]:
        """Test sequential processing (baseline)"""
        print(f"\n📊 Testing {num_operations} sequential operations...")
        
        start_time = time.perf_counter()
        operations_completed = 0
        processing_times = []
        
        for i in range(num_operations):
            try:
                op_start = time.perf_counter()
                
                # Simulate simple neural processing
                data = {"values": [i * 0.1, (i+1) * 0.1, (i+2) * 0.1]}
                
                # Direct processing (no batching)
                await asyncio.sleep(0.001)  # Simulate 1ms processing time
                result = {"processed": True, "sequential": True}
                
                op_time = (time.perf_counter() - op_start) * 1000
                processing_times.append(op_time)
                operations_completed += 1
                
            except Exception as e:
                print(f"   Error in operation {i}: {e}")
        
        total_time = time.perf_counter() - start_time
        
        results = {
            'test_type': 'sequential_processing',
            'operations_completed': operations_completed,
            'total_time_seconds': total_time,
            'operations_per_second': operations_completed / total_time if total_time > 0 else 0,
            'avg_operation_time_ms': statistics.mean(processing_times) if processing_times else 0,
            'min_operation_time_ms': min(processing_times) if processing_times else 0,
            'max_operation_time_ms': max(processing_times) if processing_times else 0
        }
        
        print(f"   ✅ Completed: {operations_completed} operations")
        print(f"   ⚡ Rate: {results['operations_per_second']:.0f} ops/sec")
        print(f"   ⏱️  Avg time: {results['avg_operation_time_ms']:.2f}ms per operation")
        
        return results
    
    async def test_batch_processing(self, num_operations: int = 1000) -> Dict[str, Any]:
        """Test batch processing"""
        print(f"\n🔄 Testing {num_operations} batch operations...")
        
        start_time = time.perf_counter()
        
        # Create all operations concurrently
        tasks = []
        for i in range(num_operations):
            data = {"values": [i * 0.1, (i+1) * 0.1, (i+2) * 0.1], "sequence": [i, i+1, i+2]}
            
            task = process_with_batching(
                operation_id=f"test_op_{i}",
                batch_type=BatchType.NEURAL_FORWARD,
                input_data=data,
                component_id="test_component",
                priority=0
            )
            tasks.append(task)
        
        # Wait for all operations to complete
        try:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            successful_ops = sum(1 for r in results if not isinstance(r, Exception))
            errors = sum(1 for r in results if isinstance(r, Exception))
            
            total_time = time.perf_counter() - start_time
            
            batch_results = {
                'test_type': 'batch_processing',
                'operations_requested': num_operations,
                'operations_completed': successful_ops,
                'errors': errors,
                'total_time_seconds': total_time,
                'operations_per_second': successful_ops / total_time if total_time > 0 else 0,
                'avg_operation_time_ms': (total_time / successful_ops * 1000) if successful_ops > 0 else 0
            }
            
            print(f"   ✅ Completed: {successful_ops}/{num_operations} operations")
            print(f"   ❌ Errors: {errors}")
            print(f"   ⚡ Rate: {batch_results['operations_per_second']:.0f} ops/sec")
            print(f"   ⏱️  Avg time: {batch_results['avg_operation_time_ms']:.2f}ms per operation")
            
            return batch_results
            
        except Exception as e:
            print(f"   ❌ Batch processing failed: {e}")
            return {'test_type': 'batch_processing', 'error': str(e)}
    
    async def test_mixed_batch_types(self, num_each: int = 200) -> Dict[str, Any]:
        """Test mixed batch types processing"""
        print(f"\n🎭 Testing mixed batch types ({num_each} each)...")
        
        start_time = time.perf_counter()
        
        tasks = []
        batch_types = [
            BatchType.NEURAL_FORWARD,
            BatchType.BERT_ATTENTION,
            BatchType.LNN_PROCESSING,
            BatchType.NEURAL_ODE,
            BatchType.GENERAL_COMPUTE
        ]
        
        for batch_type in batch_types:
            for i in range(num_each):
                if batch_type == BatchType.BERT_ATTENTION:
                    data = {"text": f"test text for attention processing {i}"}
                elif batch_type == BatchType.NEURAL_ODE:
                    data = {"values": [i * 0.01, (i+1) * 0.01, (i+2) * 0.01, (i+3) * 0.01]}
                elif batch_type == BatchType.LNN_PROCESSING:
                    data = {"sequence": [i, i+1, i+2, i+3]}
                else:
                    data = {"values": [i * 0.1, (i+1) * 0.1], "operation": batch_type.value}
                
                task = process_with_batching(
                    operation_id=f"{batch_type.value}_{i}",
                    batch_type=batch_type,
                    input_data=data,
                    component_id=f"test_{batch_type.value}",
                    priority=1 if batch_type in [BatchType.BERT_ATTENTION, BatchType.NEURAL_ODE] else 0
                )
                tasks.append((batch_type, task))
        
        # Wait for all operations
        try:
            all_results = await asyncio.gather(*[task for _, task in tasks], return_exceptions=True)
            total_time = time.perf_counter() - start_time
            
            # Analyze results by batch type
            type_stats = {}
            for (batch_type, _), result in zip(tasks, all_results):
                type_name = batch_type.value
                if type_name not in type_stats:
                    type_stats[type_name] = {'completed': 0, 'errors': 0}
                
                if isinstance(result, Exception):
                    type_stats[type_name]['errors'] += 1
                else:
                    type_stats[type_name]['completed'] += 1
            
            total_completed = sum(stats['completed'] for stats in type_stats.values())
            total_errors = sum(stats['errors'] for stats in type_stats.values())
            
            results = {
                'test_type': 'mixed_batch_types',
                'total_operations': len(tasks),
                'total_completed': total_completed,
                'total_errors': total_errors,
                'total_time_seconds': total_time,
                'operations_per_second': total_completed / total_time if total_time > 0 else 0,
                'type_stats': type_stats
            }
            
            print(f"   ✅ Total completed: {total_completed}/{len(tasks)} operations")
            print(f"   ⚡ Overall rate: {results['operations_per_second']:.0f} ops/sec")
            
            for type_name, stats in type_stats.items():
                print(f"   📋 {type_name}: {stats['completed']} completed, {stats['errors']} errors")
            
            return results
            
        except Exception as e:
            print(f"   ❌ Mixed batch processing failed: {e}")
            return {'test_type': 'mixed_batch_types', 'error': str(e)}
    
    async def test_adaptive_batching(self, num_operations: int = 500) -> Dict[str, Any]:
        """Test adaptive batch sizing"""
        print(f"\n📈 Testing adaptive batching with {num_operations} operations...")
        
        # Get processor metrics before test
        initial_metrics = await self.processor.get_metrics()
        
        start_time = time.perf_counter()
        
        # Create operations with varying complexity
        tasks = []
        for i in range(num_operations):
            # Vary complexity to trigger adaptive sizing
            if i % 4 == 0:  # High complexity
                data = {"values": list(range(20)), "complexity": "high"}
                batch_type = BatchType.NEURAL_ODE
                priority = 2
            elif i % 4 == 1:  # Medium complexity
                data = {"text": f"medium complexity text processing for operation {i}", "complexity": "medium"}
                batch_type = BatchType.BERT_ATTENTION  
                priority = 1
            else:  # Low complexity
                data = {"values": [i, i+1], "complexity": "low"}
                batch_type = BatchType.NEURAL_FORWARD
                priority = 0
            
            task = process_with_batching(
                operation_id=f"adaptive_op_{i}",
                batch_type=batch_type,
                input_data=data,
                component_id="adaptive_test",
                priority=priority
            )
            tasks.append(task)
        
        # Process all operations
        try:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            total_time = time.perf_counter() - start_time
            
            successful_ops = sum(1 for r in results if not isinstance(r, Exception))
            errors = sum(1 for r in results if isinstance(r, Exception))
            
            # Get final metrics
            final_metrics = await self.processor.get_metrics()
            
            adaptive_results = {
                'test_type': 'adaptive_batching',
                'operations_completed': successful_ops,
                'errors': errors,
                'total_time_seconds': total_time,
                'operations_per_second': successful_ops / total_time if total_time > 0 else 0,
                'initial_batch_sizes': initial_metrics.get('adaptive_batch_sizes', {}),
                'final_batch_sizes': final_metrics.get('adaptive_batch_sizes', {}),
                'metrics_improvement': {
                    'operations_per_second': final_metrics.get('operations_per_second', 0) - initial_metrics.get('operations_per_second', 0),
                    'avg_batch_size': final_metrics.get('avg_batch_size', 0) - initial_metrics.get('avg_batch_size', 0),
                    'batch_efficiency': final_metrics.get('batch_efficiency', 0) - initial_metrics.get('batch_efficiency', 0)
                }
            }
            
            print(f"   ✅ Completed: {successful_ops}/{num_operations} operations")
            print(f"   ⚡ Rate: {adaptive_results['operations_per_second']:.0f} ops/sec")
            print(f"   📊 Batch size changes:")
            
            for batch_type, initial_size in initial_metrics.get('adaptive_batch_sizes', {}).items():
                final_size = final_metrics.get('adaptive_batch_sizes', {}).get(batch_type, initial_size)
                if final_size != initial_size:
                    print(f"     {batch_type}: {initial_size} → {final_size}")
            
            return adaptive_results
            
        except Exception as e:
            print(f"   ❌ Adaptive batching test failed: {e}")
            return {'test_type': 'adaptive_batching', 'error': str(e)}
    
    async def run_comprehensive_test(self) -> Dict[str, Any]:
        """Run comprehensive batch processing test suite"""
        print("🧪 Async Batch Processing Test Suite")
        print("=" * 50)
        
        if not await self.initialize():
            return {'error': 'Failed to initialize batch processor'}
        
        test_results = {}
        
        try:
            # 1. Sequential baseline
            test_results['sequential'] = await self.test_sequential_processing(500)
            
            # 2. Basic batching
            test_results['batch'] = await self.test_batch_processing(1000)
            
            # 3. Mixed batch types
            test_results['mixed'] = await self.test_mixed_batch_types(200)
            
            # 4. Adaptive batching
            test_results['adaptive'] = await self.test_adaptive_batching(500)
            
            # 5. Get final metrics
            test_results['final_metrics'] = await self.processor.get_metrics()
            
        except Exception as e:
            print(f"❌ Test suite error: {e}")
            test_results['error'] = str(e)
        
        return test_results
    
    async def close(self):
        """Clean up processor"""
        if self.processor:
            await self.processor.stop()

async def main():
    """Run async batch processing tests"""
    tester = BatchComponentTester()
    
    try:
        results = await tester.run_comprehensive_test()
        
        # Print summary
        print("\n" + "=" * 50)
        print("🏆 ASYNC BATCH PROCESSING SUMMARY")
        print("=" * 50)
        
        if 'error' not in results:
            if 'sequential' in results and 'batch' in results:
                seq_ops = results['sequential']['operations_per_second']
                batch_ops = results['batch']['operations_per_second']
                speedup = batch_ops / seq_ops if seq_ops > 0 else 0
                
                print(f"📊 Sequential Processing: {seq_ops:.0f} ops/sec")
                print(f"🚀 Batch Processing: {batch_ops:.0f} ops/sec")
                print(f"⚡ Batch Speedup: {speedup:.1f}x")
            
            if 'mixed' in results:
                mixed = results['mixed']
                print(f"🎭 Mixed Batch Types: {mixed['operations_per_second']:.0f} ops/sec")
                print(f"   Success Rate: {mixed['total_completed']}/{mixed['total_operations']} ({mixed['total_completed']/mixed['total_operations']*100:.1f}%)")
            
            if 'adaptive' in results:
                adaptive = results['adaptive']
                print(f"📈 Adaptive Batching: {adaptive['operations_per_second']:.0f} ops/sec")
            
            if 'final_metrics' in results:
                metrics = results['final_metrics']
                print(f"📊 Final Metrics:")
                print(f"   Total Operations: {metrics.get('total_operations', 0)}")
                print(f"   Total Batches: {metrics.get('total_batches', 0)}")
                print(f"   Avg Batch Size: {metrics.get('avg_batch_size', 0):.1f}")
                print(f"   Batch Efficiency: {metrics.get('batch_efficiency', 0)*100:.1f}%")
                print(f"   GPU Available: {metrics.get('gpu_available', False)}")
        
        # Save results
        results_file = Path("async_batch_results.json")
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\n💾 Detailed results saved to: {results_file}")
        
        # Performance verdict
        if 'batch' in results:
            ops_per_second = results['batch']['operations_per_second']
            if ops_per_second > 5000:
                print("\n🎉 EXCELLENT - Ultra-high batch performance!")
            elif ops_per_second > 2000:
                print("\n✅ GOOD - High batch performance!")
            elif ops_per_second > 1000:
                print("\n⚠️  FAIR - Moderate batch performance")
            else:
                print("\n❌ POOR - Batch performance needs improvement")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
    finally:
        await tester.close()

if __name__ == "__main__":
    asyncio.run(main())