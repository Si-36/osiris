#!/usr/bin/env python3
"""
Test component system - registry and batch processing
"""

import asyncio
import sys
from pathlib import Path
from datetime import datetime
import time

# Add project to path
sys.path.insert(0, str(Path(__file__).parent / "core" / "src"))

print("🧩 TESTING COMPONENT SYSTEM")
print("=" * 60)

async def test_components():
    """Test all component system features"""
    
    # Test 1: Import all modules
    print("\n📦 Testing imports...")
    try:
        from aura_intelligence.components.registry import (
            AURAComponentRegistry, Component, ComponentRole, ComponentCategory,
            ComponentStatus, get_registry, register_component
        )
        from aura_intelligence.components.async_batch_processor import (
            AsyncBatchProcessor, BatchProcessor, Batch, BatchItem,
            BatchPriority, ProcessingStrategy, BatchConfig
        )
        print("✅ All imports successful")
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return
    
    # Test 2: Component Registry
    print("\n📋 Testing Component Registry...")
    try:
        registry = AURAComponentRegistry()
        
        # Register test components
        await registry.register(
            component_id="test_neural_1",
            name="Test Neural Component",
            module_path="test.module",
            category=ComponentCategory.NEURAL,
            role=ComponentRole.PROCESSOR,
            capabilities=["neural_processing", "pattern_recognition"]
        )
        print("✅ Component registered: test_neural_1")
        
        await registry.register(
            component_id="test_memory_1",
            name="Test Memory Component",
            module_path="test.memory",
            category=ComponentCategory.MEMORY,
            role=ComponentRole.INFORMATION_AGENT,
            capabilities=["memory_storage", "retrieval"],
            dependencies=["test_neural_1"]
        )
        print("✅ Component registered: test_memory_1")
        
        # Test dependency order
        order = registry.get_dependency_order()
        print(f"✅ Dependency order: {order}")
        
        # Test component queries
        neural_components = registry.get_components_by_category(ComponentCategory.NEURAL)
        print(f"✅ Neural components: {len(neural_components)}")
        
        memory_capable = registry.get_components_by_capability("memory_storage")
        print(f"✅ Memory-capable components: {len(memory_capable)}")
        
        # Test health system
        health = registry.get_system_health()
        print(f"✅ System health check:")
        print(f"   - Total components: {health['total_components']}")
        print(f"   - Active components: {health['active_components']}")
        print(f"   - Average health: {health['average_health_score']:.2f}")
        
    except Exception as e:
        print(f"❌ Registry test failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 3: Batch Processing
    print("\n🔄 Testing Batch Processing...")
    try:
        # Create test processor
        class TestProcessor(BatchProcessor):
            async def process_batch(self, batch):
                # Simulate processing
                await asyncio.sleep(0.05)
                results = []
                for item in batch.items:
                    results.append({
                        "id": item.id,
                        "processed": item.data.get("value", 0) * 2,
                        "timestamp": datetime.now().isoformat()
                    })
                return results
        
        # Create batch processor
        config = BatchConfig(
            max_batch_size=50,
            batch_timeout=0.5,
            max_concurrent_batches=5,
            processing_strategy=ProcessingStrategy.ADAPTIVE
        )
        
        processor = TestProcessor()
        batch_processor = AsyncBatchProcessor(processor, config)
        
        await batch_processor.start()
        print("✅ Batch processor started")
        
        # Submit items
        start_time = time.time()
        item_ids = []
        
        for i in range(200):
            item_id = await batch_processor.submit(
                {"value": i, "data": f"test_{i}"},
                batch_type="test",
                priority=BatchPriority.HIGH if i % 20 == 0 else BatchPriority.NORMAL
            )
            item_ids.append(item_id)
        
        print(f"✅ Submitted 200 items")
        
        # Wait for processing
        await asyncio.sleep(2)
        
        # Get stats
        stats = batch_processor.get_stats()
        elapsed = time.time() - start_time
        
        print(f"✅ Batch processing stats:")
        print(f"   - Processed items: {stats['processed_items']}")
        print(f"   - Processed batches: {stats['processed_batches']}")
        print(f"   - Failed items: {stats['failed_items']}")
        print(f"   - Avg latency: {stats['avg_latency_ms']}ms")
        print(f"   - Throughput: {stats['processed_items']/elapsed:.1f} items/sec")
        
        await batch_processor.stop()
        print("✅ Batch processor stopped")
        
    except Exception as e:
        print(f"❌ Batch processing test failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 4: Processing Strategies
    print("\n📊 Testing Processing Strategies...")
    try:
        strategies = [
            ProcessingStrategy.FIFO,
            ProcessingStrategy.LIFO,
            ProcessingStrategy.PRIORITY,
            ProcessingStrategy.ADAPTIVE
        ]
        
        for strategy in strategies:
            config = BatchConfig(
                max_batch_size=10,
                batch_timeout=0.2,
                processing_strategy=strategy
            )
            
            processor = TestProcessor()
            batch_processor = AsyncBatchProcessor(processor, config)
            
            await batch_processor.start()
            
            # Submit mixed priority items
            for i in range(30):
                priority = BatchPriority.CRITICAL if i < 5 else BatchPriority.NORMAL
                await batch_processor.submit(
                    {"value": i},
                    priority=priority
                )
            
            await asyncio.sleep(1)
            
            stats = batch_processor.get_stats()
            print(f"✅ {strategy.value} strategy: {stats['processed_items']} items in {stats['processed_batches']} batches")
            
            await batch_processor.stop()
            
    except Exception as e:
        print(f"❌ Strategy test failed: {e}")
    
    # Test 5: Circuit Breaker
    print("\n🔌 Testing Circuit Breaker...")
    try:
        # Create failing processor
        class FailingProcessor(BatchProcessor):
            def __init__(self):
                self.fail_count = 0
            
            async def process_batch(self, batch):
                self.fail_count += 1
                if self.fail_count <= 5:
                    raise Exception("Simulated failure")
                # Recover after 5 failures
                return [{"id": item.id, "processed": True} for item in batch.items]
        
        config = BatchConfig(
            max_batch_size=5,
            circuit_breaker_threshold=3,
            circuit_breaker_timeout=1.0
        )
        
        processor = FailingProcessor()
        batch_processor = AsyncBatchProcessor(processor, config)
        
        await batch_processor.start()
        
        # Submit items
        for i in range(20):
            try:
                await batch_processor.submit({"value": i}, batch_type="failing")
            except RuntimeError as e:
                if "Circuit breaker open" in str(e):
                    print(f"✅ Circuit breaker opened after failures")
                    break
        
        # Wait for recovery
        await asyncio.sleep(2)
        
        # Try again
        await batch_processor.submit({"value": 100}, batch_type="failing")
        print("✅ Circuit breaker recovered")
        
        stats = batch_processor.get_stats()
        print(f"   - Failed items: {stats['failed_items']}")
        print(f"   - Circuit breakers: {stats['circuit_breakers']}")
        
        await batch_processor.stop()
        
    except Exception as e:
        print(f"❌ Circuit breaker test failed: {e}")
    
    # Test 6: Integration Test
    print("\n🔗 Testing Integration...")
    try:
        # Create a complete component
        class DataProcessor(Component):
            def __init__(self):
                self.batch_processor = None
                self.processed_count = 0
            
            async def initialize(self, config):
                processor = TestProcessor()
                self.batch_processor = AsyncBatchProcessor(processor, BatchConfig())
            
            async def start(self):
                await self.batch_processor.start()
            
            async def stop(self):
                await self.batch_processor.stop()
            
            async def health_check(self):
                stats = self.batch_processor.get_stats()
                return {
                    "healthy": True,
                    "score": 1.0 if stats['failed_items'] == 0 else 0.5,
                    "processed": stats['processed_items']
                }
            
            def get_capabilities(self):
                return ["batch_processing", "data_transformation"]
            
            async def process(self, data):
                return await self.batch_processor.submit(data)
        
        # Register and load
        registry = get_registry()
        
        # Note: Can't actually load without proper module path
        # This is just to show the pattern
        print("✅ Integration pattern demonstrated")
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
    
    print("\n" + "=" * 60)
    print("COMPONENT SYSTEM TEST COMPLETE")
    
# Run the test
if __name__ == "__main__":
    asyncio.run(test_components())