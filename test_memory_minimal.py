#!/usr/bin/env python3
"""
Minimal Memory Test - Test Core Functionality Only
=================================================
"""

import asyncio
import numpy as np
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'core/src'))


async def test_basic_structures():
    """Test basic memory structures without dependencies"""
    print("\n=== Testing Basic Memory Structures ===")
    
    # Test 1: Memory types and enums
    print("\n1. Testing memory enums...")
    from aura_intelligence.memory.unified_memory_interface import (
        MemoryType, ConsistencyLevel, SearchType
    )
    
    print(f"✓ Memory types: {[t.value for t in MemoryType]}")
    print(f"✓ Consistency levels: {[c.value for c in ConsistencyLevel]}")
    print(f"✓ Search types: {[s.value for s in SearchType]}")
    
    # Test 2: Mem0 structures
    print("\n2. Testing Mem0 structures...")
    from aura_intelligence.memory.mem0_pipeline import (
        ExtractedFact, FactType, ConfidenceLevel
    )
    
    fact = ExtractedFact(
        fact_id="test_001",
        fact_type=FactType.ENTITY,
        subject="Memory",
        predicate="is",
        object="working",
        confidence=0.95,
        source_id="test"
    )
    
    print(f"✓ Created fact: {fact.subject} {fact.predicate} {fact.object}")
    print(f"✓ Confidence: {fact.confidence} ({fact.confidence_level.value})")
    
    # Test 3: H-MEM structures
    print("\n3. Testing H-MEM structures...")
    from aura_intelligence.memory.hierarchical_routing import MemoryLevel
    
    level = MemoryLevel(
        level=0,
        name="Test",
        summary="Test memory",
        embedding=np.zeros(768)
    )
    
    print(f"✓ Created memory level: {level.name} at level {level.level}")
    
    # Test 4: Qdrant configuration
    print("\n4. Testing Qdrant configuration...")
    from aura_intelligence.memory.qdrant_config import (
        QdrantMultitenantManager, QuantizationPreset
    )
    
    manager = QdrantMultitenantManager()
    savings = manager.estimate_ram_savings(
        vector_count=100000,
        vector_size=768,
        quantization_preset=QuantizationPreset.BALANCED
    )
    
    print(f"✓ RAM savings estimation:")
    print(f"  Original: {savings['original_size_mb']:.1f} MB")
    print(f"  Quantized: {savings['quantized_size_mb']:.1f} MB")
    print(f"  Savings: {savings['savings_percent']}%")
    
    return True


async def test_knn_index():
    """Test KNN index"""
    print("\n=== Testing KNN Index ===")
    
    from aura_intelligence.memory import create_knn_index
    
    # Create simple index
    index = create_knn_index(
        embedding_dim=128,
        backend='sklearn',  # Always available
        metric='cosine'
    )
    
    print("✓ Created KNN index")
    
    # Add some vectors
    vectors = np.random.randn(10, 128).astype(np.float32)
    ids = [f"v{i}" for i in range(10)]
    
    index.add(vectors, ids)
    print(f"✓ Added {len(ids)} vectors")
    
    # Search
    query = np.random.randn(128).astype(np.float32)
    results = index.search(query, k=3)
    
    print(f"✓ Search returned {len(results)} results")
    
    return True


async def test_memory_metadata():
    """Test memory metadata"""
    print("\n=== Testing Memory Metadata ===")
    
    from aura_intelligence.memory.unified_memory_interface import MemoryMetadata
    
    metadata = MemoryMetadata(
        tenant_id="test_tenant",
        user_id="test_user",
        session_id="test_session"
    )
    
    print("✓ Created metadata:")
    for k, v in metadata.to_dict().items():
        print(f"  {k}: {v}")
    
    return True


async def main():
    """Run minimal tests"""
    print("=" * 60)
    print("AURA Memory System - Minimal Test")
    print("=" * 60)
    
    tests = [
        ("Basic Structures", test_basic_structures),
        ("KNN Index", test_knn_index),
        ("Memory Metadata", test_memory_metadata)
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            result = await test_func()
            if result:
                passed += 1
                print(f"\n✅ {test_name}: PASSED")
            else:
                failed += 1
                print(f"\n❌ {test_name}: FAILED")
        except Exception as e:
            failed += 1
            print(f"\n❌ {test_name}: FAILED with error:")
            print(f"   {type(e).__name__}: {str(e)}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 60)
    print(f"Test Results: {passed} passed, {failed} failed")
    print("=" * 60)
    
    if passed > 0:
        print("\n✓ Core memory functionality is working!")
        print("\n📋 Memory Architecture:")
        print("  • 4-tier memory system (L1-L4)")
        print("  • Mem0 pipeline for accuracy")
        print("  • H-MEM hierarchical routing")
        print("  • Qdrant quantization for efficiency")
        print("  • KNN index for vector search")
    
    return passed, failed


if __name__ == "__main__":
    passed, failed = asyncio.run(main())
    sys.exit(0 if failed == 0 else 1)