#!/usr/bin/env python3
"""
Basic Memory System Test - No External Dependencies
==================================================

Tests core memory functionality without requiring FAISS/Annoy.
"""

import asyncio
import time
import numpy as np
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'core/src'))


async def test_memory_components():
    """Test individual memory components"""
    print("\n=== Testing Memory Components ===")
    
    # Test 1: Memory metadata
    print("\n1. Testing memory metadata...")
    from aura_intelligence.memory.unified_memory_interface import (
        MemoryMetadata, MemoryType
    )
    
    metadata = MemoryMetadata(
        tenant_id="test_tenant",
        user_id="test_user",
        type=MemoryType.EPISODIC
    )
    print(f"✓ Created metadata: {metadata.to_dict()}")
    
    # Test 2: Mem0 fact extraction structures
    print("\n2. Testing Mem0 structures...")
    from aura_intelligence.memory.mem0_pipeline import (
        ExtractedFact, FactType, ConfidenceLevel
    )
    
    fact = ExtractedFact(
        fact_id="fact_001",
        fact_type=FactType.ENTITY,
        subject="AURA",
        predicate="is",
        object="AI system",
        confidence=0.95,
        source_id="test"
    )
    print(f"✓ Created fact: {fact.subject} {fact.predicate} {fact.object}")
    print(f"  Confidence level: {fact.confidence_level.value}")
    
    # Test 3: H-MEM memory levels
    print("\n3. Testing H-MEM structures...")
    from aura_intelligence.memory.hierarchical_routing import (
        MemoryLevel, PositionalEncoding
    )
    
    level = MemoryLevel(
        level=0,
        name="Test Memory",
        summary="Test summary",
        embedding=np.random.randn(768)
    )
    print(f"✓ Created memory level: {level.name} (level {level.level})")
    
    pos_encoding = PositionalEncoding(
        level_encoding=np.random.randn(32),
        temporal_encoding=np.random.randn(32),
        semantic_encoding=np.random.randn(32)
    )
    combined = pos_encoding.to_vector()
    print(f"✓ Created positional encoding: shape {combined.shape}")
    
    # Test 4: Qdrant configuration
    print("\n4. Testing Qdrant configuration...")
    from aura_intelligence.memory.qdrant_config import (
        QdrantMultitenantManager, QuantizationPreset
    )
    
    manager = QdrantMultitenantManager()
    
    # Test RAM savings estimation
    for preset in [QuantizationPreset.BALANCED, QuantizationPreset.MAXIMUM_COMPRESSION]:
        savings = manager.estimate_ram_savings(
            vector_count=100_000,
            vector_size=768,
            quantization_preset=preset
        )
        print(f"\n✓ {preset.value}:")
        print(f"  Original: {savings['original_size_mb']:.1f} MB")
        print(f"  Quantized: {savings['quantized_size_mb']:.1f} MB")
        print(f"  Savings: {savings['savings_percent']}%")
    
    # Test tenant filter
    tenant_filter = manager.get_tenant_filter("tenant_123")
    print(f"\n✓ Tenant filter: {tenant_filter}")
    
    # Test search params
    search_params = manager.get_search_params("balanced")
    print(f"✓ Search params: hnsw_ef={search_params['hnsw_ef']}")
    
    return True


async def test_memory_pipeline():
    """Test memory pipeline logic without dependencies"""
    print("\n=== Testing Memory Pipeline Logic ===")
    
    from aura_intelligence.memory.mem0_pipeline import Mem0Pipeline
    
    pipeline = Mem0Pipeline()
    
    # Test confidence scoring
    print("\n1. Testing confidence logic...")
    
    # Test extraction patterns
    content = "AURA is an AI system"
    extractions = await pipeline._pattern_extraction(content)
    print(f"✓ Pattern extraction: {len(extractions)} patterns found")
    
    # Test confidence scoring
    if extractions:
        confidence = await pipeline._score_confidence(
            extractions[0], content, None
        )
        print(f"✓ Confidence scoring: {confidence:.2f}")
    
    # Test token savings computation
    print("\n2. Testing token savings...")
    
    from aura_intelligence.memory.mem0_pipeline import ExtractedFact, FactType
    
    facts = [
        ExtractedFact(
            fact_id=f"f{i}",
            fact_type=FactType.RELATION,
            subject=f"subject_{i}",
            predicate="relates_to",
            object=f"object_{i}",
            confidence=0.8,
            source_id="test"
        )
        for i in range(5)
    ]
    
    raw_candidates = ["long raw text " * 20 for _ in range(10)]
    
    savings = pipeline._compute_token_savings(facts, raw_candidates)
    print(f"✓ Token savings: {savings['percentage']}%")
    print(f"  Raw tokens: {savings['raw_tokens']}")
    print(f"  Structured tokens: {savings['structured_tokens']}")
    
    return True


async def test_hierarchical_routing():
    """Test H-MEM routing logic"""
    print("\n=== Testing H-MEM Routing Logic ===")
    
    from aura_intelligence.memory.hierarchical_routing import HMemSystem
    
    hmem = HMemSystem(max_levels=3)
    
    # Test similarity computation
    print("\n1. Testing similarity computation...")
    
    vec1 = np.random.randn(768)
    vec2 = np.random.randn(768)
    
    # Normalize for cosine similarity test
    vec1 = vec1 / np.linalg.norm(vec1)
    vec2 = vec2 / np.linalg.norm(vec2)
    
    similarity = hmem._compute_similarity(vec1, vec2)
    print(f"✓ Cosine similarity: {similarity:.3f}")
    
    # Test clustering logic
    print("\n2. Testing clustering logic...")
    
    memory_indices = [f"mem_{i}" for i in range(20)]
    clusters = await hmem._cluster_memories(memory_indices, target_level=1)
    
    print(f"✓ Created {len(clusters)} clusters:")
    for cluster_id, members in list(clusters.items())[:3]:
        print(f"  Cluster {cluster_id}: {len(members)} members")
    
    # Test metrics
    print("\n3. H-MEM metrics:")
    metrics = hmem.get_metrics()
    for key, value in metrics.items():
        print(f"  {key}: {value}")
    
    return True


async def test_configuration():
    """Test configuration and setup"""
    print("\n=== Testing Configuration ===")
    
    # Test Qdrant collection config
    from aura_intelligence.memory.qdrant_config import (
        QdrantCollectionConfig,
        QuantizationPreset,
        HNSWConfig,
        ShardingConfig
    )
    
    config = QdrantCollectionConfig(
        name="test_collection",
        vector_size=768,
        quantization_preset=QuantizationPreset.BALANCED,
        hnsw_config=HNSWConfig(m=16, ef_construct=200),
        sharding_config=ShardingConfig(shard_key="region", shard_number=3)
    )
    
    qdrant_config = config.to_qdrant_config()
    
    print("✓ Created Qdrant configuration:")
    print(f"  Vector size: {qdrant_config['vectors']['size']}")
    print(f"  Distance: {qdrant_config['vectors']['distance']}")
    print(f"  HNSW M: {qdrant_config['hnsw_config']['m']}")
    print(f"  Quantization: {list(qdrant_config.get('quantization_config', {}).keys())}")
    
    return True


async def main():
    """Run all tests"""
    print("=" * 60)
    print("AURA Memory System - Basic Test Suite")
    print("=" * 60)
    
    tests = [
        ("Memory Components", test_memory_components),
        ("Memory Pipeline", test_memory_pipeline),
        ("Hierarchical Routing", test_hierarchical_routing),
        ("Configuration", test_configuration)
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
    
    # Architecture summary
    print("\n📐 Memory Architecture Summary:")
    print("  L1 Hot: Redis for <1ms working memory")
    print("  L2 Warm: Qdrant vectors with quantization")
    print("  L3 Semantic: Neo4j GraphRAG")
    print("  L4 Cold: Iceberg for archival")
    print("\n  Mem0: Extract→Update→Retrieve pipeline")
    print("  H-MEM: Hierarchical routing for efficiency")
    print("  Multitenancy: Single collection with filters")
    
    return passed, failed


if __name__ == "__main__":
    passed, failed = asyncio.run(main())
    sys.exit(0 if failed == 0 else 1)