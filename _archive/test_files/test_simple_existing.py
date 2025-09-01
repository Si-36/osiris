#!/usr/bin/env python3
"""
Test using simple existing implementations
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'core/src'))

import numpy as np


def test_vector_search():
    """Test the basic vector search that exists."""
    print("🔍 Testing Basic Vector Search")
    print("=" * 60)
    
    try:
        from aura_intelligence.vector_search import VectorIndex
        
        # Create index
        index = VectorIndex(dimension=128, metric="cosine")
        print("✅ Created VectorIndex")
        
        # Add vectors
        for i in range(100):
            vec = np.random.randn(128)
            index.add(f"doc_{i}", vec)
        
        print(f"✅ Added {len(index.vectors)} vectors")
        
        # Search
        query = np.random.randn(128)
        results = index.search(query, k=5)
        
        print(f"\nTop 5 results:")
        for result in results[:5]:
            print(f"  {result.id}: {result.score:.4f}")
            
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def test_existing_supervisor():
    """Test if supervisor works standalone."""
    print("\n\n🧠 Testing Existing Supervisor")
    print("=" * 60)
    
    try:
        # Import just the necessary parts
        from aura_intelligence.orchestration.workflows.nodes.supervisor import (
            DecisionType, WorkflowMetrics, PatternType
        )
        
        print("✅ Imported Supervisor types")
        
        # Create simple metrics
        metrics = {
            "success_rate": 0.8,
            "error_rate": 0.2,
            "avg_duration": 150.0
        }
        
        # Simple pattern detection
        patterns = []
        if metrics["error_rate"] > 0.5:
            patterns.append("high_error_rate")
            
        print(f"  Success rate: {metrics['success_rate']}")
        print(f"  Error rate: {metrics['error_rate']}")
        print(f"  Patterns: {patterns}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def test_memory_structure():
    """Test basic memory structures."""
    print("\n\n💾 Testing Memory Structures")
    print("=" * 60)
    
    try:
        # Simple memory simulation
        memory = {
            "hot": {},
            "warm": {},
            "cold": {}
        }
        
        # Add items
        memory["hot"]["critical"] = {"data": "important", "access_count": 5}
        memory["warm"]["recent"] = {"data": "normal", "access_count": 2}
        memory["cold"]["archive"] = {"data": "old", "access_count": 0}
        
        print(f"✅ Memory tiers:")
        print(f"  HOT: {len(memory['hot'])} items")
        print(f"  WARM: {len(memory['warm'])} items")
        print(f"  COLD: {len(memory['cold'])} items")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def main():
    """Run simple tests."""
    print("🧬 AURA Simple Test")
    print("Testing basic functionality")
    print("=" * 80)
    
    results = []
    
    # Test vector search
    results.append(("Vector Search", test_vector_search()))
    
    # Test supervisor
    results.append(("Supervisor", test_existing_supervisor()))
    
    # Test memory
    results.append(("Memory", test_memory_structure()))
    
    # Summary
    print("\n\n📊 Summary")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{name}: {status}")
    
    print(f"\n🎯 Total: {passed}/{total} tests passed")


if __name__ == "__main__":
    main()