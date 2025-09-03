#!/usr/bin/env python3
"""
Test UnifiedCognitiveMemory - The Complete Memory System
========================================================

This tests the full cognitive memory architecture:
- Experience processing (write path)
- Query execution (read path)
- Memory transfers (lifecycle)
- Consolidation (learning)
"""

import asyncio
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'core/src'))

print("=" * 60)
print("UNIFIED COGNITIVE MEMORY TEST")
print("=" * 60)


async def test_unified_memory():
    """Test the complete memory system"""
    
    print("\n1️⃣ Initializing UnifiedCognitiveMemory...")
    
    try:
        from aura_intelligence.memory.unified_cognitive_memory import (
            UnifiedCognitiveMemory,
            QueryType,
            MemoryContext
        )
        
        # Initialize with config
        config = {
            'working': {'capacity': 7},
            'episodic': {
                'redis_host': 'localhost',
                'redis_port': 6379,
                'lmdb_path': '/tmp/test_episodic_lmdb',
                'duckdb_path': '/tmp/test_episodic.duckdb'
            },
            'semantic': {
                'neo4j_uri': 'bolt://localhost:7687',
                'neo4j_user': 'neo4j',
                'neo4j_password': 'password'
            },
            'use_cuda': False,
            'cache_size': 50,
            'sleep_interval_hours': 0.01  # Short for testing
        }
        
        memory = UnifiedCognitiveMemory(config)
        print("✅ UnifiedCognitiveMemory initialized")
        
        # Start the system
        await memory.start()
        print("✅ Memory system started")
        
    except Exception as e:
        print(f"❌ Initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # ==================== Test Write Path ====================
    
    print("\n2️⃣ Testing WRITE path (Experience → Memory)...")
    
    try:
        # Process various experiences
        experiences = [
            {
                'content': "Learning about neural networks",
                'context': {
                    'importance': 0.9,
                    'emotional': {'valence': 0.8, 'arousal': 0.6},
                    'spatial': {'location_id': 'lab'},
                    'social': {'participants': ['assistant', 'user']}
                }
            },
            {
                'content': "Debugging memory overflow issue",
                'context': {
                    'importance': 0.7,
                    'surprise': 0.8,
                    'causal': {'trigger_episodes': ['learning_phase']}
                }
            },
            {
                'content': "Successfully implemented episodic memory",
                'context': {
                    'importance': 0.95,
                    'emotional': {'valence': 0.9, 'arousal': 0.8}
                }
            }
        ]
        
        for i, exp in enumerate(experiences, 1):
            result = await memory.process_experience(
                content=exp['content'],
                context=exp['context']
            )
            
            if result['success']:
                print(f"  ✅ Experience {i} processed: {exp['content'][:30]}...")
                if 'episode_id' in result:
                    print(f"     → High importance triggered immediate episodic storage")
            else:
                print(f"  ❌ Failed to process experience {i}: {result['error']}")
        
        print(f"\n  📊 Metrics: {memory.metrics['total_writes']} total writes")
        
    except Exception as e:
        print(f"❌ Write path test failed: {e}")
    
    # ==================== Test Read Path ====================
    
    print("\n3️⃣ Testing READ path (Query → Retrieval → Synthesis)...")
    
    queries = [
        "What did I learn about neural networks?",
        "Why did the memory overflow happen?",
        "What are my successful implementations?",
        "What do I know about debugging?",
        "How do episodic and semantic memory relate?"
    ]
    
    for query in queries:
        try:
            print(f"\n  🔍 Query: '{query}'")
            
            # Execute query
            context = await memory.query(query, timeout=3.0)
            
            print(f"     → Sources: {context.total_sources} memories retrieved")
            print(f"     → Confidence: {context.confidence:.2f}")
            print(f"     → Grounding: {context.grounding_strength:.2f}")
            
            if context.synthesized_answer:
                answer = context.synthesized_answer[:100]
                print(f"     → Answer: {answer}...")
            
            print(f"     → Time: {context.retrieval_time_ms}ms retrieval, {context.synthesis_time_ms}ms synthesis")
            
        except Exception as e:
            print(f"  ❌ Query failed: {e}")
    
    print(f"\n  📊 Query metrics: {memory.metrics['successful_queries']}/{memory.metrics['total_queries']} successful")
    
    # ==================== Test Memory Lifecycle ====================
    
    print("\n4️⃣ Testing Memory Lifecycle (Transfers & Consolidation)...")
    
    try:
        # Fill working memory to trigger overflow
        print("  📝 Filling working memory to trigger overflow...")
        for i in range(10):
            await memory.process_experience(
                content=f"Overflow test item {i}",
                context={'importance': 0.5}
            )
        
        # Check transfer counts
        transfers = memory.lifecycle_manager.transfer_counts
        print(f"  ✅ Transfers: {dict(transfers)}")
        
        # Trigger awake consolidation
        print("\n  🧠 Triggering awake consolidation...")
        await memory.run_awake_consolidation()
        print("  ✅ Awake consolidation completed")
        
        # Run sleep cycle
        print("\n  😴 Running sleep cycle...")
        await memory.run_sleep_cycle()
        print(f"  ✅ Sleep cycle completed")
        print(f"     → Consolidation cycles: {memory.metrics['consolidation_cycles']}")
        
    except Exception as e:
        print(f"❌ Lifecycle test failed: {e}")
    
    # ==================== Test Different Query Types ====================
    
    print("\n5️⃣ Testing Different Query Types...")
    
    query_tests = [
        ("What is episodic memory?", QueryType.FACTUAL),
        ("What happened during debugging?", QueryType.EPISODIC),
        ("Why did the implementation succeed?", QueryType.CAUSAL),
        ("What patterns exist in my learning?", QueryType.ANALYTICAL),
        ("What do I know about my own knowledge?", QueryType.METACOGNITIVE)
    ]
    
    for query_text, expected_type in query_tests:
        try:
            print(f"\n  🎯 {expected_type.value}: '{query_text}'")
            
            context = await memory.query(query_text, timeout=2.0)
            
            if context.synthesized_answer:
                print(f"     → {context.synthesized_answer[:80]}...")
                
        except Exception as e:
            print(f"  ❌ {expected_type.value} query failed: {e}")
    
    # ==================== Test System Statistics ====================
    
    print("\n6️⃣ System Statistics & Health Check...")
    
    try:
        # Get statistics
        stats = await memory.get_statistics()
        
        print("\n  📊 System Statistics:")
        print(f"     • Working Memory: {stats['working_memory']['current_items']} items")
        print(f"     • Episodic Memory: {stats['episodic_memory'].get('total_episodes', 0)} episodes")
        print(f"     • Semantic Memory: {stats['semantic_memory'].get('total_concepts', 0)} concepts")
        print(f"     • Cache: {stats['cache_size']} cached queries")
        print(f"     • Last Sleep: {stats['last_sleep']}")
        
        # Health check
        health = await memory.health_check()
        print("\n  🏥 Health Check:")
        for component, status in health.items():
            icon = "✅" if status else "❌"
            print(f"     {icon} {component}: {'healthy' if status else 'unhealthy'}")
        
    except Exception as e:
        print(f"❌ Statistics failed: {e}")
    
    # ==================== Cleanup ====================
    
    print("\n7️⃣ Cleanup...")
    
    try:
        await memory.stop()
        print("✅ Memory system stopped successfully")
        
    except Exception as e:
        print(f"❌ Cleanup failed: {e}")
    
    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)


# Run the test
if __name__ == "__main__":
    print("\n⚠️  Note: This test requires Redis and Neo4j to be running")
    print("   Run ./setup_databases.sh if needed\n")
    
    try:
        asyncio.run(test_unified_memory())
    except KeyboardInterrupt:
        print("\n\n⚠️ Test interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()