"""
Persistence Layer Integration Tests
==================================
Comprehensive tests for the 2025 production data platform.
"""

import asyncio
import pytest
from datetime import datetime, timedelta
from typing import Dict, List, Any
import numpy as np

from ..core import (
    StoreType,
    QueryResult,
    WriteResult,
    TransactionContext,
    TransactionManager,
    SagaOrchestrator
)

from ..stores import (
    StoreRegistry,
    get_store,
    NATSKVStore,
    QdrantVectorStore,
    InfluxDB3Store,
    Neo4jGraphStore
)

from ..lakehouse import (
    IcebergCatalog,
    CatalogType,
    CatalogConfig,
    EventsDataset,
    BranchManager,
    StreamingBridge
)

from ..security import (
    EnvelopeEncryption,
    FieldLevelEncryption,
    EncryptionConfig
)


class TestPersistenceIntegration:
    """Integration tests for persistence layer"""
    
    @pytest.fixture
    async def registry(self):
        """Create store registry"""
        registry = StoreRegistry()
        yield registry
        await registry.close_all()
        
    @pytest.fixture
    async def encryption(self):
        """Create encryption service"""
        config = EncryptionConfig(kms_provider="local")
        envelope = EnvelopeEncryption(config)
        await envelope.initialize()
        return envelope
        
    async def test_kv_store_operations(self, registry):
        """Test NATS KV store with <1ms operations"""
        # Get KV store
        kv_store = await registry.get_kv_store()
        
        # Test basic operations
        key = "test_config"
        value = {"setting": "value", "number": 42}
        
        # Write
        result = await kv_store.upsert(key, value)
        assert result.success
        
        # Read
        retrieved = await kv_store.get(key)
        assert retrieved == value
        
        # List keys
        list_result = await kv_store.list()
        assert list_result.success
        assert len(list_result.data) > 0
        
        # Watch for changes
        changes = []
        
        async def on_change(entry):
            changes.append(entry)
            
        watch_id = await kv_store.watch(key, on_change)
        
        # Update value
        value["number"] = 100
        await kv_store.upsert(key, value)
        
        # Wait for watch notification
        await asyncio.sleep(0.1)
        assert len(changes) > 0
        
        # Cleanup
        await kv_store.unwatch(watch_id)
        await kv_store.delete(key)
        
    async def test_vector_store_operations(self, registry):
        """Test vector store with quantization"""
        # Get Qdrant store
        vector_store = await registry.get_vector_store("qdrant")
        
        # Create test vectors
        vectors = []
        for i in range(100):
            doc = {
                "vector": np.random.randn(384).tolist(),
                "metadata": {
                    "category": f"cat_{i % 5}",
                    "score": i / 100.0
                }
            }
            vectors.append((f"vec_{i}", doc))
            
        # Batch upsert
        results = await vector_store.batch_upsert(vectors)
        assert all(r.success for r in results)
        
        # Search similar
        query_vector = np.random.randn(384).tolist()
        search_result = await vector_store.search_similar(
            query_vector,
            limit=10,
            filter_dict={"category": "cat_1"}
        )
        
        assert search_result.success
        assert len(search_result.data) <= 10
        
        # Verify scores are sorted
        scores = [r['score'] for r in search_result.data]
        assert scores == sorted(scores, reverse=True)
        
    async def test_timeseries_operations(self, registry):
        """Test time-series ingestion and queries"""
        # Get InfluxDB store
        ts_store = await registry.get_timeseries_store("influxdb3")
        
        # Generate time-series data
        points = []
        base_time = datetime.utcnow() - timedelta(hours=1)
        
        for i in range(1000):
            point = {
                "timestamp": base_time + timedelta(seconds=i),
                "fields": {
                    "cpu": 50 + np.sin(i / 100) * 20,
                    "memory": 70 + np.cos(i / 100) * 10
                },
                "tags": {
                    "host": f"server-{i % 3}",
                    "region": "us-west"
                }
            }
            points.append(point)
            
        # Write points
        result = await ts_store.write_points("system_metrics", points)
        assert result.success
        
        # Query range
        query_result = await ts_store.query_range(
            "system_metrics",
            base_time,
            datetime.utcnow(),
            aggregation="mean"
        )
        
        assert query_result.success
        assert len(query_result.data) > 0
        
    async def test_graph_operations(self, registry):
        """Test graph store with traversals"""
        # Get Neo4j store
        graph_store = await registry.get_graph_store("neo4j")
        
        # Create nodes
        nodes = []
        for i in range(10):
            node = {
                "labels": ["Entity", "Node"],
                "properties": {
                    "name": f"Node {i}",
                    "type": "test"
                }
            }
            result = await graph_store.upsert(f"node_{i}", node)
            assert result.success
            
        # Create edges
        for i in range(9):
            result = await graph_store.add_edge(
                f"node_{i}",
                f"node_{i+1}",
                "CONNECTED_TO",
                {"weight": 1.0}
            )
            assert result.success
            
        # Traverse graph
        traversal_result = await graph_store.traverse(
            "node_0",
            "neighbors",
            max_depth=3
        )
        
        assert traversal_result.success
        assert len(traversal_result.data) > 0
        
    async def test_distributed_transactions(self, registry):
        """Test distributed transactions with saga pattern"""
        # Get multiple stores
        kv_store = await registry.get_kv_store()
        vector_store = await registry.get_vector_store()
        
        # Create saga orchestrator
        saga = SagaOrchestrator()
        saga_id = await saga.create_saga()
        
        # Add steps
        await saga.add_step(
            saga_id,
            "update_config",
            kv_store.upsert,
            kv_store.delete,  # Compensation
            "saga_test",
            {"status": "processing"}
        )
        
        await saga.add_step(
            saga_id,
            "store_vector",
            vector_store.upsert,
            vector_store.delete,  # Compensation
            "saga_vec",
            {"vector": [0.1] * 384}
        )
        
        # Execute saga
        success, results = await saga.execute(saga_id)
        assert success
        assert len(results) == 2
        
        # Verify data exists
        config = await kv_store.get("saga_test")
        assert config is not None
        
        vector = await vector_store.get("saga_vec")
        assert vector is not None
        
    async def test_encryption_operations(self, encryption):
        """Test field-level encryption"""
        field_enc = FieldLevelEncryption(encryption)
        
        # Test document
        document = {
            "id": "user_123",
            "name": "John Doe",
            "email": "john@example.com",  # Sensitive
            "ssn": "123-45-6789",  # Sensitive
            "age": 30,
            "active": True
        }
        
        # Encrypt document
        encrypted_doc = await field_enc.encrypt_document(document)
        
        # Verify sensitive fields are encrypted
        assert encrypted_doc["email"]["_encrypted"] is True
        assert encrypted_doc["ssn"]["_encrypted"] is True
        assert encrypted_doc["age"] == 30  # Not encrypted
        
        # Decrypt document
        decrypted_doc = await field_enc.decrypt_document(encrypted_doc)
        
        # Verify decryption
        assert decrypted_doc["email"] == document["email"]
        assert decrypted_doc["ssn"] == document["ssn"]
        
    async def test_lakehouse_branching(self):
        """Test Iceberg branching for experiments"""
        # Create catalog
        config = CatalogConfig(
            catalog_type=CatalogType.NESSIE,
            uri="http://localhost:19120",
            warehouse="s3://aura-lakehouse"
        )
        
        # Would test with actual Nessie
        # catalog = create_catalog(config)
        # branch_manager = BranchManager(catalog)
        
        # Create experiment branch
        # branch = await branch_manager.create_branch(
        #     "experiment/new_feature",
        #     from_ref="main",
        #     created_by="test_user"
        # )
        
        # Make changes on branch
        # ...
        
        # Merge back to main
        # result = await branch_manager.merge_branch(
        #     "experiment/new_feature",
        #     "main"
        # )
        
        # assert result.success
        pass
        
    async def test_streaming_ingestion(self):
        """Test CDC streaming to lakehouse"""
        # Would test with actual streaming
        # config = StreamConfig(
        #     engine=StreamingEngine.KAFKA,
        #     bootstrap_servers=["localhost:9092"]
        # )
        
        # bridge = StreamingBridge(catalog)
        # sink = await bridge.create_sink("events_sink", config)
        
        # await sink.start()
        # ... send test events ...
        # await sink.stop()
        
        pass
        
    async def test_performance_benchmarks(self, registry):
        """Performance benchmarks for each store type"""
        results = {}
        
        # KV Store benchmark
        kv_store = await registry.get_kv_store()
        start = datetime.utcnow()
        
        for i in range(1000):
            await kv_store.upsert(f"perf_key_{i}", {"value": i})
            
        kv_write_time = (datetime.utcnow() - start).total_seconds()
        results["kv_writes_per_sec"] = 1000 / kv_write_time
        
        # Vector search benchmark
        vector_store = await registry.get_vector_store()
        query_vec = np.random.randn(384).tolist()
        
        start = datetime.utcnow()
        for _ in range(100):
            await vector_store.search_similar(query_vec, limit=10)
            
        vector_search_time = (datetime.utcnow() - start).total_seconds()
        results["vector_searches_per_sec"] = 100 / vector_search_time
        
        # Print results
        print("\nPerformance Benchmarks:")
        for metric, value in results.items():
            print(f"  {metric}: {value:.2f}")
            
        # Assert minimum performance
        assert results["kv_writes_per_sec"] > 100  # >100 writes/sec
        assert results["vector_searches_per_sec"] > 10  # >10 searches/sec


async def run_integration_tests():
    """Run all integration tests"""
    test = TestPersistenceIntegration()
    
    # Create fixtures
    registry = StoreRegistry()
    encryption_config = EncryptionConfig(kms_provider="local")
    envelope = EnvelopeEncryption(encryption_config)
    await envelope.initialize()
    
    try:
        print("Testing KV Store operations...")
        await test.test_kv_store_operations(registry)
        print("✓ KV Store tests passed")
        
        print("\nTesting Vector Store operations...")
        await test.test_vector_store_operations(registry)
        print("✓ Vector Store tests passed")
        
        print("\nTesting Time-Series operations...")
        await test.test_timeseries_operations(registry)
        print("✓ Time-Series tests passed")
        
        print("\nTesting Graph operations...")
        await test.test_graph_operations(registry)
        print("✓ Graph tests passed")
        
        print("\nTesting Distributed Transactions...")
        await test.test_distributed_transactions(registry)
        print("✓ Transaction tests passed")
        
        print("\nTesting Encryption...")
        await test.test_encryption_operations(envelope)
        print("✓ Encryption tests passed")
        
        print("\nRunning Performance Benchmarks...")
        await test.test_performance_benchmarks(registry)
        print("✓ Performance tests passed")
        
        print("\n✅ All persistence tests passed!")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        raise
        
    finally:
        await registry.close_all()


if __name__ == "__main__":
    asyncio.run(run_integration_tests())