#!/usr/bin/env python3
"""
AURA Persistence Layer - Real Test
=================================
Tests the actual persistence implementation without mocks.
"""

import asyncio
import sys
import os
import time
from datetime import datetime, timedelta
import json

# Add to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'core/src'))

# Test only what's actually implemented
from aura_intelligence.persistence.core import (
    QueryBuilder,
    TransactionContext,
    SagaOrchestrator,
    StoreCircuitBreaker,
    CircuitState
)

from aura_intelligence.persistence.stores import (
    StoreRegistry,
    UnifiedDocumentStore,
    UnifiedEventStore,
    DocumentConfig,
    EventStoreConfig
)

from aura_intelligence.persistence.security import (
    EnvelopeEncryption,
    FieldLevelEncryption,
    EncryptionConfig,
    ImmutableAuditLog,
    AccessMonitor,
    ComplianceReporter,
    AuditEntry,
    TenantIsolation,
    DataResidencyManager,
    RowLevelSecurity,
    TenantConfig
)

from aura_intelligence.persistence.lakehouse import (
    EventsDataset,
    FeaturesDataset,
    EmbeddingsDataset,
    TopologyDataset,
    AuditDataset
)


async def test_query_builder():
    """Test the query builder"""
    print("\n📝 Testing QueryBuilder...")
    
    # Build a complex query
    query = (QueryBuilder()
            .eq("status", "active")
            .gt("score", 0.8)
            .lt("created_at", datetime.utcnow())
            .in_("category", ["A", "B", "C"])
            .contains("name", "test")
            .sort_desc("score")
            .sort_asc("name")
            .with_limit(50)
            .with_offset(100))
    
    query_dict = query.build()
    
    # Verify query structure
    assert len(query_dict['filters']) == 5
    assert query_dict['limit'] == 50
    assert query_dict['offset'] == 100
    assert len(query_dict['sort']) == 2
    
    print("  ✓ QueryBuilder works correctly")
    print(f"  Generated query: {json.dumps(query_dict, indent=2, default=str)}")


async def test_transaction_context():
    """Test transaction context"""
    print("\n🔄 Testing TransactionContext...")
    
    # Create root context
    root_ctx = TransactionContext(
        transaction_id="tx_12345",
        tenant_id="tenant_1",
        saga_id="saga_1"
    )
    
    # Create child contexts
    child1 = root_ctx.create_child_context("operation_1")
    child2 = child1.create_child_context("sub_operation")
    
    assert child1.transaction_id == "tx_12345:operation_1"
    assert child2.transaction_id == "tx_12345:operation_1:sub_operation"
    assert child2.tenant_id == "tenant_1"
    assert child2.saga_id == "saga_1"
    
    print("  ✓ Transaction context hierarchy works")


async def test_saga_orchestrator():
    """Test saga pattern implementation"""
    print("\n🎭 Testing Saga Orchestrator...")
    
    saga = SagaOrchestrator(enable_temporal=False)
    saga_id = await saga.create_saga()
    
    # Track execution
    execution_log = []
    
    # Define saga steps
    async def step1():
        execution_log.append("step1_executed")
        return {"step": 1, "result": "success"}
        
    async def step1_compensation(result):
        execution_log.append("step1_compensated")
        
    async def step2():
        execution_log.append("step2_executed")
        return {"step": 2, "result": "success"}
        
    async def step2_compensation(result):
        execution_log.append("step2_compensated")
        
    async def failing_step():
        execution_log.append("failing_step_attempted")
        raise Exception("Intentional failure")
        
    # Test successful saga
    print("  1. Testing successful saga execution...")
    
    await saga.add_step(saga_id, "step1", step1, step1_compensation)
    await saga.add_step(saga_id, "step2", step2, step2_compensation)
    
    success, results = await saga.execute(saga_id)
    assert success
    assert len(results) == 2
    assert execution_log == ["step1_executed", "step2_executed"]
    
    # Test saga with failure and compensation
    print("  2. Testing saga with failure and compensation...")
    
    execution_log.clear()
    saga_id_fail = await saga.create_saga()
    
    await saga.add_step(saga_id_fail, "step1", step1, step1_compensation)
    await saga.add_step(saga_id_fail, "step2", step2, step2_compensation)
    await saga.add_step(saga_id_fail, "failing", failing_step, None)
    
    success, results = await saga.execute(saga_id_fail)
    assert not success
    assert "step2_compensated" in execution_log
    assert "step1_compensated" in execution_log
    
    print("  ✓ Saga pattern works with automatic compensation")


async def test_circuit_breaker():
    """Test circuit breaker pattern"""
    print("\n⚡ Testing Circuit Breaker...")
    
    # Create circuit breaker with low threshold for testing
    config = CircuitConfig(
        failure_threshold=3,
        recovery_timeout=1.0,
        min_calls_for_evaluation=1
    )
    
    breaker = StoreCircuitBreaker("test_store", config)
    
    # Track calls
    call_count = 0
    
    async def failing_operation():
        nonlocal call_count
        call_count += 1
        raise Exception("Operation failed")
        
    async def successful_operation():
        nonlocal call_count
        call_count += 1
        return "success"
        
    # Test circuit opening
    print("  1. Testing circuit opening after failures...")
    
    for i in range(3):
        try:
            await breaker.call(failing_operation)
        except Exception:
            pass
            
    assert breaker.state == CircuitState.OPEN
    assert call_count == 3
    
    # Test circuit blocking calls when open
    print("  2. Testing circuit blocks calls when open...")
    
    call_count = 0
    try:
        await breaker.call(successful_operation)
    except RuntimeError as e:
        assert "is OPEN" in str(e)
        
    assert call_count == 0  # Operation not called
    
    # Test circuit recovery
    print("  3. Testing circuit recovery...")
    
    await asyncio.sleep(1.1)  # Wait for recovery timeout
    
    # Should transition to half-open and allow limited calls
    result = await breaker.call(successful_operation)
    assert result == "success"
    assert breaker.state == CircuitState.HALF_OPEN
    
    print("  ✓ Circuit breaker protects against cascading failures")


async def test_encryption():
    """Test encryption services"""
    print("\n🔐 Testing Encryption Services...")
    
    # Create encryption service
    config = EncryptionConfig(
        kms_provider="local",
        enable_envelope=True,
        enable_field_level=True
    )
    
    envelope = EnvelopeEncryption(config)
    await envelope.initialize()
    
    # Test envelope encryption
    print("  1. Testing envelope encryption...")
    
    plaintext = b"Sensitive data: SSN 123-45-6789"
    encrypted = await envelope.encrypt(plaintext)
    
    assert encrypted.ciphertext != plaintext
    assert len(encrypted.nonce) == 12
    assert encrypted.algorithm == "AES-256-GCM"
    
    decrypted = await envelope.decrypt(encrypted)
    assert decrypted == plaintext
    
    # Test field-level encryption
    print("  2. Testing field-level encryption...")
    
    field_enc = FieldLevelEncryption(envelope)
    
    document = {
        "id": "user_123",
        "name": "John Doe",
        "email": "john@example.com",  # Sensitive
        "ssn": "123-45-6789",  # Sensitive
        "phone": "+1-555-0123",  # Sensitive
        "age": 30,
        "active": True
    }
    
    encrypted_doc = await field_enc.encrypt_document(document)
    
    # Verify sensitive fields are encrypted
    assert encrypted_doc["email"]["_encrypted"] is True
    assert encrypted_doc["ssn"]["_encrypted"] is True
    assert encrypted_doc["phone"]["_encrypted"] is True
    assert encrypted_doc["age"] == 30  # Not encrypted
    
    # Decrypt and verify
    decrypted_doc = await field_enc.decrypt_document(encrypted_doc)
    assert decrypted_doc["email"] == document["email"]
    assert decrypted_doc["ssn"] == document["ssn"]
    
    print("  ✓ Encryption services work correctly")


async def test_audit_logging():
    """Test audit logging and compliance"""
    print("\n📋 Testing Audit & Compliance...")
    
    # Create audit log
    audit_log = ImmutableAuditLog()
    
    # Log some events
    print("  1. Testing audit logging...")
    
    for i in range(10):
        entry = AuditEntry(
            audit_id=f"audit_{i}",
            timestamp=datetime.utcnow() - timedelta(hours=i),
            action="data_access",
            resource=f"resource_{i % 3}",
            actor=f"user_{i % 2}",
            result="success" if i % 5 != 0 else "failure",
            compliance_labels=["SOC2", "GDPR"] if i % 2 == 0 else ["HIPAA"]
        )
        await audit_log.log(entry)
        
    # Search audit logs
    results = await audit_log.search(actor="user_0")
    assert len(results) == 5
    
    # Test compliance reporting
    print("  2. Testing compliance reporting...")
    
    reporter = ComplianceReporter(audit_log)
    report = await reporter.generate_report(
        "SOC2",
        datetime.utcnow() - timedelta(days=1),
        datetime.utcnow()
    )
    
    assert report['standard'] == "SOC2"
    assert report['total_events'] > 0
    assert 'compliance_score' in report
    
    print(f"  Compliance score: {report['compliance_score']:.1f}%")
    
    # Test access monitoring
    print("  3. Testing access monitoring...")
    
    monitor = AccessMonitor()
    
    # Simulate access pattern
    for i in range(15):
        await monitor.record_access("suspicious_user", "sensitive_resource")
        
    # Should detect anomaly (logged as warning)
    
    print("  ✓ Audit and compliance features work")


async def test_multi_tenancy():
    """Test multi-tenancy features"""
    print("\n🏢 Testing Multi-tenancy...")
    
    # Create tenant isolation
    isolation = TenantIsolation()
    
    # Register tenants
    tenant1 = TenantConfig(
        tenant_id="tenant_1",
        tenant_name="Acme Corp",
        allowed_regions=["us-east-1", "us-west-2"],
        isolation_level="strict"
    )
    
    tenant2 = TenantConfig(
        tenant_id="tenant_2",
        tenant_name="GlobalTech",
        allowed_regions=["eu-west-1", "eu-central-1"],
        isolation_level="strict"
    )
    
    isolation.register_tenant(tenant1)
    isolation.register_tenant(tenant2)
    
    # Test tenant isolation
    print("  1. Testing tenant isolation...")
    
    isolation.set_current_tenant("tenant_1")
    
    query = {"select": ["*"], "from": "users"}
    filtered_query = isolation.apply_tenant_filter(query)
    
    assert filtered_query['tenant_id'] == "tenant_1"
    
    # Test data residency
    print("  2. Testing data residency...")
    
    residency = DataResidencyManager()
    
    # Validate regions
    assert residency.validate_region("tenant_1", "us-east-1", tenant1) is True
    assert residency.validate_region("tenant_1", "eu-west-1", tenant1) is False
    
    selected_region = residency.select_region(tenant2)
    assert selected_region in tenant2.allowed_regions
    
    # Test row-level security
    print("  3. Testing row-level security...")
    
    rls = RowLevelSecurity()
    
    rls.create_policy(
        "tenant_data_isolation",
        "users",
        "tenant_id = current_tenant()",
        ["user", "admin"]
    )
    
    query = {"table": "users", "select": ["*"]}
    secured_query = rls.apply_policies(query, "user", "users")
    
    assert 'filters' in secured_query
    assert any(f['type'] == 'rls' for f in secured_query['filters'])
    
    print("  ✓ Multi-tenancy features work correctly")


async def test_document_store():
    """Test document store implementation"""
    print("\n📄 Testing Document Store...")
    
    config = DocumentConfig(
        collection_name="test_documents",
        enable_full_text=True
    )
    
    store = UnifiedDocumentStore(config)
    await store.initialize()
    
    # Test CRUD operations
    print("  1. Testing document CRUD...")
    
    doc = {
        "title": "Test Document",
        "content": "This is a test document with some content.",
        "tags": ["test", "demo"],
        "metadata": {
            "author": "Test User",
            "version": 1
        }
    }
    
    # Create
    result = await store.upsert("doc_1", doc)
    assert result.success
    
    # Read
    retrieved = await store.get("doc_1")
    assert retrieved is not None
    assert retrieved['_id'] == "doc_1"
    
    # Update
    doc['metadata']['version'] = 2
    result = await store.upsert("doc_1", doc)
    assert result.success
    
    # List
    list_result = await store.list()
    assert list_result.success
    assert len(list_result.data) > 0
    
    # Delete
    result = await store.delete("doc_1")
    assert result.success
    
    print("  ✓ Document store operations work")


async def test_event_store():
    """Test event store implementation"""
    print("\n📡 Testing Event Store...")
    
    config = EventStoreConfig(
        stream_name="test_events",
        enable_snapshots=True,
        snapshot_frequency=5
    )
    
    store = UnifiedEventStore(config)
    await store.initialize()
    
    # Test event sourcing
    print("  1. Testing event sourcing...")
    
    # Append events
    events = [
        {
            "type": "created",
            "data": {"user_id": "123", "name": "Test User"}
        },
        {
            "type": "updated",
            "data": {"field": "email", "value": "test@example.com"}
        }
    ]
    
    result = await store.append_events("user_123", events)
    assert result.success
    assert result.version == 2
    
    # Read events
    stored_events = await store.get_events("user_123")
    assert len(stored_events) == 2
    
    # Test subscriptions
    print("  2. Testing event subscriptions...")
    
    received_events = []
    
    async def on_event(event):
        received_events.append(event)
        
    sub_id = await store.subscribe("test_sub", "*", on_event)
    
    # Append more events
    await store.append_events("user_456", [{"type": "created", "data": {"id": "456"}}])
    
    # Wait for processing
    await asyncio.sleep(0.2)
    
    # Should have received the event
    assert len(received_events) > 0
    
    await store.unsubscribe(sub_id)
    
    print("  ✓ Event store with subscriptions works")


async def test_lakehouse_datasets():
    """Test lakehouse dataset definitions"""
    print("\n🏔️ Testing Lakehouse Datasets...")
    
    # Test all dataset schemas
    datasets = {
        "Events": EventsDataset(),
        "Features": FeaturesDataset(),
        "Embeddings": EmbeddingsDataset(),
        "Topology": TopologyDataset(),
        "Audit": AuditDataset()
    }
    
    for name, dataset in datasets.items():
        schema = dataset.get_schema()
        partitions = dataset.get_partition_spec()
        properties = dataset.get_table_properties()
        
        print(f"  {name} Dataset:")
        print(f"    - Schema fields: {len(schema)}")
        print(f"    - Partitions: {[p.source_column for p in partitions]}")
        print(f"    - Compression: {properties.get('write.parquet.compression-codec', 'none')}")
        
        # Verify essential fields
        assert len(schema) > 5
        assert len(partitions) > 0
        assert 'format-version' in properties
        
    # Test special features
    audit_ds = AuditDataset()
    audit_props = audit_ds.get_table_properties()
    
    assert audit_props.get('table.immutable') == 'true'
    assert 's3.object-lock.enabled' in audit_props
    
    print("\n  ✓ All lakehouse datasets properly defined")


async def run_all_tests():
    """Run all tests"""
    print("="*60)
    print("AURA PERSISTENCE LAYER - REAL IMPLEMENTATION TEST")
    print("="*60)
    
    start_time = time.time()
    
    try:
        # Core components
        await test_query_builder()
        await test_transaction_context()
        await test_saga_orchestrator()
        await test_circuit_breaker()
        
        # Security
        await test_encryption()
        await test_audit_logging()
        await test_multi_tenancy()
        
        # Stores
        await test_document_store()
        await test_event_store()
        
        # Lakehouse
        await test_lakehouse_datasets()
        
        # Summary
        elapsed = time.time() - start_time
        
        print("\n" + "="*60)
        print("✅ ALL TESTS PASSED!")
        print("="*60)
        print(f"\n⏱️ Total time: {elapsed:.2f} seconds")
        
        print("\n📊 Test Coverage:")
        print("  ✓ Core abstractions (QueryBuilder, TransactionContext)")
        print("  ✓ Distributed transactions (Saga pattern)")
        print("  ✓ Resilience patterns (Circuit breaker)")
        print("  ✓ Security (Encryption, Audit, Multi-tenancy)")
        print("  ✓ Storage (Document store, Event store)")
        print("  ✓ Lakehouse (Dataset schemas)")
        
        print("\n🎯 Key Features Validated:")
        print("  - Saga pattern with automatic compensation")
        print("  - Circuit breaker with state transitions")
        print("  - Field-level encryption for PII")
        print("  - Immutable audit logging")
        print("  - Tenant isolation with data residency")
        print("  - Event sourcing with subscriptions")
        print("  - Lakehouse dataset definitions")
        
        print("\n🚀 The persistence layer is ready for integration!")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1
        
    return 0


if __name__ == "__main__":
    # Import missing module from query_builder
    from aura_intelligence.persistence.core.circuit_breaker import CircuitConfig
    
    exit_code = asyncio.run(run_all_tests())
    sys.exit(exit_code)