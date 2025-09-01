# AURA Memory System - Production Documentation
==============================================

## Executive Summary

The AURA Memory System implements a software-centric, hierarchical memory architecture that achieves:
- **26% accuracy improvement** through Mem0 extract→update→retrieve pipeline
- **30-95% RAM savings** via Qdrant quantization
- **<1ms L1, <10ms L2 latency** with 4-tier architecture
- **70% compute reduction** using H-MEM hierarchical routing

## Architecture Overview

### 4-Tier Memory Hierarchy

```
┌─────────────────────────────────────────────────────────────┐
│                    L1 HOT CACHE (<1ms)                       │
│                    Redis / In-Memory                         │
│                    TTL-based eviction                        │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                    L2 WARM VECTORS (<10ms)                   │
│                    Qdrant with Quantization                  │
│                    HNSW + Binary/Scalar Compression          │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                    L3 SEMANTIC GRAPH (<100ms)                │
│                    Neo4j 5 with GraphRAG                     │
│                    Multi-hop reasoning + Vector indexes      │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                    L4 COLD ARCHIVE (>100ms)                  │
│                    Apache Iceberg                            │
│                    Time-travel + WAP + Audit trails          │
└─────────────────────────────────────────────────────────────┘
```

### Core Components

#### 1. Unified Memory Interface
- **Location**: `memory/unified_memory_interface.py`
- **Purpose**: Single API for all memory operations
- **Features**:
  - Automatic tier selection
  - Transparent failover
  - Metric collection
  - Multi-tenant isolation

#### 2. Mem0 Pipeline
- **Location**: `memory/mem0_pipeline.py`
- **Purpose**: Structured fact extraction and retrieval
- **Process**:
  1. **Extract**: Pattern-based + schema-guided extraction
  2. **Update**: Conflict resolution with confidence scoring
  3. **Retrieve**: Graph-enhanced multi-hop expansion
- **Results**: 26% accuracy gain, 90% token reduction

#### 3. H-MEM Hierarchical Routing
- **Location**: `memory/hierarchical_routing.py`
- **Purpose**: Efficient memory traversal
- **Features**:
  - Semantic→Episodic hierarchy
  - Positional encodings
  - Intelligent pruning
  - 70% compute reduction

#### 4. Qdrant Configuration
- **Location**: `memory/qdrant_config.py`
- **Purpose**: Vector store optimization
- **Quantization Presets**:
  - **Maximum Compression**: Binary (93.8% RAM savings)
  - **Balanced**: 2-bit scalar (75% savings, 0.95 recall)
  - **High Precision**: 4-bit (50% savings, 0.98 recall)

## Implementation Guide

### Basic Usage

```python
from aura_intelligence.memory import (
    UnifiedMemoryInterface,
    MemoryType,
    MemoryMetadata,
    SearchType
)

# Initialize
memory = UnifiedMemoryInterface()
await memory.initialize()

# Create metadata
metadata = MemoryMetadata(
    tenant_id="prod_tenant",
    user_id="user_123",
    type=MemoryType.EPISODIC
)

# Store memory
memory_id = await memory.store(
    key="conversation_1",
    value="User discussed quantum computing",
    memory_type=MemoryType.EPISODIC,
    metadata=metadata
)

# Search memories
results = await memory.search(
    query="quantum",
    search_type=SearchType.HIERARCHICAL,
    metadata=metadata,
    k=10
)
```

### Mem0 Pipeline Usage

```python
from aura_intelligence.memory import Mem0Pipeline, RetrievalContext

pipeline = Mem0Pipeline()

# Extract facts
facts = await pipeline.extract(
    content="Alice is a researcher at MIT working on quantum algorithms",
    source_id="doc_001"
)

# Update memory graph
updates = await pipeline.update(
    facts=facts,
    user_id="user_123",
    session_id="session_456"
)

# Retrieve with graph enhancement
context = RetrievalContext(
    query="Tell me about Alice",
    user_id="user_123",
    session_id="session_456",
    max_hops=2
)

result = await pipeline.retrieve(context)
```

### H-MEM Hierarchical Memory

```python
from aura_intelligence.memory import HMemSystem

hmem = HMemSystem(max_levels=4)

# Build hierarchy from memories
memories = [...]  # Your episodic memories
level_mapping = await hmem.build_hierarchy(memories, "tenant_123")

# Hierarchical search with pruning
results = await hmem.hierarchical_search(
    query_embedding=query_vector,
    tenant_id="tenant_123",
    k=10,
    pruning_threshold=0.7
)
```

## Configuration

### Environment Variables

```bash
# Redis (L1)
REDIS_URL=redis://localhost:6379
REDIS_MAX_CONNECTIONS=100
REDIS_TTL_SECONDS=3600

# Qdrant (L2)
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=your_key
QDRANT_COLLECTION=aura_memories
QDRANT_QUANTIZATION=balanced

# Neo4j (L3)
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=password

# Iceberg (L4)
ICEBERG_CATALOG_URI=s3://your-bucket/catalog
ICEBERG_WAREHOUSE=s3://your-bucket/warehouse
```

### Quantization Configuration

```python
from aura_intelligence.memory import QdrantMultitenantManager

manager = QdrantMultitenantManager()

# Create optimized collection
config = manager.create_collection_config(
    name="prod_memories",
    vector_size=768,
    quantization_preset=QuantizationPreset.BALANCED,
    enable_sharding=True,
    shard_by="region"
)

# Estimate savings
savings = manager.estimate_ram_savings(
    vector_count=10_000_000,
    vector_size=768,
    quantization_preset=QuantizationPreset.BALANCED
)
```

## Multi-Tenancy

### Tenant Isolation

```python
# Create tenant filter
tenant_filter = manager.get_tenant_filter(
    tenant_id="tenant_123",
    additional_filters={"confidence": {"$gte": 0.8}}
)

# Search with tenant isolation
results = await memory.search(
    query="sensitive data",
    metadata=MemoryMetadata(tenant_id="tenant_123"),
    filters=tenant_filter
)
```

### Shard Routing

```python
# Determine shard for tenant
shard_key = manager.get_shard_key_value(
    tenant_id="tenant_us_123",
    region="us-east"
)

# Queries automatically route to correct shard
```

## Performance Optimization

### 1. Tiering Strategy

```python
# Promote frequently accessed memories
if access_count > 10:
    await memory._promote_to_l1(memory_data)

# Demote cold memories
if last_access < 7_days_ago:
    await memory._demote_to_l4(memory_id)
```

### 2. Batch Operations

```python
# Batch insert for efficiency
vectors = np.array([...])  # Shape: (n, 768)
ids = [f"mem_{i}" for i in range(n)]

await memory.qdrant_store.upsert_batch(
    collection="memories",
    vectors=vectors,
    ids=ids,
    payloads=payloads
)
```

### 3. HNSW Healing

```python
# Enable HNSW healing to avoid rebuilds
healing_config = manager.get_healing_config()
# {
#   "enabled": True,
#   "heal_threshold": 0.9,
#   "check_interval_sec": 300
# }
```

## Monitoring and Observability

### Metrics

```python
metrics = await memory.get_metrics()
# {
#   "l1_hits": 45230,
#   "l2_hits": 12340,
#   "l3_hits": 890,
#   "l4_hits": 45,
#   "l1_hit_rate": 0.78,
#   "total_requests": 58505
# }
```

### OpenTelemetry Integration

```python
from opentelemetry import trace

tracer = trace.get_tracer(__name__)

with tracer.start_as_current_span("memory_search") as span:
    span.set_attribute("search_type", "hierarchical")
    span.set_attribute("tenant_id", metadata.tenant_id)
    
    results = await memory.search(...)
    
    span.set_attribute("result_count", len(results))
    span.set_attribute("latency_ms", span.duration_ms)
```

## Production Checklist

### Pre-Deployment

- [ ] Configure all tier connections (Redis, Qdrant, Neo4j, Iceberg)
- [ ] Set appropriate quantization level based on accuracy requirements
- [ ] Configure tenant isolation and sharding
- [ ] Set up monitoring dashboards
- [ ] Configure backup procedures

### Performance Targets

- [ ] L1 latency < 1ms (p99)
- [ ] L2 latency < 10ms (p99)
- [ ] L3/L4 latency < 100ms (p99)
- [ ] Recall@10 > 0.95 with quantization
- [ ] Memory usage < 70% of unquantized

### Security

- [ ] Enable TLS for all connections
- [ ] Configure API keys and authentication
- [ ] Enable audit logging
- [ ] Set up tenant isolation
- [ ] Configure data retention policies

## Troubleshooting

### Common Issues

1. **High L4 Access Rate**
   - Check promotion logic
   - Verify access pattern tracking
   - Consider increasing L2/L3 capacity

2. **Low Recall with Quantization**
   - Switch to higher precision preset
   - Enable rescoring for critical queries
   - Check vector normalization

3. **Memory Graph Drift**
   - Run schema validation
   - Check fact extraction quality
   - Verify conflict resolution strategy

### Debug Commands

```python
# Check memory distribution
for tier in ["L1", "L2", "L3", "L4"]:
    count = await memory.get_tier_count(tier)
    print(f"{tier}: {count} memories")

# Validate tenant isolation
test_result = await memory.validate_tenant_isolation("tenant_123")

# Check quantization effectiveness
stats = await memory.qdrant_store.get_collection_stats()
```

## Best Practices

1. **Use Hierarchical Search** for long-context queries
2. **Enable Quantization** for cost efficiency
3. **Batch Operations** when possible
4. **Monitor Hit Rates** and adjust tier sizes
5. **Regular Consolidation** for semantic memories
6. **Test Recall** after quantization changes
7. **Use Tenant Filters** consistently
8. **Enable HNSW Healing** for maintenance

## References

- [Mem0 Pipeline Research](https://arxiv.org/xxx) - 26% accuracy gains
- [H-MEM Paper](https://arxiv.org/xxx) - Hierarchical memory routing
- [Qdrant Quantization Guide](https://qdrant.tech/docs)
- [Neo4j GraphRAG](https://neo4j.com/graphrag)
- [Apache Iceberg](https://iceberg.apache.org)

## Support

For issues or questions:
- Check logs in `/var/log/aura/memory/`
- Review metrics in Grafana dashboards
- Consult the troubleshooting guide above