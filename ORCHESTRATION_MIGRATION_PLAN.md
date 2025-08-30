# 🚀 AURA Orchestration Production Migration Plan

## Phase 1: Core Stabilization (Weeks 1-2)

### 1.1 PostgreSQL Persistence Migration
```python
# Replace ALL MemorySaver instances
# Before:
from langgraph.checkpoint import MemorySaver
checkpointer = MemorySaver()

# After:
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.store.postgres import PostgresStore

postgres_saver = PostgresSaver.from_conn_string(
    "postgresql://aura:aura@localhost:5432/aura_orchestration"
)
await postgres_saver.setup()

# Add long-term memory
postgres_store = PostgresStore.from_conn_string(
    "postgresql://aura:aura@localhost:5432/aura_orchestration"
)
await postgres_store.setup()
```

**Files to Update:**
- `workflows.py` - Line 18: Replace MemorySaver
- `langgraph_workflows.py` - Line 32: Replace MemorySaver
- `checkpoints.py` - Throughout: Add PostgresSaver support
- `adaptive_checkpoint.py` - Integrate with PostgreSQL

### 1.2 Complete Critical Stubs
**Priority Files with "pass" statements:**

1. **pro_orchestration_system.py** (26 stubs)
   - `execute_workflow()` - Line 234
   - `_execute_with_retry()` - Line 298
   - `EventStore.append()` - Line 412
   - `Saga.execute()` - Line 567

2. **temporal_signalfirst.py** (8 stubs)
   - `_process_priority_queue()` - Line 203
   - `_flush_batch()` - Line 267
   - `_route_to_temporal()` - Line 334

3. **adaptive_checkpoint.py** (5 stubs)
   - `_coalesce_checkpoints()` - Line 189
   - `_detect_burst()` - Line 245

### 1.3 Wire Event Bus + Metrics
```python
# In bus_metrics.py - complete the metrics server
async def start_metrics_server():
    app = web.Application()
    app.router.add_get('/metrics', handle_metrics)
    
    # Add Prometheus metrics
    app['publish_latency'] = Histogram('bus_publish_latency_ms')
    app['stream_lag'] = Gauge('bus_stream_lag')
    app['dlq_size'] = Gauge('bus_dlq_size')
    
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, 'localhost', 9091)
    await site.start()
```

## Phase 2: Durable Signaling & Efficiency (Weeks 3-4)

### 2.1 SignalFirst + Temporal Integration
```python
# In temporal_signalfirst.py
async def _route_to_temporal(self, metadata: SignalMetadata, signal_data: Any):
    """Route high-priority signals through Temporal"""
    # Complete implementation
    if metadata.priority == SignalPriority.CRITICAL:
        # Direct routing, bypass queue
        handle = await self.temporal_client.get_workflow_handle(
            metadata.workflow_id
        )
        await handle.signal(metadata.signal_type, signal_data)
        
        # Track latency
        latency = (time.time() - metadata.timestamp.timestamp()) * 1000
        self.stats["signal_latency_ms"].append(latency)
    else:
        # Queue for batching
        await self._add_to_batch(metadata, signal_data)
```

### 2.2 Checkpoint Coalescing
```python
# In adaptive_checkpoint.py
async def coalesce_checkpoints(self, checkpoints: List[Checkpoint]) -> Checkpoint:
    """Coalesce multiple checkpoints into one"""
    # Group by workflow_id/thread_id
    groups = defaultdict(list)
    for cp in checkpoints:
        key = f"{cp.workflow_id}:{cp.thread_id}"
        groups[key].append(cp)
    
    # Merge each group
    coalesced = []
    for key, group_cps in groups.items():
        merged = self._merge_checkpoint_group(group_cps)
        coalesced.append(merged)
        
        # Track savings
        self.metrics["writes_saved"] += len(group_cps) - 1
    
    return coalesced
```

### 2.3 Monitoring Dashboard
```yaml
# prometheus.yml
scrape_configs:
  - job_name: 'aura_orchestration'
    static_configs:
      - targets: 
        - 'localhost:9090'  # Main metrics
        - 'localhost:9091'  # Event bus metrics
    
    metric_relabel_configs:
      # Key metrics to track
      - source_labels: [__name__]
        regex: '(workflow_duration_ms|signal_latency_ms|checkpoint_writes_saved|saga_compensation_rate|circuit_breaker_state)'
        action: keep
```

## Phase 3: Distributed Scaling (Weeks 5-6)

### 3.1 Ray Orchestration Activation
```python
# In ray_orchestrator.py - complete actor implementation
@ray.remote
class WorkerActor:
    def __init__(self, worker_id: str):
        self.worker_id = worker_id
        self.tda_analyzer = AgentTopologyAnalyzer()
        
    async def process_task(self, task: Task) -> TaskResult:
        # Analyze topology
        topology = await self.tda_analyzer.analyze_workflow(
            task.workflow_id,
            task.data
        )
        
        # Route based on bottlenecks
        if topology.bottleneck_score > 0.7:
            # Use different strategy
            pass
            
        # Process task
        result = await self._execute_task(task)
        return result
```

### 3.2 Consensus for Critical Sections
```python
# Wire consensus to orchestrator
async def make_critical_decision(self, decision_type: str, context: Dict) -> bool:
    """Use Raft consensus for critical decisions"""
    if decision_type in ["policy_change", "compliance_update", "resource_allocation"]:
        # Route through consensus
        proposal = ConsensusProposal(
            proposer_id=self.node_id,
            decision_type=decision_type,
            context=context
        )
        
        result = await self.raft_consensus.propose(proposal)
        return result.accepted
    else:
        # Non-critical, proceed directly
        return True
```

## Key Integration Points

### 1. TDA Integration
```python
# In every workflow execution
topology = await self.tda_analyzer.analyze_workflow(workflow_id, data)

if topology.bottleneck_score > threshold:
    # Reroute or scale
    await self.hierarchical_coordinator.escalate({
        "reason": "bottleneck_detected",
        "score": topology.bottleneck_score,
        "agents": topology.bottleneck_agents
    })
```

### 2. Memory Integration
```python
# Store all workflow patterns
await self.memory_system.store(
    content={
        "workflow_id": workflow_id,
        "topology": topology.to_dict(),
        "outcome": result.status,
        "duration_ms": duration
    },
    workflow_data=workflow_graph,
    metadata={"outcome": result.status}
)

# Learn from past executions
similar_workflows = await self.memory_system.retrieve_by_topology(
    query_pattern=current_topology,
    k=10
)
```

### 3. Neural Integration
```python
# For AI-powered steps
if step.requires_llm:
    model_request = ProviderRequest(
        messages=[{"role": "system", "content": step.prompt}],
        temperature=0.7
    )
    
    response = await self.neural_router.route_request(model_request)
    step.result = response.content
```

## Success Metrics

### Week 2 Checkpoint
- [ ] All LangGraph workflows use PostgresSaver
- [ ] Zero "pass" statements in core files
- [ ] Event bus metrics dashboard live
- [ ] 10 integration tests passing

### Week 4 Checkpoint
- [ ] SignalFirst achieving <20ms p95 latency
- [ ] Checkpoint writes reduced by 40%
- [ ] Saga compensation working end-to-end
- [ ] Circuit breakers preventing cascades

### Week 6 Checkpoint
- [ ] Ray actors processing heavy workloads
- [ ] Consensus working for critical decisions
- [ ] TDA-guided routing in production
- [ ] Full observability with Grafana

## Migration Commands

```bash
# 1. Setup PostgreSQL
docker run -d --name aura-postgres \
  -e POSTGRES_PASSWORD=aura \
  -e POSTGRES_DB=aura_orchestration \
  -p 5432:5432 \
  postgres:15

# 2. Run migrations
python -m aura_intelligence.orchestration.migrations.setup_postgres

# 3. Start Temporal
docker run -d --name temporal \
  -p 7233:7233 \
  temporalio/auto-setup:latest

# 4. Launch orchestration
python -m aura_intelligence.orchestration.unified_orchestration_engine

# 5. View metrics
open http://localhost:9090/metrics
open http://localhost:3000  # Grafana
```

## File Consolidation Map

### Keep As-Is (Gold Nuggets):
- `temporal_signalfirst.py` → Core signal routing
- `saga_patterns.py` → Distributed transactions
- `pipeline_registry.py` → A/B testing & versioning
- `experiment_manager.py` → Shadow mode
- `gossip_router.py` → GABFT consensus
- `latency_scheduler.py` → SLA optimization
- `adaptive_checkpoint.py` → Write reduction
- `circuit_breaker.py` → Failure prevention

### Merge Into Unified Engine:
- `workflows.py` + `langgraph_workflows.py` → Unified workflow builder
- `hierarchical_orchestrator.py` → Decision escalation
- `tda_coordinator.py` → Topology routing

### Archive (Reference Only):
- Duplicate workflow implementations
- Test files
- Old checkpoint implementations

This migration preserves ALL the gold nuggets while creating a clean, unified API!