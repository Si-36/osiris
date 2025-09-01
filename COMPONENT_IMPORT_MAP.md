# AURA Intelligence Component Import Map

## Component Structure & Dependencies

### 1. AGENTS Component
**Location**: `core/src/aura_intelligence/agents/`
**Purpose**: Autonomous agents with LangGraph state management

**Key Files**:
- `production_langgraph_agent.py` - Main agent implementation
- `base.py` - Base agent classes
- Test agents: `test_code_agent.py`, `test_data_agent.py`, etc.

**Dependencies**:
- → Memory: `HierarchicalMemorySystem` (from `memory.advanced_memory_system`)
- → Neural: Provider adapters for LLM access
- → Orchestration: For multi-agent coordination
- → Persistence: For state checkpointing

**Import Issues Found**:
- ✅ Fixed: Was importing non-existent `HierarchicalMemoryManager`
- ⚠️ Problem: `base.py` imports from `langgraph` (external dependency)

---

### 2. NEURAL Component
**Location**: `core/src/aura_intelligence/neural/`
**Purpose**: LNN, MoE, Mamba neural architectures

**Key Files**:
- `liquid_neural_network.py` - Adaptive neural networks
- `mixture_of_experts.py` - MoE routing
- `mamba_architecture.py` - Sequence modeling
- `provider_adapters.py` - LLM provider integration

**Dependencies**:
- → Memory: For storing embeddings and model state
- → Resilience: `CircuitBreaker` for fault tolerance
- → Persistence: For model checkpointing

**Import Issues Found**:
- ✅ Fixed: `CircuitBreaker` alias added
- ⚠️ Problem: Some files import from old structure

---

### 3. MEMORY Component
**Location**: `core/src/aura_intelligence/memory/`
**Purpose**: GPU-accelerated hierarchical memory system

**Key Files**:
- `advanced_memory_system.py` - Contains `HierarchicalMemorySystem`
- `hybrid_manager.py` - Contains `HybridMemoryManager`
- `unified_memory_interface.py` - Unified API
- GPU wrappers and adapters

**Dependencies**:
- → Persistence: For durable storage
- → Infrastructure: Ray for distributed memory
- → Hardware: GPU acceleration

**Import Issues Found**:
- ✅ Fixed: Added `MemoryManager` alias for `HybridMemoryManager`
- ⚠️ Problem: Multiple memory implementations with unclear relationships

---

### 4. PERSISTENCE Component
**Location**: `core/src/aura_intelligence/persistence/`
**Purpose**: Causal tracking and memory-native persistence

**Key Files**:
- `causal_state_manager.py` - Tracks WHY decisions were made
- `memory_native.py` - Memory as computation substrate
- `migrate_from_pickle.py` - Migration tools

**Dependencies**:
- → Infrastructure: For distributed storage
- → Memory: Tight integration with memory tiers

**Import Issues Found**:
- ❌ Critical: `backup/` module import fails due to `.backup` directory
- ⚠️ Problem: Complex circular dependencies

---

### 5. ORCHESTRATION Component
**Location**: `core/src/aura_intelligence/orchestration/`
**Purpose**: Swarm coordination and strategic planning

**Key Files**:
- `strategic/strategic_planner.py` - High-level planning
- Swarm coordination modules

**Dependencies**:
- → Agents: Coordinates multiple agents
- → Memory: Uses `HierarchicalMemorySystem`
- → Consensus: For distributed decisions

**Import Issues Found**:
- ✅ Fixed: Updated to use correct memory import

---

### 6. CONSENSUS Component
**Location**: `core/src/aura_intelligence/consensus/`
**Purpose**: Distributed agreement protocols

**Key Files**:
- `raft.py` - Raft consensus
- `byzantine.py` - Byzantine fault tolerance
- `consensus_types.py` - Shared types

**Dependencies**:
- → Agents: For agent state agreement
- → Communication: Event propagation

**Import Issues Found**:
- ✅ Fixed: Import from `.types` → `.consensus_types`

---

### 7. RESILIENCE Component  
**Location**: `core/src/aura_intelligence/resilience/`
**Purpose**: Fault tolerance and recovery

**Key Files**:
- `circuit_breaker.py` - Contains `AdaptiveCircuitBreaker`
- `retry.py` - Smart retry logic
- `timeout.py` - Adaptive timeouts
- `bulkhead.py` - Resource isolation

**Dependencies**:
- → Consensus: For decisions under failure
- → Metrics: For monitoring

**Import Issues Found**:
- ✅ Fixed: All syntax errors (indentation, async issues)
- ✅ Fixed: Added `CircuitBreaker` alias

---

### 8. INFRASTRUCTURE Component
**Location**: `core/src/aura_intelligence/infrastructure/`
**Purpose**: Ray and Kubernetes integration

**Dependencies**:
- Used by all components for distributed execution

---

### 9. COMMUNICATION Component
**Location**: `core/src/aura_intelligence/communication/`
**Purpose**: NATS messaging and events

**Key Files**:
- Event producers and consumers
- NATS integration

**Dependencies**:
- → Agents: For inter-agent communication
- → Orchestration: For coordination messages

---

### 10. TDA Component
**Location**: `core/src/aura_intelligence/tda/`
**Purpose**: Topological Data Analysis

**Dependencies**:
- → Memory: For storing topological features
- → Neural: For feature extraction

---

### 11. CORE Component
**Location**: `core/src/aura_intelligence/core/`
**Purpose**: Active Inference and Free Energy Principle

**Key Files**:
- Active inference implementation
- FEP calculations

**Dependencies**:
- → Agents: For decision making
- → Memory: For belief updates

---

## Critical Issues to Fix:

### 1. The `.backup` Directory Problem
```bash
# Must remove this - it's breaking Python imports
rm -rf core/src/aura_intelligence.backup
```

### 2. Import Naming Mismatches
| Old Name (doesn't exist) | New Name (actual) | Where Used |
|--------------------------|-------------------|------------|
| HierarchicalMemoryManager | HierarchicalMemorySystem | agents, inference, orchestration |
| CircuitBreaker | AdaptiveCircuitBreaker | neural, others |
| MemoryManager | HybridMemoryManager | neural |

### 3. Circular Dependencies
- agents → memory → persistence → agents (circular!)
- Need to break these cycles

### 4. External Dependencies
- `langgraph` - Required for agents
- `ray` - Required for infrastructure
- Various ML libraries

## Recommended Fix Order:

1. **First**: Remove `.backup` directory
2. **Memory**: Fix all memory imports (it's used everywhere)
3. **Agents**: Fix base agent imports
4. **Neural**: Ensure provider adapters work
5. **Persistence**: Fix backup module issue
6. **Test**: Basic agent → memory → persistence flow

## Minimal Working System:
```
Agents (with state management)
  ↓
Memory (for agent memory)
  ↓
Persistence (for checkpointing)
  ↓
Resilience (for fault tolerance)
```

Everything else can be added incrementally.