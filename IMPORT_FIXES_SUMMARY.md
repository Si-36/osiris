# Import Fixes Summary

## Problem Overview
We built advanced components with new names but didn't update all the imports throughout the system. Additionally, external dependencies (langgraph, aiokafka) were being imported at module level, breaking the system when they weren't available.

## Fixes Applied

### 1. Removed Problematic Backup Directory
```bash
rm -rf core/src/aura_intelligence.backup
```
This directory was interfering with Python's import system.

### 2. Fixed Memory Module Aliases
**File**: `core/src/aura_intelligence/memory/__init__.py`
- Added: `HierarchicalMemoryManager = HierarchicalMemorySystem`
- Already had: `MemoryManager = HybridMemoryManager`

**Updated imports in**:
- `agents/production_langgraph_agent.py`
- `inference/free_energy_core.py`
- `inference/active_inference_lite.py`
- `orchestration/strategic/strategic_planner.py`

### 3. Fixed Circular Dependencies
**File**: `core/src/aura_intelligence/consensus/consensus_types.py`
- Removed: `from ..agents.base import AgentState` (unused import)

### 4. Made External Dependencies Optional
**EventProducer (aiokafka)**:
- `consensus/simple.py` - Made EventProducer optional, wrapped all event sends
- `consensus/byzantine.py` - Made EventProducer optional
- `consensus/raft.py` - Made EventProducer and temporal imports optional
- `consensus/manager.py` - Made EventProducer and TemporalClient optional

**Main init**:
- `__init__.py` - Made agents import optional (requires langgraph)

### 5. Fixed Resilience Module
- Already had: `CircuitBreaker = AdaptiveCircuitBreaker` alias

## Remaining Issues

### External Dependencies Still Required At Module Level:
1. **aiokafka** - Still imported in `events/producers.py`
2. **langgraph** - Still imported in `agents/production_langgraph_agent.py`
3. **asyncpg** - Required by persistence causal_state_manager

### Import Chains That Need Breaking:
```
memory → neural → resilience → consensus → events → aiokafka
                                        ↓
                                     agents → langgraph
```

## Next Steps

### Option 1: Make Events Module Optional
Make the events module imports optional throughout:
```python
try:
    from ..events import EventProducer
    EVENTS_AVAILABLE = True
except ImportError:
    EventProducer = None
    EVENTS_AVAILABLE = False
```

### Option 2: Create Minimal Core
Create a minimal import set that doesn't require external deps:
```python
# core_imports.py
from .memory.hybrid_manager import HybridMemoryManager
from .persistence.causal_state_manager import CausalStateManager
# etc - only the essentials
```

### Option 3: Fix Events Module
Make events/producers.py handle missing aiokafka gracefully:
```python
try:
    from aiokafka import AIOKafkaProducer
    KAFKA_AVAILABLE = True
except ImportError:
    KAFKA_AVAILABLE = False
    AIOKafkaProducer = None
```

## Testing Without Dependencies
With external deps mocked or when they're installed:
```python
from aura_intelligence.memory import HierarchicalMemoryManager
from aura_intelligence.persistence.causal_state_manager import get_causal_manager
from aura_intelligence.neural import LiquidNeuralNetwork
```

These should all work once the remaining import issues are fixed.