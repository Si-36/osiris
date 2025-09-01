# Fix Import Plan - Option 3 (Hybrid Approach)

## Step 1: Clean Up Interference (DO FIRST!)
```bash
# Remove the problematic .backup directory
rm -rf core/src/aura_intelligence.backup

# Keep archive for safety
# Keep: core/src/aura_intelligence_archive/
# Keep: core/src/aura_intelligence_clean/
```

## Step 2: Map What We Actually Built vs What's Imported

### What We Built (New Components):
1. **Causal Persistence** (`causal_state_manager.py`)
   - Tracks WHY decisions were made
   - Speculative branches
   - GPU memory tier

2. **Memory-Native Architecture** (`memory_native.py`)
   - Memory as computation
   - Quantum-inspired superposition
   - Compute-on-retrieval

3. **Advanced Memory System** (`advanced_memory_system.py`)
   - `HierarchicalMemorySystem` class (NOT HierarchicalMemoryManager!)
   
4. **Hybrid Memory Manager** (`hybrid_manager.py`)
   - `HybridMemoryManager` class (NOT MemoryManager!)

5. **Adaptive Resilience** 
   - `AdaptiveCircuitBreaker` (NOT CircuitBreaker!)
   - Context-aware retry
   - Adaptive timeouts

### What's Being Imported (Legacy Names):
- `HierarchicalMemoryManager` → Should be `HierarchicalMemorySystem`
- `CircuitBreaker` → Should be `AdaptiveCircuitBreaker`
- `MemoryManager` → Should be `HybridMemoryManager`

## Step 3: Fix Imports Folder by Folder

### Folder 1: MEMORY (Fix First - Everything Depends on It)
**Issues**:
- `hybrid_manager.py` - ✅ Fixed syntax errors
- Multiple memory implementations unclear which to use
- Need clear hierarchy: UnifiedMemoryInterface → HierarchicalMemorySystem → HybridMemoryManager

**Actions**:
1. Ensure `__init__.py` exports correctly
2. Add proper aliases for backward compatibility
3. Test basic memory operations

### Folder 2: AGENTS 
**Issues**:
- Imports langgraph (external dependency)
- Uses old memory import names
- Test agents might be using old patterns

**Actions**:
1. Fix base.py imports
2. Update all agents to use new memory system
3. Ensure state management works

### Folder 3: PERSISTENCE
**Issues**:
- backup/ module import broken
- Complex dependencies on memory
- New causal system not integrated

**Actions**:
1. Fix backup module import
2. Integrate causal_state_manager with agents
3. Test persistence operations

### Folder 4: NEURAL
**Issues**:
- Provider adapters using old imports
- LNN, MoE, Mamba not connected to new memory

**Actions**:
1. Update all neural imports
2. Connect to new persistence for checkpointing
3. Test model operations

### Folder 5: RESILIENCE
**Issues**:
- ✅ Syntax fixed but imports might be wrong
- Need to wrap other components properly

**Actions**:
1. Verify all imports work
2. Test circuit breaker with real operations

## Step 4: Minimal Working System Test

### Test Flow:
```python
# 1. Create an agent
agent = ProductionLangGraphAgent()

# 2. Agent stores something in memory
await agent.memory.store("test", data)

# 3. Memory persists to causal storage
await persistence.save_state(state_data)

# 4. Retrieve and verify
retrieved = await agent.memory.retrieve("test")
```

## Step 5: Integration Points to Fix

### Critical Connections:
1. **Agent → Memory**: How agents store/retrieve memories
2. **Memory → Persistence**: How memory tiers persist
3. **Neural → Memory**: How models use memory
4. **All → Resilience**: How fault tolerance wraps everything

### Import Fixes Needed:
```python
# In each file that imports old names:

# OLD (broken):
from ..memory import MemoryManager
from ..memory.hierarchical_memory import HierarchicalMemoryManager
from ..resilience import CircuitBreaker

# NEW (correct):
from ..memory import HybridMemoryManager as MemoryManager
from ..memory.advanced_memory_system import HierarchicalMemorySystem
from ..resilience import AdaptiveCircuitBreaker as CircuitBreaker
```

## Why We Have This Problem:

1. **We transformed components individually** without updating integration points
2. **We created new classes** but imports still look for old names
3. **We made backups** that interfere with Python's import system
4. **We never tested** the connections between components

## The Right Way Forward:

1. **Delete `.backup` directory** - It's poison for imports
2. **Fix one folder at a time** - Starting with Memory
3. **Test each connection** - Not just syntax
4. **Update imports systematically** - Old name → New name
5. **Document what works** - So we don't break it again

This is a systematic approach to untangle the import mess while keeping your advanced components intact.