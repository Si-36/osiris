# Final Import Instructions

## The Core Problem

The AURA codebase has deep import dependencies:
```
Main __init__.py → neural → resilience → consensus → events → aiokafka
```

Every import pulls in the entire chain, and aiokafka is required deep in the chain.

## Solutions

### Option 1: Install aiokafka in your environment (RECOMMENDED)
```bash
# You showed it was installed, but you're not in the virtual env
source /home/sina/projects/osiris-2/aura_venv/bin/activate
python3 TEST_AURA_STEP_BY_STEP.py
```

### Option 2: Use Direct Imports (Bypass __init__.py)
Instead of:
```python
from aura_intelligence import SimpleConsensus
```

Use:
```python
from aura_intelligence.consensus.simple import SimpleConsensus
```

### Option 3: Fix ALL Optional Imports (Complex)
Would need to make MANY files handle missing dependencies:
- events/streams.py (classes inherit from None)
- events/connectors.py 
- Many other files

## What I've Fixed So Far

1. ✅ Removed circular dependencies
2. ✅ Added backward compatibility aliases
3. ✅ Made many imports optional
4. ❌ But the dependency chain is too deep

## Working Components (if you activate venv)

With your virtual environment active, these should work:
- ✅ Memory (HybridMemoryManager, HierarchicalMemoryManager)
- ✅ Consensus (SimpleConsensus, RaftConsensus, ByzantineConsensus)
- ✅ Events (EventProducer, EventConsumer)
- ✅ Persistence (CausalStateManager)
- ✅ Neural (if torch installed)
- ✅ Agents (if langgraph installed)

## About simple.py

You're absolutely right - "SimpleConsensus" is not professional!

Recommend renaming:
- `simple.py` → `hybrid_consensus.py`
- `SimpleConsensus` → `HybridConsensus`

Because it's actually sophisticated:
- Uses event ordering for 95% of decisions (fast path)
- Uses Raft consensus for 5% critical decisions (slow path)
- Not "simple" but "optimized"!

## Next Steps

1. **Activate your virtual environment**:
   ```bash
   source /home/sina/projects/osiris-2/aura_venv/bin/activate
   which python3  # Should show venv path
   ```

2. **Run the test**:
   ```bash
   python3 TEST_AURA_STEP_BY_STEP.py
   ```

3. **If you still get errors**, share them and I'll fix them one by one.

The codebase is complex with many interdependencies, but with your environment active, most imports should work!