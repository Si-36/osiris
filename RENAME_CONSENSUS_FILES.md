# Consensus Module Renaming Plan

## Current Structure (Not Professional)
- `simple.py` → SimpleConsensus (too basic name)
- `byzantine.py` → ByzantineConsensus (good)
- `raft.py` → RaftConsensus (good)

## Proposed Professional Structure

### Option 1: Purpose-Based Naming
- `simple.py` → `fast_consensus.py` (FastConsensus)
  - Emphasizes it's optimized for speed (95% fast path)
- `simple.py` → `event_consensus.py` (EventConsensus)
  - Emphasizes event-ordering approach
- `simple.py` → `hybrid_consensus.py` (HybridConsensus)
  - Shows it combines event ordering + Raft

### Option 2: Technical Naming
- `simple.py` → `ordered_consensus.py` (OrderedConsensus)
  - Based on event ordering
- `simple.py` → `quorum_consensus.py` (QuorumConsensus)
  - If it uses quorum voting

### Option 3: Performance-Based Naming
- `simple.py` → `optimistic_consensus.py` (OptimisticConsensus)
  - Assumes most decisions don't need full consensus
- `simple.py` → `adaptive_consensus.py` (AdaptiveConsensus)
  - Adapts between fast/slow paths

## Recommended: HybridConsensus

Rename `simple.py` → `hybrid_consensus.py` because:
1. It's professional and descriptive
2. Shows it combines approaches (event ordering + Raft)
3. Not "simple" but "smart" - uses right tool for each case

```python
# In hybrid_consensus.py
class HybridConsensus:
    """
    Hybrid consensus implementation combining:
    - Fast path: Event ordering for 95% of decisions
    - Slow path: Raft consensus for critical 5%
    
    Optimized for AURA's mixed workload of frequent 
    low-stakes and occasional high-stakes decisions.
    """
```

## Changes Required:
1. Rename file: `simple.py` → `hybrid_consensus.py`
2. Update class: `SimpleConsensus` → `HybridConsensus`
3. Update imports everywhere
4. Update __init__.py exports