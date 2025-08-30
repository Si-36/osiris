# 🧹 CLEANUP PLAN: What to Remove and Keep

## 📊 Current State
- **684 Python files** total
- **~300 files** already transformed/consolidated
- **~400 files** remaining to process

## 🗑️ IMMEDIATE CLEANUP NEEDED

### 1. **Enterprise Folder (Mem0/GraphRAG originals)**
Since we've extracted the best parts:
- ✅ KEEP: Our extractions in `memory/enhancements/` and `memory/graph/`
- ❌ REMOVE: `enterprise/mem0_*` folders (already extracted)
- ❌ REMOVE: `enterprise/knowledge_graph.py` (we have GraphRAG now)
- ⚠️ REVIEW: Other enterprise files for unique features

### 2. **LNN Duplicates**
We extracted council to `agents/lnn_council.py`:
- ❌ REMOVE: `agents/council/lnn/` (79 files! - already extracted)
- ⚠️ REVIEW: `lnn/` folder - might have other LNN implementations
  - `core.py` (706 lines) - check if different from council
  - `advanced_lnn_system.py` (848 lines) - check for unique features

### 3. **Old Memory Files**
Keep our new structure, remove old:
- ❌ REMOVE: Old memory files that duplicate our new system
- ✅ KEEP: Our new `memory/` structure with enhancements

### 4. **TDA Legacy (Already Done)**
- ✅ Already moved 21 files to `tda/legacy/`

### 5. **Multiple "Unified" Systems**
Too many entry points:
- `unified_brain.py` (446 lines)
- `production_system_2025.py` (699 lines)
- `ultra_deep_2025.py` (439 lines)
- `enhanced_system_2025.py` (344 lines)
- ⚠️ NEED TO: Consolidate to ONE main entry point

## 📁 FILES TO DELETE NOW

```bash
# Enterprise duplicates (we extracted Mem0 and GraphRAG)
rm -rf core/src/aura_intelligence/enterprise/mem0_hot/
rm -rf core/src/aura_intelligence/enterprise/mem0_semantic/
rm -rf core/src/aura_intelligence/enterprise/mem0_search/
rm core/src/aura_intelligence/enterprise/knowledge_graph.py

# LNN council (we extracted to agents/lnn_council.py)
rm -rf core/src/aura_intelligence/agents/council/lnn/

# Old provider adapter attempt
rm core/src/aura_intelligence/neural/provider_adapters_simple.py

# Backup files
find core/src/aura_intelligence -name "*.bak" -delete
```

## 📁 FILES TO ARCHIVE (Move to legacy/)

```bash
# Research/experimental files
mkdir -p core/src/aura_intelligence/legacy/research/

# Move research files
mv core/src/aura_intelligence/research_2025/ legacy/research/
mv core/src/aura_intelligence/bio_homeostatic/ legacy/research/
mv core/src/aura_intelligence/chaos/ legacy/research/
```

## ✅ FILES TO KEEP (Our Work)

### Phase 1 & 2 Creations:
```
neural/
├── model_router.py ✅
├── provider_adapters.py ✅
├── adaptive_routing_engine.py ✅
├── cache_manager.py ✅
├── fallback_chain.py ✅
├── cost_optimizer.py ✅
├── load_balancer.py ✅
├── performance_tracker.py ✅
└── context_manager.py ✅

tda/
├── agent_topology.py ✅
├── algorithms.py ✅
├── realtime_monitor.py ✅
├── enhanced_topology.py ✅
└── legacy/ (archived) ✅

memory/
├── core/
│   ├── memory_api.py ✅
│   ├── topology_adapter.py ✅
│   └── causal_tracker.py ✅
├── routing/
│   └── hierarchical_router.py ✅
├── storage/
│   └── tier_manager.py ✅
├── operations/
│   └── monitoring.py ✅
├── enhancements/
│   └── mem0_integration.py ✅
├── graph/
│   └── graphrag_knowledge.py ✅
└── hardware/
    └── hardware_tier_manager.py ✅

orchestration/
├── unified_orchestration_engine.py ✅
└── gpu/
    └── gpu_workflow_integration.py ✅

agents/
├── lnn_council.py ✅
├── agent_core.py ✅
└── agent_templates.py ✅

persistence/
└── lakehouse_core.py ✅
```

## 🔢 Cleanup Impact

**Before Cleanup:**
- 684 Python files
- Many duplicates
- Confusing structure

**After Cleanup:**
- ~550 Python files (remove ~134 duplicates)
- Clear structure
- No confusion about what to use

## 🎯 Next Steps After Cleanup

1. **Complete Integration**
   - Connect Iceberg to Memory system
   - Wire Mem0 enhancements
   - Test GraphRAG with real data

2. **Continue Transformation**
   - Phase 3: swarm/, distributed/, consensus/
   - Phase 4: Final consolidation

3. **Create Single Entry Point**
   - One main.py or aura_system.py
   - Clear API
   - Deprecate all "unified" files

## ⚠️ IMPORTANT NOTES

1. **Before deleting:** Make sure we've extracted all valuable code
2. **Create backups:** `cp -r core/src/aura_intelligence core/src/aura_intelligence.backup`
3. **Test after cleanup:** Run our integration tests
4. **Document deletions:** Keep track of what was removed and why

## 📝 Cleanup Commands

```bash
# 1. Backup first
cp -r core/src/aura_intelligence core/src/aura_intelligence.backup

# 2. Remove duplicates
rm -rf core/src/aura_intelligence/enterprise/mem0_*
rm -rf core/src/aura_intelligence/agents/council/lnn/
rm core/src/aura_intelligence/neural/provider_adapters_simple.py

# 3. Clean backup files
find core/src/aura_intelligence -name "*.bak" -delete

# 4. Archive research
mkdir -p core/src/aura_intelligence/legacy/research/
mv core/src/aura_intelligence/research_2025/ core/src/aura_intelligence/legacy/research/

# 5. Verify
find core/src/aura_intelligence -type f -name "*.py" | wc -l
# Should show ~550 files
```

This cleanup will make the codebase much cleaner and easier to navigate!