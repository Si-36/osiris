# Legacy TDA Implementations - DEPRECATED

This directory contains legacy TDA implementations that have been superseded by the new agent-focused architecture.

## Migration Guide

### Old → New Mappings

| Old Class/Function | New Replacement | Notes |
|-------------------|-----------------|-------|
| `UnifiedTDAEngine2025` | `AgentTopologyAnalyzer` | Use for workflow analysis |
| `RealTDAEngine` | `AgentTopologyAnalyzer` | Consolidated functionality |
| `AdvancedTDAEngine` | `AgentTopologyAnalyzer` | GPU features moved to config |
| `RipsComplex` | `compute_persistence()` | Direct function call |
| `PersistentHomology` | `compute_persistence()` | Simplified API |
| `get_unified_tda_engine()` | `AgentTopologyAnalyzer()` | Create instance directly |

### Example Migration

**Old Code:**
```python
from aura_intelligence.tda.unified_engine_2025 import get_unified_tda_engine

engine = get_unified_tda_engine()
result = await engine.analyze_agent_system(data)
```

**New Code:**
```python
from aura_intelligence.tda import AgentTopologyAnalyzer

analyzer = AgentTopologyAnalyzer()
features = await analyzer.analyze_workflow(workflow_id, workflow_data)
```

## Why These Changes?

1. **Focus on Agent Systems**: The new architecture specifically targets multi-agent workflow analysis rather than generic TDA
2. **Simplified API**: Removed academic complexity in favor of practical metrics
3. **Better Performance**: CPU-first approach with optional GPU acceleration
4. **Clearer Purpose**: Each component has a specific role in agent topology analysis

## Removal Timeline

- **v2.0.0** (Current): Deprecation warnings added
- **v2.1.0**: Legacy imports will raise DeprecationWarning
- **v3.0.0**: Legacy code will be removed

## Files in This Directory

- `unified_engine_2025.py` - Latest unified engine attempt
- `real_tda_engine_2025.py` - "Real" TDA implementation
- `advanced_tda_system.py` - GPU-accelerated version
- `gpu_acceleration.py`, `cuda_kernels.py` - GPU utilities
- `streaming/` - Old streaming implementations
- Various other experimental implementations

## Need Help?

If you're having trouble migrating, please:
1. Check the new documentation in `/docs/tda/`
2. Look at examples in `/examples/tda/`
3. Contact the AURA team for assistance