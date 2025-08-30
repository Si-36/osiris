"""
⚡ Mixture of Experts (MoE) Module - Production 2025
==================================================

Switch Transformer-based MoE with LNN integration for:
- Sparse expert routing (only 2-4 experts per token)
- Load balanced computation
- Complexity-aware gating
- Expert specialization

Based on Google's Switch Transformer paper.
"""

from .google_switch_transformer import (
    SwitchTransformerMoE,
    GoogleSwitchMoESystem
)

from .enhanced_switch_moe import (
    ProductionSwitchMoE,
    SwitchMoEWithLNN,
    MoEConfig,
    ExpertType,
    ExpertModule,
    create_production_switch_moe
)

__all__ = [
    # Google's original
    'SwitchTransformerMoE',
    'GoogleSwitchMoESystem',
    
    # Enhanced production version
    'ProductionSwitchMoE',
    'SwitchMoEWithLNN',
    'MoEConfig',
    'ExpertType',
    'ExpertModule',
    'create_production_switch_moe'
]