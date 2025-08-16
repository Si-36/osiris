"""
AURA Intelligence Consciousness Layer

This module implements the consciousness layer based on Global Workspace Theory (GWT)
and other cognitive architectures, providing system-wide awareness, attention,
and executive control.

The consciousness layer includes:
- Global Workspace Theory implementation for distributed awareness
- Attention mechanisms for selective information processing
- Executive functions for high-level planning and control
- Metacognitive monitoring and self-awareness
- Consciousness stream for continuous awareness processing
"""

from .global_workspace import (
    MetaCognitiveController,
    GlobalWorkspace,
    ConsciousnessStream,
    ConsciousDecision,
    WorkspaceContent,
    ConsciousnessLevel,
    create_metacognitive_controller,
    get_global_workspace
)

from .attention import (
    AttentionMechanism,
    AttentionFocus,
    SalienceDetector,
    ResourceAllocator,
    TransformerAttention,
    create_attention_mechanism,
    get_attention_mechanism
)

from .executive_functions import (
    ExecutiveFunction,
    WorkingMemory,
    CognitiveFlexibility,
    InhibitoryControl,
    PlanningSystem,
    Goal,
    Plan,
    GoalStatus,
    ExecutiveState,
    create_executive_function,
    create_goal
)

__all__ = [
    'MetaCognitiveController',
    'GlobalWorkspace', 
    'ConsciousnessStream',
    'ConsciousDecision',
    'WorkspaceContent',
    'ConsciousnessLevel',
    'create_metacognitive_controller',
    'get_global_workspace',
    'AttentionMechanism',
    'AttentionFocus',
    'SalienceDetector',
    'ResourceAllocator',
    'TransformerAttention',
    'create_attention_mechanism',
    'get_attention_mechanism',
    'ExecutiveFunction',
    'WorkingMemory',
    'CognitiveFlexibility', 
    'InhibitoryControl',
    'PlanningSystem',
    'Goal',
    'Plan',
    'GoalStatus',
    'ExecutiveState',
    'create_executive_function',
    'create_goal'
]