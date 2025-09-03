"""
Memory Consolidation Module - September 2025 State-of-the-Art
=============================================================

Complete implementation of biologically-inspired sleep cycles for
continuous learning, memory abstraction, and creative problem solving.

Components:
- SleepConsolidation: Main orchestrator managing sleep phases
- PriorityReplayBuffer: Surprise-based memory prioritization
- DreamGenerator: VAE-based creative recombination
- SynapticHomeostasis: Real-time weight normalization
- TopologicalLaplacianExtractor: Advanced TDA beyond homology

Based on latest research:
- SESLR for 32x memory efficiency
- Topological Laplacians for geometric evolution
- Astrocyte-inspired associative validation
- Sub-minute homeostatic response
"""

from .orchestrator import (
    SleepConsolidation,
    SleepPhase,
    ConsolidationConfig,
    ConsolidationMetrics,
    create_sleep_consolidation
)

from .replay_buffer import (
    PriorityReplayBuffer,
    ReplayMemory
)

from .dream_generator import (
    DreamGenerator,
    DreamVAE,
    AstrocyteAssociativeValidator,
    Dream
)

from .homeostasis import (
    SynapticHomeostasis,
    HomeostasisMetrics,
    WeightSnapshot
)

from .laplacian_extractor import (
    TopologicalLaplacianExtractor,
    LaplacianSignature
)

__all__ = [
    # Orchestrator
    'SleepConsolidation',
    'SleepPhase',
    'ConsolidationConfig',
    'ConsolidationMetrics',
    'create_sleep_consolidation',
    
    # Replay Buffer
    'PriorityReplayBuffer',
    'ReplayMemory',
    
    # Dream Generator
    'DreamGenerator',
    'DreamVAE',
    'AstrocyteAssociativeValidator',
    'Dream',
    
    # Homeostasis
    'SynapticHomeostasis',
    'HomeostasisMetrics',
    'WeightSnapshot',
    
    # Laplacian Extractor
    'TopologicalLaplacianExtractor',
    'LaplacianSignature'
]

# Version information
__version__ = '2.0.0'
__author__ = 'AURA Intelligence Team'
__date__ = 'September 2025'