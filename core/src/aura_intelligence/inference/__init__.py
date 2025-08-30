"""
AURA Intelligence Inference Module
==================================
Implements Active Inference and advanced inference techniques
"""

from .pearl_engine import PEARLInferenceEngine, PEARLConfig, InferenceMode
from .free_energy_core import (
    FreeEnergyMinimizer,
    BeliefState,
    FreeEnergyComponents,
    GenerativeModel,
    create_free_energy_minimizer
)
from .active_inference_lite import (
    ActiveInferenceLite,
    ActiveInferenceMetrics,
    create_active_inference_system
)

__all__ = [
    # PEARL Engine
    'PEARLInferenceEngine',
    'PEARLConfig', 
    'InferenceMode',
    
    # Active Inference Core
    'FreeEnergyMinimizer',
    'BeliefState',
    'FreeEnergyComponents',
    'GenerativeModel',
    'create_free_energy_minimizer',
    
    # Active Inference Integration
    'ActiveInferenceLite',
    'ActiveInferenceMetrics',
    'create_active_inference_system'
]