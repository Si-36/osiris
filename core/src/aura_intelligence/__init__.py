"""
AURA Intelligence Core System
Advanced AI platform with real intelligence integration
"""

__version__ = "2.0.0"

# Core system components
from .core.unified_system import UnifiedSystem, get_unified_system, create_unified_system
from .core.unified_config import UnifiedConfig, get_config
from .core.unified_interfaces import (
    UnifiedComponent, ComponentStatus, ComponentMetrics,
    AgentComponent, MemoryComponent, NeuralComponent,
    OrchestrationComponent, ObservabilityComponent
)

# Infrastructure components - conditional import to avoid dependency issues
try:
    from .infrastructure.gemini_client import GeminiClient, GeminiClientManager
except ImportError:
    GeminiClient = None
    GeminiClientManager = None

# Main orchestrator
# Lazy import to avoid circular references
# from .unified_brain import UnifiedAURABrain, UnifiedConfig as BrainConfig, AnalysisResult

# Export main classes for easy import
__all__ = [
    # Core System
    "UnifiedSystem",
    "get_unified_system", 
    "create_unified_system",
    "UnifiedConfig",
    "get_config",
    
    # Component Interfaces
    "UnifiedComponent",
    "ComponentStatus", 
    "ComponentMetrics",
    "AgentComponent",
    "MemoryComponent", 
    "NeuralComponent",
    "OrchestrationComponent",
    "ObservabilityComponent",
    
    # AI Integration
    "GeminiClient",
    "GeminiClientManager",
    
    # Main Brain
    "UnifiedAURABrain",
    "BrainConfig",
    "AnalysisResult",
    
    # Version
    "__version__",
]
