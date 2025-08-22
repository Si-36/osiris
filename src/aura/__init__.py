"""
AURA Intelligence System
========================

The world's first system that uses topological data analysis to predict 
and prevent cascading failures in multi-agent AI systems.

Core Hypothesis: "Prevent agent failures through topological context intelligence"
Project ID: bc-a397ac41-47c3-4620-a5ec-c56fb1f50fd0

Components:
- 112 TDA Algorithms
- 10 Neural Networks (Liquid NN variants)
- 40 Memory Systems
- 100 Agent Systems
- 51 Infrastructure Components
Total: 213 Components

Copyright (c) 2025 AURA Intelligence
"""

__version__ = "2025.1.0"
__author__ = "AURA Intelligence Team"
__license__ = "MIT"

# Core exports
from .core.system import AURASystem
from .core.config import AURAConfig
from .api.main import AURAMainAPI

__all__ = [
    "AURASystem",
    "AURAConfig", 
    "AURAMainAPI",
    "__version__"
]