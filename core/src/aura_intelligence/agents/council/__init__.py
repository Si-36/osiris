"""
LNN Council Agent Module (2025 Modular Architecture)

Clean, modular implementation following 2025 best practices.
Each module < 150 lines, focused responsibility.
"""

from .core_agent import LNNCouncilAgent
from aura_intelligence.config import LNNCouncilConfig
from .models import GPUAllocationRequest, GPUAllocationDecision, LNNCouncilState

__all__ = [
    "LNNCouncilAgent",
    "LNNCouncilConfig", 
    "GPUAllocationRequest",
    "GPUAllocationDecision",
    "LNNCouncilState"
]

__version__ = "2.0.0"  # 2025 modular architecture