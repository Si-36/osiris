"""
CrewAI Integration Module - 2025 Modular Architecture

Exports clean public API for geometric intelligence orchestration.
"""

from .geometric_space import HyperbolicSpace, GeometricRouter
from .flow_engine import FlowEngine, FlowContext
from .orchestrator import CrewAIOrchestrator

__all__ = [
    'HyperbolicSpace',
    'GeometricRouter', 
    'FlowEngine',
    'FlowContext',
    'CrewAIOrchestrator'
]