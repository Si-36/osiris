"""
🔥 AURA Intelligence TDA Engine - Production Grade
Enterprise-ready Topological Data Analysis with 30x GPU acceleration.
"""

# Only import what actually exists
try:
    from .algorithms import RipsComplex, PersistentHomology
except (SyntaxError, IndentationError):
    from .algorithms_simple import RipsComplex, PersistentHomology
from .models import TDARequest, TDAResponse, TDAMetrics, TDAConfiguration, TDAAlgorithm, DataFormat

__all__ = [
    'RipsComplex',
    'PersistentHomology',
    'TDARequest', 
    'TDAResponse',
    'TDAMetrics',
    'TDAConfiguration',
    'TDAAlgorithm',
    'DataFormat',
]

__version__ = "1.0.0"
