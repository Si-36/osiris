"""
LNN Training - Minimal implementation for compatibility
"""

import torch
import torch.nn as nn

class LiquidTrainer:
    """Liquid Neural Network trainer."""
    def __init__(self, *args, **kwargs):
        pass

class BackpropThroughTime:
    """Backpropagation through time for LNNs."""
    def __init__(self, *args, **kwargs):
        pass

class AdjointSensitivity:
    """Adjoint sensitivity method for LNNs."""
    def __init__(self, *args, **kwargs):
        pass

class SparsityRegularizer:
    """Sparsity regularization for LNNs."""
    def __init__(self, *args, **kwargs):
        pass

class TemporalLoss(nn.Module):
    """Temporal loss function for LNNs."""
    def __init__(self, *args, **kwargs):
        super().__init__()

__all__ = ['LiquidTrainer', 'BackpropThroughTime', 'AdjointSensitivity', 'SparsityRegularizer', 'TemporalLoss']