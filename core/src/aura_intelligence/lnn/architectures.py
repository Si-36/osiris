"""
LNN Architectures - Minimal implementation for compatibility
"""

import torch
import torch.nn as nn
from .core import LiquidNeuralNetwork

class LiquidRNN(LiquidNeuralNetwork):
    """Liquid RNN architecture."""
    pass

class LiquidTransformer(nn.Module):
    """Liquid Transformer architecture."""
    def __init__(self, *args, **kwargs):
        super().__init__()

class LiquidAutoencoder(nn.Module):
    """Liquid Autoencoder architecture."""
    def __init__(self, *args, **kwargs):
        super().__init__()

class HybridLiquidNet(nn.Module):
    """Hybrid Liquid Network architecture."""
    def __init__(self, *args, **kwargs):
        super().__init__()

class StreamingLNN(nn.Module):
    """Streaming Liquid Neural Network."""
    def __init__(self, *args, **kwargs):
        super().__init__()

__all__ = ['LiquidRNN', 'LiquidTransformer', 'LiquidAutoencoder', 'HybridLiquidNet', 'StreamingLNN']