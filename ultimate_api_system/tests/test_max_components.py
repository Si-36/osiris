"""
Tests for the MAX-accelerated components.
"""

import pytest
import numpy as np
import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from ultimate_api_system.max_aura_api import MAXModelManager
from ultimate_api_system.components.neural.max_lnn import MAXLiquidNeuralNetwork

def check_max_available():
    try:
        import max.engine
        return True
    except ImportError:
        return False

MAX_AVAILABLE = check_max_available()

@pytest.fixture
def model_manager():
    """Fixture for the MAXModelManager."""
    if not MAX_AVAILABLE:
        pytest.skip("MAX engine not available")
    return MAXModelManager()

@pytest.mark.asyncio
@pytest.mark.skipif(not MAX_AVAILABLE, reason="MAX engine not available")
async def test_max_lnn_component_initialization(model_manager):
    """Test that the MAXLiquidNeuralNetwork component initializes correctly."""
    lnn_component = MAXLiquidNeuralNetwork(
        component_id="test_lnn",
        config={},
        model_manager=model_manager
    )
    assert await lnn_component.initialize()

@pytest.mark.asyncio
@pytest.mark.skipif(not MAX_AVAILABLE, reason="MAX engine not available")
async def test_max_lnn_component_process(model_manager):
    """Test that the MAXLiquidNeuralNetwork component can process a request."""
    lnn_component = MAXLiquidNeuralNetwork(
        component_id="test_lnn",
        config={},
        model_manager=model_manager
    )
    await lnn_component.initialize()

    # Create a dummy input tensor
    input_data = np.random.rand(1, 128, 768).astype(np.float32)

    # Process the input
    result = await lnn_component.process(input_data)

    # Check the output
    assert isinstance(result, np.ndarray)
    assert result.shape == (1, 128, 768) # Simple ReLU preserves input shape
