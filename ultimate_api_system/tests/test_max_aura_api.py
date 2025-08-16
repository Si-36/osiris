import pytest
import sys
import os
from httpx import AsyncClient

# Set the MODULAR_HOME environment variable before importing max
pixi_env_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '.pixi', 'envs', 'default'))
os.environ['MODULAR_HOME'] = os.path.join(pixi_env_path, 'share', 'max')
os.environ['CONDA_PREFIX'] = pixi_env_path # Some MAX components might need this too

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from ultimate_api_system.max_aura_api import app

@pytest.mark.asyncio
async def test_health_check():
    """
    Tests if the /health endpoint returns a 200 OK status.
    """
    async with AsyncClient(app=app, base_url="http://test") as ac:
        response = await ac.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
