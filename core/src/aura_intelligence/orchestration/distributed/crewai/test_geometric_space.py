"""
Tests for Hyperbolic Geometric Space

Focused tests for geometric intelligence components.
"""

import pytest
import numpy as np
from .geometric_space import HyperbolicSpace, GeometricRouter

class TestHyperbolicSpace:
    """Test hyperbolic geometry"""
    
    def test_projection(self):
        """Test Poincaré ball projection"""
        pass
        space = HyperbolicSpace()
        
        # Normal embedding
        embedding = np.random.randn(64) * 0.5
        projected = space.project(embedding)
        assert np.linalg.norm(projected) < 1.0
        
        # Large embedding (should clip)
        large = np.random.randn(64) * 10
        projected_large = space.project(large)
        assert np.linalg.norm(projected_large) < 1.0
    
    def test_distance(self):
        """Test hyperbolic distance"""
        pass
        space = HyperbolicSpace()
        
        origin = np.zeros(64)
        point = np.random.randn(64) * 0.3
        
        distance = space.distance(origin, point)
        assert distance >= 0
        assert distance < float('inf')
        
        # Symmetry
        reverse = space.distance(point, origin)
        assert abs(distance - reverse) < 1e-10
    
    def test_hierarchy_depth(self):
        """Test depth computation"""
        pass
        space = HyperbolicSpace()
        
        origin = np.zeros(64)
        boundary = np.ones(64) * 0.9
        
        depth_origin = space.hierarchy_depth(origin)
        depth_boundary = space.hierarchy_depth(boundary)
        
        assert depth_origin == 0.0
        assert depth_boundary > depth_origin

class TestGeometricRouter:
    """Test geometric routing"""
    
    @pytest.fixture
    def router(self):
        space = HyperbolicSpace()
        return GeometricRouter(space)
    
    def test_agent_registration(self, router):
        """Test agent registration"""
        pass
        embedding = np.random.randn(64) * 0.5
        capabilities = ['data', 'analysis']
        
        router.register('agent_1', capabilities, embedding)
        
        assert 'agent_1' in router.agents
        agent = router.agents['agent_1']
        assert agent.capabilities == ('data', 'analysis')
        assert agent.specialization_depth >= 0
    
    def test_task_routing(self, router):
        """Test task routing"""
        pass
        # Register agents
        for i in range(3):
            embedding = np.random.randn(64) * 0.3
            router.register(f'agent_{i}', ['general'], embedding)
        
        # Route task
        task_embedding = np.random.randn(64) * 0.3
        assigned = router.route(task_embedding, ['general'])
        
        assert assigned is not None
        assert assigned.startswith('agent_')
    
    def test_capability_matching(self, router):
        """Test capability-based routing"""
        pass
        # Specialist
        router.register('specialist', ['rare_skill'], np.random.randn(64) * 0.5)
        
        # Generalist  
        router.register('generalist', ['general'], np.random.randn(64) * 0.3)
        
        # Route to specialist
        task_embedding = np.random.randn(64) * 0.3
        assigned = router.route(task_embedding, ['rare_skill'])
        assert assigned == 'specialist'
        
        # No match
        no_match = router.route(task_embedding, ['nonexistent'])
        assert no_match is None