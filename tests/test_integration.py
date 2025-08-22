"""
Integration tests for AURA system
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import unittest
from aura.core.system import AURASystem
from aura.core.config import AURAConfig

class TestAURAIntegration(unittest.TestCase):
    """Test full AURA system integration"""
    
    def setUp(self):
        """Set up test environment"""
        self.config = AURAConfig()
        self.system = AURASystem(self.config)
    
    def test_component_count(self):
        """Test that all 213 components are registered"""
        stats = self.system.component_stats
        total = sum(stats.values())
        self.assertEqual(total, 213, f"Expected 213 components, found {total}")
    
    def test_tda_algorithms(self):
        """Test TDA algorithm count"""
        self.assertEqual(len(self.system.tda_algorithms), 112)
    
    def test_neural_networks(self):
        """Test neural network variants"""
        self.assertEqual(len(self.system.neural_networks), 10)
    
    def test_memory_systems(self):
        """Test memory system components"""
        self.assertEqual(len(self.system.memory_components), 40)
    
    def test_agent_systems(self):
        """Test agent creation"""
        self.assertEqual(len(self.system.agents), 100)
        
        # Check specific agents exist
        self.assertIn("pattern_ia_001", self.system.agents)
        self.assertIn("resource_ca_001", self.system.agents)
    
    def test_infrastructure(self):
        """Test infrastructure components"""
        self.assertEqual(len(self.system.infrastructure), 51)
    
    async def test_pipeline(self):
        """Test complete pipeline"""
        # Create test data
        agent_data = {
            "agents": [{"id": i, "health": 0.8} for i in range(30)],
            "connections": [(i, i+1) for i in range(29)],
        }
        
        # Run pipeline
        result = await self.system.execute_pipeline(agent_data)
        
        # Check result
        self.assertIn("risk_level", result)
        self.assertIn("action", result)
        self.assertIn("topology", result)

if __name__ == "__main__":
    unittest.main()
