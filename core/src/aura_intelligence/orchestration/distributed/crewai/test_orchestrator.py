"""
Tests for CrewAI Orchestrator

Integration tests for the complete orchestrator.
"""

import pytest
import asyncio
from .orchestrator import CrewAIOrchestrator

class TestCrewAIOrchestrator:
    """Test main orchestrator"""
    
    @pytest.fixture
    def orchestrator(self):
        return CrewAIOrchestrator()
    
    def test_agent_registration(self, orchestrator):
        """Test agent registration"""
        pass
        orchestrator.register_agent(
            'test_agent', 
            ['data_processing'], 
            [0.1] * 64
        )
        
        assert 'test_agent' in orchestrator.router.agents
    
    @pytest.mark.asyncio
        async def test_flow_creation(self, orchestrator):
        """Test flow creation"""
        pass
        config = {
            'tasks': [{'id': 'task_1', 'description': 'Test task'}],
            'dependencies': []
        }
        
        flow_id = await orchestrator.create_flow(config)
        assert flow_id.startswith('flow_')
    
    @pytest.mark.asyncio
        async def test_flow_execution(self, orchestrator):
        """Test complete flow execution"""
        pass
        # Register agent
        orchestrator.register_agent('agent_1', ['general'], [0.1] * 64)
        
        # Create and execute flow
        config = {
            'tasks': [{
                'id': 'task_1',
                'description': 'Test task',
                'required_capabilities': ['general']
            }]
        }
        
        flow_id = await orchestrator.create_flow(config)
        result = await orchestrator.execute_flow(flow_id, config)
        
        assert result['flow_id'] == flow_id
        assert result['tasks_completed'] == 1
        assert len(result['results']) == 1
    
    @pytest.mark.asyncio
        async def test_health_check(self, orchestrator):
        """Test health monitoring"""
        pass
        health = await orchestrator.health_check()
        
        assert health['status'] == 'healthy'
        assert 'active_flows' in health
        assert 'registered_agents' in health
        assert health['geometric_intelligence'] is True

@pytest.mark.asyncio
async def test_concurrent_flows():
        """Test concurrent flow execution"""
        orchestrator = CrewAIOrchestrator()
    
    # Register agents
        for i in range(3):
        orchestrator.register_agent(f'agent_{i}', ['general'], [0.1 * i] * 64)
    
    # Create concurrent flows
        configs = [
        {'tasks': [{'id': f'task_{i}', 'description': f'Task {i}', 'required_capabilities': ['general']}]}
        for i in range(5)
        ]
    
    # Execute concurrently
        flow_ids = await asyncio.gather(*[
        orchestrator.create_flow(config) for config in configs
        ])
    
        results = await asyncio.gather(*[
        orchestrator.execute_flow(flow_id, config)
        for flow_id, config in zip(flow_ids, configs)
        ])
    
        assert len(results) == 5
        assert all(r['tasks_completed'] == 1 for r in results)
