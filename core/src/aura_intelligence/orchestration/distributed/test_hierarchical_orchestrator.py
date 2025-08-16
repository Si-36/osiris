"""
Tests for Hierarchical Orchestrator

Focused tests for strategic/tactical/operational layer coordination.
"""

import pytest
import asyncio
from datetime import datetime

from .hierarchical_orchestrator import (
    HierarchicalOrchestrator,
    OrchestrationLayer,
    DecisionScope,
    EscalationRequest,
    create_hierarchical_orchestrator
)

class TestHierarchicalOrchestrator:
    """Test hierarchical orchestration system"""
    
    @pytest.fixture
    async def orchestrator(self):
        """Create orchestrator for testing"""
        orch = create_hierarchical_orchestrator()
        await orch.start()
        yield orch
        await orch.stop()
    
    def test_layer_initialization(self, orchestrator):
        """Test that all layers are properly initialized"""
        assert len(orchestrator.layers) == 3
        assert OrchestrationLayer.STRATEGIC in orchestrator.layers
        assert OrchestrationLayer.TACTICAL in orchestrator.layers
        assert OrchestrationLayer.OPERATIONAL in orchestrator.layers
        
        # Check authority levels
        strategic = orchestrator.layers[OrchestrationLayer.STRATEGIC]
        tactical = orchestrator.layers[OrchestrationLayer.TACTICAL]
        operational = orchestrator.layers[OrchestrationLayer.OPERATIONAL]
        
        assert strategic.authority_level > tactical.authority_level
        assert tactical.authority_level > operational.authority_level
    
    @pytest.mark.asyncio
    async def test_operational_layer_processing(self, orchestrator):
        """Test operational layer task execution"""
        context = {
            'tasks': [
                {'id': 'task_1', 'description': 'Process data'},
                {'id': 'task_2', 'description': 'Generate report'}
            ]
        }
        
        result = await orchestrator.submit_request(
            OrchestrationLayer.OPERATIONAL,
            'task_execution',
            context
        )
        
        assert result['status'] == 'executed'
        assert result['layer'] == 'operational'
        assert len(result['results']) == 2
        assert all(r['status'] == 'completed' for r in result['results'])
    
    @pytest.mark.asyncio
    async def test_tactical_layer_coordination(self, orchestrator):
        """Test tactical layer agent coordination"""
        context = {
            'agents': ['agent_1', 'agent_2', 'agent_3', 'agent_4', 'agent_5']
        }
        
        result = await orchestrator.submit_request(
            OrchestrationLayer.TACTICAL,
            'agent_coordination',
            context
        )
        
        assert result['status'] == 'coordinated'
        assert result['layer'] == 'tactical'
        assert 'coordination_plan' in result
        assert len(result['coordination_plan']['primary_agents']) <= 3
    
    @pytest.mark.asyncio
    async def test_strategic_layer_resource_allocation(self, orchestrator):
        """Test strategic layer resource allocation"""
        context = {
            'agents': ['agent_1', 'agent_2'],
            'urgency': 0.9
        }
        
        result = await orchestrator.submit_request(
            OrchestrationLayer.STRATEGIC,
            'resource_allocation',
            context
        )
        
        assert result['status'] == 'approved'
        assert result['layer'] == 'strategic'
        assert 'allocated_resources' in result
        assert result['allocated_resources']['priority'] == 'high'
    
    def test_complexity_assessment(self, orchestrator):
        """Test request complexity assessment"""
        # Simple request
        simple_context = {
            'agents': ['agent_1'],
            'resources': {'cpu_hours': 1, 'memory_gb': 2},
            'deadline_hours': 24
        }
        complexity = orchestrator._assess_complexity(simple_context)
        assert complexity < 0.5
        
        # Complex request
        complex_context = {
            'agents': [f'agent_{i}' for i in range(15)],
            'resources': {'cpu_hours': 200, 'memory_gb': 100},
            'deadline_hours': 0.5
        }
        complexity = orchestrator._assess_complexity(complex_context)
        assert complexity > 0.8
    
    @pytest.mark.asyncio
    async def test_escalation_flow(self, orchestrator):
        """Test escalation from operational to tactical layer"""
        # Create high-complexity request at operational layer
        complex_context = {
            'agents': [f'agent_{i}' for i in range(20)],
            'resources': {'cpu_hours': 500},
            'deadline_hours': 0.1,
            'urgency': 0.95
        }
        
        result = await orchestrator.submit_request(
            OrchestrationLayer.OPERATIONAL,
            'resource_allocation',
            complex_context
        )
        
        # Should be escalated and processed at higher layer
        assert result['status'] in ['approved', 'processed']
        # Note: In this test, it will be processed at tactical or strategic level
    
    @pytest.mark.asyncio
    async def test_layer_status(self, orchestrator):
        """Test layer status reporting"""
        status = await orchestrator.get_layer_status(OrchestrationLayer.TACTICAL)
        
        assert status['layer'] == 'tactical'
        assert 'authority_level' in status
        assert 'active_workflows' in status
        assert 'escalation_threshold' in status
        assert 'decision_scope' in status
    
    @pytest.mark.asyncio
    async def test_system_status(self, orchestrator):
        """Test overall system status"""
        status = await orchestrator.get_system_status()
        
        assert status['status'] == 'operational'
        assert 'layers' in status
        assert len(status['layers']) == 3
        assert 'escalation_queue_size' in status
        assert 'coordination_channels' in status
    
    def test_layer_hierarchy(self, orchestrator):
        """Test layer hierarchy and escalation paths"""
        # Test escalation path from operational
        higher = orchestrator._get_higher_layer(OrchestrationLayer.OPERATIONAL)
        assert higher == OrchestrationLayer.TACTICAL
        
        # Test escalation path from tactical
        higher = orchestrator._get_higher_layer(OrchestrationLayer.TACTICAL)
        assert higher == OrchestrationLayer.STRATEGIC
        
        # Test no escalation from strategic (top layer)
        higher = orchestrator._get_higher_layer(OrchestrationLayer.STRATEGIC)
        assert higher is None

class TestEscalationRequest:
    """Test escalation request functionality"""
    
    def test_escalation_request_creation(self):
        """Test escalation request creation"""
        escalation = EscalationRequest(
            request_id='test_123',
            from_layer=OrchestrationLayer.OPERATIONAL,
            to_layer=OrchestrationLayer.TACTICAL,
            decision_type='resource_allocation',
            context={'test': 'data'},
            urgency=0.8
        )
        
        assert escalation.request_id == 'test_123'
        assert escalation.from_layer == OrchestrationLayer.OPERATIONAL
        assert escalation.to_layer == OrchestrationLayer.TACTICAL
        assert escalation.urgency == 0.8
        assert isinstance(escalation.created_at, datetime)

@pytest.mark.asyncio
async def test_concurrent_requests():
    """Test handling multiple concurrent requests"""
    orchestrator = create_hierarchical_orchestrator()
    await orchestrator.start()
    
    try:
        # Submit multiple requests concurrently
        contexts = [
            {'tasks': [{'id': f'task_{i}', 'description': f'Task {i}'}]}
            for i in range(5)
        ]
        
        results = await asyncio.gather(*[
            orchestrator.submit_request(
                OrchestrationLayer.OPERATIONAL,
                'task_execution',
                context
            )
            for context in contexts
        ])
        
        assert len(results) == 5
        assert all(r['status'] == 'executed' for r in results)
        
    finally:
        await orchestrator.stop()

@pytest.mark.asyncio
async def test_tda_integration():
    """Test TDA integration (mock)"""
    # Mock TDA integration
    mock_tda = {'anomaly_detection': True}
    
    orchestrator = create_hierarchical_orchestrator(tda_integration=mock_tda)
    await orchestrator.start()
    
    try:
        status = await orchestrator.get_system_status()
        assert status['tda_integration'] is True
        
    finally:
        await orchestrator.stop()

def test_factory_function():
    """Test factory function"""
    orchestrator = create_hierarchical_orchestrator()
    
    assert isinstance(orchestrator, HierarchicalOrchestrator)
    assert len(orchestrator.layers) == 3
    assert orchestrator.tda_integration is None

if __name__ == "__main__":
    pytest.main([__file__, "-v"])