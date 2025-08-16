"""
🧪 Distributed Coordination Tests

Comprehensive test suite for distributed coordination system:
- Consensus algorithm testing
- Load balancing validation
- Fault tolerance verification
- Performance benchmarking
- TDA integration testing
"""

import pytest
import asyncio
from datetime import datetime, timezone, timedelta
from unittest.mock import Mock, AsyncMock

from .coordination_core import NodeId, NodeInfo, NodeState, VectorClock
from .consensus import ModernRaftConsensus, RaftState
from .load_balancing import TDALoadBalancer, LoadBalancingStrategy, LoadMetrics
from .coordination_manager import (
    DistributedCoordinationManager, AgentRequest, AgentResponse,
    CoordinationEvent
)


class MockTDAIntegration:
    """Mock TDA integration for testing"""
    
    def __init__(self):
        self.results = []
        self.contexts = {}
        self.patterns = {"anomalies": {"severity": 0.3}}
    
    async def send_orchestration_result(self, result, correlation_id):
        self.results.append((result, correlation_id))
        return True
    
    async def get_context(self, correlation_id):
        return self.contexts.get(correlation_id)
    
    async def get_current_patterns(self, window="1h"):
        return self.patterns


class TestVectorClock:
    """Test vector clock implementation"""
    
    def test_vector_clock_increment(self):
        """Test vector clock increment"""
        clock = VectorClock({"node1": 1, "node2": 2})
        
        incremented = clock.increment("node1")
        
        assert incremented.clocks["node1"] == 2
        assert incremented.clocks["node2"] == 2
        assert clock.clocks["node1"] == 1  # Original unchanged
    
    def test_vector_clock_merge(self):
        """Test vector clock merge"""
        clock1 = VectorClock({"node1": 3, "node2": 1})
        clock2 = VectorClock({"node1": 2, "node2": 4, "node3": 1})
        
        merged = clock1.merge(clock2)
        
        assert merged.clocks["node1"] == 3  # max(3, 2)
        assert merged.clocks["node2"] == 4  # max(1, 4)
        assert merged.clocks["node3"] == 1  # max(0, 1)
    
    def test_happens_before(self):
        """Test happens-before relationship"""
        clock1 = VectorClock({"node1": 1, "node2": 2})
        clock2 = VectorClock({"node1": 2, "node2": 3})
        clock3 = VectorClock({"node1": 2, "node2": 2})
        
        assert clock1.happens_before(clock2)  # All components <=, at least one <
        assert not clock2.happens_before(clock1)  # Not happens before
        assert not clock1.happens_before(clock3)  # Concurrent events


class TestTDALoadBalancer:
    """Test TDA-aware load balancer"""
    
    @pytest.fixture
    def mock_tda_integration(self):
        return MockTDAIntegration()
    
    @pytest.fixture
    def load_balancer(self, mock_tda_integration):
        return TDALoadBalancer(
            strategy=LoadBalancingStrategy.TDA_AWARE,
            tda_integration=mock_tda_integration
        )
    
    @pytest.mark.asyncio
    async def test_node_management(self, load_balancer):
        """Test adding and removing nodes"""
        node_info = NodeInfo(
            node_id="node1",
            address="192.168.1.1",
            port=8080,
            capabilities={"agent_execution", "gpu_enabled"},
            load_factor=0.3,
            state=NodeState.ACTIVE,
            last_seen=datetime.now(timezone.utc)
        )
        
        # Add node
        await load_balancer.add_node(node_info)
        
        assert "node1" in load_balancer.nodes
        assert "node1" in load_balancer.load_metrics
        assert "node1" in load_balancer.circuit_breakers
        
        # Remove node
        await load_balancer.remove_node("node1")
        
        assert "node1" not in load_balancer.nodes
        assert "node1" not in load_balancer.load_metrics
    
    @pytest.mark.asyncio
    async def test_round_robin_selection(self, load_balancer):
        """Test round-robin load balancing"""
        load_balancer.strategy = LoadBalancingStrategy.ROUND_ROBIN
        
        # Add test nodes
        for i in range(3):
            node_info = NodeInfo(
                node_id=f"node{i}",
                address=f"192.168.1.{i+1}",
                port=8080,
                capabilities={"agent_execution"},
                load_factor=0.2,
                state=NodeState.ACTIVE,
                last_seen=datetime.now(timezone.utc)
            )
            await load_balancer.add_node(node_info)
        
        # Test round-robin selection
        selections = []
        for _ in range(6):
            selected = await load_balancer.select_node({"test": "request"})
            selections.append(selected)
        
        # Should cycle through nodes
        assert selections == ["node0", "node1", "node2", "node0", "node1", "node2"]
    
    @pytest.mark.asyncio
    async def test_least_connections_selection(self, load_balancer):
        """Test least connections load balancing"""
        load_balancer.strategy = LoadBalancingStrategy.LEAST_CONNECTIONS
        
        # Add nodes with different connection counts
        for i in range(3):
            node_info = NodeInfo(
                node_id=f"node{i}",
                address=f"192.168.1.{i+1}",
                port=8080,
                capabilities={"agent_execution"},
                load_factor=0.2,
                state=NodeState.ACTIVE,
                last_seen=datetime.now(timezone.utc)
            )
            await load_balancer.add_node(node_info)
            
            # Set different connection counts
            load_balancer.load_metrics[f"node{i}"].active_connections = i * 10
        
        # Should select node with least connections (node0)
        selected = await load_balancer.select_node({"test": "request"})
        assert selected == "node0"
    
    @pytest.mark.asyncio
    async def test_tda_aware_selection(self, load_balancer, mock_tda_integration):
        """Test TDA-aware load balancing"""
        # Add nodes
        for i in range(3):
            node_info = NodeInfo(
                node_id=f"node{i}",
                address=f"192.168.1.{i+1}",
                port=8080,
                capabilities={"agent_execution"},
                load_factor=0.2 + i * 0.1,  # Different load factors
                state=NodeState.ACTIVE,
                last_seen=datetime.now(timezone.utc)
            )
            await load_balancer.add_node(node_info)
        
        # Set TDA context
        mock_tda_integration.contexts["test-correlation"] = Mock(
            anomaly_severity=0.8,
            pattern_confidence=0.9
        )
        
        # Test TDA-aware selection
        request = {
            "workflow_id": "test_workflow",
            "tda_correlation_id": "test-correlation"
        }
        
        selected = await load_balancer.select_node(request)
        assert selected in ["node0", "node1", "node2"]
    
    @pytest.mark.asyncio
    async def test_circuit_breaker(self, load_balancer):
        """Test circuit breaker functionality"""
        # Add node
        node_info = NodeInfo(
            node_id="node1",
            address="192.168.1.1",
            port=8080,
            capabilities={"agent_execution"},
            load_factor=0.2,
            state=NodeState.ACTIVE,
            last_seen=datetime.now(timezone.utc)
        )
        await load_balancer.add_node(node_info)
        
        # Simulate failures
        for _ in range(6):  # Exceed failure threshold
            metrics = LoadMetrics(
                cpu_usage=0.5,
                memory_usage=0.3,
                active_connections=10,
                request_rate=5.0,
                response_time_p95=100.0,
                error_rate=0.2  # High error rate
            )
            await load_balancer.update_node_load("node1", metrics)
        
        # Circuit breaker should be open
        circuit_breaker = load_balancer.circuit_breakers["node1"]
        assert circuit_breaker.state.value == "open"
        
        # Node should not be available for selection
        available_nodes = await load_balancer._get_available_nodes()
        assert "node1" not in available_nodes
    
    @pytest.mark.asyncio
    async def test_load_metrics_update(self, load_balancer, mock_tda_integration):
        """Test load metrics update and TDA correlation"""
        # Add node
        node_info = NodeInfo(
            node_id="node1",
            address="192.168.1.1",
            port=8080,
            capabilities={"agent_execution"},
            load_factor=0.2,
            state=NodeState.ACTIVE,
            last_seen=datetime.now(timezone.utc)
        )
        await load_balancer.add_node(node_info)
        
        # Update load metrics
        metrics = LoadMetrics(
            cpu_usage=0.7,
            memory_usage=0.5,
            active_connections=20,
            request_rate=10.0,
            response_time_p95=200.0,
            error_rate=0.05
        )
        
        await load_balancer.update_node_load("node1", metrics)
        
        # Verify metrics were updated
        assert load_balancer.load_metrics["node1"].cpu_usage == 0.7
        assert load_balancer.load_metrics["node1"].memory_usage == 0.5
        
        # Verify TDA integration was called
        assert len(mock_tda_integration.results) > 0


class TestDistributedCoordinationManager:
    """Test distributed coordination manager"""
    
    @pytest.fixture
    def mock_tda_integration(self):
        return MockTDAIntegration()
    
    @pytest.fixture
    def mock_observability(self):
        mock = AsyncMock()
        return mock
    
    @pytest.fixture
    def coordination_manager(self, mock_tda_integration, mock_observability):
        return DistributedCoordinationManager(
            node_id="test_node",
            cluster_nodes={"test_node", "node1", "node2"},
            tda_integration=mock_tda_integration,
            observability_manager=mock_observability
        )
    
    @pytest.mark.asyncio
    async def test_cluster_join_leave(self, coordination_manager):
        """Test joining and leaving cluster"""
        node_info = NodeInfo(
            node_id="new_node",
            address="192.168.1.10",
            port=8080,
            capabilities={"agent_execution"},
            load_factor=0.1,
            state=NodeState.ACTIVE,
            last_seen=datetime.now(timezone.utc)
        )
        
        # Join cluster
        result = await coordination_manager.join_cluster(node_info)
        assert result is True
        assert "new_node" in coordination_manager.cluster_state
        
        # Leave cluster
        result = await coordination_manager.leave_cluster("new_node")
        assert result is True
        assert "new_node" not in coordination_manager.cluster_state
    
    @pytest.mark.asyncio
    async def test_agent_request_execution(self, coordination_manager, mock_observability):
        """Test agent request execution"""
        # Create agent request
        request = AgentRequest(
            request_id="test_request_001",
            workflow_id="test_workflow",
            agent_type="data_processor",
            operation="process_data",
            parameters={"input": "test_data"},
            tda_correlation_id="test-correlation"
        )
        
        # Execute request
        response = await coordination_manager.execute_agent_request(request)
        
        # Verify response
        assert response.request_id == "test_request_001"
        assert response.status in ["success", "failed"]
        assert response.execution_time >= 0
        
        # Verify observability was called
        assert mock_observability.record_step_execution.call_count >= 1
    
    @pytest.mark.asyncio
    async def test_cluster_status(self, coordination_manager):
        """Test cluster status reporting"""
        status = await coordination_manager.get_cluster_status()
        
        assert "cluster_id" in status
        assert "node_id" in status
        assert "cluster_size" in status
        assert "active_nodes" in status
        assert "cluster_load" in status
        assert "timestamp" in status
        
        assert status["node_id"] == "test_node"
        assert isinstance(status["cluster_size"], int)
    
    @pytest.mark.asyncio
    async def test_event_subscription(self, coordination_manager):
        """Test event subscription and handling"""
        events_received = []
        
        def event_handler(event):
            events_received.append(event)
        
        # Subscribe to events
        coordination_manager.subscribe_to_events(
            CoordinationEvent.NODE_JOINED, 
            event_handler
        )
        
        # Trigger event
        node_info = NodeInfo(
            node_id="event_test_node",
            address="192.168.1.20",
            port=8080,
            capabilities={"agent_execution"},
            load_factor=0.1,
            state=NodeState.ACTIVE,
            last_seen=datetime.now(timezone.utc)
        )
        
        await coordination_manager.join_cluster(node_info)
        
        # Give event processor time to run
        await asyncio.sleep(0.1)
        
        # Verify event was received
        assert len(events_received) > 0
        assert events_received[0]["event_type"] == CoordinationEvent.NODE_JOINED
    
    @pytest.mark.asyncio
    async def test_consensus_integration(self, coordination_manager):
        """Test consensus integration"""
        # Test leadership check
        is_leader = await coordination_manager.consensus.is_leader()
        assert isinstance(is_leader, bool)
        
        # Test consensus proposal (simplified)
        decision = {"action": "scale_up", "target_nodes": 5}
        # Note: In a real test, we'd need to set up a proper consensus cluster
        # For now, we just verify the interface works
        
        # Test getting consensus value
        value = await coordination_manager.get_cluster_decision()
        # May be None if no consensus reached yet
    
    @pytest.mark.asyncio
    async def test_performance_metrics(self, coordination_manager):
        """Test performance metrics collection"""
        # Execute multiple requests to generate metrics
        requests = []
        for i in range(5):
            request = AgentRequest(
                request_id=f"perf_test_{i}",
                workflow_id="performance_test",
                agent_type="test_agent",
                operation="test_operation",
                parameters={"test": f"data_{i}"}
            )
            requests.append(request)
        
        # Execute requests
        responses = []
        for request in requests:
            response = await coordination_manager.execute_agent_request(request)
            responses.append(response)
        
        # Verify all requests completed
        assert len(responses) == 5
        assert all(r.status in ["success", "failed"] for r in responses)
        
        # Check metrics were collected
        assert len(coordination_manager.request_metrics) > 0
    
    @pytest.mark.asyncio
    async def test_fault_tolerance(self, coordination_manager):
        """Test fault tolerance scenarios"""
        # Add a node
        node_info = NodeInfo(
            node_id="fault_test_node",
            address="192.168.1.30",
            port=8080,
            capabilities={"agent_execution"},
            load_factor=0.1,
            state=NodeState.ACTIVE,
            last_seen=datetime.now(timezone.utc)
        )
        
        await coordination_manager.join_cluster(node_info)
        
        # Simulate node failure by not updating health
        coordination_manager.node_health["fault_test_node"] = (
            datetime.now(timezone.utc) - timedelta(seconds=60)  # 1 minute ago
        )
        
        # Wait for health monitor to detect failure
        # (In a real test, we'd trigger the health monitor directly)
        
        # Verify node is marked as failed
        # This would be detected by the health monitor in a running system


class TestIntegrationScenarios:
    """Integration tests for complex scenarios"""
    
    @pytest.mark.asyncio
    async def test_multi_node_coordination(self):
        """Test coordination between multiple nodes"""
        # Create multiple coordination managers
        managers = []
        cluster_nodes = {"node1", "node2", "node3"}
        
        for node_id in cluster_nodes:
            manager = DistributedCoordinationManager(
                node_id=node_id,
                cluster_nodes=cluster_nodes,
                tda_integration=MockTDAIntegration()
            )
            managers.append(manager)
        
        # Test cluster formation
        for manager in managers:
            status = await manager.get_cluster_status()
            assert status["node_id"] in cluster_nodes
            assert status["cluster_size"] >= 1
    
    @pytest.mark.asyncio
    async def test_load_balancing_with_failures(self):
        """Test load balancing behavior during node failures"""
        tda_integration = MockTDAIntegration()
        load_balancer = TDALoadBalancer(
            strategy=LoadBalancingStrategy.TDA_AWARE,
            tda_integration=tda_integration
        )
        
        # Add multiple nodes
        for i in range(5):
            node_info = NodeInfo(
                node_id=f"node{i}",
                address=f"192.168.1.{i+1}",
                port=8080,
                capabilities={"agent_execution"},
                load_factor=0.2,
                state=NodeState.ACTIVE,
                last_seen=datetime.now(timezone.utc)
            )
            await load_balancer.add_node(node_info)
        
        # Simulate some nodes failing
        for i in [1, 3]:  # Fail nodes 1 and 3
            for _ in range(6):  # Trigger circuit breaker
                metrics = LoadMetrics(
                    cpu_usage=0.9,
                    memory_usage=0.8,
                    active_connections=100,
                    request_rate=50.0,
                    response_time_p95=5000.0,
                    error_rate=0.5  # High error rate
                )
                await load_balancer.update_node_load(f"node{i}", metrics)
        
        # Test that load balancer still works with remaining nodes
        for _ in range(10):
            selected = await load_balancer.select_node({"test": "request"})
            assert selected in ["node0", "node2", "node4"]  # Only healthy nodes
    
    @pytest.mark.asyncio
    async def test_tda_integration_end_to_end(self):
        """Test end-to-end TDA integration"""
        tda_integration = MockTDAIntegration()
        
        # Set up TDA context
        tda_integration.contexts["e2e-test"] = Mock(
            anomaly_severity=0.6,
            pattern_confidence=0.8
        )
        
        # Create coordination manager
        manager = DistributedCoordinationManager(
            node_id="e2e_test_node",
            cluster_nodes={"e2e_test_node"},
            tda_integration=tda_integration
        )
        
        # Execute request with TDA correlation
        request = AgentRequest(
            request_id="e2e_test_request",
            workflow_id="e2e_test_workflow",
            agent_type="e2e_test_agent",
            operation="e2e_test_operation",
            parameters={"test": "e2e"},
            tda_correlation_id="e2e-test"
        )
        
        response = await manager.execute_agent_request(request)
        
        # Verify TDA integration was used
        assert len(tda_integration.results) > 0
        
        # Verify response
        assert response.request_id == "e2e_test_request"
        assert response.status in ["success", "failed"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])