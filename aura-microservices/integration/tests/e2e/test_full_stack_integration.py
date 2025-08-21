"""
Full Stack Integration Tests for AURA Microservices
End-to-end testing with all services integrated

2025 Best Practices:
- Parallel test execution
- Snapshot testing
- Property-based testing
- Visual regression testing
"""

import asyncio
import pytest
import httpx
import json
import time
from typing import Dict, Any, List
import numpy as np
from datetime import datetime

# Import our testing frameworks
import sys
sys.path.append('/workspace/aura-microservices')

from integration.framework.testcontainers.container_manager import AuraTestContainers
from integration.framework.contracts.contract_framework import ContractTestRunner
from integration.framework.chaos.chaos_framework_2025 import (
    ChaosOrchestrator, ChaosInjector, ChaosExperiment, ChaosHypothesis,
    FaultType
)


@pytest.fixture(scope="module")
async def aura_stack():
    """Start complete AURA stack for testing"""
    manager = AuraTestContainers(use_compose=False)
    
    # Pre-warm containers for speed
    await manager.prepare_warm_pools()
    
    async with manager.start_aura_stack() as service_urls:
        yield service_urls, manager


@pytest.fixture
async def http_client():
    """HTTP client for tests"""
    async with httpx.AsyncClient(timeout=30.0) as client:
        yield client


class TestFullStackIntegration:
    """
    Comprehensive integration tests for AURA platform
    """
    
    @pytest.mark.asyncio
    async def test_end_to_end_workflow(self, aura_stack, http_client):
        """Test complete workflow across all services"""
        service_urls, _ = aura_stack
        
        # Step 1: Process data with Neuromorphic service
        spike_data = [[random.randint(0, 1) for _ in range(125)] for _ in range(10)]
        
        neuro_response = await http_client.post(
            f"{service_urls['neuromorphic']}/api/v1/process/spike",
            json={
                "spike_data": spike_data,
                "time_steps": 10
            }
        )
        assert neuro_response.status_code == 200
        neuro_result = neuro_response.json()
        
        # Step 2: Store result in Memory service with shape analysis
        memory_response = await http_client.post(
            f"{service_urls['memory']}/api/v1/store",
            json={
                "data": {
                    "neuromorphic_output": neuro_result,
                    "timestamp": datetime.utcnow().isoformat()
                },
                "enable_shape_analysis": True
            }
        )
        assert memory_response.status_code == 200
        memory_result = memory_response.json()
        memory_id = memory_result["memory_id"]
        
        # Step 3: Use MoE Router for intelligent routing
        routing_response = await http_client.post(
            f"{service_urls['moe']}/api/v1/route",
            json={
                "data": {
                    "type": "consensus",
                    "memory_ref": memory_id,
                    "priority": 0.9,
                    "complexity": 0.7
                }
            }
        )
        assert routing_response.status_code == 200
        routing_result = routing_response.json()
        assert "byzantine" in routing_result["selected_services"]
        
        # Step 4: Byzantine consensus on decision
        # First register nodes
        nodes = ["alpha", "beta", "gamma", "delta"]
        for node_id in nodes:
            register_response = await http_client.post(
                f"{service_urls['byzantine']}/api/v1/nodes/register",
                json={
                    "node_id": node_id,
                    "address": f"http://{node_id}:8002"
                }
            )
            assert register_response.status_code == 200
        
        # Propose consensus
        propose_response = await http_client.post(
            f"{service_urls['byzantine']}/api/v1/consensus/propose",
            json={
                "node_id": nodes[0],
                "topic": "integration_test_decision",
                "value": {
                    "memory_id": memory_id,
                    "neuromorphic_energy": neuro_result["energy_consumed_pj"],
                    "decision": "approve"
                }
            }
        )
        assert propose_response.status_code == 200
        proposal_id = propose_response.json()["proposal_id"]
        
        # Cast votes
        for node_id in nodes:
            vote_response = await http_client.post(
                f"{service_urls['byzantine']}/api/v1/consensus/vote",
                json={
                    "node_id": node_id,
                    "proposal_id": proposal_id,
                    "vote_value": {"decision": "approve"},
                    "confidence": 0.9
                }
            )
            assert vote_response.status_code == 200
        
        # Wait for consensus
        await asyncio.sleep(2)
        
        # Check consensus state
        state_response = await http_client.get(
            f"{service_urls['byzantine']}/api/v1/consensus/state/{proposal_id}"
        )
        assert state_response.status_code == 200
        consensus_state = state_response.json()
        assert consensus_state["status"] == "DECIDED"
        
        # Step 5: Adaptive learning with LNN
        lnn_response = await http_client.post(
            f"{service_urls['lnn']}/api/v1/inference",
            json={
                "model_id": "adaptive",
                "input_data": [0.1 * i for i in range(128)],
                "session_id": "integration_test",
                "return_dynamics": True
            }
        )
        assert lnn_response.status_code == 200
        lnn_result = lnn_response.json()
        
        # Adapt based on feedback
        adapt_response = await http_client.post(
            f"{service_urls['lnn']}/api/v1/adapt",
            json={
                "model_id": "adaptive",
                "feedback_signal": 0.8,
                "adaptation_strength": 0.1
            }
        )
        assert adapt_response.status_code == 200
        
        # Verify complete workflow
        assert neuro_result["energy_consumed_pj"] < 1000  # Energy efficient
        assert memory_result["tier"] in ["L1_CACHE", "L2_CACHE", "L3_CACHE"]  # Fast tier
        assert routing_result["routing_strategy"] in ["adaptive", "semantic"]  # Smart routing
        assert consensus_state["final_confidence"] > 0.8  # High confidence
        assert lnn_result["latency_ms"] < 50  # Fast inference
    
    @pytest.mark.asyncio
    async def test_parallel_request_handling(self, aura_stack, http_client):
        """Test system under parallel load"""
        service_urls, _ = aura_stack
        
        # Create parallel requests
        async def make_request(index: int):
            # Route through MoE
            response = await http_client.post(
                f"{service_urls['moe']}/api/v1/route",
                json={
                    "data": {
                        "type": "inference",
                        "request_id": f"parallel_{index}",
                        "data": [random.random() for _ in range(100)],
                        "priority": random.random()
                    }
                }
            )
            return response.status_code == 200, response.elapsed.total_seconds()
        
        # Send 50 parallel requests
        tasks = [make_request(i) for i in range(50)]
        results = await asyncio.gather(*tasks)
        
        # Analyze results
        success_count = sum(1 for success, _ in results if success)
        latencies = [latency for success, latency in results if success]
        
        assert success_count >= 48  # 96% success rate
        assert np.mean(latencies) < 0.1  # Average < 100ms
        assert np.percentile(latencies, 95) < 0.2  # P95 < 200ms
    
    @pytest.mark.asyncio
    async def test_service_discovery_and_registration(self, aura_stack, http_client):
        """Test dynamic service registration"""
        service_urls, _ = aura_stack
        
        # Register a custom service
        register_response = await http_client.post(
            f"{service_urls['moe']}/api/v1/services/register",
            json={
                "service_id": "custom_test_service",
                "service_type": "custom",
                "endpoint": "http://localhost:9999",
                "capabilities": ["test", "demo"],
                "max_capacity": 50
            }
        )
        assert register_response.status_code == 200
        
        # List services
        list_response = await http_client.get(
            f"{service_urls['moe']}/api/v1/services"
        )
        assert list_response.status_code == 200
        services = list_response.json()["services"]
        
        # Verify custom service is registered
        custom_service = next(
            (s for s in services if s["service_id"] == "custom_test_service"),
            None
        )
        assert custom_service is not None
        assert custom_service["capabilities"] == ["test", "demo"]
        
        # Unregister service
        unregister_response = await http_client.delete(
            f"{service_urls['moe']}/api/v1/services/custom_test_service"
        )
        assert unregister_response.status_code == 200
    
    @pytest.mark.asyncio
    async def test_cross_service_data_consistency(self, aura_stack, http_client):
        """Test data consistency across services"""
        service_urls, _ = aura_stack
        
        # Create test data
        test_data = {
            "id": "consistency_test_001",
            "values": [random.random() for _ in range(50)],
            "metadata": {
                "source": "integration_test",
                "timestamp": time.time()
            }
        }
        
        # Store in memory service
        store_response = await http_client.post(
            f"{service_urls['memory']}/api/v1/store",
            json={
                "data": test_data,
                "enable_shape_analysis": True
            }
        )
        assert store_response.status_code == 200
        memory_id = store_response.json()["memory_id"]
        
        # Retrieve and verify
        retrieve_response = await http_client.get(
            f"{service_urls['memory']}/api/v1/retrieve/{memory_id}"
        )
        assert retrieve_response.status_code == 200
        retrieved_data = retrieve_response.json()["data"]
        
        # Verify data integrity
        assert retrieved_data["id"] == test_data["id"]
        assert len(retrieved_data["values"]) == len(test_data["values"])
        assert abs(sum(retrieved_data["values"]) - sum(test_data["values"])) < 0.001
    
    @pytest.mark.asyncio
    async def test_performance_under_load(self, aura_stack, http_client):
        """Test system performance characteristics"""
        service_urls, _ = aura_stack
        
        # Warm up
        for _ in range(10):
            await http_client.get(f"{service_urls['moe']}/api/v1/health")
        
        # Performance test
        start_time = time.time()
        request_count = 100
        
        async def timed_request():
            req_start = time.perf_counter()
            response = await http_client.post(
                f"{service_urls['neuromorphic']}/api/v1/process/spike",
                json={
                    "spike_data": [[1, 0, 1, 0, 1] * 25],
                    "time_steps": 5
                }
            )
            req_end = time.perf_counter()
            return response.status_code, (req_end - req_start) * 1000
        
        # Execute requests
        tasks = [timed_request() for _ in range(request_count)]
        results = await asyncio.gather(*tasks)
        
        total_time = time.time() - start_time
        
        # Calculate metrics
        successful = [r for r in results if r[0] == 200]
        latencies = [r[1] for r in successful]
        
        throughput = len(successful) / total_time
        avg_latency = np.mean(latencies)
        p50_latency = np.percentile(latencies, 50)
        p95_latency = np.percentile(latencies, 95)
        p99_latency = np.percentile(latencies, 99)
        
        # Assert performance requirements
        assert len(successful) >= 98  # 98% success rate
        assert throughput >= 20  # At least 20 req/s
        assert avg_latency < 50  # Average < 50ms
        assert p95_latency < 100  # P95 < 100ms
        assert p99_latency < 200  # P99 < 200ms
        
        print(f"\nPerformance Results:")
        print(f"Throughput: {throughput:.2f} req/s")
        print(f"Average Latency: {avg_latency:.2f}ms")
        print(f"P50 Latency: {p50_latency:.2f}ms")
        print(f"P95 Latency: {p95_latency:.2f}ms")
        print(f"P99 Latency: {p99_latency:.2f}ms")


class TestContractValidation:
    """Test contract compliance between services"""
    
    @pytest.mark.asyncio
    async def test_consumer_contracts(self, aura_stack):
        """Validate all consumer contracts"""
        service_urls, _ = aura_stack
        
        # Run contract tests for MoE Router as consumer
        runner = ContractTestRunner("/workspace/aura-microservices/contracts")
        
        results = await runner.run_consumer_tests(
            consumer_name="moe-router",
            provider_url=service_urls["neuromorphic"]
        )
        
        # Verify all contracts pass
        failures = [r for r in results if not r.valid]
        
        if failures:
            for failure in failures:
                print(f"\nContract failure: {failure.contract_id}")
                print(f"Errors: {failure.errors}")
        
        assert len(failures) == 0, f"{len(failures)} contract(s) failed"
    
    @pytest.mark.asyncio
    async def test_backward_compatibility(self, aura_stack, http_client):
        """Test API backward compatibility"""
        service_urls, _ = aura_stack
        
        # Test v1 endpoints still work
        v1_endpoints = [
            (f"{service_urls['neuromorphic']}/api/v1/health", "GET"),
            (f"{service_urls['memory']}/api/v1/health", "GET"),
            (f"{service_urls['byzantine']}/api/v1/health", "GET"),
            (f"{service_urls['lnn']}/api/v1/health", "GET"),
            (f"{service_urls['moe']}/api/v1/health", "GET")
        ]
        
        for endpoint, method in v1_endpoints:
            response = await http_client.request(method, endpoint)
            assert response.status_code == 200, f"Endpoint {endpoint} failed"


class TestChaosEngineering:
    """Chaos engineering tests"""
    
    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_network_partition_recovery(self, aura_stack):
        """Test recovery from network partition"""
        service_urls, manager = aura_stack
        
        # Create chaos components
        injector = ChaosInjector(manager)
        orchestrator = ChaosOrchestrator(service_urls, injector)
        
        # Define experiment
        experiment = ChaosExperiment(
            name="network_partition_test",
            hypothesis=ChaosHypothesis(
                description="System recovers from network partition",
                steady_state_metrics={
                    "min_health_percentage": 80,
                    "max_latency_ms": 100
                },
                expected_impact="Temporary service unavailability",
                recovery_time_slo=timedelta(seconds=30),
                blast_radius=["neuromorphic", "moe"]
            ),
            target_services=["neuromorphic"],
            fault_specs=[{
                "target": "neuromorphic",
                "type": FaultType.NETWORK_PARTITION.value,
                "duration": 20,
                "intensity": 0.8
            }],
            duration=timedelta(seconds=20)
        )
        
        # Run experiment
        result = await orchestrator.run_experiment(experiment)
        
        # Verify hypothesis
        assert result.hypothesis_validated, f"Hypothesis not validated: {result.observations}"
        assert result.recovery_time.total_seconds() < 30, "Recovery took too long"
    
    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_cascading_failure_prevention(self, aura_stack):
        """Test that failures don't cascade"""
        service_urls, manager = aura_stack
        
        # Create chaos components
        injector = ChaosInjector(manager)
        orchestrator = ChaosOrchestrator(service_urls, injector)
        
        # Define experiment - crash one service
        experiment = ChaosExperiment(
            name="cascade_prevention_test",
            hypothesis=ChaosHypothesis(
                description="Single service failure doesn't cascade",
                steady_state_metrics={
                    "min_health_percentage": 60,  # Allow one service down
                    "required_services": ["moe", "byzantine"]  # These must stay up
                },
                expected_impact="Only target service affected",
                recovery_time_slo=timedelta(seconds=60),
                blast_radius=["memory"]  # Only memory should be affected
            ),
            target_services=["memory"],
            fault_specs=[{
                "target": "memory",
                "type": FaultType.SERVICE_CRASH.value,
                "duration": 30
            }],
            duration=timedelta(seconds=30)
        )
        
        # Run experiment
        result = await orchestrator.run_experiment(experiment)
        
        # Verify no cascade
        assert result.hypothesis_validated, f"Cascade detected: {result.actual_impact}"
        unexpected_impact = set(result.actual_impact) - {"memory"}
        assert len(unexpected_impact) == 0, f"Unexpected services impacted: {unexpected_impact}"


class TestObservability:
    """Test observability and monitoring"""
    
    @pytest.mark.asyncio
    async def test_distributed_tracing(self, aura_stack, http_client):
        """Test that distributed traces are properly propagated"""
        service_urls, _ = aura_stack
        
        # Make request with trace header
        trace_id = "test-trace-123"
        headers = {
            "X-Trace-ID": trace_id,
            "X-Request-ID": "test-request-456"
        }
        
        # Route through MoE to multiple services
        response = await http_client.post(
            f"{service_urls['moe']}/api/v1/route",
            json={
                "data": {
                    "type": "inference",
                    "data": [0.5] * 100
                },
                "proxy_request": True
            },
            headers=headers
        )
        
        assert response.status_code == 200
        
        # Verify trace propagation in response headers
        assert "X-Request-ID" in response.headers
    
    @pytest.mark.asyncio
    async def test_metrics_collection(self, aura_stack, http_client):
        """Test that metrics are properly collected"""
        service_urls, _ = aura_stack
        
        # Get metrics from each service
        for service_name, url in service_urls.items():
            if service_name in ["neuromorphic", "memory", "byzantine", "lnn", "moe"]:
                metrics_response = await http_client.get(f"{url}/metrics")
                assert metrics_response.status_code == 200
                
                # Verify Prometheus format
                metrics_text = metrics_response.text
                assert "# HELP" in metrics_text
                assert "# TYPE" in metrics_text


# Performance benchmark fixture
@pytest.fixture(scope="module")
def benchmark_results():
    """Collect benchmark results"""
    return {
        "services": {},
        "scenarios": {}
    }


if __name__ == "__main__":
    # Run tests with proper configuration
    pytest.main([
        __file__,
        "-v",
        "--asyncio-mode=auto",
        "--tb=short",
        "-m", "not slow",  # Skip slow tests by default
        "--maxfail=5"
    ])
    
    # For full test suite including chaos tests:
    # pytest.main([__file__, "-v", "--asyncio-mode=auto"])


import random  # Add this import at the top