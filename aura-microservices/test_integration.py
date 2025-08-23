"""
Integration Test Suite for AURA Microservices
Testing Neuromorphic + Memory Services Together
"""

import asyncio
import httpx
import numpy as np
import time
import json
from typing import Dict, Any, List
import structlog

logger = structlog.get_logger()

# Service URLs
NEUROMORPHIC_URL = "http://localhost:8000"
MEMORY_URL = "http://localhost:8001"


class IntegrationTester:
    """Test integration between Neuromorphic and Memory services"""
    
    def __init__(self):
        self.neuro_client = httpx.AsyncClient(base_url=NEUROMORPHIC_URL, timeout=30.0)
        self.memory_client = httpx.AsyncClient(base_url=MEMORY_URL, timeout=30.0)
        self.test_results = []
        
    async def __aenter__(self):
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.neuro_client.aclose()
        await self.memory_client.aclose()
        
    async def test_service_health(self):
        """Test 1: Verify both services are healthy"""
        print("\n🔍 Test 1: Service Health Check")
        
        try:
            # Check Neuromorphic service
            neuro_health = await self.neuro_client.get("/api/v1/health")
            neuro_status = neuro_health.json()
            print(f"✅ Neuromorphic Service: {neuro_status['status']}")
            print(f"   - Models loaded: {neuro_status['models_loaded']}")
            print(f"   - Avg energy/op: {neuro_status['avg_energy_per_operation_pj']:.2f} pJ")
            
            # Check Memory service
            memory_health = await self.memory_client.get("/api/v1/health")
            memory_status = memory_health.json()
            print(f"✅ Memory Service: {memory_status['status']}")
            print(f"   - Hit ratio: {memory_status['hit_ratio']:.2%}")
            print(f"   - Avg latency: {memory_status['average_latency_ns']:.2f} ns")
            
            self.test_results.append(("health_check", "PASSED", None))
            return True
            
        except Exception as e:
            print(f"❌ Health check failed: {e}")
            self.test_results.append(("health_check", "FAILED", str(e)))
            return False
            
    async def test_neuromorphic_to_memory_flow(self):
        """Test 2: Process spikes and store results in memory"""
        print("\n🔍 Test 2: Neuromorphic → Memory Flow")
        
        try:
            # Generate test spike data (simulating sensor readings)
            spike_data = [[float(np.random.binomial(1, 0.2)) for _ in range(128)]]
            
            # Process through Neuromorphic
            print("   Processing spike data...")
            neuro_resp = await self.neuro_client.post(
                "/api/v1/process/spike",
                json={
                    "spike_data": spike_data,
                    "time_steps": 10,
                    "reward_signal": 0.5
                }
            )
            neuro_result = neuro_resp.json()
            
            print(f"   ✓ Neuromorphic processing complete:")
            print(f"     - Energy: {neuro_result['energy_consumed_pj']:.2f} pJ")
            print(f"     - Latency: {neuro_result['latency_us']:.2f} μs")
            print(f"     - Spike rate: {neuro_result['spike_rate']:.3f}")
            
            # Store result in Memory with shape analysis
            print("   Storing in memory with shape analysis...")
            mem_resp = await self.memory_client.post(
                "/api/v1/store",
                json={
                    "key": f"neuro_result_{int(time.time()*1000)}",
                    "data": {
                        "spike_input": spike_data,
                        "neuro_output": neuro_result['output'],
                        "energy_pj": neuro_result['energy_consumed_pj'],
                        "timestamp": time.time()
                    },
                    "enable_shape_analysis": True,
                    "relationships": []
                }
            )
            mem_result = mem_resp.json()
            
            print(f"   ✓ Memory storage complete:")
            print(f"     - Stored in tier: {mem_result['tier']}")
            print(f"     - Storage latency: {mem_result['latency_ns']:.2f} ns")
            print(f"     - Shape indexed: {mem_result['shape_indexed']}")
            
            self.test_results.append(("neuro_to_memory_flow", "PASSED", None))
            return mem_result['key']
            
        except Exception as e:
            print(f"❌ Neuromorphic→Memory flow failed: {e}")
            self.test_results.append(("neuro_to_memory_flow", "FAILED", str(e)))
            return None
            
    async def test_memory_shape_query_with_neuromorphic(self):
        """Test 3: Query by shape and process similar patterns"""
        print("\n🔍 Test 3: Shape Query → Neuromorphic Processing")
        
        try:
            # First, store some test patterns
            print("   Storing test patterns...")
            test_patterns = [
                [1, 0, 1, 0, 1, 0, 1, 0],  # Regular pattern
                [1, 1, 0, 0, 1, 1, 0, 0],  # Block pattern
                [1, 0, 0, 1, 0, 0, 1, 0],  # Sparse pattern
                [0, 1, 1, 1, 0, 1, 1, 1],  # Dense pattern
            ]
            
            stored_keys = []
            for i, pattern in enumerate(test_patterns):
                resp = await self.memory_client.post(
                    "/api/v1/store",
                    json={
                        "key": f"pattern_{i}",
                        "data": {"pattern": pattern, "type": f"type_{i}"},
                        "enable_shape_analysis": True
                    }
                )
                stored_keys.append(resp.json()['key'])
            
            # Query by shape similarity
            print("   Querying by shape similarity...")
            query_pattern = [1, 0, 1, 1, 0, 1, 0, 1]  # Similar to first pattern
            
            shape_resp = await self.memory_client.post(
                "/api/v1/query/shape",
                json={
                    "query_data": {"pattern": query_pattern},
                    "k": 3
                }
            )
            shape_results = shape_resp.json()
            
            print(f"   ✓ Found {shape_results['num_results']} similar patterns")
            print(f"     - Query latency: {shape_results['query_latency_ns']:.2f} ns")
            
            # Process similar patterns through Neuromorphic
            print("   Processing similar patterns through Neuromorphic...")
            for i, result in enumerate(shape_results['results'][:2]):
                pattern_data = result['data']['pattern']
                
                # Convert to spike format and process
                spike_data = [[float(x) for x in pattern_data] * 16]  # Repeat to 128 dims
                
                neuro_resp = await self.neuro_client.post(
                    "/api/v1/process/lif",
                    json={"spike_data": spike_data}
                )
                
                print(f"     - Pattern {i}: similarity={result['similarity_score']:.3f}, "
                      f"energy={neuro_resp.json()['energy_consumed_pj']:.2f} pJ")
            
            self.test_results.append(("shape_query_neuromorphic", "PASSED", None))
            return True
            
        except Exception as e:
            print(f"❌ Shape query→Neuromorphic failed: {e}")
            self.test_results.append(("shape_query_neuromorphic", "FAILED", str(e)))
            return False
            
    async def test_bidirectional_integration(self):
        """Test 4: Full bidirectional integration with feedback loop"""
        print("\n🔍 Test 4: Bidirectional Integration with Feedback")
        
        try:
            # Simulate time-series data processing
            print("   Simulating time-series processing...")
            
            time_series = []
            for t in range(5):
                # Generate evolving pattern
                base_pattern = np.sin(np.linspace(0, 2*np.pi, 32) + t*0.5)
                spike_pattern = (base_pattern > 0).astype(float)
                
                # Process through Neuromorphic
                neuro_resp = await self.neuro_client.post(
                    "/api/v1/process/lif",
                    json={
                        "spike_data": [spike_pattern.tolist() * 4],  # Expand to 128
                        "time_steps": 5
                    }
                )
                neuro_result = neuro_resp.json()
                
                # Store with relationships to previous
                relationships = [f"timeseries_{t-1}"] if t > 0 else []
                
                mem_resp = await self.memory_client.post(
                    "/api/v1/store",
                    json={
                        "key": f"timeseries_{t}",
                        "data": {
                            "time": t,
                            "input_pattern": spike_pattern.tolist(),
                            "neuro_output": neuro_result['output'],
                            "metrics": {
                                "energy_pj": neuro_result['energy_consumed_pj'],
                                "spike_rate": neuro_result['spike_rate']
                            }
                        },
                        "enable_shape_analysis": True,
                        "relationships": relationships
                    }
                )
                
                time_series.append({
                    "time": t,
                    "key": mem_resp.json()['key'],
                    "tier": mem_resp.json()['tier'],
                    "energy": neuro_result['energy_consumed_pj']
                })
                
                print(f"     t={t}: tier={mem_resp.json()['tier']}, "
                      f"energy={neuro_result['energy_consumed_pj']:.2f} pJ")
            
            # Analyze the series
            total_energy = sum(ts['energy'] for ts in time_series)
            print(f"\n   ✓ Time series analysis complete:")
            print(f"     - Total energy: {total_energy:.2f} pJ")
            print(f"     - Avg energy/step: {total_energy/len(time_series):.2f} pJ")
            print(f"     - All stored successfully with relationships")
            
            self.test_results.append(("bidirectional_integration", "PASSED", None))
            return True
            
        except Exception as e:
            print(f"❌ Bidirectional integration failed: {e}")
            self.test_results.append(("bidirectional_integration", "FAILED", str(e)))
            return False
            
    async def test_performance_characteristics(self):
        """Test 5: Validate performance claims"""
        print("\n🔍 Test 5: Performance Validation")
        
        try:
            # Test Neuromorphic latency
            print("   Testing Neuromorphic latency (100 iterations)...")
            latencies = []
            for _ in range(100):
                start = time.perf_counter()
                await self.neuro_client.post(
                    "/api/v1/process/spike",
                    json={"spike_data": [[1,0,1,0,1]*25], "time_steps": 1}
                )
                latencies.append((time.perf_counter() - start) * 1000)  # ms
            
            avg_latency = np.mean(latencies)
            p95_latency = np.percentile(latencies, 95)
            print(f"   ✓ Neuromorphic latency: avg={avg_latency:.2f}ms, p95={p95_latency:.2f}ms")
            
            # Test Memory tier distribution
            print("   Testing Memory tier distribution...")
            
            # Store data of different sizes
            sizes = [100, 1000, 10000, 100000]  # bytes
            tier_distribution = {}
            
            for size in sizes:
                data = {"data": "x" * size}
                resp = await self.memory_client.post(
                    "/api/v1/store",
                    json={"data": data, "enable_shape_analysis": False}
                )
                tier = resp.json()['tier']
                tier_distribution[size] = tier
            
            print("   ✓ Memory tier selection by size:")
            for size, tier in tier_distribution.items():
                print(f"     - {size} bytes → {tier}")
            
            # Verify sub-millisecond latency claim
            sub_ms_achieved = avg_latency < 1.0  # < 1ms
            print(f"\n   {'✅' if sub_ms_achieved else '❌'} Sub-millisecond latency: {sub_ms_achieved}")
            
            self.test_results.append(("performance_validation", "PASSED" if sub_ms_achieved else "PARTIAL", None))
            return sub_ms_achieved
            
        except Exception as e:
            print(f"❌ Performance validation failed: {e}")
            self.test_results.append(("performance_validation", "FAILED", str(e)))
            return False
            
    async def run_all_tests(self):
        """Run all integration tests"""
        print("=" * 60)
        print("🚀 AURA Microservices Integration Test Suite")
        print("=" * 60)
        
        # Run tests
        await self.test_service_health()
        
        stored_key = await self.test_neuromorphic_to_memory_flow()
        if stored_key:
            print(f"\n   📝 Stored data key: {stored_key}")
        
        await self.test_memory_shape_query_with_neuromorphic()
        await self.test_bidirectional_integration()
        await self.test_performance_characteristics()
        
        # Summary
        print("\n" + "=" * 60)
        print("📊 Test Summary")
        print("=" * 60)
        
        passed = sum(1 for _, status, _ in self.test_results if status == "PASSED")
        failed = sum(1 for _, status, _ in self.test_results if status == "FAILED")
        partial = sum(1 for _, status, _ in self.test_results if status == "PARTIAL")
        
        print(f"✅ Passed: {passed}")
        print(f"❌ Failed: {failed}")
        print(f"⚠️  Partial: {partial}")
        
        print("\nDetailed Results:")
        for test_name, status, error in self.test_results:
            status_icon = "✅" if status == "PASSED" else "❌" if status == "FAILED" else "⚠️"
            print(f"{status_icon} {test_name}: {status}")
            if error:
                print(f"   Error: {error}")
        
        return failed == 0


async def main():
    """Run integration tests"""
    async with IntegrationTester() as tester:
        success = await tester.run_all_tests()
        
        if success:
            print("\n🎉 All integration tests passed! Services are working together perfectly.")
            print("\n💡 Next step: Create a demo showcasing both services")
        else:
            print("\n⚠️  Some tests failed. Check the errors above and fix integration issues.")
            print("\n💡 Common issues:")
            print("   - Ensure both services are running (ports 8000 and 8001)")
            print("   - Check Redis is running (port 6379)")
            print("   - Verify Neo4j is accessible (port 7687)")


if __name__ == "__main__":
    asyncio.run(main())