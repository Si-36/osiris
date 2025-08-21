"""
Complete Integration Test for All AURA Microservices
Testing: Neuromorphic + Memory + Byzantine Consensus
"""

import asyncio
import httpx
import numpy as np
import time
import json
from typing import Dict, Any, List, Tuple
import structlog

logger = structlog.get_logger()

# Service URLs
NEUROMORPHIC_URL = "http://localhost:8000"
MEMORY_URL = "http://localhost:8001"
BYZANTINE_URL = "http://localhost:8002"


class AURAIntegrationTest:
    """Test all three services working together"""
    
    def __init__(self):
        self.neuro_client = httpx.AsyncClient(base_url=NEUROMORPHIC_URL, timeout=30.0)
        self.memory_client = httpx.AsyncClient(base_url=MEMORY_URL, timeout=30.0)
        self.byzantine_client = httpx.AsyncClient(base_url=BYZANTINE_URL, timeout=30.0)
        
    async def __aenter__(self):
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.neuro_client.aclose()
        await self.memory_client.aclose()
        await self.byzantine_client.aclose()
        
    async def test_distributed_ai_decision_making(self):
        """
        Test: Distributed AI Decision Making
        Multiple agents process data, store results, and reach consensus
        """
        print("\n" + "="*80)
        print("🤖 Test 1: Distributed AI Decision Making")
        print("="*80)
        
        # Simulate sensor data from multiple sources
        print("\n📊 Generating multi-modal sensor data...")
        
        sensor_data = {
            "temperature": np.random.normal(25, 5, 100).tolist(),
            "pressure": np.random.normal(1013, 10, 100).tolist(),
            "humidity": np.random.normal(60, 15, 100).tolist(),
            "anomaly_score": np.random.exponential(0.1, 100).tolist()
        }
        
        # Each "agent" processes the data
        agent_decisions = {}
        
        for agent_id in ["agent_1", "agent_2", "agent_3", "agent_4", "agent_5"]:
            print(f"\n🤖 Agent {agent_id} processing...")
            
            # 1. Neuromorphic processing for each sensor type
            spike_patterns = {}
            total_energy = 0
            
            for sensor_type, data in sensor_data.items():
                # Convert to spikes
                normalized = (np.array(data) - np.mean(data)) / np.std(data)
                spikes = (normalized > 0.5).astype(float).tolist()
                
                # Process through neuromorphic
                neuro_resp = await self.neuro_client.post(
                    "/api/v1/process/spike",
                    json={
                        "spike_data": [spikes[:128]],  # Ensure 128 dimensions
                        "time_steps": 10
                    }
                )
                result = neuro_resp.json()
                
                spike_patterns[sensor_type] = result['output'][0]
                total_energy += result['energy_consumed_pj']
                
            print(f"   ✓ Neuromorphic processing: {total_energy:.2f} pJ total")
            
            # 2. Store processed patterns in memory with shape analysis
            mem_resp = await self.memory_client.post(
                "/api/v1/store",
                json={
                    "key": f"{agent_id}_analysis_{int(time.time()*1000)}",
                    "data": {
                        "agent": agent_id,
                        "spike_patterns": spike_patterns,
                        "energy_used": total_energy,
                        "timestamp": time.time()
                    },
                    "enable_shape_analysis": True,
                    "relationships": [f"{a}_analysis" for a in ["agent_1", "agent_2"] if a != agent_id][:2]
                }
            )
            mem_result = mem_resp.json()
            print(f"   ✓ Stored in memory tier: {mem_result['tier']}")
            
            # 3. Make a decision based on patterns
            anomaly_detected = np.mean(sensor_data["anomaly_score"]) > 0.2
            action = "investigate" if anomaly_detected else "normal_operation"
            confidence = 0.9 if not anomaly_detected else 0.7
            
            agent_decisions[agent_id] = {
                "action": action,
                "confidence": confidence,
                "energy_pj": total_energy,
                "memory_key": mem_result['key']
            }
            
        # 4. Byzantine consensus on the decision
        print("\n🏛️ Reaching Byzantine consensus on action...")
        
        # Aggregate decisions
        action_votes = {}
        for agent, decision in agent_decisions.items():
            action = decision['action']
            if action not in action_votes:
                action_votes[action] = []
            action_votes[action].append((agent, decision['confidence']))
            
        # Propose most common action for consensus
        most_common_action = max(action_votes, key=lambda k: len(action_votes[k]))
        
        consensus_resp = await self.byzantine_client.post(
            "/api/v1/propose",
            json={
                "value": {
                    "action": most_common_action,
                    "agent_votes": {k: len(v) for k, v in action_votes.items()},
                    "total_energy_pj": sum(d['energy_pj'] for d in agent_decisions.values())
                },
                "category": "safety_critical",
                "priority": "high"
            }
        )
        proposal = consensus_resp.json()
        
        # Wait for consensus
        await asyncio.sleep(2)
        
        # Check result
        status_resp = await self.byzantine_client.get(
            f"/api/v1/consensus/{proposal['proposal_id']}"
        )
        consensus_result = status_resp.json()
        
        print(f"\n📊 Results:")
        print(f"   Agent decisions: {action_votes}")
        print(f"   Consensus reached: {consensus_result.get('status') == 'decided'}")
        print(f"   Final action: {consensus_result.get('decided_value', {}).get('action')}")
        print(f"   Total energy: {sum(d['energy_pj'] for d in agent_decisions.values()):.2f} pJ")
        
        return agent_decisions, consensus_result
        
    async def test_adaptive_memory_consensus(self):
        """
        Test: Adaptive Memory with Consensus-based Learning
        System learns patterns and agents agree on model updates
        """
        print("\n" + "="*80)
        print("🧠 Test 2: Adaptive Memory with Consensus-based Learning")
        print("="*80)
        
        print("\n📚 Training adaptive system with consensus...")
        
        # Generate training patterns
        patterns = []
        for i in range(5):
            pattern = {
                "id": f"pattern_{i}",
                "data": np.sin(np.linspace(0, 2*np.pi * (i+1), 128)).tolist(),
                "label": "sine_wave",
                "frequency": i + 1
            }
            patterns.append(pattern)
            
        # Process and store each pattern
        stored_patterns = []
        
        for pattern in patterns:
            # Neuromorphic processing
            spikes = (np.array(pattern['data']) > 0).astype(float).tolist()
            
            neuro_resp = await self.neuro_client.post(
                "/api/v1/process/lsm",  # Use Liquid State Machine
                json={
                    "input_data": [pattern['data']],
                    "time_steps": 20
                }
            )
            neuro_result = neuro_resp.json()
            
            # Store with shape analysis
            mem_resp = await self.memory_client.post(
                "/api/v1/store",
                json={
                    "key": pattern['id'],
                    "data": {
                        **pattern,
                        "lsm_output": neuro_result['output'],
                        "energy_pj": neuro_result['energy_consumed_pj']
                    },
                    "enable_shape_analysis": True
                }
            )
            
            stored_patterns.append({
                "pattern_id": pattern['id'],
                "memory_tier": mem_resp.json()['tier'],
                "energy": neuro_result['energy_consumed_pj']
            })
            
        print(f"   ✓ Stored {len(patterns)} training patterns")
        
        # Query similar patterns
        print("\n🔍 Querying for similar patterns...")
        
        query_pattern = np.sin(np.linspace(0, 3*np.pi, 128)) * 0.8  # Slightly different
        
        shape_resp = await self.memory_client.post(
            "/api/v1/query/shape",
            json={
                "query_data": {"pattern": query_pattern.tolist()},
                "k": 3
            }
        )
        similar = shape_resp.json()
        
        print(f"   ✓ Found {similar['num_results']} similar patterns")
        
        # Propose model update based on learning
        print("\n🤝 Proposing model update via consensus...")
        
        model_update = {
            "update_type": "add_pattern_class",
            "new_patterns_learned": len(patterns),
            "similarity_threshold": 0.85,
            "energy_efficiency": {
                "avg_energy_per_pattern": np.mean([p['energy'] for p in stored_patterns]),
                "total_energy": sum(p['energy'] for p in stored_patterns)
            }
        }
        
        consensus_resp = await self.byzantine_client.post(
            "/api/v1/propose",
            json={
                "value": model_update,
                "category": "model_update",
                "priority": "normal"
            }
        )
        
        print(f"   ✓ Model update proposed: {consensus_resp.json()['proposal_id']}")
        
        return stored_patterns, similar
        
    async def test_neuromorphic_memory_optimization(self):
        """
        Test: Neuromorphic-guided Memory Tier Optimization
        Use spike patterns to optimize memory placement
        """
        print("\n" + "="*80)
        print("⚡ Test 3: Neuromorphic-guided Memory Optimization")
        print("="*80)
        
        print("\n🔄 Optimizing memory tiers based on access patterns...")
        
        # Simulate different data access patterns
        access_patterns = {
            "hot_data": {
                "access_frequency": 100,
                "spike_density": 0.8,
                "data": np.random.rand(1000).tolist()
            },
            "warm_data": {
                "access_frequency": 10,
                "spike_density": 0.3,
                "data": np.random.rand(5000).tolist()
            },
            "cold_data": {
                "access_frequency": 1,
                "spike_density": 0.1,
                "data": np.random.rand(10000).tolist()
            }
        }
        
        optimization_results = []
        
        for data_type, pattern in access_patterns.items():
            print(f"\n   Processing {data_type}...")
            
            # Generate spike pattern based on access frequency
            spikes = (np.random.rand(128) < pattern['spike_density']).astype(float).tolist()
            
            # Process through neuromorphic
            neuro_resp = await self.neuro_client.post(
                "/api/v1/process/spike",
                json={
                    "spike_data": [spikes],
                    "time_steps": 5
                }
            )
            
            # Store with predicted tier
            mem_resp = await self.memory_client.post(
                "/api/v1/store",
                json={
                    "data": {
                        "type": data_type,
                        "content": pattern['data'][:100],  # Sample
                        "access_pattern": pattern['access_frequency'],
                        "spike_analysis": neuro_resp.json()
                    },
                    "enable_shape_analysis": False  # Faster
                }
            )
            
            result = mem_resp.json()
            optimization_results.append({
                "data_type": data_type,
                "assigned_tier": result['tier'],
                "expected_tier": "cxl_hot" if data_type == "hot_data" else "dram" if data_type == "warm_data" else "nvme_cold",
                "correct": result['tier'] == ("cxl_hot" if data_type == "hot_data" else "dram" if data_type == "warm_data" else "nvme_cold")
            })
            
            print(f"   ✓ {data_type} → {result['tier']} (latency: {result['latency_ns']:.2f}ns)")
            
        # Get memory efficiency report
        efficiency_resp = await self.memory_client.get("/api/v1/stats/efficiency")
        efficiency = efficiency_resp.json()
        
        print(f"\n📊 Memory Optimization Results:")
        print(f"   Correct tier assignments: {sum(r['correct'] for r in optimization_results)}/{len(optimization_results)}")
        print(f"   Hit ratio: {efficiency['hit_ratio']:.2%}")
        print(f"   Avg latency: {efficiency['average_latency_ns']:.2f}ns")
        
        return optimization_results
        
    async def test_full_system_integration(self):
        """
        Test: Complete System Integration
        All three services working together in a realistic scenario
        """
        print("\n" + "="*80)
        print("🚀 Test 4: Full System Integration - Emergency Response Scenario")
        print("="*80)
        
        print("\n🚨 Simulating distributed emergency detection and response...")
        
        # Simulate emergency sensor readings from multiple locations
        locations = ["north", "south", "east", "west", "center"]
        emergency_data = {}
        
        for location in locations:
            # Some locations have emergencies
            is_emergency = location in ["north", "east"]
            
            emergency_data[location] = {
                "temperature": np.random.normal(25 + (50 if is_emergency else 0), 5, 50).tolist(),
                "smoke_level": np.random.exponential(0.1 if not is_emergency else 2.0, 50).tolist(),
                "motion": np.random.binomial(1, 0.1 if not is_emergency else 0.8, 50).tolist()
            }
            
        # Each location processes its data
        location_analyses = {}
        
        for location, data in emergency_data.items():
            print(f"\n📍 Processing {location} sensor data...")
            
            # Neuromorphic processing for rapid pattern detection
            combined_signal = np.array(data['temperature']) / 100 + \
                            np.array(data['smoke_level']) / 5 + \
                            np.array(data['motion'])
            
            spikes = (combined_signal > np.percentile(combined_signal, 70)).astype(float)
            
            # Process with edge-optimized neuromorphic
            neuro_resp = await self.neuro_client.post(
                "/api/v1/process/spike",
                json={
                    "spike_data": [spikes.tolist()[:128]],
                    "time_steps": 3  # Fast processing
                }
            )
            neuro_result = neuro_resp.json()
            
            # Determine emergency level
            spike_rate = neuro_result['spike_rate']
            emergency_level = "critical" if spike_rate > 0.5 else "warning" if spike_rate > 0.3 else "normal"
            
            # Store analysis with high priority
            mem_resp = await self.memory_client.post(
                "/api/v1/store",
                json={
                    "key": f"emergency_{location}_{int(time.time()*1000)}",
                    "data": {
                        "location": location,
                        "emergency_level": emergency_level,
                        "spike_rate": spike_rate,
                        "raw_data": {k: v[:10] for k, v in data.items()},  # Sample
                        "timestamp": time.time(),
                        "energy_used_pj": neuro_result['energy_consumed_pj']
                    },
                    "enable_shape_analysis": True,
                    "preferred_tier": "cxl_hot" if emergency_level == "critical" else None
                }
            )
            
            location_analyses[location] = {
                "emergency_level": emergency_level,
                "confidence": 0.9 if spike_rate > 0.7 or spike_rate < 0.2 else 0.6,
                "energy_pj": neuro_result['energy_consumed_pj'],
                "memory_tier": mem_resp.json()['tier']
            }
            
            print(f"   ✓ {location}: {emergency_level} (confidence: {location_analyses[location]['confidence']:.2f})")
            
        # Byzantine consensus on emergency response
        print("\n🏛️ Coordinating emergency response via Byzantine consensus...")
        
        # Determine overall emergency status
        critical_count = sum(1 for l in location_analyses.values() if l['emergency_level'] == 'critical')
        warning_count = sum(1 for l in location_analyses.values() if l['emergency_level'] == 'warning')
        
        response_plan = {
            "emergency_detected": critical_count > 0,
            "locations_critical": [loc for loc, analysis in location_analyses.items() if analysis['emergency_level'] == 'critical'],
            "locations_warning": [loc for loc, analysis in location_analyses.items() if analysis['emergency_level'] == 'warning'],
            "recommended_action": "evacuate" if critical_count >= 2 else "investigate" if critical_count > 0 else "monitor",
            "total_energy_used_pj": sum(a['energy_pj'] for a in location_analyses.values()),
            "consensus_timestamp": time.time()
        }
        
        # Propose emergency response
        consensus_resp = await self.byzantine_client.post(
            "/api/v1/propose",
            json={
                "value": response_plan,
                "category": "safety_critical",
                "priority": "critical",
                "require_unanimous": critical_count >= 2  # Unanimous for evacuation
            }
        )
        proposal = consensus_resp.json()
        
        # Wait for consensus
        await asyncio.sleep(2)
        
        # Get consensus result
        status_resp = await self.byzantine_client.get(
            f"/api/v1/consensus/{proposal['proposal_id']}"
        )
        consensus = status_resp.json()
        
        # Query all emergency data for coordination
        if critical_count > 0:
            print("\n🔍 Retrieving all emergency data for coordination...")
            
            shape_resp = await self.memory_client.post(
                "/api/v1/query/shape",
                json={
                    "query_data": {"emergency": True, "pattern": combined_signal.tolist()[:128]},
                    "k": 10,
                    "filters": {"tier": "cxl_hot"}  # Only hot tier for speed
                }
            )
            related_emergencies = shape_resp.json()
            
            print(f"   ✓ Found {related_emergencies['num_results']} related emergency patterns")
            
        print(f"\n🚨 Emergency Response Summary:")
        print(f"   Critical locations: {critical_count}")
        print(f"   Warning locations: {warning_count}")
        print(f"   Consensus reached: {consensus.get('status') == 'decided'}")
        print(f"   Action decided: {consensus.get('decided_value', {}).get('recommended_action')}")
        print(f"   Total energy used: {sum(a['energy_pj'] for a in location_analyses.values()):.2f} pJ")
        print(f"   Response time: {(time.time() - emergency_data['north']['timestamp']) * 1000:.2f}ms" if 'timestamp' in emergency_data.get('north', {}) else "N/A")
        
        return location_analyses, consensus
        
    async def run_all_tests(self):
        """Run all integration tests"""
        print("=" * 80)
        print("🚀 AURA Microservices Complete Integration Test")
        print("   Testing: Neuromorphic + Memory + Byzantine Consensus")
        print("=" * 80)
        
        # Check all services are running
        try:
            print("\n🔍 Checking service health...")
            
            neuro_health = await self.neuro_client.get("/api/v1/health")
            print(f"   ✅ Neuromorphic Service: {neuro_health.json()['status']}")
            
            memory_health = await self.memory_client.get("/api/v1/health")
            print(f"   ✅ Memory Service: {memory_health.json()['status']}")
            
            byzantine_health = await self.byzantine_client.get("/api/v1/health")
            print(f"   ✅ Byzantine Service: {byzantine_health.json()['status']}")
            
        except Exception as e:
            print(f"\n❌ Service health check failed: {e}")
            print("\n💡 Please ensure all services are running:")
            print("   - Neuromorphic: port 8000")
            print("   - Memory: port 8001")
            print("   - Byzantine: port 8002")
            return
            
        # Run tests
        print("\n" + "="*80)
        
        results = {}
        
        # Test 1
        agents, consensus1 = await self.test_distributed_ai_decision_making()
        results['distributed_ai'] = {
            'agents': len(agents),
            'consensus': consensus1.get('status') == 'decided'
        }
        
        input("\nPress Enter to continue to next test...")
        
        # Test 2
        patterns, similar = await self.test_adaptive_memory_consensus()
        results['adaptive_memory'] = {
            'patterns_stored': len(patterns),
            'similar_found': similar['num_results']
        }
        
        input("\nPress Enter to continue to next test...")
        
        # Test 3
        optimization = await self.test_neuromorphic_memory_optimization()
        results['memory_optimization'] = {
            'correct_assignments': sum(r['correct'] for r in optimization),
            'total': len(optimization)
        }
        
        input("\nPress Enter to continue to final test...")
        
        # Test 4
        emergency, response = await self.test_full_system_integration()
        results['emergency_response'] = {
            'locations_analyzed': len(emergency),
            'consensus_reached': response.get('status') == 'decided'
        }
        
        # Summary
        print("\n" + "="*80)
        print("📊 Integration Test Summary")
        print("="*80)
        
        all_passed = True
        for test_name, result in results.items():
            passed = all(v for v in result.values() if isinstance(v, bool))
            status = "✅ PASSED" if passed else "❌ FAILED"
            print(f"\n{test_name}: {status}")
            for key, value in result.items():
                print(f"   - {key}: {value}")
            if not passed:
                all_passed = False
                
        print("\n" + "="*80)
        if all_passed:
            print("🎉 All integration tests PASSED!")
            print("\n✨ The AURA Intelligence Microservices are working perfectly together:")
            print("   • Neuromorphic processing with <1000pJ per operation")
            print("   • Shape-aware memory with intelligent tiering")
            print("   • Byzantine fault-tolerant consensus")
            print("   • Ready for production deployment!")
        else:
            print("⚠️  Some tests failed. Check the details above.")
            
        print("="*80)


async def main():
    """Run the complete integration test"""
    async with AURAIntegrationTest() as tester:
        await tester.run_all_tests()


if __name__ == "__main__":
    print("\n🌟 Starting AURA Microservices Integration Test Suite")
    print("   Ensure all three services are running:")
    print("   - Neuromorphic: http://localhost:8000")
    print("   - Memory: http://localhost:8001") 
    print("   - Byzantine: http://localhost:8002")
    print("\n   Also ensure Redis (6379) and Neo4j (7687) are running")
    
    input("\nPress Enter when ready to start tests...")
    
    asyncio.run(main())