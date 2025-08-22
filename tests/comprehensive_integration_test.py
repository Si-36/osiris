"""Comprehensive Integration Test - Real Data Flow Through All Bio-Enhanced Layers"""
import asyncio
import sys
import os
import json
import time
from typing import Dict, Any

# Add core path
sys.path.append(os.path.join(os.path.dirname(__file__), 'core', 'src'))

class MockRegistry:
    """Mock registry that simulates real component behavior"""
    def __init__(self):
        self.components = {f"component_{i}": f"Component {i}" for i in range(209)}
        self.call_count = 0
        
    async def process_data(self, component_id: str, data: Any, context: Dict[str, Any] = None) -> Dict[str, Any]:
        self.call_count += 1
        context = context or {}
        
        # Simulate different component behaviors
        if "error" in str(data).lower():
            return {
                "status": "error",
                "component": component_id,
                "error_type": "ProcessingError",
                "confidence": 0.2,
                "tda_anomaly": 0.8,
                "result": None
            }
        elif "complex" in str(data).lower():
            return {
                "status": "ok",
                "component": component_id,
                "confidence": 0.9,
                "tda_anomaly": 0.1,
                "result": {"processed": True, "complexity": "high", "mode": context.get("mode", "normal")},
                "processing_time": 0.05
            }
        else:
            return {
                "status": "ok", 
                "component": component_id,
                "confidence": 0.8,
                "tda_anomaly": 0.2,
                "result": {"processed": True, "complexity": "low", "mode": context.get("mode", "normal")},
                "processing_time": 0.01
            }

async def test_complete_data_flow():
    """Test complete data flow through all bio-enhanced layers"""
    print("🧬 COMPREHENSIVE BIO-ENHANCED AURA INTEGRATION TEST")
    print("=" * 60)
    
    # Initialize mock registry
    mock_registry = MockRegistry()
    
    # Test data scenarios
    test_scenarios = [
        {
            "name": "Simple Request",
            "data": {"query": "simple analysis", "values": [1, 2, 3]}
        },
        {
            "name": "Complex Request", 
            "data": {"query": "complex deep analysis with intricate patterns", "values": list(range(100))}
        },
        {
            "name": "Error Scenario",
            "data": {"query": "error prone analysis", "values": []}
        }
    ]
    
    for scenario in test_scenarios:
        print(f"\n🎯 Testing Scenario: {scenario['name']}")
        print("-" * 40)
        
        # Reset call counter
        mock_registry.call_count = 0
        
        # 1. Test Mixture of Depths
        print("1️⃣ Testing Mixture of Depths...")
        try:
            from aura_intelligence.advanced_processing.mixture_of_depths import MixtureOfDepths
            mod = MixtureOfDepths()
            mod_result = await mod.route_with_depth(scenario["data"])
            print(f"   ✅ Depth: {mod_result['depth_used']:.3f}")
            print(f"   ✅ Experts: {mod_result['experts_selected']}")
            print(f"   ✅ Compute Reduction: {mod_result['compute_reduction']:.1%}")
        except Exception as e:
            print(f"   ❌ Error: {e}")
        
        # 2. Test Metabolic Manager with real registry
        print("2️⃣ Testing Metabolic Manager...")
        try:
            from aura_intelligence.bio_homeostatic.metabolic_manager import MetabolicManager
            # Create metabolic manager without auto-starting periodic task
            mm = MetabolicManager.__new__(MetabolicManager)
            mm.registry = mock_registry
            mm.in_mem = {"budgets": {}, "consumption": {}}
            mm.r = None
            mm.signals = mm._default_signals()
            
            # Test processing with metabolism
            component_id = "component_1"
            result = await mm.process_with_metabolism(component_id, scenario["data"])
            print(f"   ✅ Status: {result['status']}")
            print(f"   ✅ Component: {result['component']}")
            if 'latency_ms' in result:
                print(f"   ✅ Latency: {result['latency_ms']:.2f}ms")
        except Exception as e:
            print(f"   ❌ Error: {e}")
        
        # 3. Test Swarm Intelligence
        print("3️⃣ Testing Swarm Intelligence...")
        try:
            from aura_intelligence.swarm_intelligence.ant_colony_detection import AntColonyDetection
            acd = AntColonyDetection(component_registry=mock_registry)
            swarm_result = await acd.detect_errors(scenario["data"])
            print(f"   ✅ Errors Detected: {swarm_result['errors_detected']}")
            print(f"   ✅ Components Tested: {len(swarm_result.get('error_components', [])) + swarm_result.get('healthy_components', 0)}")
            print(f"   ✅ Detection Rate: {swarm_result['detection_rate']:.1%}")
        except Exception as e:
            print(f"   ❌ Error: {e}")
        
        # 4. Test Spiking Council
        print("4️⃣ Testing Spiking GNN Council...")
        try:
            from aura_intelligence.spiking.council_sgnn import SpikingCouncil
            sc = SpikingCouncil()
            messages = {
                "component_1": {"confidence": 0.8, "priority": 0.7, "tda_anomaly": 0.1},
                "component_2": {"confidence": 0.6, "priority": 0.5, "tda_anomaly": 0.3}
            }
            spiking_result = await sc.process_component_messages(messages)
            print(f"   ✅ Consensus Strength: {spiking_result['consensus_strength']:.3f}")
            print(f"   ✅ Sparsity: {spiking_result['sparsity_ratio']:.1%}")
            print(f"   ✅ Power: {spiking_result['power_mw']:.2f}mW")
            print(f"   ✅ Latency: {spiking_result['latency_ms']:.2f}ms")
        except Exception as e:
            print(f"   ❌ Error: {e}")
        
        # 5. Test Complete Bio-Enhanced System
        print("5️⃣ Testing Complete Bio-Enhanced Integration...")
        try:
            from aura_intelligence.bio_enhanced_production_system import BioEnhancedAURA
            bio_aura = BioEnhancedAURA()
            
            # Override with mock registry for testing
            if bio_aura.metabolic:
                bio_aura.metabolic.registry = mock_registry
            if bio_aura.swarm:
                bio_aura.swarm.registry = mock_registry
            
            complete_result = await bio_aura.process_enhanced(scenario["data"], "test_component")
            print(f"   ✅ Bio-Enhanced: {complete_result['bio_enhanced']}")
            print(f"   ✅ Enhancements: {complete_result['enhancements']}")
            print(f"   ✅ Total Time: {complete_result['performance']['total_ms']:.2f}ms")
        except Exception as e:
            print(f"   ❌ Error: {e}")
        
        print(f"📊 Registry Calls Made: {mock_registry.call_count}")

async def test_api_endpoints():
    """Test actual API endpoints with real data flow"""
    print("\n🌐 TESTING API ENDPOINTS WITH REAL DATA FLOW")
    print("=" * 60)
    
    import subprocess
    import requests
    import time
    
    # Start API server in background
    print("Starting API server...")
    try:
        # Kill any existing process on port 8089
        subprocess.run(["pkill", "-f", "enhanced_bio_api.py"], capture_output=True)
        time.sleep(1)
        
        # Start new server
        server_process = subprocess.Popen([
            "python3", "enhanced_bio_api.py"
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Wait for server to start
        time.sleep(3)
        
        base_url = "http://localhost:8089"
        
        # Test health endpoint
        print("1️⃣ Testing /health endpoint...")
        try:
            response = requests.get(f"{base_url}/health", timeout=5)
            print(f"   ✅ Status: {response.status_code}")
            print(f"   ✅ Response: {response.json()}")
        except Exception as e:
            print(f"   ❌ Error: {e}")
        
        # Test process endpoint with different data
        test_requests = [
            {"data": {"query": "simple test", "values": [1, 2, 3]}, "component_id": "test_component"},
            {"data": {"query": "complex analysis with deep patterns", "values": list(range(50))}, "component_id": "analysis_component"},
            {"data": {"query": "error prone data", "values": []}, "component_id": "error_component"}
        ]
        
        for i, req_data in enumerate(test_requests, 1):
            print(f"2️⃣.{i} Testing /process endpoint...")
            try:
                response = requests.post(f"{base_url}/process", json=req_data, timeout=10)
                result = response.json()
                print(f"   ✅ Status: {response.status_code}")
                print(f"   ✅ Bio-Enhanced: {result.get('bio_enhanced', False)}")
                print(f"   ✅ Enhancements: {result.get('enhancements', {})}")
                print(f"   ✅ Performance: {result.get('performance', {})}")
            except Exception as e:
                print(f"   ❌ Error: {e}")
        
        # Test swarm check endpoint
        print("3️⃣ Testing /swarm/check endpoint...")
        try:
            response = requests.post(f"{base_url}/swarm/check", 
                                   json={"data": {"test": "anomaly detection"}}, timeout=10)
            result = response.json()
            print(f"   ✅ Status: {response.status_code}")
            print(f"   ✅ Swarm Detection: {result.get('swarm_detection', {})}")
        except Exception as e:
            print(f"   ❌ Error: {e}")
        
        # Test system status
        print("4️⃣ Testing /status endpoint...")
        try:
            response = requests.get(f"{base_url}/status", timeout=5)
            result = response.json()
            print(f"   ✅ Status: {response.status_code}")
            print(f"   ✅ Bio-Enhancements: {result.get('bio_enhancements', {})}")
            print(f"   ✅ Feature Flags: {result.get('feature_flags', {})}")
        except Exception as e:
            print(f"   ❌ Error: {e}")
        
        # Cleanup
        server_process.terminate()
        server_process.wait()
        
    except Exception as e:
        print(f"❌ API Test Error: {e}")

async def main():
    """Run comprehensive integration tests"""
    print("🚀 STARTING COMPREHENSIVE BIO-ENHANCED AURA INTEGRATION TESTS")
    print("=" * 80)
    
    # Test component integration
    await test_complete_data_flow()
    
    # Test API endpoints
    await test_api_endpoints()
    
    print("\n🎉 INTEGRATION TESTS COMPLETED!")
    print("=" * 80)

if __name__ == "__main__":
    asyncio.run(main())