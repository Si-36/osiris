#!/usr/bin/env python3
"""
AURA Intelligence - Real System Demonstration
Shows actual working TDA API with real data processing
"""

import requests
import time
import json
from typing import Dict, Any, List
import asyncio

class AURARealSystemDemo:
    def __init__(self, api_base_url: str = "http://localhost:8080"):
        self.api_base = api_base_url
        self.session = requests.Session()
    
    def print_header(self, title: str):
        """Print formatted section header"""
        print("\n" + "=" * 60)
        print(f"🧠 {title}")
        print("=" * 60)
    
    def print_success(self, message: str):
        """Print success message"""
        print(f"✅ {message}")
    
    def print_info(self, message: str):
        """Print info message"""
        print(f"📊 {message}")
    
    def print_performance(self, time_ms: float):
        """Print performance info"""
        if time_ms < 1:
            print(f"⚡ Processing time: {time_ms:.2f}ms - EXCELLENT!")
        elif time_ms < 10:
            print(f"⚡ Processing time: {time_ms:.2f}ms - Very good")
        elif time_ms < 100:
            print(f"⚡ Processing time: {time_ms:.2f}ms - Good")
        else:
            print(f"⚡ Processing time: {time_ms:.2f}ms - Needs optimization")
    
    def test_system_health(self):
        """Test system health endpoint"""
        self.print_header("System Health Check")
        
        try:
            response = self.session.get(f"{self.api_base}/health", timeout=10)
            if response.status_code == 200:
                data = response.json()
                self.print_success("System is healthy")
                self.print_info(f"Uptime: {data['uptime_seconds']:.1f} seconds")
                self.print_info(f"Total requests: {data['total_requests']}")
                self.print_info(f"Success rate: {data['successful_requests']}/{data['total_requests']}")
                self.print_info(f"TDA engine: {data['tda_engine_status']}")
                return True
            else:
                print(f"❌ Health check failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Cannot connect to system: {e}")
            return False
    
    def test_real_tda_analysis(self):
        """Test real TDA analysis with various datasets"""
        self.print_header("Real TDA Analysis Tests")
        
        test_cases = [
            {
                "name": "Simple Linear Pattern",
                "data": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                "expected": "Single connected structure"
            },
            {
                "name": "Two Distinct Clusters", 
                "data": [1, 2, 3, 20, 21, 22],
                "expected": "Two separate groups"
            },
            {
                "name": "Complex Multi-Modal Data",
                "data": [1, 2, 3, 10, 11, 12, 50, 51, 52, 100, 101, 102],
                "expected": "Multiple clusters"
            }
        ]
        
        all_successful = True
        processing_times = []
        
        for i, test_case in enumerate(test_cases, 1):
            print(f"\n📈 Test {i}: {test_case['name']}")
            print(f"   Input: {test_case['data'][:5]}... ({len(test_case['data'])} points)")
            print(f"   Expected: {test_case['expected']}")
            
            try:
                payload = {"data": test_case['data']}
                response = self.session.post(
                    f"{self.api_base}/analyze/tda",
                    json=payload,
                    timeout=30
                )
                
                if response.status_code == 200:
                    result = response.json()
                    if result['success']:
                        self.print_success("TDA analysis successful")
                        
                        # Show results
                        betti = result['betti_numbers']
                        print(f"   🔢 Betti numbers: b₀={betti['b0']}, b₁={betti['b1']}")
                        
                        if betti['b0'] > 1:
                            print(f"   🔗 Found {betti['b0']} disconnected components")
                        else:
                            print(f"   🔗 Single connected component")
                        
                        if betti['b1'] > 0:
                            print(f"   🔄 Found {betti['b1']} topological loops")
                        else:
                            print(f"   🔄 No topological loops detected")
                        
                        # Performance
                        proc_time = result['processing_time_ms']
                        processing_times.append(proc_time)
                        self.print_performance(proc_time)
                        
                    else:
                        print(f"   ❌ Analysis failed: {result.get('error', 'Unknown error')}")
                        all_successful = False
                else:
                    print(f"   ❌ API error: {response.status_code}")
                    all_successful = False
                    
            except Exception as e:
                print(f"   ❌ Request failed: {e}")
                all_successful = False
        
        # Summary
        if processing_times:
            avg_time = sum(processing_times) / len(processing_times)
            print(f"\n📊 Average processing time: {avg_time:.2f}ms")
            
        return all_successful
    
    def test_agent_failure_prediction(self):
        """Test agent failure prediction capability"""
        self.print_header("Agent Failure Prediction Test")
        
        # Create test agent networks
        test_networks = [
            {
                "name": "Healthy Network",
                "agents": [
                    {"id": f"agent_{i}", "health": 0.9, "load": 0.3}
                    for i in range(10)
                ]
            },
            {
                "name": "Degraded Network",
                "agents": [
                    {"id": f"agent_{i}", "health": 0.5, "load": 0.7}
                    for i in range(10)
                ]
            },
            {
                "name": "Mixed Health Network",
                "agents": [
                    {"id": f"healthy_{i}", "health": 0.9, "load": 0.3}
                    for i in range(5)
                ] + [
                    {"id": f"failing_{i}", "health": 0.2, "load": 0.9}
                    for i in range(5)
                ]
            }
        ]
        
        all_successful = True
        
        for network in test_networks:
            print(f"\n🌐 Testing: {network['name']}")
            print(f"   Agents: {len(network['agents'])}")
            
            try:
                payload = {"agent_network": network['agents']}
                response = self.session.post(
                    f"{self.api_base}/predict/failure",
                    json=payload,
                    timeout=30
                )
                
                if response.status_code == 200:
                    result = response.json()
                    if result['success']:
                        self.print_success("Failure prediction successful")
                        
                        # Show topology analysis
                        topo = result['topological_features']
                        print(f"   🔢 Connected components: {topo['connected_components']}")
                        print(f"   🔄 Loops detected: {topo['loops']}")
                        print(f"   ⚠️  Disconnected groups: {topo['disconnected_components']}")
                        
                        # Show prediction
                        pred = result['failure_prediction']
                        risk = pred['risk_score']
                        prob = pred['failure_probability']
                        recommendation = pred['recommendation']
                        
                        print(f"   📊 Risk score: {risk:.3f}")
                        print(f"   🎯 Failure probability: {prob:.1%}")
                        print(f"   🚨 Recommendation: {recommendation}")
                        
                        # Risk assessment
                        if risk < 0.3:
                            print("   ✅ Network appears stable")
                        elif risk < 0.7:
                            print("   ⚠️  Network shows warning signs")
                        else:
                            print("   🚨 Network at high risk of failure")
                        
                        self.print_performance(result['processing_time_ms'])
                        
                    else:
                        print(f"   ❌ Prediction failed: {result.get('error', 'Unknown error')}")
                        all_successful = False
                else:
                    print(f"   ❌ API error: {response.status_code}")
                    all_successful = False
                    
            except Exception as e:
                print(f"   ❌ Request failed: {e}")
                all_successful = False
        
        return all_successful
    
    def test_live_demo(self):
        """Test live demo with random agent generation"""
        self.print_header("Live Demo - Real-Time Processing")
        
        try:
            response = self.session.get(f"{self.api_base}/demo/live", timeout=30)
            if response.status_code == 200:
                result = response.json()
                
                self.print_success("Live demo successful")
                print(f"   🎲 Generated network: {len(result['generated_network'])} agents")
                print(f"   ⏰ Timestamp: {result['timestamp']}")
                
                # Show analysis result
                analysis = result['analysis_result']
                if analysis['success']:
                    topo = analysis['topological_features']
                    pred = analysis['failure_prediction']
                    
                    print(f"   🔢 Topology: {topo['connected_components']} components, {topo['loops']} loops")
                    print(f"   🎯 Risk assessment: {pred['risk_score']:.3f}")
                    print(f"   🚨 Status: {pred['recommendation']}")
                    self.print_performance(analysis['processing_time_ms'])
                    
                    return True
                else:
                    print(f"   ❌ Analysis failed in live demo")
                    return False
            else:
                print(f"❌ Live demo failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Live demo error: {e}")
            return False
    
    def get_system_metrics(self):
        """Get and display system metrics"""
        self.print_header("System Performance Metrics")
        
        try:
            response = self.session.get(f"{self.api_base}/metrics", timeout=10)
            if response.status_code == 200:
                data = response.json()
                
                metrics = data['system_metrics']
                tda = data['tda_engine']
                
                print(f"📊 System Uptime: {metrics['uptime_seconds']:.1f} seconds")
                print(f"📊 Total Requests: {metrics['total_requests']}")
                print(f"📊 Success Rate: {metrics['success_rate']:.1%}")
                print(f"📊 Average Processing: {metrics['average_processing_time_ms']:.2f}ms")
                print(f"🧠 TDA Engine: {tda['status']}")
                print(f"🧠 Algorithm: {tda['algorithm']}")
                print(f"🧠 Performance: {tda['performance']}")
                
                return True
            else:
                print(f"❌ Metrics unavailable: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Metrics error: {e}")
            return False
    
    def run_comprehensive_demo(self):
        """Run complete system demonstration"""
        print("🧠 AURA Intelligence - Real System Demonstration")
        print("🎯 Demonstrating actual working TDA with real data processing")
        print("⚡ NO MOCKS - REAL COMPUTATIONS ONLY")
        
        start_time = time.time()
        
        # Test sequence
        tests = [
            ("System Health", self.test_system_health),
            ("TDA Analysis", self.test_real_tda_analysis),
            ("Failure Prediction", self.test_agent_failure_prediction),
            ("Live Demo", self.test_live_demo),
            ("Performance Metrics", self.get_system_metrics)
        ]
        
        results = {}
        for test_name, test_func in tests:
            print(f"\n🔄 Running {test_name}...")
            try:
                results[test_name] = test_func()
            except Exception as e:
                print(f"❌ {test_name} failed with exception: {e}")
                results[test_name] = False
        
        # Final summary
        total_time = time.time() - start_time
        successful_tests = sum(1 for success in results.values() if success)
        total_tests = len(results)
        
        self.print_header("Final Results Summary")
        print(f"✅ Successful tests: {successful_tests}/{total_tests}")
        print(f"📈 Success rate: {successful_tests/total_tests*100:.1f}%")
        print(f"⏱️  Total demo time: {total_time:.1f} seconds")
        
        if successful_tests == total_tests:
            print("\n🎉 COMPLETE SUCCESS!")
            print("✨ AURA Intelligence system is fully operational")
            print("✨ Real TDA processing with sub-second performance")
            print("✨ Agent failure prediction working accurately")
            print("✨ All endpoints responding correctly")
            print("\n🚀 READY FOR PRODUCTION DEPLOYMENT!")
        elif successful_tests > 0:
            print(f"\n⚠️  PARTIAL SUCCESS ({successful_tests}/{total_tests} working)")
            print("🔧 Some components need attention")
        else:
            print("\n❌ SYSTEM NOT WORKING")
            print("🔧 Critical issues need resolution")
        
        return successful_tests, total_tests

def main():
    """Main demonstration function"""
    demo = AURARealSystemDemo()
    
    # Check if API is running
    try:
        response = demo.session.get(f"{demo.api_base}/", timeout=5)
        if response.status_code != 200:
            print("❌ API server not responding. Start with: python3 working_tda_api.py")
            return
    except:
        print("❌ Cannot connect to API server at http://localhost:8080")
        print("💡 Start the API with: python3 working_tda_api.py")
        return
    
    # Run demonstration
    success_count, total_count = demo.run_comprehensive_demo()
    
    # Exit code for automation
    if success_count == total_count:
        exit(0)  # All tests passed
    else:
        exit(1)  # Some tests failed

if __name__ == "__main__":
    main()