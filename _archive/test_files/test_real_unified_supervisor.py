#!/usr/bin/env python3
"""
Comprehensive Test Suite for Real Unified AURA Supervisor
Tests real TDA + LNN integration with various workflow scenarios
"""

import asyncio
import json
import time
from typing import Dict, Any, List
import requests
from datetime import datetime

# Test scenarios
class SupervisorTestSuite:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.results = []
        
    def run_all_tests(self):
        """Run comprehensive test suite"""
        print("🧪 UNIFIED AURA SUPERVISOR - COMPREHENSIVE TEST SUITE")
        print("=" * 70)
        print(f"Testing against: {self.base_url}")
        print(f"Timestamp: {datetime.now().isoformat()}")
        print("=" * 70 + "\n")
        
        # Test 1: API Health
        self.test_api_health()
        
        # Test 2: Simple workflow
        self.test_simple_workflow()
        
        # Test 3: Complex workflow with errors
        self.test_complex_workflow_with_errors()
        
        # Test 4: High-risk scenario
        self.test_high_risk_scenario()
        
        # Test 5: Disconnected workflow
        self.test_disconnected_workflow()
        
        # Test 6: Performance test
        self.test_performance()
        
        # Test 7: Edge cases
        self.test_edge_cases()
        
        # Generate report
        self.generate_report()
    
    def test_api_health(self):
        """Test API health and components"""
        print("Test 1: API Health Check")
        print("-" * 40)
        
        try:
            # Test root endpoint
            resp = requests.get(f"{self.base_url}/")
            assert resp.status_code == 200
            data = resp.json()
            print(f"✅ Service: {data['service']}")
            print(f"✅ Status: {data['status']}")
            print(f"✅ Components: {json.dumps(data['components'], indent=2)}")
            
            # Test health endpoint
            resp = requests.get(f"{self.base_url}/health")
            assert resp.status_code == 200
            health = resp.json()
            print(f"✅ Health status: {health['status']}")
            
            self.results.append(("API Health", "PASSED", None))
        except Exception as e:
            print(f"❌ API Health Check failed: {e}")
            self.results.append(("API Health", "FAILED", str(e)))
        
        print()
    
    def test_simple_workflow(self):
        """Test simple linear workflow"""
        print("Test 2: Simple Linear Workflow")
        print("-" * 40)
        
        workflow = {
            "workflow_id": "simple_001",
            "current_step": "processing",
            "evidence_log": [
                {"type": "observation", "data": "Data collected"},
                {"type": "analysis", "data": "Analysis complete"}
            ],
            "error_log": [],
            "messages": ["Workflow progressing normally"],
            "steps": [
                {"id": "collect", "status": "complete"},
                {"id": "analyze", "status": "complete"},
                {"id": "process", "status": "active"}
            ],
            "agents": [
                {"id": "agent_1", "status": "active", "assigned_steps": [0, 1, 2]}
            ]
        }
        
        try:
            resp = requests.post(f"{self.base_url}/supervise", json=workflow)
            assert resp.status_code == 200
            decision = resp.json()
            
            print(f"✅ Decision: {decision['decision']}")
            print(f"✅ Confidence: {decision['confidence']:.3f}")
            print(f"✅ Risk Score: {decision['risk_score']:.3f}")
            print(f"✅ Complexity: {decision['topology_analysis']['complexity_score']:.3f}")
            print(f"✅ Processing Time: {decision['topology_analysis']['processing_time_ms']:.2f}ms")
            
            self.results.append(("Simple Workflow", "PASSED", decision))
        except Exception as e:
            print(f"❌ Simple workflow test failed: {e}")
            self.results.append(("Simple Workflow", "FAILED", str(e)))
        
        print()
    
    def test_complex_workflow_with_errors(self):
        """Test complex workflow with errors"""
        print("Test 3: Complex Workflow with Errors")
        print("-" * 40)
        
        workflow = {
            "workflow_id": "complex_001",
            "current_step": "error_recovery",
            "evidence_log": [
                {"type": "observation", "data": "Initial data"},
                {"type": "analysis", "data": "Anomaly detected"},
                {"type": "error", "data": "Processing failed"}
            ],
            "error_log": [
                {"timestamp": time.time(), "error": "Connection timeout"},
                {"timestamp": time.time(), "error": "Data validation failed"}
            ],
            "messages": ["Multiple errors encountered", "Attempting recovery"],
            "steps": [
                {"id": "init", "status": "complete"},
                {"id": "process_1", "status": "failed"},
                {"id": "process_2", "status": "failed"},
                {"id": "recover", "status": "active"},
                {"id": "finalize", "status": "pending"}
            ],
            "agents": [
                {"id": "agent_1", "status": "error", "assigned_steps": [0, 1]},
                {"id": "agent_2", "status": "error", "assigned_steps": [2]},
                {"id": "agent_3", "status": "active", "assigned_steps": [3, 4]}
            ]
        }
        
        try:
            resp = requests.post(f"{self.base_url}/supervise", json=workflow)
            assert resp.status_code == 200
            decision = resp.json()
            
            print(f"✅ Decision: {decision['decision']}")
            print(f"✅ Confidence: {decision['confidence']:.3f}")
            print(f"✅ Risk Score: {decision['risk_score']:.3f}")
            print(f"✅ Anomaly Score: {decision['topology_analysis']['anomaly_score']:.3f}")
            print(f"✅ Recommendations: {decision['recommendations']}")
            
            # Expect higher risk for error scenario
            assert decision['risk_score'] > 0.4, "Risk score should be elevated for error scenario"
            
            self.results.append(("Complex Workflow with Errors", "PASSED", decision))
        except Exception as e:
            print(f"❌ Complex workflow test failed: {e}")
            self.results.append(("Complex Workflow with Errors", "FAILED", str(e)))
        
        print()
    
    def test_high_risk_scenario(self):
        """Test high-risk workflow scenario"""
        print("Test 4: High-Risk Scenario")
        print("-" * 40)
        
        workflow = {
            "workflow_id": "highrisk_001",
            "current_step": "critical_operation",
            "evidence_log": [
                {"type": "warning", "data": "Resource exhaustion"},
                {"type": "error", "data": "Critical failure"},
                {"type": "anomaly", "data": "Unexpected behavior"}
            ],
            "error_log": [
                {"error": "Memory overflow"},
                {"error": "Process crash"},
                {"error": "Data corruption detected"},
                {"error": "Security breach attempt"}
            ],
            "messages": ["CRITICAL: System instability", "Multiple failures cascading"],
            "steps": [
                {"id": "step_1", "status": "failed"},
                {"id": "step_2", "status": "failed"},
                {"id": "step_3", "status": "error"}
            ],
            "agents": [
                {"id": "agent_1", "status": "crashed"},
                {"id": "agent_2", "status": "unresponsive"}
            ]
        }
        
        try:
            resp = requests.post(f"{self.base_url}/supervise", json=workflow)
            assert resp.status_code == 200
            decision = resp.json()
            
            print(f"✅ Decision: {decision['decision']}")
            print(f"✅ Risk Score: {decision['risk_score']:.3f}")
            print(f"✅ Anomaly Score: {decision['topology_analysis']['anomaly_score']:.3f}")
            print(f"✅ Recommendations: {decision['recommendations']}")
            
            # Expect very high risk
            assert decision['risk_score'] > 0.7, "Risk score should be very high"
            assert "High risk" in str(decision['recommendations']), "Should recommend caution"
            
            self.results.append(("High-Risk Scenario", "PASSED", decision))
        except Exception as e:
            print(f"❌ High-risk scenario test failed: {e}")
            self.results.append(("High-Risk Scenario", "FAILED", str(e)))
        
        print()
    
    def test_disconnected_workflow(self):
        """Test workflow with disconnected components"""
        print("Test 5: Disconnected Workflow")
        print("-" * 40)
        
        workflow = {
            "workflow_id": "disconnected_001",
            "current_step": "isolated_process",
            "evidence_log": [],
            "error_log": [],
            "messages": ["Components not communicating"],
            "steps": [
                {"id": "island_1", "status": "active"},
                {"id": "island_2", "status": "active"},
                {"id": "island_3", "status": "pending"}
            ],
            "agents": [
                {"id": "agent_1", "status": "active", "assigned_steps": [0]},
                {"id": "agent_2", "status": "active", "assigned_steps": [1]},
                {"id": "agent_3", "status": "idle", "assigned_steps": []}
            ]
        }
        
        try:
            resp = requests.post(f"{self.base_url}/supervise", json=workflow)
            assert resp.status_code == 200
            decision = resp.json()
            
            print(f"✅ Decision: {decision['decision']}")
            print(f"✅ Connected: {decision['topology_analysis']['graph_properties']['is_connected']}")
            print(f"✅ Components: {decision['topology_analysis']['graph_properties']['connected_components']}")
            print(f"✅ Insights: {decision['topology_analysis']['insights']}")
            
            # Should detect disconnection
            assert not decision['topology_analysis']['graph_properties']['is_connected']
            
            self.results.append(("Disconnected Workflow", "PASSED", decision))
        except Exception as e:
            print(f"❌ Disconnected workflow test failed: {e}")
            self.results.append(("Disconnected Workflow", "FAILED", str(e)))
        
        print()
    
    def test_performance(self):
        """Test supervisor performance with multiple requests"""
        print("Test 6: Performance Test")
        print("-" * 40)
        
        num_requests = 10
        response_times = []
        
        workflow = {
            "workflow_id": "perf_test",
            "current_step": "processing",
            "evidence_log": [{"type": "test", "data": f"item_{i}"} for i in range(5)],
            "error_log": [],
            "messages": ["Performance test"],
            "steps": [{"id": f"step_{i}", "status": "active"} for i in range(3)],
            "agents": [{"id": "agent_1", "status": "active", "assigned_steps": [0, 1, 2]}]
        }
        
        try:
            print(f"Running {num_requests} requests...")
            
            for i in range(num_requests):
                start = time.time()
                resp = requests.post(f"{self.base_url}/supervise", json=workflow)
                end = time.time()
                
                assert resp.status_code == 200
                response_times.append((end - start) * 1000)
                
                if i % 5 == 0:
                    print(f"  Completed {i+1}/{num_requests} requests")
            
            avg_time = sum(response_times) / len(response_times)
            min_time = min(response_times)
            max_time = max(response_times)
            
            print(f"✅ Average response time: {avg_time:.2f}ms")
            print(f"✅ Min response time: {min_time:.2f}ms")
            print(f"✅ Max response time: {max_time:.2f}ms")
            print(f"✅ Throughput: {1000/avg_time:.1f} requests/second")
            
            # Check metrics endpoint
            resp = requests.get(f"{self.base_url}/metrics")
            metrics = resp.json()
            print(f"✅ Total decisions made: {metrics['total_decisions']}")
            
            self.results.append(("Performance Test", "PASSED", {
                "avg_ms": avg_time,
                "throughput": 1000/avg_time
            }))
        except Exception as e:
            print(f"❌ Performance test failed: {e}")
            self.results.append(("Performance Test", "FAILED", str(e)))
        
        print()
    
    def test_edge_cases(self):
        """Test edge cases and boundary conditions"""
        print("Test 7: Edge Cases")
        print("-" * 40)
        
        # Empty workflow
        empty_workflow = {
            "workflow_id": "empty_001",
            "current_step": "",
            "evidence_log": [],
            "error_log": [],
            "messages": [],
            "steps": [],
            "agents": []
        }
        
        try:
            resp = requests.post(f"{self.base_url}/supervise", json=empty_workflow)
            assert resp.status_code == 200
            print("✅ Empty workflow handled")
            
            # Very large workflow
            large_workflow = {
                "workflow_id": "large_001",
                "current_step": "processing",
                "evidence_log": [{"type": "data", "value": i} for i in range(100)],
                "error_log": [],
                "messages": [f"Message {i}" for i in range(50)],
                "steps": [{"id": f"step_{i}", "status": "active"} for i in range(20)],
                "agents": [{"id": f"agent_{i}", "status": "active", "assigned_steps": list(range(i, min(i+5, 20)))} 
                          for i in range(10)]
            }
            
            resp = requests.post(f"{self.base_url}/supervise", json=large_workflow)
            assert resp.status_code == 200
            decision = resp.json()
            print(f"✅ Large workflow handled - {decision['topology_analysis']['graph_properties']['num_nodes']} nodes")
            
            self.results.append(("Edge Cases", "PASSED", None))
        except Exception as e:
            print(f"❌ Edge case test failed: {e}")
            self.results.append(("Edge Cases", "FAILED", str(e)))
        
        print()
    
    def generate_report(self):
        """Generate final test report"""
        print("\n" + "=" * 70)
        print("📊 TEST SUMMARY")
        print("=" * 70)
        
        passed = sum(1 for _, status, _ in self.results if status == "PASSED")
        failed = sum(1 for _, status, _ in self.results if status == "FAILED")
        total = len(self.results)
        
        print(f"Total Tests: {total}")
        print(f"Passed: {passed} ✅")
        print(f"Failed: {failed} ❌")
        print(f"Success Rate: {(passed/total)*100:.1f}%")
        
        print("\nDetailed Results:")
        print("-" * 40)
        for test_name, status, _ in self.results:
            emoji = "✅" if status == "PASSED" else "❌"
            print(f"{emoji} {test_name}: {status}")
        
        # Save detailed report
        report = {
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "total": total,
                "passed": passed,
                "failed": failed,
                "success_rate": passed/total
            },
            "results": [
                {
                    "test": name,
                    "status": status,
                    "data": data if isinstance(data, (dict, str)) else None
                }
                for name, status, data in self.results
            ]
        }
        
        with open("/workspace/supervisor_test_report.json", "w") as f:
            json.dump(report, f, indent=2)
        
        print(f"\n💾 Detailed report saved to: /workspace/supervisor_test_report.json")
        
        if passed == total:
            print("\n🎉 ALL TESTS PASSED! Unified AURA Supervisor is production ready!")
        else:
            print(f"\n⚠️ {failed} tests failed. Review the detailed report.")

if __name__ == "__main__":
    # Run comprehensive test suite
    tester = SupervisorTestSuite()
    tester.run_all_tests()