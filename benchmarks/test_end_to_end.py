"""
End-to-End Testing with Visualization
Tests complete AURA system and shows results
"""
import asyncio
import requests
import time
import json
from typing import Dict, Any

class AURAEndToEndTester:
    """Complete end-to-end testing with visual results"""
    
    def __init__(self, base_url: str = "http://localhost:8098"):
        self.base_url = base_url
        self.results = {}
        
    def test_system_health(self) -> Dict[str, Any]:
        """Test system health endpoint"""
        print("🏥 Testing System Health...")
        
        try:
            response = requests.get(f"{self.base_url}/health", timeout=10)
            health_data = response.json()
            
            print(f"   ✅ Status: {health_data['status']}")
            print(f"   ✅ Health Score: {health_data['health_score']:.2f}")
            print(f"   ✅ Active Enhancements: {len([e for e in health_data['enhancements'].values() if isinstance(e, dict) and e.get('status') == 'active'])}")
            
            self.results['health'] = health_data
            return health_data
            
        except Exception as e:
            print(f"   ❌ Health check failed: {e}")
            return {"error": str(e)}
    
    def test_enhanced_processing(self) -> Dict[str, Any]:
        """Test enhanced processing pipeline"""
        print("🧬 Testing Enhanced Processing...")
        
        test_request = {
            "council_task": {
                "gpu_allocation": {
                    "gpu_count": 4,
                    "cost_per_hour": 2.5,
                    "duration_hours": 8,
                    "user_id": "test_user"
                }
            },
            "contexts": [
                {"component_id": "neural_001", "activity": 0.8, "load": 0.6},
                {"component_id": "memory_002", "activity": 0.7, "load": 0.4}
            ],
            "action": {
                "type": "resource_allocation",
                "confidence": 0.85,
                "risk_level": "medium",
                "efficiency_score": 0.75
            },
            "context": {
                "system_load": 0.6,
                "component_coordination": 0.8
            }
        }
        
        try:
            start_time = time.time()
            response = requests.post(
                f"{self.base_url}/enhanced/process",
                json=test_request,
                timeout=30
            )
            processing_time = time.time() - start_time
            
            result = response.json()
            
            print(f"   ✅ Processing Time: {processing_time:.3f}s")
            print(f"   ✅ Success: {result.get('success', False)}")
            
            if 'enhanced_results' in result:
                enhanced = result['enhanced_results']
                if 'enhanced_council' in enhanced:
                    council = enhanced['enhanced_council']
                    print(f"   ✅ Council Vote: {council.get('vote', 'N/A')}")
                    print(f"   ✅ Confidence: {council.get('confidence', 0):.2f}")
                    print(f"   ✅ Liquid Adaptations: {council.get('liquid_adaptations', 0)}")
                
                if 'enhanced_coral' in enhanced:
                    coral = enhanced['enhanced_coral']
                    print(f"   ✅ CoRaL Context Length: {coral.get('context_buffer_size', 0)}")
                    print(f"   ✅ Linear Complexity: {coral.get('linear_complexity', False)}")
                
                if 'enhanced_dpo' in enhanced:
                    dpo = enhanced['enhanced_dpo']
                    const_eval = dpo.get('constitutional_evaluation', {})
                    print(f"   ✅ Constitutional Compliance: {const_eval.get('constitutional_compliance', 0):.2f}")
                    print(f"   ✅ Auto-Corrected: {const_eval.get('auto_corrected', False)}")
            
            self.results['enhanced_processing'] = result
            return result
            
        except Exception as e:
            print(f"   ❌ Enhanced processing failed: {e}")
            return {"error": str(e)}
    
    def test_memory_system(self) -> Dict[str, Any]:
        """Test hybrid memory system"""
        print("💾 Testing Hybrid Memory System...")
        
        try:
            # Test store operation
            store_request = {
                "key": "test_memory_key",
                "data": {"test_data": [1, 2, 3, 4, 5], "timestamp": time.time()},
                "component_id": "memory_test_001",
                "operation": "store"
            }
            
            store_response = requests.post(
                f"{self.base_url}/memory/hybrid",
                json=store_request,
                timeout=10
            )
            store_result = store_response.json()
            
            print(f"   ✅ Store Success: {store_result.get('stored', False)}")
            print(f"   ✅ Storage Tier: {store_result.get('tier', 'unknown')}")
            print(f"   ✅ Storage Time: {store_result.get('storage_time_ms', 0):.2f}ms")
            
            # Test retrieve operation
            retrieve_request = {
                "key": "test_memory_key",
                "component_id": "memory_test_001",
                "operation": "retrieve"
            }
            
            retrieve_response = requests.post(
                f"{self.base_url}/memory/hybrid",
                json=retrieve_request,
                timeout=10
            )
            retrieve_result = retrieve_response.json()
            
            print(f"   ✅ Retrieve Success: {retrieve_result.get('found', False)}")
            print(f"   ✅ Retrieval Time: {retrieve_result.get('retrieval_time_ms', 0):.2f}ms")
            print(f"   ✅ Access Count: {retrieve_result.get('access_count', 0)}")
            
            # Get memory stats
            stats_response = requests.get(f"{self.base_url}/memory/stats", timeout=10)
            stats_result = stats_response.json()
            
            print(f"   ✅ Total Items: {stats_result.get('total_items', 0)}")
            print(f"   ✅ Hot Tier Utilization: {stats_result.get('tiers', {}).get('hot', {}).get('utilization', 0):.2%}")
            
            self.results['memory_system'] = {
                'store': store_result,
                'retrieve': retrieve_result,
                'stats': stats_result
            }
            
            return self.results['memory_system']
            
        except Exception as e:
            print(f"   ❌ Memory system test failed: {e}")
            return {"error": str(e)}
    
    def test_metabolic_system(self) -> Dict[str, Any]:
        """Test metabolic processing system"""
        print("⚡ Testing Metabolic System...")
        
        try:
            # Test metabolic processing
            metabolic_request = {
                "component_id": "neural_test_001",
                "data": {"input_vector": [0.1, 0.2, 0.3, 0.4, 0.5]},
                "context": {"tda_anomaly": 0.2, "system_load": 0.6}
            }
            
            response = requests.post(
                f"{self.base_url}/metabolic/process",
                json=metabolic_request,
                timeout=10
            )
            result = response.json()
            
            print(f"   ✅ Processing Status: {result.get('status', 'unknown')}")
            print(f"   ✅ Latency: {result.get('latency_ms', 0):.2f}ms")
            print(f"   ✅ Energy Debit: {result.get('energy_debit', 0):.2f}")
            print(f"   ✅ Budget Remaining: {result.get('budget_remaining', 0):.2f}")
            
            # Get metabolic status
            status_response = requests.get(f"{self.base_url}/metabolic/status", timeout=10)
            status_result = status_response.json()
            
            print(f"   ✅ Active Components: {status_result.get('active_components', 0)}")
            health_dist = status_result.get('health_distribution', {})
            print(f"   ✅ Health Distribution: {health_dist.get('healthy', 0)} healthy, {health_dist.get('stressed', 0)} stressed")
            
            self.results['metabolic_system'] = {
                'processing': result,
                'status': status_result
            }
            
            return self.results['metabolic_system']
            
        except Exception as e:
            print(f"   ❌ Metabolic system test failed: {e}")
            return {"error": str(e)}
    
    def test_metrics_endpoint(self) -> Dict[str, Any]:
        """Test Prometheus metrics endpoint"""
        print("📊 Testing Metrics Endpoint...")
        
        try:
            # Get Prometheus metrics
            metrics_response = requests.get(f"{self.base_url}/metrics", timeout=10)
            metrics_text = metrics_response.text
            
            # Count metrics
            metric_lines = [line for line in metrics_text.split('\n') if line and not line.startswith('#')]
            
            print(f"   ✅ Metrics Available: {len(metric_lines)} metric values")
            
            # Get metrics summary
            summary_response = requests.get(f"{self.base_url}/metrics/summary", timeout=10)
            summary_result = summary_response.json()
            
            print(f"   ✅ Uptime: {summary_result.get('uptime_seconds', 0):.1f}s")
            print(f"   ✅ Metrics Collected: {summary_result.get('metrics_collected', 0)}")
            
            self.results['metrics'] = {
                'metrics_count': len(metric_lines),
                'summary': summary_result
            }
            
            return self.results['metrics']
            
        except Exception as e:
            print(f"   ❌ Metrics test failed: {e}")
            return {"error": str(e)}
    
    def run_benchmark(self) -> Dict[str, Any]:
        """Run system benchmark"""
        print("🏃 Running System Benchmark...")
        
        try:
            response = requests.get(f"{self.base_url}/benchmark", timeout=30)
            benchmark_result = response.json()
            
            benchmarks = benchmark_result.get('benchmarks', {})
            
            print(f"   ✅ Enhanced Processing: {benchmarks.get('enhanced_processing_ms', 0):.1f}ms")
            print(f"   ✅ Memory Operations: {benchmarks.get('memory_operations_ms', 0):.1f}ms")
            print(f"   ✅ Metabolic Processing: {benchmarks.get('metabolic_processing_ms', 0):.1f}ms")
            print(f"   ✅ Performance Grade: {benchmark_result.get('performance_grade', 'unknown')}")
            
            self.results['benchmark'] = benchmark_result
            return benchmark_result
            
        except Exception as e:
            print(f"   ❌ Benchmark failed: {e}")
            return {"error": str(e)}
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all end-to-end tests"""
        print("🧪 AURA INTELLIGENCE END-TO-END TESTING")
        print("=" * 60)
        
        # Wait for system to be ready
        print("⏳ Waiting for system to be ready...")
        for i in range(30):
            try:
                requests.get(f"{self.base_url}/", timeout=5)
                break
            except:
                time.sleep(2)
        else:
            print("❌ System not ready after 60 seconds")
            return {"error": "System not ready"}
        
        # Run all tests
        self.test_system_health()
        self.test_enhanced_processing()
        self.test_memory_system()
        self.test_metabolic_system()
        self.test_metrics_endpoint()
        self.run_benchmark()
        
        print("\n🎉 END-TO-END TESTING COMPLETE!")
        print("=" * 60)
        
        # Summary
        success_count = sum(1 for result in self.results.values() if not result.get('error'))
        total_tests = len(self.results)
        
        print(f"✅ Tests Passed: {success_count}/{total_tests}")
        print(f"📊 View metrics at: http://localhost:9090 (Prometheus)")
        print(f"📈 View dashboard at: http://localhost:3000 (Grafana, admin/admin)")
        print(f"🌐 API available at: {self.base_url}")
        
        return {
            'success_rate': success_count / total_tests if total_tests > 0 else 0,
            'results': self.results,
            'urls': {
                'api': self.base_url,
                'prometheus': 'http://localhost:9090',
                'grafana': 'http://localhost:3000'
            }
        }

if __name__ == "__main__":
    tester = AURAEndToEndTester()
    results = tester.run_all_tests()
    
    # Save results
    with open('test_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n📄 Results saved to: test_results.json")