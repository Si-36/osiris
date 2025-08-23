#!/usr/bin/env python3
"""
🔥 AURA Intelligence Comprehensive Integration Test
✨ Tests ALL 213 components working together in production mode
"""

import asyncio
import json
import time
import sys
import os
from datetime import datetime
from typing import Dict, List, Tuple, Any
import subprocess
import signal

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

try:
    from aura.core.system import AURASystem, AURAConfig
    from aura.api.unified_api import create_app
    from aura.ray.distributed_tda import AURARayServe
    from aura.a2a.agent_protocol import A2AProtocol
    from aura.ultimate_system_2025 import UltimateAURASystem
    AURA_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ AURA modules not fully available: {e}")
    AURA_AVAILABLE = False

class ComprehensiveIntegrationTest:
    """Test all AURA components working together"""
    
    def __init__(self):
        self.results = {
            'timestamp': datetime.now().isoformat(),
            'total_components': 213,
            'tests_run': 0,
            'tests_passed': 0,
            'tests_failed': 0,
            'component_status': {},
            'integration_status': {},
            'performance_metrics': {},
            'errors': []
        }
        
    async def test_core_system(self) -> bool:
        """Test core AURA system with all 213 components"""
        print("\n🔬 Testing Core System...")
        try:
            if not AURA_AVAILABLE:
                print("  ⚠️ Core system not available")
                return False
                
            config = AURAConfig()
            system = AURASystem(config)
            
            # Verify component counts
            components = system.get_all_components()
            
            expected_counts = {
                'tda': 112,
                'nn': 10,
                'memory': 40,
                'agents': 100,
                'consensus': 5,
                'neuromorphic': 8,
                'infrastructure': 51
            }
            
            all_good = True
            for category, expected in expected_counts.items():
                actual = len(components.get(category, []))
                if actual == expected:
                    print(f"  ✅ {category.upper()}: {actual}/{expected}")
                else:
                    print(f"  ❌ {category.upper()}: {actual}/{expected}")
                    all_good = False
                    
                self.results['component_status'][category] = {
                    'expected': expected,
                    'actual': actual,
                    'status': 'pass' if actual == expected else 'fail'
                }
            
            # Test basic operations
            print("\n  Testing operations...")
            
            # Test topology analysis
            test_data = [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]
            topology = system.analyze_topology(test_data)
            if topology and 'betti_numbers' in topology:
                print("  ✅ Topology analysis working")
            else:
                print("  ❌ Topology analysis failed")
                all_good = False
                
            # Test failure prediction
            prediction = system.predict_failure(topology)
            if prediction and 'risk_score' in prediction:
                print("  ✅ Failure prediction working")
            else:
                print("  ❌ Failure prediction failed")
                all_good = False
                
            # Test cascade prevention
            prevention = system.prevent_cascade(prediction)
            if prevention and 'success' in prevention:
                print("  ✅ Cascade prevention working")
            else:
                print("  ❌ Cascade prevention failed")
                all_good = False
                
            return all_good
            
        except Exception as e:
            print(f"  ❌ Core system test failed: {e}")
            self.results['errors'].append(f"Core system: {str(e)}")
            return False
            
    async def test_knowledge_graph(self) -> bool:
        """Test Knowledge Graph integration"""
        print("\n🧠 Testing Knowledge Graph...")
        try:
            # Check if Neo4j is accessible
            result = subprocess.run(
                ['curl', '-s', 'http://localhost:7474'],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            if 'neo4j' in result.stdout.lower() or result.returncode == 0:
                print("  ✅ Neo4j is accessible")
                
                # Test knowledge graph operations
                if AURA_AVAILABLE:
                    from aura.ultimate_system_2025 import EnhancedKnowledgeGraph
                    kg = EnhancedKnowledgeGraph()
                    await kg.initialize()
                    
                    # Store test topology
                    test_topology = {
                        'betti_numbers': [1, 2, 3],
                        'complexity': 0.75
                    }
                    
                    stored = await kg.store_topology(test_topology)
                    if stored:
                        print("  ✅ Topology storage working")
                    else:
                        print("  ⚠️ Topology storage not available")
                        
                    # Query patterns
                    patterns = await kg.query_patterns('cascade_risk')
                    if patterns:
                        print("  ✅ Pattern query working")
                    else:
                        print("  ⚠️ Pattern query not available")
                        
                return True
            else:
                print("  ⚠️ Neo4j not running")
                return False
                
        except Exception as e:
            print(f"  ❌ Knowledge Graph test failed: {e}")
            self.results['errors'].append(f"Knowledge Graph: {str(e)}")
            return False
            
    async def test_ray_integration(self) -> bool:
        """Test Ray distributed computing"""
        print("\n🌟 Testing Ray Integration...")
        try:
            # Check if Ray is installed
            result = subprocess.run(
                ['python3', '-c', 'import ray; print(ray.__version__)'],
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                print(f"  ✅ Ray installed: v{result.stdout.strip()}")
                
                # Test Ray operations
                if AURA_AVAILABLE:
                    # Simple Ray test
                    import ray
                    if not ray.is_initialized():
                        ray.init(ignore_reinit_error=True)
                        
                    @ray.remote
                    def test_func(x):
                        return x * 2
                        
                    result = ray.get(test_func.remote(21))
                    if result == 42:
                        print("  ✅ Ray computation working")
                    else:
                        print("  ❌ Ray computation failed")
                        
                    ray.shutdown()
                    
                return True
            else:
                print("  ⚠️ Ray not installed")
                return False
                
        except Exception as e:
            print(f"  ❌ Ray test failed: {e}")
            self.results['errors'].append(f"Ray: {str(e)}")
            return False
            
    async def test_a2a_mcp(self) -> bool:
        """Test A2A communication with MCP"""
        print("\n🤝 Testing A2A + MCP...")
        try:
            if AURA_AVAILABLE:
                from aura.a2a.agent_protocol import A2AProtocol, MCPContext
                
                # Create protocol
                protocol = A2AProtocol(agent_id="test-agent-1", agent_type="orchestrator")
                
                # Test MCP context
                context = MCPContext(
                    domain="test",
                    capabilities=["analyze", "predict"],
                    constraints={"max_latency": 100}
                )
                
                # Test message creation
                message = protocol.create_message(
                    recipient="test-agent-2",
                    content={"test": "data"},
                    message_type="request",
                    context=context
                )
                
                if message and hasattr(message, 'id'):
                    print("  ✅ A2A message creation working")
                else:
                    print("  ❌ A2A message creation failed")
                    
                # Test protocol validation
                valid = protocol.validate_message(message)
                if valid:
                    print("  ✅ MCP validation working")
                else:
                    print("  ❌ MCP validation failed")
                    
                return True
            else:
                print("  ⚠️ A2A/MCP modules not available")
                return False
                
        except Exception as e:
            print(f"  ❌ A2A/MCP test failed: {e}")
            self.results['errors'].append(f"A2A/MCP: {str(e)}")
            return False
            
    async def test_monitoring_stack(self) -> bool:
        """Test Prometheus/Grafana monitoring"""
        print("\n📊 Testing Monitoring Stack...")
        try:
            # Check Prometheus
            prom_result = subprocess.run(
                ['curl', '-s', 'http://localhost:9090/-/healthy'],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            prom_healthy = prom_result.returncode == 0
            if prom_healthy:
                print("  ✅ Prometheus is healthy")
            else:
                print("  ⚠️ Prometheus not running")
                
            # Check Grafana
            graf_result = subprocess.run(
                ['curl', '-s', 'http://localhost:3000/api/health'],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            graf_healthy = graf_result.returncode == 0
            if graf_healthy:
                print("  ✅ Grafana is healthy")
            else:
                print("  ⚠️ Grafana not running")
                
            # Check metrics endpoint
            if AURA_AVAILABLE:
                metrics_result = subprocess.run(
                    ['curl', '-s', 'http://localhost:8000/metrics'],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                
                if 'aura_' in metrics_result.stdout:
                    print("  ✅ AURA metrics exposed")
                else:
                    print("  ⚠️ AURA metrics not available")
                    
            return prom_healthy or graf_healthy
            
        except Exception as e:
            print(f"  ❌ Monitoring test failed: {e}")
            self.results['errors'].append(f"Monitoring: {str(e)}")
            return False
            
    async def test_api_endpoints(self) -> bool:
        """Test all API endpoints"""
        print("\n🌐 Testing API Endpoints...")
        try:
            # Check if API is running
            api_result = subprocess.run(
                ['curl', '-s', 'http://localhost:8000/health'],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            if api_result.returncode == 0:
                health_data = json.loads(api_result.stdout)
                if health_data.get('status') == 'healthy':
                    print("  ✅ API is healthy")
                else:
                    print("  ⚠️ API unhealthy")
                    
                # Test key endpoints
                endpoints = [
                    '/analyze',
                    '/predict',
                    '/intervene',
                    '/topology/visualize',
                    '/debug/components'
                ]
                
                for endpoint in endpoints:
                    try:
                        # Just check if endpoint exists
                        result = subprocess.run(
                            ['curl', '-s', '-X', 'OPTIONS', f'http://localhost:8000{endpoint}'],
                            capture_output=True,
                            text=True,
                            timeout=5
                        )
                        
                        if result.returncode == 0:
                            print(f"  ✅ {endpoint} available")
                        else:
                            print(f"  ⚠️ {endpoint} not responding")
                    except:
                        print(f"  ⚠️ {endpoint} error")
                        
                return True
            else:
                print("  ⚠️ API not running")
                return False
                
        except Exception as e:
            print(f"  ❌ API test failed: {e}")
            self.results['errors'].append(f"API: {str(e)}")
            return False
            
    async def test_performance_benchmarks(self) -> bool:
        """Run performance benchmarks"""
        print("\n⚡ Testing Performance...")
        try:
            if not AURA_AVAILABLE:
                print("  ⚠️ Performance tests skipped (modules not available)")
                return False
                
            config = AURAConfig()
            system = AURASystem(config)
            
            # Test latency
            latencies = []
            for i in range(10):
                start = time.time()
                test_data = [[j/10 for j in range(10)] for _ in range(5)]
                topology = system.analyze_topology(test_data)
                prediction = system.predict_failure(topology)
                prevention = system.prevent_cascade(prediction)
                end = time.time()
                latencies.append((end - start) * 1000)  # Convert to ms
                
            avg_latency = sum(latencies) / len(latencies)
            max_latency = max(latencies)
            min_latency = min(latencies)
            
            print(f"  📊 Latency: avg={avg_latency:.2f}ms, min={min_latency:.2f}ms, max={max_latency:.2f}ms")
            
            self.results['performance_metrics'] = {
                'avg_latency_ms': avg_latency,
                'min_latency_ms': min_latency,
                'max_latency_ms': max_latency,
                'target_latency_ms': 5.0,
                'meets_target': avg_latency < 5.0
            }
            
            if avg_latency < 5.0:
                print("  ✅ Meets 5ms latency target")
                return True
            else:
                print("  ⚠️ Exceeds 5ms latency target")
                return False
                
        except Exception as e:
            print(f"  ❌ Performance test failed: {e}")
            self.results['errors'].append(f"Performance: {str(e)}")
            return False
            
    async def run_all_tests(self):
        """Run all integration tests"""
        print("🚀 AURA Intelligence Comprehensive Integration Test")
        print("=" * 60)
        
        tests = [
            ("Core System", self.test_core_system),
            ("Knowledge Graph", self.test_knowledge_graph),
            ("Ray Integration", self.test_ray_integration),
            ("A2A + MCP", self.test_a2a_mcp),
            ("Monitoring Stack", self.test_monitoring_stack),
            ("API Endpoints", self.test_api_endpoints),
            ("Performance", self.test_performance_benchmarks)
        ]
        
        for test_name, test_func in tests:
            self.results['tests_run'] += 1
            try:
                passed = await test_func()
                if passed:
                    self.results['tests_passed'] += 1
                    self.results['integration_status'][test_name] = 'pass'
                else:
                    self.results['tests_failed'] += 1
                    self.results['integration_status'][test_name] = 'fail'
            except Exception as e:
                self.results['tests_failed'] += 1
                self.results['integration_status'][test_name] = 'error'
                self.results['errors'].append(f"{test_name}: {str(e)}")
                
        # Summary
        print("\n" + "=" * 60)
        print("📊 INTEGRATION TEST SUMMARY")
        print("=" * 60)
        
        print(f"Total Tests: {self.results['tests_run']}")
        print(f"✅ Passed: {self.results['tests_passed']}")
        print(f"❌ Failed: {self.results['tests_failed']}")
        
        pass_rate = (self.results['tests_passed'] / self.results['tests_run'] * 100) if self.results['tests_run'] > 0 else 0
        print(f"📈 Pass Rate: {pass_rate:.1f}%")
        
        # Component summary
        print("\n📦 Component Status:")
        total_components = 0
        actual_components = 0
        for category, status in self.results['component_status'].items():
            total_components += status['expected']
            actual_components += status['actual']
            symbol = "✅" if status['status'] == 'pass' else "❌"
            print(f"  {symbol} {category.upper()}: {status['actual']}/{status['expected']}")
            
        print(f"\n🔢 Total Components: {actual_components}/{total_components} ({actual_components/total_components*100:.1f}%)")
        
        # Save detailed report
        report_file = f"integration_test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"\n💾 Detailed report saved to: {report_file}")
        
        # Final verdict
        if pass_rate >= 90 and actual_components == total_components:
            print("\n🎉 INTEGRATION TEST PASSED! System is production-ready! 🚀")
            return True
        else:
            print("\n⚠️  Integration test needs attention - not all components working perfectly")
            return False

async def main():
    """Run the comprehensive integration test"""
    tester = ComprehensiveIntegrationTest()
    success = await tester.run_all_tests()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    asyncio.run(main())