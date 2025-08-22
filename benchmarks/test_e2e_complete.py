#!/usr/bin/env python3
"""
🧪 AURA Intelligence - Complete E2E Test Suite
Tests all components working together with real data and scenarios
"""

import asyncio
import time
import json
import sys
import logging
import requests
import websockets
from typing import Dict, List, Any
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import subprocess
import signal
import threading
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class E2ETestRunner:
    """Complete end-to-end test runner for AURA Intelligence"""
    
    def __init__(self):
        self.base_url = "http://localhost:8080"
        self.ws_url = "ws://localhost:8080/ws"
        self.dashboard_url = "http://localhost:8766"
        self.demo_process = None
        self.test_results = {}
        self.performance_data = []
        
    async def run_all_tests(self):
        """Run the complete E2E test suite"""
        logger.info("🧪 AURA Intelligence - Complete E2E Test Suite")
        logger.info("=" * 70)
        
        start_time = time.time()
        
        try:
            # Step 1: Start the demo application
            await self.start_demo_application()
            
            # Step 2: Wait for system to be ready
            await self.wait_for_system_ready()
            
            # Step 3: Run health checks
            health_results = await self.test_system_health()
            self.test_results['health'] = health_results
            
            # Step 4: Test individual components
            component_results = await self.test_components()
            self.test_results['components'] = component_results
            
            # Step 5: Test complete scenarios
            scenario_results = await self.test_demo_scenarios()
            self.test_results['scenarios'] = scenario_results
            
            # Step 6: Test real-time features
            realtime_results = await self.test_realtime_features()
            self.test_results['realtime'] = realtime_results
            
            # Step 7: Performance stress test
            performance_results = await self.test_performance()
            self.test_results['performance'] = performance_results
            
            # Step 8: Test concurrent processing
            concurrency_results = await self.test_concurrent_processing()
            self.test_results['concurrency'] = concurrency_results
            
            # Generate final report
            total_time = time.time() - start_time
            final_score = await self.generate_final_report(total_time)
            
            return final_score
            
        except Exception as e:
            logger.error(f"❌ E2E test suite failed: {e}")
            raise
        finally:
            await self.cleanup()
    
    async def start_demo_application(self):
        """Start the AURA demo application in background"""
        logger.info("🚀 Starting AURA Intelligence demo application...")
        
        # Start the demo in background
        self.demo_process = subprocess.Popen([
            sys.executable, "aura_complete_demo.py"
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        logger.info("✅ Demo application starting in background")
    
    async def wait_for_system_ready(self, timeout=120):
        """Wait for the system to be ready"""
        logger.info("⏳ Waiting for system to be ready...")
        
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                response = requests.get(f"{self.base_url}/health", timeout=5)
                if response.status_code == 200:
                    health_data = response.json()
                    if health_data.get('status') == 'healthy':
                        logger.info(f"✅ System ready after {time.time() - start_time:.1f}s")
                        return True
                    else:
                        logger.info(f"   System status: {health_data.get('status', 'unknown')}")
            except requests.exceptions.RequestException:
                pass
            
            await asyncio.sleep(2)
        
        raise TimeoutError("System failed to become ready within timeout")
    
    async def test_system_health(self) -> Dict[str, Any]:
        """Test system health endpoints"""
        logger.info("🏥 Testing system health...")
        
        results = {
            'health_endpoint': False,
            'component_health': {},
            'gpu_available': False,
            'response_time_ms': 0
        }
        
        try:
            start_time = time.time()
            response = requests.get(f"{self.base_url}/health", timeout=10)
            response_time = (time.time() - start_time) * 1000
            
            if response.status_code == 200:
                health_data = response.json()
                results['health_endpoint'] = True
                results['component_health'] = health_data.get('components', {})
                results['gpu_available'] = health_data.get('gpu_available', False)
                results['response_time_ms'] = response_time
                
                logger.info(f"   ✅ Health check passed ({response_time:.1f}ms)")
                logger.info(f"   ✅ GPU available: {results['gpu_available']}")
                logger.info(f"   ✅ Components healthy: {len(results['component_health'])}")
            else:
                logger.error(f"   ❌ Health check failed: {response.status_code}")
                
        except Exception as e:
            logger.error(f"   ❌ Health check error: {e}")
            
        return results
    
    async def test_components(self) -> Dict[str, Any]:
        """Test individual component functionality"""
        logger.info("🧩 Testing individual components...")
        
        results = {
            'scenarios_endpoint': False,
            'available_scenarios': 0,
            'performance_endpoint': False
        }
        
        try:
            # Test scenarios endpoint
            response = requests.get(f"{self.base_url}/scenarios", timeout=10)
            if response.status_code == 200:
                scenarios_data = response.json()
                results['scenarios_endpoint'] = True
                results['available_scenarios'] = len(scenarios_data.get('scenarios', {}))
                logger.info(f"   ✅ Scenarios endpoint: {results['available_scenarios']} scenarios")
            
            # Test performance endpoint
            response = requests.get(f"{self.base_url}/performance", timeout=10)
            if response.status_code == 200:
                results['performance_endpoint'] = True
                logger.info("   ✅ Performance metrics endpoint")
                
        except Exception as e:
            logger.error(f"   ❌ Component test error: {e}")
            
        return results
    
    async def test_demo_scenarios(self) -> Dict[str, Any]:
        """Test all demo scenarios"""
        logger.info("🎬 Testing demo scenarios...")
        
        scenarios = {
            'ai_reasoning': {
                'task': 'ai_reasoning',
                'data': {'text': 'Test AI reasoning with quantum computing implications', 'complexity': 'high'},
                'use_gpu': True,
                'use_agents': True
            },
            'real_time_analysis': {
                'task': 'real_time_analysis',
                'data': {'stream': list(np.random.randn(100)), 'window': 10},
                'use_gpu': True,
                'use_agents': False
            },
            'multi_agent_decision': {
                'task': 'multi_agent_decision',
                'data': {'decision_context': 'resource_allocation', 'agents': 3, 'confidence_threshold': 0.8},
                'use_gpu': False,
                'use_agents': True
            },
            'performance_benchmark': {
                'task': 'performance_benchmark',
                'data': {'iterations': 10, 'load_test': True, 'comprehensive': True},
                'use_gpu': True,
                'use_agents': True
            }
        }
        
        results = {}
        
        for scenario_name, scenario_data in scenarios.items():
            logger.info(f"   🔬 Testing {scenario_name}...")
            
            try:
                start_time = time.time()
                response = requests.post(
                    f"{self.base_url}/demo",
                    json=scenario_data,
                    timeout=30
                )
                response_time = (time.time() - start_time) * 1000
                
                if response.status_code == 200:
                    demo_result = response.json()
                    if demo_result.get('status') == 'success':
                        results[scenario_name] = {
                            'success': True,
                            'response_time_ms': response_time,
                            'processing_time_ms': demo_result['performance_metrics']['total_processing_time_ms'],
                            'pipeline_steps': len(demo_result['processing_pipeline']),
                            'components_used': demo_result['performance_metrics']['component_count'],
                            'efficiency_score': demo_result['performance_metrics']['efficiency_score']
                        }
                        logger.info(f"      ✅ {scenario_name}: {response_time:.1f}ms, {results[scenario_name]['pipeline_steps']} steps")
                    else:
                        results[scenario_name] = {'success': False, 'error': demo_result.get('results', {}).get('error', 'Unknown error')}
                        logger.error(f"      ❌ {scenario_name} failed: {results[scenario_name]['error']}")
                else:
                    results[scenario_name] = {'success': False, 'error': f'HTTP {response.status_code}'}
                    logger.error(f"      ❌ {scenario_name} HTTP error: {response.status_code}")
                    
            except Exception as e:
                results[scenario_name] = {'success': False, 'error': str(e)}
                logger.error(f"      ❌ {scenario_name} exception: {e}")
        
        success_count = sum(1 for r in results.values() if r.get('success', False))
        logger.info(f"   📊 Scenario tests: {success_count}/{len(scenarios)} passed")
        
        return results
    
    async def test_realtime_features(self) -> Dict[str, Any]:
        """Test real-time WebSocket features"""
        logger.info("🔄 Testing real-time features...")
        
        results = {
            'websocket_connection': False,
            'messages_received': 0,
            'average_latency_ms': 0
        }
        
        try:
            messages = []
            message_times = []
            
            async with websockets.connect(self.ws_url, timeout=10) as websocket:
                results['websocket_connection'] = True
                logger.info("   ✅ WebSocket connection established")
                
                # Listen for messages for 10 seconds
                start_time = time.time()
                while time.time() - start_time < 10:
                    try:
                        message = await asyncio.wait_for(websocket.recv(), timeout=3)
                        message_time = time.time()
                        messages.append(json.loads(message))
                        message_times.append(message_time)
                    except asyncio.TimeoutError:
                        break
                
                results['messages_received'] = len(messages)
                if len(message_times) > 1:
                    # Calculate average time between messages
                    intervals = [message_times[i] - message_times[i-1] for i in range(1, len(message_times))]
                    results['average_latency_ms'] = np.mean(intervals) * 1000
                
                logger.info(f"   ✅ Received {results['messages_received']} real-time messages")
                
        except Exception as e:
            logger.error(f"   ❌ Real-time test error: {e}")
        
        return results
    
    async def test_performance(self) -> Dict[str, Any]:
        """Test system performance under load"""
        logger.info("⚡ Testing performance under load...")
        
        results = {
            'single_request_avg_ms': 0,
            'requests_completed': 0,
            'requests_failed': 0,
            'throughput_req_per_sec': 0,
            'p95_latency_ms': 0
        }
        
        # Performance test scenario
        test_scenario = {
            'task': 'performance_test',
            'data': {'test_size': 'medium', 'complexity': 'standard'},
            'use_gpu': True,
            'use_agents': False
        }
        
        # Run 20 requests sequentially to measure performance
        request_times = []
        
        for i in range(20):
            try:
                start_time = time.time()
                response = requests.post(
                    f"{self.base_url}/demo",
                    json=test_scenario,
                    timeout=15
                )
                end_time = time.time()
                
                if response.status_code == 200:
                    request_times.append((end_time - start_time) * 1000)
                    results['requests_completed'] += 1
                else:
                    results['requests_failed'] += 1
                    
            except Exception as e:
                results['requests_failed'] += 1
                logger.error(f"   Request {i+1} failed: {e}")
        
        if request_times:
            results['single_request_avg_ms'] = np.mean(request_times)
            results['p95_latency_ms'] = np.percentile(request_times, 95)
            
            # Calculate throughput (assuming sequential processing)
            total_time_sec = sum(request_times) / 1000
            results['throughput_req_per_sec'] = len(request_times) / total_time_sec if total_time_sec > 0 else 0
            
            logger.info(f"   📊 Average latency: {results['single_request_avg_ms']:.1f}ms")
            logger.info(f"   📊 95th percentile: {results['p95_latency_ms']:.1f}ms")
            logger.info(f"   📊 Throughput: {results['throughput_req_per_sec']:.1f} req/s")
            logger.info(f"   📊 Success rate: {results['requests_completed']}/{results['requests_completed'] + results['requests_failed']}")
        
        return results
    
    async def test_concurrent_processing(self) -> Dict[str, Any]:
        """Test concurrent request handling"""
        logger.info("🚀 Testing concurrent processing...")
        
        results = {
            'concurrent_requests': 5,
            'successful_concurrent': 0,
            'failed_concurrent': 0,
            'average_concurrent_latency_ms': 0,
            'concurrency_efficiency': 0
        }
        
        # Test concurrent scenario
        test_scenario = {
            'task': 'concurrent_test',
            'data': {'concurrent_id': 0},
            'use_gpu': True,
            'use_agents': True
        }
        
        async def make_concurrent_request(request_id):
            scenario = test_scenario.copy()
            scenario['data']['concurrent_id'] = request_id
            
            try:
                loop = asyncio.get_event_loop()
                
                # Run request in thread pool to avoid blocking
                def make_request():
                    return requests.post(
                        f"{self.base_url}/demo",
                        json=scenario,
                        timeout=20
                    )
                
                with ThreadPoolExecutor() as executor:
                    response = await loop.run_in_executor(executor, make_request)
                
                return response.status_code == 200, time.time()
            except Exception as e:
                logger.error(f"   Concurrent request {request_id} failed: {e}")
                return False, time.time()
        
        # Run concurrent requests
        start_time = time.time()
        tasks = [make_concurrent_request(i) for i in range(results['concurrent_requests'])]
        concurrent_results = await asyncio.gather(*tasks)
        end_time = time.time()
        
        # Analyze results
        successful = sum(1 for success, _ in concurrent_results if success)
        failed = len(concurrent_results) - successful
        
        results['successful_concurrent'] = successful
        results['failed_concurrent'] = failed
        results['average_concurrent_latency_ms'] = (end_time - start_time) * 1000
        
        # Calculate efficiency (how much faster than sequential)
        if successful > 0:
            results['concurrency_efficiency'] = successful / ((end_time - start_time) * 1000) * 1000
        
        logger.info(f"   📊 Concurrent success: {successful}/{len(concurrent_results)}")
        logger.info(f"   📊 Concurrent latency: {results['average_concurrent_latency_ms']:.1f}ms")
        logger.info(f"   📊 Concurrency efficiency: {results['concurrency_efficiency']:.1f}")
        
        return results
    
    async def generate_final_report(self, total_time: float) -> Dict[str, Any]:
        """Generate comprehensive test report"""
        logger.info("📋 Generating final E2E test report...")
        logger.info("=" * 70)
        
        # Calculate overall scores
        health_score = 1.0 if self.test_results.get('health', {}).get('health_endpoint', False) else 0.0
        
        scenario_results = self.test_results.get('scenarios', {})
        scenario_score = sum(1 for r in scenario_results.values() if r.get('success', False)) / max(len(scenario_results), 1)
        
        realtime_score = 1.0 if self.test_results.get('realtime', {}).get('websocket_connection', False) else 0.0
        
        performance_results = self.test_results.get('performance', {})
        performance_score = 1.0 if performance_results.get('single_request_avg_ms', 0) < 1000 else 0.5
        
        concurrent_results = self.test_results.get('concurrency', {})
        concurrent_score = concurrent_results.get('successful_concurrent', 0) / max(concurrent_results.get('concurrent_requests', 1), 1)
        
        # Weighted overall score
        overall_score = (
            health_score * 0.2 +
            scenario_score * 0.4 +
            realtime_score * 0.1 +
            performance_score * 0.2 +
            concurrent_score * 0.1
        )
        
        # Determine grade and status
        if overall_score >= 0.9:
            grade = "A+"
            status = "EXCELLENT"
        elif overall_score >= 0.8:
            grade = "A"
            status = "VERY GOOD"
        elif overall_score >= 0.7:
            grade = "B"
            status = "GOOD"
        elif overall_score >= 0.6:
            grade = "C"
            status = "SATISFACTORY"
        else:
            grade = "D"
            status = "NEEDS IMPROVEMENT"
        
        final_report = {
            'overall_score': overall_score,
            'grade': grade,
            'status': status,
            'total_test_time_sec': total_time,
            'test_breakdown': {
                'health': health_score,
                'scenarios': scenario_score,
                'realtime': realtime_score,
                'performance': performance_score,
                'concurrency': concurrent_score
            },
            'key_metrics': {
                'avg_processing_time_ms': performance_results.get('single_request_avg_ms', 0),
                'throughput_req_per_sec': performance_results.get('throughput_req_per_sec', 0),
                'scenarios_passed': sum(1 for r in scenario_results.values() if r.get('success', False)),
                'gpu_available': self.test_results.get('health', {}).get('gpu_available', False)
            }
        }
        
        # Display report
        logger.info("🏆 FINAL E2E TEST RESULTS")
        logger.info("=" * 70)
        logger.info(f"📊 Overall Score: {overall_score*100:.1f}% ({grade})")
        logger.info(f"🎯 Status: {status}")
        logger.info(f"⏱️  Total Test Time: {total_time:.1f}s")
        logger.info("")
        logger.info("📋 Test Breakdown:")
        logger.info(f"   🏥 Health Tests:      {health_score*100:.0f}%")
        logger.info(f"   🎬 Scenario Tests:    {scenario_score*100:.0f}%")
        logger.info(f"   🔄 Real-time Tests:   {realtime_score*100:.0f}%")
        logger.info(f"   ⚡ Performance Tests: {performance_score*100:.0f}%")
        logger.info(f"   🚀 Concurrency Tests: {concurrent_score*100:.0f}%")
        logger.info("")
        logger.info("🔑 Key Metrics:")
        logger.info(f"   📊 Average Processing: {final_report['key_metrics']['avg_processing_time_ms']:.1f}ms")
        logger.info(f"   📊 Throughput: {final_report['key_metrics']['throughput_req_per_sec']:.1f} req/s")
        logger.info(f"   📊 Scenarios Passed: {final_report['key_metrics']['scenarios_passed']}/4")
        logger.info(f"   📊 GPU Available: {final_report['key_metrics']['gpu_available']}")
        
        # Save detailed results
        with open('e2e_test_results.json', 'w') as f:
            json.dump({
                'final_report': final_report,
                'detailed_results': self.test_results,
                'timestamp': datetime.now().isoformat()
            }, f, indent=2)
        
        logger.info("")
        logger.info("💾 Detailed results saved to: e2e_test_results.json")
        logger.info("=" * 70)
        
        if overall_score >= 0.8:
            logger.info("🎉 E2E TESTS PASSED! SYSTEM IS PRODUCTION READY!")
        else:
            logger.warning("⚠️  E2E TESTS NEED IMPROVEMENT")
        
        return final_report
    
    async def cleanup(self):
        """Cleanup test resources"""
        logger.info("🧹 Cleaning up test resources...")
        
        if self.demo_process:
            try:
                self.demo_process.terminate()
                self.demo_process.wait(timeout=5)
            except:
                self.demo_process.kill()
            logger.info("✅ Demo application stopped")


async def main():
    """Main test execution"""
    test_runner = E2ETestRunner()
    
    try:
        final_report = await test_runner.run_all_tests()
        
        # Return appropriate exit code
        if final_report['overall_score'] >= 0.8:
            return 0
        else:
            return 1
            
    except Exception as e:
        logger.error(f"❌ E2E test execution failed: {e}")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)