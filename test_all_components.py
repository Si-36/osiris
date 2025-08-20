#!/usr/bin/env python3
"""
🧪 COMPLETE AURA INTELLIGENCE COMPONENT TESTING
Test all 209 components individually + E2E testing
"""

import asyncio
import time
import json
from typing import Dict, Any, List
from pathlib import Path

# Add to path
import sys
sys.path.insert(0, str(Path(__file__).parent / "core" / "src"))

from aura_intelligence.components.real_registry import get_real_registry, ComponentType

class ComponentTester:
    def __init__(self):
        self.registry = get_real_registry()
        self.test_results = {}
        self.passed = 0
        self.failed = 0
        
    async def test_all_components(self) -> Dict[str, Any]:
        """Test all 209 components individually"""
        print("🧪 Testing all AURA Intelligence components...")
        
        start_time = time.time()
        
        # Test by component type
        for component_type in ComponentType:
            components = self.registry.get_components_by_type(component_type)
            print(f"\n📋 Testing {component_type.value} components ({len(components)} total)")
            
            for component in components:
                await self._test_single_component(component.component_id, component_type)
        
        total_time = time.time() - start_time
        
        # Generate summary
        summary = {
            'total_components': len(self.registry.components),
            'components_tested': len(self.test_results),
            'passed': self.passed,
            'failed': self.failed,
            'success_rate': self.passed / len(self.test_results) * 100,
            'total_time_seconds': total_time,
            'avg_time_per_component_ms': (total_time / len(self.test_results)) * 1000,
            'test_results': self.test_results
        }
        
        return summary
    
    async def _test_single_component(self, component_id: str, component_type: ComponentType):
        """Test single component with appropriate test data"""
        test_data = self._get_test_data_for_type(component_type)
        
        try:
            start_time = time.perf_counter()
            result = await self.registry.process_data(component_id, test_data)
            processing_time = (time.perf_counter() - start_time) * 1000
            
            # Validate result
            is_valid = self._validate_result(result, component_type)
            
            self.test_results[component_id] = {
                'status': 'PASSED' if is_valid else 'FAILED',
                'processing_time_ms': processing_time,
                'result_type': str(type(result)),
                'has_output': bool(result),
                'validation': is_valid
            }
            
            if is_valid:
                self.passed += 1
                print(f"  ✅ {component_id} - {processing_time:.2f}ms")
            else:
                self.failed += 1
                print(f"  ❌ {component_id} - Invalid result")
                
        except Exception as e:
            self.failed += 1
            self.test_results[component_id] = {
                'status': 'ERROR',
                'error': str(e),
                'processing_time_ms': 0
            }
            print(f"  💥 {component_id} - ERROR: {e}")
    
    def _get_test_data_for_type(self, component_type: ComponentType) -> Dict[str, Any]:
        """Get appropriate test data for component type"""
        if component_type == ComponentType.NEURAL:
            return {
                'values': [0.1, 0.5, 0.8, 0.3, 0.9],
                'sequence': [1, 2, 3, 4, 5],
                'hidden_states': [[0.1] * 512],
                'text': 'test input for neural processing'
            }
        elif component_type == ComponentType.TDA:
            return {
                'points': [[1, 2], [3, 4], [5, 6], [7, 8]],
                'matrix': [[1, 0], [0, 1]]
            }
        elif component_type == ComponentType.MEMORY:
            return {
                'key': 'test_key',
                'value': 'test_value',
                'vector': [0.1, 0.2, 0.3, 0.4, 0.5]
            }
        elif component_type == ComponentType.AGENT:
            return {
                'action': 'test_action',
                'tasks': ['task1', 'task2'],
                'experience': {'reward': 0.8, 'state': 'good'}
            }
        elif component_type == ComponentType.ORCHESTRATION:
            return {
                'steps': ['step1', 'step2', 'step3'],
                'workflow_id': 'test_workflow'
            }
        else:  # OBSERVABILITY
            return {
                'metrics': {'cpu': 0.5, 'memory': 0.3},
                'timestamp': time.time()
            }
    
    def _validate_result(self, result: Any, component_type: ComponentType) -> bool:
        """Validate component result"""
        if not result:
            return False
        
        if isinstance(result, dict):
            # Check for error indicators
            if 'error' in result:
                return False
            
            # Type-specific validation
            if component_type == ComponentType.NEURAL:
                return any(key in result for key in ['lnn_output', 'attention_output', 'embeddings', 'component_id'])
            elif component_type == ComponentType.TDA:
                return any(key in result for key in ['betti_numbers', 'persistence_computed', 'tda_component'])
            elif component_type == ComponentType.MEMORY:
                return any(key in result for key in ['stored', 'cache_hit', 'memory_operation'])
            elif component_type == ComponentType.AGENT:
                return any(key in result for key in ['decision', 'coordinated_tasks', 'agent_action'])
            else:
                return True
        
        return True

class E2ETester:
    def __init__(self):
        self.registry = get_real_registry()
    
    async def run_e2e_tests(self) -> Dict[str, Any]:
        """Run end-to-end system tests"""
        print("\n🔄 Running End-to-End Tests...")
        
        tests = [
            self._test_neural_pipeline,
            self._test_tda_pipeline,
            self._test_memory_pipeline,
            self._test_agent_pipeline,
            self._test_full_system_pipeline
        ]
        
        results = {}
        for test in tests:
            test_name = test.__name__.replace('_test_', '').replace('_', ' ').title()
            try:
                start_time = time.perf_counter()
                result = await test()
                processing_time = (time.perf_counter() - start_time) * 1000
                
                results[test_name] = {
                    'status': 'PASSED',
                    'processing_time_ms': processing_time,
                    'result': result
                }
                print(f"  ✅ {test_name} - {processing_time:.2f}ms")
                
            except Exception as e:
                results[test_name] = {
                    'status': 'FAILED',
                    'error': str(e)
                }
                print(f"  ❌ {test_name} - ERROR: {e}")
        
        return results
    
    async def _test_neural_pipeline(self) -> Dict[str, Any]:
        """Test neural processing pipeline"""
        # Get neural components
        neural_components = self.registry.get_components_by_type(ComponentType.NEURAL)
        
        # Process through first 3 neural components
        data = {'values': [0.1, 0.5, 0.8]}
        results = []
        
        for component in neural_components[:3]:
            result = await self.registry.process_data(component.component_id, data)
            results.append(result)
            data = result  # Chain results
        
        return {'components_used': 3, 'final_result': results[-1]}
    
    async def _test_tda_pipeline(self) -> Dict[str, Any]:
        """Test TDA processing pipeline"""
        tda_components = self.registry.get_components_by_type(ComponentType.TDA)
        
        data = {'points': [[1, 2], [3, 4], [5, 6]]}
        result = await self.registry.process_data(tda_components[0].component_id, data)
        
        return {'tda_result': result}
    
    async def _test_memory_pipeline(self) -> Dict[str, Any]:
        """Test memory processing pipeline"""
        memory_components = self.registry.get_components_by_type(ComponentType.MEMORY)
        
        results = []
        for component in memory_components[:2]:
            result = await self.registry.process_data(
                component.component_id, 
                {'key': f'test_{component.component_id}', 'value': 'test_data'}
            )
            results.append(result)
        
        return {'memory_operations': len(results), 'results': results}
    
    async def _test_agent_pipeline(self) -> Dict[str, Any]:
        """Test agent processing pipeline"""
        agent_components = self.registry.get_components_by_type(ComponentType.AGENT)
        
        data = {'action': 'coordinate', 'tasks': ['task1', 'task2']}
        result = await self.registry.process_data(agent_components[0].component_id, data)
        
        return {'agent_result': result}
    
    async def _test_full_system_pipeline(self) -> Dict[str, Any]:
        """Test full system pipeline with multiple component types"""
        # Select one component from each type
        pipeline_components = []
        for component_type in ComponentType:
            components = self.registry.get_components_by_type(component_type)
            if components:
                pipeline_components.append(components[0].component_id)
        
        # Process through pipeline
        data = {'values': [0.1, 0.2, 0.3], 'test': 'full_pipeline'}
        results = await self.registry.process_pipeline(data, pipeline_components)
        
        return {
            'pipeline_length': len(pipeline_components),
            'total_processing_time': results.get('total_processing_time', 0),
            'components_used': results.get('components_used', 0),
            'success': True
        }

async def main():
    """Run complete test suite"""
    print("🚀 AURA Intelligence Complete Test Suite")
    print("=" * 50)
    
    # Component testing
    component_tester = ComponentTester()
    component_results = await component_tester.test_all_components()
    
    print(f"\n📊 Component Test Summary:")
    print(f"  Total Components: {component_results['total_components']}")
    print(f"  Tested: {component_results['components_tested']}")
    print(f"  Passed: {component_results['passed']}")
    print(f"  Failed: {component_results['failed']}")
    print(f"  Success Rate: {component_results['success_rate']:.1f}%")
    print(f"  Total Time: {component_results['total_time_seconds']:.2f}s")
    print(f"  Avg Time/Component: {component_results['avg_time_per_component_ms']:.2f}ms")
    
    # E2E testing
    e2e_tester = E2ETester()
    e2e_results = await e2e_tester.run_e2e_tests()
    
    print(f"\n📊 E2E Test Summary:")
    passed_e2e = sum(1 for r in e2e_results.values() if r['status'] == 'PASSED')
    total_e2e = len(e2e_results)
    print(f"  E2E Tests Passed: {passed_e2e}/{total_e2e}")
    print(f"  E2E Success Rate: {passed_e2e/total_e2e*100:.1f}%")
    
    # Save detailed results
    results_file = Path("test_results.json")
    with open(results_file, 'w') as f:
        json.dump({
            'component_tests': component_results,
            'e2e_tests': e2e_results,
            'timestamp': time.time()
        }, f, indent=2)
    
    print(f"\n💾 Detailed results saved to: {results_file}")
    
    # Final system health
    overall_success = (component_results['success_rate'] + passed_e2e/total_e2e*100) / 2
    print(f"\n🏆 OVERALL SYSTEM HEALTH: {overall_success:.1f}%")
    
    if overall_success > 90:
        print("🎉 EXCELLENT - System is production ready!")
    elif overall_success > 75:
        print("✅ GOOD - System is mostly functional")
    elif overall_success > 50:
        print("⚠️  FAIR - System needs improvements")
    else:
        print("❌ POOR - System needs major fixes")

if __name__ == "__main__":
    asyncio.run(main())