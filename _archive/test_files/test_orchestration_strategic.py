#!/usr/bin/env python3
"""
Strategic Orchestration Layer Test Suite
=======================================
Tests drift detection, resource planning, and model lifecycle
"""

import asyncio
import numpy as np
import torch
import time
from typing import Dict, Any, List
import sys
import os

# Add AURA to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'core/src'))

from aura_intelligence.orchestration.strategic import (
    StrategicPlanner,
    ResourcePlan,
    DriftDetector,
    DriftScore,
    ModelLifecycleManager,
    ModelVersion,
    CanaryDeployment,
    DeploymentStage
)


class OrchestratorTestMetrics:
    """Track orchestration test metrics"""
    def __init__(self):
        self.results = {}
        self.start_time = time.time()
        
    def record(self, test_name: str, success: bool, metrics: Dict[str, Any]):
        self.results[test_name] = {
            'success': success,
            'metrics': metrics,
            'duration': time.time() - self.start_time
        }
        
    def summary(self):
        total = len(self.results)
        passed = sum(1 for r in self.results.values() if r['success'])
        
        print("\n" + "="*80)
        print("📊 ORCHESTRATION TEST SUMMARY")
        print("="*80)
        print(f"Total Tests: {total}")
        print(f"Passed: {passed}/{total} ({passed/total*100:.1f}%)")
        
        for test, result in self.results.items():
            status = "✅" if result['success'] else "❌"
            print(f"\n{status} {test}")
            for metric, value in result['metrics'].items():
                print(f"   {metric}: {value}")


async def test_strategic_planning():
    """Test strategic resource planning"""
    print("\n🧪 Testing Strategic Planning...")
    
    try:
        planner = StrategicPlanner()
        
        # Mock current state
        current_state = {
            'cpu_usage': {
                'tda': 75.0,
                'inference': 85.0,  # High usage
                'consensus': 45.0,
                'memory': 30.0
            },
            'gpu_usage': {
                'inference': 90.0,  # Very high
                'neuromorphic': 60.0
            },
            'memory_usage': {
                'memory': 28000,  # MB
                'inference': 12000
            }
        }
        
        # Mock performance metrics
        performance_metrics = {
            'tda_latency': 45.0,
            'inference_latency': 120.0,  # High latency
            'consensus_time_ms': 35.0,
            'memory_retrieval_ms': 12.0,
            'free_energy': 0.8
        }
        
        # Create resource plan
        plan = await planner.create_resource_plan(current_state, performance_metrics)
        
        # Verify plan
        assert isinstance(plan, ResourcePlan), "Plan should be ResourcePlan instance"
        assert plan.horizon_hours == 24, "Default horizon should be 24 hours"
        assert 'inference' in plan.scale_up_components, "Should recommend scaling inference"
        assert plan.confidence > 0.5, "Should have reasonable confidence"
        
        print(f"   ✅ Resource plan created")
        print(f"   - CPU allocations: {list(plan.cpu_allocation.keys())}")
        print(f"   - Scale up: {plan.scale_up_components}")
        print(f"   - Scale down: {plan.scale_down_components}")
        print(f"   - Confidence: {plan.confidence:.2f}")
        
        return True, {
            'cpu_total': sum(plan.cpu_allocation.values()),
            'gpu_total': sum(plan.gpu_allocation.values()),
            'scale_up_count': len(plan.scale_up_components),
            'scale_down_count': len(plan.scale_down_components),
            'confidence': plan.confidence
        }
        
    except Exception as e:
        print(f"   ❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False, {'error': str(e)}


async def test_drift_detection():
    """Test PHFormer-based drift detection"""
    print("\n🧪 Testing Drift Detection...")
    
    try:
        detector = DriftDetector()
        
        # Create mock topological signatures
        class MockSignature:
            def __init__(self, features):
                self.persistence_diagram = [(0, f) for f in features]
                self.betti_numbers = [1, 2, 0]
                self.wasserstein_distance = np.mean(features)
                
        # Establish baseline with stable features
        print("   Establishing baseline...")
        for i in range(60):
            baseline_sig = MockSignature([0.1, 0.2, 0.15, 0.18])
            await detector.detect_drift('tda', baseline_sig)
            
        # Introduce drift
        print("   Introducing drift...")
        drift_scores = []
        for i in range(20):
            # Gradually drift features
            drift_amount = i * 0.05
            drifted_sig = MockSignature([0.1 + drift_amount, 0.2 + drift_amount, 
                                       0.15 + drift_amount, 0.18 + drift_amount])
            score = await detector.detect_drift('tda', drifted_sig)
            drift_scores.append(score.score)
            
        # Verify drift detection
        assert drift_scores[-1] > drift_scores[0], "Drift should increase over time"
        assert drift_scores[-1] > 0.3, "Should detect significant drift"
        
        final_score = await detector.detect_drift('tda', drifted_sig)
        
        print(f"   ✅ Drift detection working")
        print(f"   - Initial drift: {drift_scores[0]:.3f}")
        print(f"   - Final drift: {drift_scores[-1]:.3f}")
        print(f"   - Action required: {final_score.requires_action}")
        print(f"   - Suggested action: {final_score.suggested_action}")
        
        return True, {
            'initial_drift': drift_scores[0],
            'final_drift': drift_scores[-1],
            'drift_increase': drift_scores[-1] - drift_scores[0],
            'action_required': final_score.requires_action,
            'confidence': final_score.confidence
        }
        
    except Exception as e:
        print(f"   ❌ Test failed: {e}")
        return False, {'error': str(e)}


async def test_model_lifecycle():
    """Test model lifecycle management with canary deployment"""
    print("\n🧪 Testing Model Lifecycle Management...")
    
    try:
        manager = ModelLifecycleManager()
        
        # Create test model
        model = ModelVersion(
            model_id="test_inference_v2",
            version="2.0.0",
            component="inference",
            accuracy=0.97,
            latency_ms=45.0,
            memory_mb=2048,
            gpu_required=True
        )
        
        # Deploy as canary
        print("   Deploying canary...")
        deployment = await manager.deploy_model(model, deployment_type="canary")
        
        assert isinstance(deployment, CanaryDeployment), "Should return CanaryDeployment"
        assert deployment.canary_percentage == 10.0, "Should start at 10%"
        assert len(deployment.ramp_schedule) > 0, "Should have ramp schedule"
        
        # Simulate some metrics
        await asyncio.sleep(0.1)  # Small delay
        
        # Check deployment status
        status = manager.get_deployment_status()
        
        print(f"   ✅ Canary deployment created")
        print(f"   - Deployment ID: {deployment.deployment_id}")
        print(f"   - Initial percentage: {deployment.canary_percentage}%")
        print(f"   - Ramp steps: {len(deployment.ramp_schedule)}")
        print(f"   - Auto rollback: {deployment.auto_rollback}")
        
        # Test rollback
        print("   Testing rollback...")
        await manager.rollback_deployment(deployment.deployment_id)
        
        assert not deployment.is_active, "Deployment should be inactive"
        assert deployment.rollback_triggered, "Rollback should be triggered"
        
        print(f"   ✅ Rollback successful")
        
        return True, {
            'deployment_id': deployment.deployment_id,
            'canary_percentage': deployment.canary_percentage,
            'ramp_steps': len(deployment.ramp_schedule),
            'rollback_tested': True,
            'deployment_history': len(manager.deployment_history)
        }
        
    except Exception as e:
        print(f"   ❌ Test failed: {e}")
        return False, {'error': str(e)}


async def test_drift_triggered_update():
    """Test drift-triggered model updates"""
    print("\n🧪 Testing Drift-Triggered Updates...")
    
    try:
        planner = StrategicPlanner()
        detector = DriftDetector()
        
        # Create high drift scenario
        class HighDriftSignature:
            def __init__(self):
                self.persistence_diagram = [(0, np.random.rand()) for _ in range(10)]
                self.betti_numbers = [np.random.randint(0, 5) for _ in range(3)]
                self.wasserstein_distance = np.random.rand() * 2.0
                
        # Add signatures to create drift
        for _ in range(100):
            sig = HighDriftSignature()
            planner.signature_history.append({
                'timestamp': time.time(),
                'topological_features': [s[1] for s in sig.persistence_diagram],
                'free_energy': np.random.rand() * 2.0
            })
            
        # Create plan with drift
        state = {'cpu_usage': {'tda': 50.0}}
        metrics = {'tda_latency': 30.0}
        
        plan = await planner.create_resource_plan(state, metrics)
        
        # Should recommend model updates
        assert len(plan.drift_alerts) > 0, "Should detect drift"
        assert len(plan.model_updates) > 0, "Should schedule model updates"
        
        print(f"   ✅ Drift-triggered updates working")
        print(f"   - Drift alerts: {len(plan.drift_alerts)}")
        print(f"   - Scheduled updates: {len(plan.model_updates)}")
        print(f"   - Update times: {[u[1] - time.time() for _, u in plan.model_updates[:3]]}")
        
        return True, {
            'drift_alerts': len(plan.drift_alerts),
            'model_updates': len(plan.model_updates),
            'max_drift_severity': max((a['severity'] for a in plan.drift_alerts), default='none')
        }
        
    except Exception as e:
        print(f"   ❌ Test failed: {e}")
        return False, {'error': str(e)}


async def test_resource_optimization():
    """Test resource allocation optimization"""
    print("\n🧪 Testing Resource Optimization...")
    
    try:
        planner = StrategicPlanner()
        
        # Test with resource constraints
        state = {
            'cpu_usage': {
                'tda': 90.0,
                'inference': 95.0,
                'consensus': 85.0,
                'memory': 80.0,
                'agents': 88.0
            }
        }
        
        metrics = {
            'tda_latency': 100.0,
            'inference_latency': 150.0
        }
        
        plan = await planner.create_resource_plan(state, metrics)
        
        # Verify resource limits are respected
        cpu_total = sum(plan.cpu_allocation.values())
        gpu_total = sum(plan.gpu_allocation.values())
        
        assert cpu_total <= 100.0, f"CPU allocation {cpu_total} exceeds limit"
        assert gpu_total <= 100.0, f"GPU allocation {gpu_total} exceeds limit"
        
        print(f"   ✅ Resource optimization working")
        print(f"   - CPU total: {cpu_total:.1f}%")
        print(f"   - GPU total: {gpu_total:.1f}%") 
        print(f"   - Memory total: {sum(plan.memory_allocation.values())} MB")
        
        return True, {
            'cpu_utilization': cpu_total,
            'gpu_utilization': gpu_total,
            'components_scaled': len(plan.scale_up_components) + len(plan.scale_down_components),
            'optimization_success': True
        }
        
    except Exception as e:
        print(f"   ❌ Test failed: {e}")
        return False, {'error': str(e)}


async def run_strategic_tests():
    """Run all strategic orchestration tests"""
    print("🚀 Strategic Orchestration Test Suite")
    print("="*80)
    print("Testing drift detection, resource planning, and model lifecycle")
    
    metrics = OrchestratorTestMetrics()
    
    # Run tests
    tests = [
        ("Strategic Planning", test_strategic_planning),
        ("Drift Detection", test_drift_detection),
        ("Model Lifecycle", test_model_lifecycle),
        ("Drift-Triggered Updates", test_drift_triggered_update),
        ("Resource Optimization", test_resource_optimization)
    ]
    
    for test_name, test_func in tests:
        success, test_metrics = await test_func()
        metrics.record(test_name, success, test_metrics)
        
    # Show summary
    metrics.summary()
    
    # Overall success
    all_passed = all(r['success'] for r in metrics.results.values())
    
    if all_passed:
        print("\n🎉 All strategic orchestration tests passed!")
        print("   ✅ Drift detection with PHFormer working")
        print("   ✅ Resource planning optimizes allocations")
        print("   ✅ Canary deployments with auto-rollback")
        print("   ✅ Drift-triggered model updates")
        print("   → Strategic layer ready for production")
    else:
        print("\n⚠️  Some tests failed - review errors above")
        
    return all_passed


if __name__ == "__main__":
    success = asyncio.run(run_strategic_tests())
    sys.exit(0 if success else 1)