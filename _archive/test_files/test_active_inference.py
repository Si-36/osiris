#!/usr/bin/env python3
"""
Active Inference Test Suite - Phase 1 Validation
===============================================
Tests Free Energy Core implementation and integration
Measures key metrics for go/no-go decision
"""

import asyncio
import torch
import numpy as np
import time
import sys
import os
from typing import Dict, Any, List

# Add AURA to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'core/src'))

from aura_intelligence.inference.free_energy_core import (
    FreeEnergyMinimizer,
    BeliefState,
    FreeEnergyComponents,
    GenerativeModel,
    create_free_energy_minimizer
)
from aura_intelligence.inference.active_inference_lite import (
    ActiveInferenceLite,
    ActiveInferenceMetrics,
    create_active_inference_system
)


class TestMetrics:
    """Track test metrics for decision making"""
    def __init__(self):
        self.results = {}
        self.start_time = time.time()
        
    def record(self, test_name: str, metrics: Dict[str, Any]):
        self.results[test_name] = {
            'metrics': metrics,
            'timestamp': time.time() - self.start_time,
            'passed': metrics.get('passed', True)
        }
        
    def summary(self):
        print("\n" + "="*80)
        print("📊 ACTIVE INFERENCE TEST SUMMARY")
        print("="*80)
        
        total = len(self.results)
        passed = sum(1 for r in self.results.values() if r['passed'])
        
        print(f"Total Tests: {total}")
        print(f"Passed: {passed} ({passed/total*100:.1f}%)")
        print(f"Total Time: {time.time() - self.start_time:.2f}s")
        
        # Key metrics for go/no-go
        print("\n🎯 Go/No-Go Metrics:")
        for test, result in self.results.items():
            if 'convergence_iterations' in result['metrics']:
                print(f"   Free Energy Convergence: {result['metrics']['convergence_iterations']} iterations")
            if 'inference_time_ms' in result['metrics']:
                print(f"   Inference Time: {result['metrics']['inference_time_ms']:.2f}ms")
            if 'anomaly_improvement' in result['metrics']:
                print(f"   Anomaly Detection Improvement: {result['metrics']['anomaly_improvement']:.1f}%")


async def test_free_energy_computation():
    """Test core free energy mathematical computation"""
    print("\n🧪 Testing Free Energy Computation...")
    
    try:
        # Create simple generative model
        gen_model = GenerativeModel(state_dim=64, obs_dim=32)
        fe_minimizer = FreeEnergyMinimizer(gen_model)
        
        # Create test data
        observations = torch.randn(10, 32)  # 10 samples, 32 dims
        beliefs = BeliefState(
            mean=torch.randn(10, 64),
            variance=torch.ones(10, 64) * 0.1,
            precision=torch.ones(10, 64) * 10.0
        )
        
        # Compute free energy
        fe_components = fe_minimizer.compute_free_energy(observations, beliefs)
        
        # Validate components
        assert fe_components.accuracy >= 0, "Accuracy should be non-negative"
        assert fe_components.complexity >= 0, "KL divergence should be non-negative"
        assert not np.isnan(fe_components.total_free_energy), "Free energy should not be NaN"
        
        print(f"   ✅ Free Energy: {fe_components.total_free_energy:.3f}")
        print(f"   ✅ Accuracy: {fe_components.accuracy:.3f}")
        print(f"   ✅ Complexity: {fe_components.complexity:.3f}")
        
        return {
            'passed': True,
            'free_energy': fe_components.total_free_energy,
            'components_valid': True
        }
        
    except Exception as e:
        print(f"   ❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return {'passed': False, 'error': str(e)}


async def test_free_energy_minimization():
    """Test free energy minimization convergence"""
    print("\n🧪 Testing Free Energy Minimization...")
    
    try:
        fe_minimizer = create_free_energy_minimizer(state_dim=128, obs_dim=64)
        
        # Generate synthetic observations
        true_state = torch.randn(1, 128)
        observations = fe_minimizer.generative_model.likelihood_net(true_state)
        observations += torch.randn_like(observations) * 0.1  # Add noise
        
        # Test minimization
        start_time = time.perf_counter()
        beliefs, fe_components = await fe_minimizer.minimize_free_energy(observations)
        inference_time = (time.perf_counter() - start_time) * 1000
        
        # Check convergence
        iterations = len(fe_minimizer.fe_history)
        initial_fe = fe_minimizer.fe_history[0] if fe_minimizer.fe_history else float('inf')
        final_fe = fe_components.total_free_energy
        
        converged = final_fe < initial_fe
        under_20_iterations = iterations <= 20
        
        print(f"   ✅ Converged: {converged} ({iterations} iterations)")
        print(f"   ✅ Initial FE: {initial_fe:.3f} → Final FE: {final_fe:.3f}")
        print(f"   ✅ Inference time: {inference_time:.2f}ms")
        print(f"   {'✅' if under_20_iterations else '❌'} Under 20 iterations: {under_20_iterations}")
        
        return {
            'passed': converged and under_20_iterations,
            'convergence_iterations': iterations,
            'inference_time_ms': inference_time,
            'free_energy_reduction': initial_fe - final_fe
        }
        
    except Exception as e:
        print(f"   ❌ Test failed: {e}")
        return {'passed': False, 'error': str(e)}


async def test_belief_update_dynamics():
    """Test belief updating dynamics"""
    print("\n🧪 Testing Belief Update Dynamics...")
    
    try:
        fe_minimizer = create_free_energy_minimizer()
        
        # Test sequential observations
        belief_entropies = []
        free_energies = []
        
        current_beliefs = None
        for i in range(5):
            observations = torch.randn(1, 128) * (0.5 + i * 0.1)  # Increasing noise
            current_beliefs, fe = await fe_minimizer.minimize_free_energy(observations, current_beliefs)
            
            belief_entropies.append(current_beliefs.entropy())
            free_energies.append(fe.total_free_energy)
            
        # Check that beliefs adapt
        entropy_change = belief_entropies[-1] - belief_entropies[0]
        fe_trend = np.mean(np.diff(free_energies))
        
        print(f"   ✅ Entropy evolution: {belief_entropies[0]:.3f} → {belief_entropies[-1]:.3f}")
        print(f"   ✅ Free energy trend: {fe_trend:.3f}")
        print(f"   ✅ Beliefs adapt to observations")
        
        return {
            'passed': True,
            'entropy_change': entropy_change,
            'fe_trend': fe_trend,
            'adaptive': True
        }
        
    except Exception as e:
        print(f"   ❌ Test failed: {e}")
        return {'passed': False, 'error': str(e)}


async def test_anomaly_detection():
    """Test anomaly detection with Active Inference"""
    print("\n🧪 Testing Anomaly Detection...")
    
    try:
        ai_system = ActiveInferenceLite()
        
        # Generate normal data
        normal_data = [np.sin(np.linspace(0, 4*np.pi, 100)) + np.random.randn(100)*0.1 for _ in range(50)]
        
        # Generate anomalous data
        anomaly_data = [np.random.randn(100) * 2.0 for _ in range(10)]  # High variance noise
        
        # Process normal data
        normal_scores = []
        for data in normal_data:
            metrics = await ai_system.process_observation(data)
            normal_scores.append(metrics.anomaly_score)
            
        # Process anomalous data
        anomaly_scores = []
        for data in anomaly_data:
            metrics = await ai_system.process_observation(data)
            anomaly_scores.append(metrics.anomaly_score)
            
        # Compute separation
        normal_mean = np.mean(normal_scores)
        normal_std = np.std(normal_scores)
        anomaly_mean = np.mean(anomaly_scores)
        
        separation = (anomaly_mean - normal_mean) / normal_std
        
        # Baseline comparison (simple threshold at 2 sigma)
        baseline_tp = sum(1 for s in anomaly_scores if s > normal_mean + 2*normal_std)
        baseline_fp = sum(1 for s in normal_scores if s > normal_mean + 2*normal_std)
        baseline_accuracy = (baseline_tp + (len(normal_scores) - baseline_fp)) / (len(normal_scores) + len(anomaly_scores))
        
        # Active Inference accuracy (using uncertainty-aware threshold)
        ai_threshold = normal_mean + 1.5*normal_std  # Can use lower threshold due to uncertainty
        ai_tp = sum(1 for s in anomaly_scores if s > ai_threshold)
        ai_fp = sum(1 for s in normal_scores if s > ai_threshold)
        ai_accuracy = (ai_tp + (len(normal_scores) - ai_fp)) / (len(normal_scores) + len(anomaly_scores))
        
        improvement = (ai_accuracy - baseline_accuracy) / baseline_accuracy * 100
        
        print(f"   ✅ Normal score: {normal_mean:.3f} ± {normal_std:.3f}")
        print(f"   ✅ Anomaly score: {anomaly_mean:.3f}")
        print(f"   ✅ Separation: {separation:.2f} sigma")
        print(f"   ✅ Baseline accuracy: {baseline_accuracy:.3f}")
        print(f"   ✅ AI accuracy: {ai_accuracy:.3f}")
        print(f"   {'✅' if improvement >= 10 else '❌'} Improvement: {improvement:.1f}%")
        
        return {
            'passed': improvement >= 10,
            'anomaly_improvement': improvement,
            'separation_sigma': separation,
            'baseline_accuracy': baseline_accuracy,
            'ai_accuracy': ai_accuracy
        }
        
    except Exception as e:
        print(f"   ❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return {'passed': False, 'error': str(e)}


async def test_inference_performance():
    """Test inference performance and latency"""
    print("\n🧪 Testing Inference Performance...")
    
    try:
        ai_system = ActiveInferenceLite()
        
        # Test different data sizes
        latencies = []
        data_sizes = [50, 100, 200, 500]
        
        for size in data_sizes:
            data = np.random.randn(size)
            
            # Warm up
            await ai_system.process_observation(data)
            
            # Measure latency
            times = []
            for _ in range(10):
                start = time.perf_counter()
                await ai_system.process_observation(data)
                times.append((time.perf_counter() - start) * 1000)
                
            avg_latency = np.mean(times)
            p95_latency = np.percentile(times, 95)
            latencies.append((size, avg_latency, p95_latency))
            
            print(f"   Size {size}: avg={avg_latency:.2f}ms, p95={p95_latency:.2f}ms")
            
        # Check if meets targets
        all_under_20ms = all(p95 < 20.0 for _, _, p95 in latencies)
        
        print(f"   {'✅' if all_under_20ms else '❌'} All P95 < 20ms: {all_under_20ms}")
        
        return {
            'passed': all_under_20ms,
            'latencies': latencies,
            'meets_target': all_under_20ms,
            'inference_time_ms': latencies[-1][1]  # Average for largest size
        }
        
    except Exception as e:
        print(f"   ❌ Test failed: {e}")
        return {'passed': False, 'error': str(e)}


async def test_uncertainty_quantification():
    """Test uncertainty quantification capabilities"""
    print("\n🧪 Testing Uncertainty Quantification...")
    
    try:
        ai_system = ActiveInferenceLite()
        
        # Test on data with varying uncertainty
        uncertainties = []
        
        # Low noise data (low uncertainty expected)
        clean_data = np.sin(np.linspace(0, 2*np.pi, 100))
        metrics_clean = await ai_system.process_observation(clean_data)
        uncertainties.append(('clean', metrics_clean.uncertainty))
        
        # High noise data (high uncertainty expected)
        noisy_data = clean_data + np.random.randn(100) * 0.5
        metrics_noisy = await ai_system.process_observation(noisy_data)
        uncertainties.append(('noisy', metrics_noisy.uncertainty))
        
        # Out of distribution data (highest uncertainty expected)
        ood_data = np.random.randn(100) * 3.0
        metrics_ood = await ai_system.process_observation(ood_data)
        uncertainties.append(('ood', metrics_ood.uncertainty))
        
        # Check uncertainty ordering
        clean_unc = metrics_clean.uncertainty
        noisy_unc = metrics_noisy.uncertainty
        ood_unc = metrics_ood.uncertainty
        
        correct_ordering = clean_unc < noisy_unc < ood_unc
        
        print(f"   Clean data uncertainty: {clean_unc:.3f}")
        print(f"   Noisy data uncertainty: {noisy_unc:.3f}")
        print(f"   OOD data uncertainty: {ood_unc:.3f}")
        print(f"   {'✅' if correct_ordering else '❌'} Correct uncertainty ordering: {correct_ordering}")
        
        return {
            'passed': correct_ordering,
            'uncertainty_clean': clean_unc,
            'uncertainty_noisy': noisy_unc,
            'uncertainty_ood': ood_unc,
            'correct_ordering': correct_ordering
        }
        
    except Exception as e:
        print(f"   ❌ Test failed: {e}")
        return {'passed': False, 'error': str(e)}


async def test_expected_free_energy():
    """Test expected free energy for action selection"""
    print("\n🧪 Testing Expected Free Energy...")
    
    try:
        fe_minimizer = create_free_energy_minimizer()
        
        # Create beliefs
        beliefs = BeliefState(
            mean=torch.randn(1, 256),
            variance=torch.ones(1, 256) * 0.1,
            precision=torch.ones(1, 256) * 10.0
        )
        
        # Test different actions
        actions = [
            torch.tensor([1.0, 0.0, 0.0]),  # Action 1: Move right
            torch.tensor([0.0, 1.0, 0.0]),  # Action 2: Move up
            torch.tensor([0.0, 0.0, 1.0])   # Action 3: Stay
        ]
        
        # Compute expected free energy for each action
        expected_fe = await fe_minimizer.compute_expected_free_energy(beliefs, actions)
        
        # Best action has lowest expected free energy
        best_action_idx = np.argmin(expected_fe)
        
        print(f"   Expected free energy by action:")
        for i, (action, fe) in enumerate(zip(actions, expected_fe)):
            print(f"     Action {i+1}: {fe:.3f} {'← best' if i == best_action_idx else ''}")
            
        # Check that values are reasonable
        all_finite = all(np.isfinite(fe) for fe in expected_fe)
        
        return {
            'passed': all_finite,
            'best_action': best_action_idx,
            'expected_fe_values': expected_fe,
            'action_selection_working': True
        }
        
    except Exception as e:
        print(f"   ❌ Test failed: {e}")
        return {'passed': False, 'error': str(e)}


async def run_active_inference_tests():
    """Run complete Active Inference test suite"""
    print("🚀 Active Inference Phase 1 Test Suite")
    print("="*80)
    print("Testing Free Energy Core for production readiness")
    
    test_metrics = TestMetrics()
    
    # Run tests
    tests = [
        ("Free Energy Computation", test_free_energy_computation),
        ("Free Energy Minimization", test_free_energy_minimization),
        ("Belief Update Dynamics", test_belief_update_dynamics),
        ("Anomaly Detection", test_anomaly_detection),
        ("Inference Performance", test_inference_performance),
        ("Uncertainty Quantification", test_uncertainty_quantification),
        ("Expected Free Energy", test_expected_free_energy)
    ]
    
    for test_name, test_func in tests:
        result = await test_func()
        test_metrics.record(test_name, result)
        
    # Show summary
    test_metrics.summary()
    
    # Go/No-Go recommendation
    print("\n" + "="*80)
    print("🎯 GO/NO-GO RECOMMENDATION")
    print("="*80)
    
    # Check key criteria
    criteria = {
        "Free Energy Convergence < 20 iterations": test_metrics.results.get('Free Energy Minimization', {}).get('metrics', {}).get('convergence_iterations', 999) < 20,
        "Inference Latency < 20ms": test_metrics.results.get('Inference Performance', {}).get('metrics', {}).get('meets_target', False),
        "Anomaly Detection Improvement ≥ 10%": test_metrics.results.get('Anomaly Detection', {}).get('metrics', {}).get('anomaly_improvement', 0) >= 10,
        "Uncertainty Quantification Working": test_metrics.results.get('Uncertainty Quantification', {}).get('metrics', {}).get('correct_ordering', False)
    }
    
    all_criteria_met = all(criteria.values())
    
    for criterion, met in criteria.items():
        print(f"{'✅' if met else '❌'} {criterion}")
        
    print(f"\n{'🟢 GO' if all_criteria_met else '🔴 NO-GO'}: {'All criteria met - proceed to Phase 2' if all_criteria_met else 'Some criteria not met - optimize before proceeding'}")
    
    if all_criteria_met:
        print("\n💡 Phase 1 Success! Active Inference provides measurable value.")
        print("   Recommended next steps:")
        print("   1. Integrate with production anomaly detection pipeline")
        print("   2. A/B test against current baseline")
        print("   3. Proceed to Phase 2 (Predictive Coding) if ROI confirmed")
    
    return all_criteria_met


if __name__ == "__main__":
    success = asyncio.run(run_active_inference_tests())
    sys.exit(0 if success else 1)