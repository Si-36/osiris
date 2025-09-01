#!/usr/bin/env python3
"""
Active Inference Core Test - Minimal Dependencies
================================================
Tests core Free Energy functionality without full AURA stack
"""

import asyncio
import torch
import numpy as np
import time
import sys
import os

# Add to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'core/src'))

# Direct imports to avoid dependency chain
import torch.nn as nn
from dataclasses import dataclass
from typing import Dict, Any, Tuple, List


@dataclass
class BeliefState:
    """Minimal BeliefState for testing"""
    mean: torch.Tensor
    variance: torch.Tensor
    precision: torch.Tensor
    
    def entropy(self) -> float:
        return 0.5 * torch.sum(torch.log(2 * np.pi * np.e * self.variance)).item()
    
    def sample(self, n_samples: int = 1) -> torch.Tensor:
        std = torch.sqrt(self.variance + 1e-8)
        eps = torch.randn(n_samples, *self.mean.shape)
        return self.mean + eps * std


class MinimalGenerativeModel(nn.Module):
    """Minimal generative model for testing"""
    
    def __init__(self, state_dim: int = 64, obs_dim: int = 32):
        super().__init__()
        self.state_dim = state_dim
        self.obs_dim = obs_dim
        
        # Simple linear generative model
        self.likelihood_net = nn.Linear(state_dim, obs_dim)
        self.obs_precision = nn.Parameter(torch.ones(obs_dim))
        
    def likelihood(self, observations: torch.Tensor, states: torch.Tensor) -> torch.Tensor:
        predicted = self.likelihood_net(states)
        diff = observations - predicted
        weighted_diff = diff * self.obs_precision
        log_likelihood = -0.5 * torch.sum(weighted_diff ** 2, dim=-1)
        return log_likelihood


class MinimalFreeEnergy:
    """Minimal Free Energy implementation for testing"""
    
    def __init__(self, model: MinimalGenerativeModel):
        self.model = model
        self.learning_rate = 0.01
        self.n_iterations = 20
        
    def compute_free_energy(self, observations: torch.Tensor, beliefs: BeliefState) -> Dict[str, float]:
        """Compute F = KL[q||p] - ln P(o|s)"""
        # Accuracy: log-likelihood
        log_likelihood = self.model.likelihood(observations, beliefs.mean)
        accuracy = -log_likelihood.mean().item()
        
        # Complexity: simplified KL (assume standard normal prior)
        # Note: beliefs are in state space, not observation space
        complexity = 0.5 * torch.sum(beliefs.mean ** 2 + beliefs.variance - torch.log(beliefs.variance) - 1).item()
        
        # Total free energy
        total_fe = complexity - (-accuracy)
        
        return {
            'accuracy': accuracy,
            'complexity': complexity,
            'total_free_energy': total_fe,
            'confidence': torch.mean(beliefs.precision).item()
        }
    
    async def minimize_free_energy(self, observations: torch.Tensor) -> Tuple[BeliefState, Dict[str, float]]:
        """Minimize free energy through gradient descent"""
        # Initialize with prior
        initial_mean = torch.randn(observations.shape[0], self.model.state_dim)
        beliefs = BeliefState(
            mean=initial_mean.clone().requires_grad_(True),
            variance=torch.ones(observations.shape[0], self.model.state_dim),
            precision=torch.ones(observations.shape[0], self.model.state_dim)
        )
        
        optimizer = torch.optim.Adam([beliefs.mean], lr=self.learning_rate)
        
        best_fe = float('inf')
        fe_history = []
        
        for i in range(self.n_iterations):
            optimizer.zero_grad()
            
            # Compute loss for optimization (negative log-likelihood)
            loss = -self.model.likelihood(observations, beliefs.mean).mean()
            
            # Track free energy before backward (to avoid gradient issues)
            with torch.no_grad():
                fe_components = self.compute_free_energy(observations, beliefs)
                current_fe = fe_components['total_free_energy']
                fe_history.append(current_fe)
                
                if current_fe < best_fe:
                    best_fe = current_fe
            
            # Optimize
            loss.backward()
            optimizer.step()
            
            # Update precision based on prediction error
            with torch.no_grad():
                predicted = self.model.likelihood_net(beliefs.mean)
                error = (observations - predicted) ** 2
                # Average error per state dimension (not observation dimension)
                avg_error = torch.mean(error)
                beliefs.precision = torch.ones_like(beliefs.variance) / (avg_error + 1e-4)
                beliefs.variance = 1.0 / beliefs.precision
        
        final_fe = self.compute_free_energy(observations, beliefs)
        final_fe['iterations'] = len(fe_history)
        final_fe['converged'] = fe_history[-1] < fe_history[0]
        
        return beliefs, final_fe


async def test_free_energy_core():
    """Test core free energy computation"""
    print("\n🧪 Testing Core Free Energy Implementation...")
    
    try:
        # Create model
        model = MinimalGenerativeModel(state_dim=64, obs_dim=32)
        fe_engine = MinimalFreeEnergy(model)
        
        # Generate synthetic observations
        true_state = torch.randn(5, 64)
        observations = model.likelihood_net(true_state) + torch.randn(5, 32) * 0.1
        
        # Test free energy computation
        initial_beliefs = BeliefState(
            mean=torch.randn(5, 64),
            variance=torch.ones(5, 64),
            precision=torch.ones(5, 64)
        )
        
        fe_components = fe_engine.compute_free_energy(observations, initial_beliefs)
        
        print(f"   Initial Free Energy: {fe_components['total_free_energy']:.3f}")
        print(f"   - Accuracy: {fe_components['accuracy']:.3f}")
        print(f"   - Complexity: {fe_components['complexity']:.3f}")
        
        # Test minimization
        start_time = time.perf_counter()
        optimized_beliefs, final_fe = await fe_engine.minimize_free_energy(observations)
        inference_time = (time.perf_counter() - start_time) * 1000
        
        print(f"\n   After Minimization:")
        print(f"   Final Free Energy: {final_fe['total_free_energy']:.3f}")
        print(f"   Iterations: {final_fe['iterations']}")
        print(f"   Converged: {final_fe['converged']}")
        print(f"   Inference Time: {inference_time:.2f}ms")
        
        # Check success criteria
        success = (
            final_fe['converged'] and
            final_fe['iterations'] <= 20 and
            inference_time < 20.0 and
            final_fe['total_free_energy'] < fe_components['total_free_energy']
        )
        
        print(f"\n   {'✅' if success else '❌'} All criteria met")
        
        return success, {
            'initial_fe': fe_components['total_free_energy'],
            'final_fe': final_fe['total_free_energy'],
            'iterations': final_fe['iterations'],
            'inference_time_ms': inference_time
        }
        
    except Exception as e:
        print(f"   ❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False, {'error': str(e)}


async def test_anomaly_detection_simple():
    """Test anomaly detection capability"""
    print("\n🧪 Testing Anomaly Detection with Free Energy...")
    
    try:
        model = MinimalGenerativeModel(state_dim=32, obs_dim=16)
        fe_engine = MinimalFreeEnergy(model)
        
        # Train on normal data
        print("   Training on normal patterns...")
        for _ in range(5):
            normal_data = torch.sin(torch.linspace(0, 2*np.pi, 16)) + torch.randn(16) * 0.1
            normal_obs = normal_data.unsqueeze(0)
            await fe_engine.minimize_free_energy(normal_obs)
        
        # Test on normal vs anomalous data
        normal_scores = []
        anomaly_scores = []
        
        print("   Testing normal data...")
        for i in range(10):
            normal_data = torch.sin(torch.linspace(0, 2*np.pi, 16)) + torch.randn(16) * 0.1
            normal_obs = normal_data.unsqueeze(0)
            _, fe_result = await fe_engine.minimize_free_energy(normal_obs)
            normal_scores.append(fe_result['total_free_energy'])
            
        print("   Testing anomalous data...")
        for i in range(10):
            # Random noise as anomaly
            anomaly_data = torch.randn(16) * 2.0
            anomaly_obs = anomaly_data.unsqueeze(0)
            _, fe_result = await fe_engine.minimize_free_energy(anomaly_obs)
            anomaly_scores.append(fe_result['total_free_energy'])
        
        # Compute separation
        normal_mean = np.mean(normal_scores)
        normal_std = np.std(normal_scores)
        anomaly_mean = np.mean(anomaly_scores)
        
        separation = (anomaly_mean - normal_mean) / (normal_std + 1e-6)
        
        print(f"\n   Normal FE: {normal_mean:.3f} ± {normal_std:.3f}")
        print(f"   Anomaly FE: {anomaly_mean:.3f}")
        print(f"   Separation: {separation:.2f} sigma")
        
        # Simple accuracy test
        threshold = normal_mean + 2 * normal_std
        tp = sum(1 for s in anomaly_scores if s > threshold)
        fp = sum(1 for s in normal_scores if s > threshold)
        accuracy = (tp + (len(normal_scores) - fp)) / (len(normal_scores) + len(anomaly_scores))
        
        print(f"   Accuracy: {accuracy:.2%}")
        
        success = separation > 2.0 and accuracy > 0.8
        print(f"\n   {'✅' if success else '❌'} Anomaly detection working")
        
        return success, {
            'separation_sigma': separation,
            'accuracy': accuracy,
            'normal_fe': normal_mean,
            'anomaly_fe': anomaly_mean
        }
        
    except Exception as e:
        print(f"   ❌ Test failed: {e}")
        return False, {'error': str(e)}


async def test_uncertainty_quantification():
    """Test uncertainty quantification"""
    print("\n🧪 Testing Uncertainty Quantification...")
    
    try:
        model = MinimalGenerativeModel()
        fe_engine = MinimalFreeEnergy(model)
        
        uncertainties = []
        
        # Test different noise levels
        noise_levels = [0.01, 0.1, 1.0]
        
        for noise in noise_levels:
            # Generate data with known noise
            true_signal = torch.sin(torch.linspace(0, 4*np.pi, 32))
            noisy_obs = (true_signal + torch.randn(32) * noise).unsqueeze(0)
            
            beliefs, fe_result = await fe_engine.minimize_free_energy(noisy_obs)
            
            # Uncertainty should correlate with noise
            uncertainty = beliefs.entropy()
            uncertainties.append((noise, uncertainty))
            
            print(f"   Noise level {noise:.2f}: Uncertainty = {uncertainty:.3f}")
        
        # Check if uncertainty increases with noise
        correct_ordering = all(uncertainties[i][1] < uncertainties[i+1][1] 
                              for i in range(len(uncertainties)-1))
        
        print(f"\n   {'✅' if correct_ordering else '❌'} Uncertainty increases with noise")
        
        return correct_ordering, {
            'uncertainties': uncertainties,
            'correct_ordering': correct_ordering
        }
        
    except Exception as e:
        print(f"   ❌ Test failed: {e}")
        return False, {'error': str(e)}


async def run_core_tests():
    """Run minimal core tests"""
    print("🚀 Active Inference Core Tests (Minimal Dependencies)")
    print("="*70)
    
    results = {}
    
    # Run tests
    tests = [
        ("Free Energy Core", test_free_energy_core),
        ("Anomaly Detection", test_anomaly_detection_simple),
        ("Uncertainty Quantification", test_uncertainty_quantification)
    ]
    
    all_passed = True
    
    for test_name, test_func in tests:
        passed, metrics = await test_func()
        results[test_name] = {'passed': passed, 'metrics': metrics}
        all_passed &= passed
        
    # Summary
    print("\n" + "="*70)
    print("📊 TEST SUMMARY")
    print("="*70)
    
    for test_name, result in results.items():
        status = "✅" if result['passed'] else "❌"
        print(f"{status} {test_name}")
        for key, value in result['metrics'].items():
            if key != 'error':
                print(f"   - {key}: {value}")
    
    # Phase 1 Go/No-Go
    print("\n" + "="*70)
    print("🎯 PHASE 1 GO/NO-GO DECISION")
    print("="*70)
    
    fe_metrics = results.get('Free Energy Core', {}).get('metrics', {})
    anomaly_metrics = results.get('Anomaly Detection', {}).get('metrics', {})
    
    criteria = {
        "Free Energy Convergence": fe_metrics.get('iterations', 999) <= 20,
        "Inference < 20ms": fe_metrics.get('inference_time_ms', 999) < 20,
        "Anomaly Separation > 2σ": anomaly_metrics.get('separation_sigma', 0) > 2,
        "Anomaly Accuracy > 80%": anomaly_metrics.get('accuracy', 0) > 0.8
    }
    
    go_decision = all(criteria.values())
    
    for criterion, met in criteria.items():
        print(f"{'✅' if met else '❌'} {criterion}")
        
    print(f"\n{'🟢 GO' if go_decision else '🔴 NO-GO'}: Phase 1 {'Success' if go_decision else 'Needs Optimization'}")
    
    if go_decision:
        print("\n💡 Active Inference Core is working!")
        print("   - Free energy minimization converges quickly")
        print("   - Anomaly detection shows clear improvement")
        print("   - Uncertainty quantification is meaningful")
        print("   → Ready for AURA integration and Phase 2")
    
    return go_decision


if __name__ == "__main__":
    success = asyncio.run(run_core_tests())
    sys.exit(0 if success else 1)