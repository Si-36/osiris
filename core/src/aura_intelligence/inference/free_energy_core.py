"""
Free Energy Core Implementation for AURA Active Inference
=========================================================
Mathematical foundation for variational free energy minimization
Integrates with AURA's TDA features and existing components
Based on 2025 research with pragmatic production focus
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass, field
import time
from collections import deque
import asyncio

# AURA imports
try:
    from ..tda.persistence import TDAProcessor
except ImportError:
    from ..tda.persistence_simple import TDAProcessor
from ..memory.advanced_memory_system import HierarchicalMemorySystem as HierarchicalMemoryManager
from ..components.registry import get_registry

import logging
logger = logging.getLogger(__name__)


@dataclass
class BeliefState:
    """Maintains q(s) - approximate posterior distributions over latent states"""
    mean: torch.Tensor
    variance: torch.Tensor  # Diagonal covariance for efficiency
    precision: torch.Tensor  # Inverse variance (attention mechanism)
    timestamp: float = field(default_factory=time.time)
    
    def sample(self, n_samples: int = 1) -> torch.Tensor:
        """Sample from belief distribution"""
        std = torch.sqrt(self.variance + 1e-8)
        eps = torch.randn(n_samples, *self.mean.shape)
        return self.mean + eps * std
    
    def entropy(self) -> float:
        """Compute entropy of belief distribution"""
        # For diagonal Gaussian: H = 0.5 * sum(log(2πe * variance))
        return 0.5 * torch.sum(torch.log(2 * np.pi * np.e * self.variance)).item()


@dataclass 
class FreeEnergyComponents:
    """Mathematical decomposition of variational free energy"""
    accuracy: float  # -ln P(o|s) - How well beliefs explain observations
    complexity: float  # KL[q(s)||p(s)] - Deviation from prior beliefs
    total_free_energy: float  # F = complexity - accuracy
    
    # Expected free energy for action selection
    epistemic_value: float  # Information gain / uncertainty reduction
    pragmatic_value: float  # Goal achievement / preference satisfaction
    
    # Additional metrics for debugging
    prediction_error: torch.Tensor
    confidence: float


class GenerativeModel(nn.Module):
    """
    Generative model P(o|s) and P(s) for Active Inference
    Maps from latent states to observations
    """
    
    def __init__(self, state_dim: int = 256, obs_dim: int = 128, hidden_dim: int = 512):
        super().__init__()
        self.state_dim = state_dim
        self.obs_dim = obs_dim
        
        # Generative network: states -> observations
        self.likelihood_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, obs_dim)
        )
        
        # Prior network: learns state priors P(s)
        self.prior_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, state_dim * 2)  # Mean and log-variance
        )
        
        # Learned observation noise (precision)
        self.obs_precision = nn.Parameter(torch.ones(obs_dim))
        
    def likelihood(self, observations: torch.Tensor, states: BeliefState) -> torch.Tensor:
        """Compute P(o|s) - likelihood of observations given states"""
        # Generate predicted observations
        predicted_obs = self.likelihood_net(states.mean)
        
        # Compute log-likelihood under Gaussian noise model
        diff = observations - predicted_obs
        weighted_diff = diff * self.obs_precision
        log_likelihood = -0.5 * torch.sum(weighted_diff ** 2, dim=-1)
        
        return log_likelihood
    
    def prior(self, states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute P(s) - prior distribution over states"""
        prior_params = self.prior_net(states)
        prior_mean = prior_params[:, :self.state_dim]
        prior_log_var = prior_params[:, self.state_dim:]
        prior_var = torch.exp(prior_log_var) + 1e-8
        
        return prior_mean, prior_var


class FreeEnergyMinimizer:
    """
    Core Active Inference engine implementing Free Energy Principle
    F = KL[q(s)||p(s)] - ln P(o|s)
    """
    
    def __init__(self, generative_model: GenerativeModel, config: Optional[Dict] = None):
        self.generative_model = generative_model
        self.config = config or self._default_config()
        
        # Belief optimization
        self.learning_rate = self.config['learning_rate']
        self.n_iterations = self.config['n_iterations']
        
        # History for temporal smoothing
        self.belief_history = deque(maxlen=self.config['history_size'])
        self.fe_history = deque(maxlen=100)
        
        # Integration with AURA
        self.registry = get_registry()
        self.tda_processor = None  # Lazy load
        self.memory_manager = None  # Lazy load
        
        logger.info("Free Energy Minimizer initialized")
        
    def _default_config(self) -> Dict[str, Any]:
        return {
            'learning_rate': 0.01,
            'n_iterations': 20,  # Target: <20 iterations
            'history_size': 10,
            'min_variance': 1e-4,
            'epistemic_weight': 0.5,
            'pragmatic_weight': 0.5
        }
    
    def compute_free_energy(self, 
                          observations: torch.Tensor,
                          beliefs: BeliefState) -> FreeEnergyComponents:
        """
        Compute variational free energy F = KL[q(s)||p(s)] - ln P(o|s)
        This is the core mathematical operation of Active Inference
        """
        with torch.no_grad():
            # Accuracy term: log-likelihood of observations
            log_likelihood = self.generative_model.likelihood(observations, beliefs)
            accuracy = -log_likelihood.mean().item()
            
            # Complexity term: KL divergence from prior
            prior_mean, prior_var = self.generative_model.prior(beliefs.mean)
            complexity = self._kl_divergence(
                beliefs.mean, beliefs.variance,
                prior_mean, prior_var
            ).item()
            
            # Total free energy
            total_fe = complexity - (-accuracy)  # Note: accuracy is negative log-likelihood
            
            # Prediction error for monitoring
            predicted_obs = self.generative_model.likelihood_net(beliefs.mean)
            prediction_error = observations - predicted_obs
            
            # Epistemic value: uncertainty reduction potential
            epistemic_value = self._compute_epistemic_value(beliefs)
            
            # Pragmatic value: goal-directed value (placeholder - integrate with DPO)
            pragmatic_value = self._compute_pragmatic_value(beliefs, observations)
            
            # Confidence based on precision
            confidence = torch.mean(beliefs.precision).item()
            
        return FreeEnergyComponents(
            accuracy=accuracy,
            complexity=complexity,
            total_free_energy=total_fe,
            epistemic_value=epistemic_value,
            pragmatic_value=pragmatic_value,
            prediction_error=prediction_error,
            confidence=confidence
        )
    
    async def minimize_free_energy(self,
                                 observations: torch.Tensor,
                                 initial_beliefs: Optional[BeliefState] = None) -> Tuple[BeliefState, FreeEnergyComponents]:
        """
        Minimize free energy through gradient descent on beliefs
        This is perception: inferring hidden states from observations
        """
        # Initialize beliefs
        if initial_beliefs is None:
            # Use prior as initial belief
            state_sample = torch.randn(observations.shape[0], self.generative_model.state_dim)
            prior_mean, prior_var = self.generative_model.prior(state_sample)
            initial_beliefs = BeliefState(
                mean=prior_mean,
                variance=prior_var,
                precision=1.0 / prior_var
            )
        
        beliefs = BeliefState(
            mean=initial_beliefs.mean.clone().requires_grad_(True),
            variance=initial_beliefs.variance.clone(),
            precision=initial_beliefs.precision.clone()
        )
        
        # Optimization loop
        optimizer = torch.optim.Adam([beliefs.mean], lr=self.learning_rate)
        
        best_fe = float('inf')
        best_beliefs = None
        
        for iteration in range(self.n_iterations):
            optimizer.zero_grad()
            
            # Compute free energy
            fe_components = self.compute_free_energy(observations, beliefs)
            
            if fe_components.total_free_energy < best_fe:
                best_fe = fe_components.total_free_energy
                best_beliefs = BeliefState(
                    mean=beliefs.mean.clone().detach(),
                    variance=beliefs.variance.clone(),
                    precision=beliefs.precision.clone()
                )
            
            # Compute gradient of free energy w.r.t beliefs
            fe_tensor = torch.tensor(fe_components.total_free_energy, requires_grad=True)
            
            # Approximate gradient through prediction error
            pred_error = fe_components.prediction_error
            belief_gradient = -torch.matmul(pred_error.T, beliefs.mean) / observations.shape[0]
            
            # Update beliefs
            beliefs.mean.backward(belief_gradient)
            optimizer.step()
            
            # Update precision based on prediction accuracy
            error_magnitude = torch.mean(pred_error ** 2, dim=0)
            beliefs.precision = 1.0 / (error_magnitude + self.config['min_variance'])
            beliefs.variance = 1.0 / beliefs.precision
            
            # Early stopping if converged
            if iteration > 5 and abs(fe_components.total_free_energy - best_fe) < 1e-4:
                logger.debug(f"Free energy converged at iteration {iteration}")
                break
        
        # Store in history
        self.belief_history.append(best_beliefs)
        self.fe_history.append(best_fe)
        
        # Return optimized beliefs
        final_fe = self.compute_free_energy(observations, best_beliefs)
        
        logger.debug(f"Free energy minimization: {initial_beliefs.entropy():.3f} -> {best_beliefs.entropy():.3f}")
        logger.debug(f"Final free energy: {final_fe.total_free_energy:.3f} (accuracy: {final_fe.accuracy:.3f}, complexity: {final_fe.complexity:.3f})")
        
        return best_beliefs, final_fe
    
    def _kl_divergence(self, 
                      q_mean: torch.Tensor, q_var: torch.Tensor,
                      p_mean: torch.Tensor, p_var: torch.Tensor) -> torch.Tensor:
        """Compute KL[q||p] for diagonal Gaussians"""
        # KL = 0.5 * (log(|Σ_p|/|Σ_q|) + tr(Σ_p^{-1}Σ_q) + (μ_p-μ_q)^T Σ_p^{-1} (μ_p-μ_q) - d)
        
        log_var_ratio = torch.sum(torch.log(p_var / q_var))
        trace_term = torch.sum(q_var / p_var)
        mean_diff = p_mean - q_mean
        quad_term = torch.sum((mean_diff ** 2) / p_var)
        d = q_mean.shape[-1]
        
        kl = 0.5 * (log_var_ratio + trace_term + quad_term - d)
        return kl
    
    def _compute_epistemic_value(self, beliefs: BeliefState) -> float:
        """
        Compute epistemic value: expected information gain
        Higher uncertainty = higher epistemic value
        """
        # Use entropy as proxy for epistemic value
        return beliefs.entropy()
    
    def _compute_pragmatic_value(self, beliefs: BeliefState, observations: torch.Tensor) -> float:
        """
        Compute pragmatic value: goal achievement
        Placeholder - will integrate with DPO preferences
        """
        # For now, use negative prediction error as pragmatic value
        predicted_obs = self.generative_model.likelihood_net(beliefs.mean)
        pred_error = torch.mean((observations - predicted_obs) ** 2)
        return -pred_error.item()
    
    async def compute_expected_free_energy(self,
                                         beliefs: BeliefState,
                                         possible_actions: List[torch.Tensor],
                                         n_samples: int = 10) -> List[float]:
        """
        Compute expected free energy G(a) for each possible action
        G = epistemic value + pragmatic value
        This drives action selection in Active Inference
        """
        expected_fe = []
        
        for action in possible_actions:
            # Predict future observations given action
            # This is a simplified version - full implementation would use 
            # transition model P(s'|s,a) and observation model P(o'|s')
            
            epistemic_sum = 0.0
            pragmatic_sum = 0.0
            
            for _ in range(n_samples):
                # Sample future state
                future_state = beliefs.sample()
                
                # Add action effect (placeholder - integrate with your action model)
                future_state = future_state + 0.1 * action
                
                # Compute expected epistemic value
                future_beliefs = BeliefState(
                    mean=future_state,
                    variance=beliefs.variance * 1.1,  # Uncertainty grows
                    precision=beliefs.precision * 0.9
                )
                epistemic_sum += self._compute_epistemic_value(future_beliefs)
                
                # Compute expected pragmatic value (placeholder)
                pragmatic_sum += torch.sum(action * future_state).item()
            
            # Average over samples
            epistemic_avg = epistemic_sum / n_samples
            pragmatic_avg = pragmatic_sum / n_samples
            
            # Total expected free energy (negative because we minimize)
            G = -(self.config['epistemic_weight'] * epistemic_avg +
                  self.config['pragmatic_weight'] * pragmatic_avg)
            
            expected_fe.append(G)
        
        return expected_fe
    
    def integrate_tda_features(self, tda_features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Convert TDA topological features to observations for Active Inference
        This is the key integration point with AURA's TDA pipeline
        """
        # Extract relevant features
        persistence_image = tda_features.get('persistence_image', torch.zeros(128))
        betti_numbers = tda_features.get('betti_numbers', torch.zeros(3))
        wasserstein_distance = tda_features.get('wasserstein_distance', torch.tensor([0.0]))
        
        # Concatenate and normalize
        observations = torch.cat([
            persistence_image.flatten(),
            betti_numbers,
            wasserstein_distance.unsqueeze(0) if wasserstein_distance.dim() == 0 else wasserstein_distance
        ])
        
        # Ensure correct dimension
        if observations.shape[0] != self.generative_model.obs_dim:
            # Project to observation space
            projection = torch.randn(observations.shape[0], self.generative_model.obs_dim)
            observations = torch.matmul(observations.unsqueeze(0), projection).squeeze(0)
        
        return observations
    
    def get_uncertainty_estimate(self, beliefs: BeliefState) -> Dict[str, float]:
        """
        Provide uncertainty quantification for anomaly detection
        Key metric for measuring Active Inference value
        """
        return {
            'total_uncertainty': beliefs.entropy(),
            'aleatoric_uncertainty': torch.mean(beliefs.variance).item(),
            'epistemic_uncertainty': torch.std(beliefs.mean).item(),
            'confidence': torch.mean(beliefs.precision).item(),
            'free_energy': self.fe_history[-1] if self.fe_history else 0.0
        }


def create_free_energy_minimizer(
    state_dim: int = 256,
    obs_dim: int = 128,
    config: Optional[Dict] = None
) -> FreeEnergyMinimizer:
    """Factory function for creating Free Energy Minimizer"""
    generative_model = GenerativeModel(state_dim, obs_dim)
    return FreeEnergyMinimizer(generative_model, config)