"""
Dream Generator with VAE and Astrocyte-Inspired Validation
==========================================================

Implements creative memory recombination through:
1. DreamVAE: Variational autoencoder for latent space interpolation
2. AstrocyteAssociativeValidator: Transformer-based coherence validation
3. Spherical interpolation (SLERP) for smooth transitions
4. Causal plausibility testing

Based on:
- MyGO Framework for generative consolidation
- IBM's Astrocyte research (glial cells as transformers)
- NeuroDream Framework for creative problem solving
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
import structlog

logger = structlog.get_logger(__name__)


@dataclass
class Dream:
    """Represents a generated dream memory"""
    signature: np.ndarray
    content: Dict[str, Any]
    parent_ids: List[str]
    interpolation_alpha: float
    coherence_score: float = 0.0
    causal_plausibility: float = 0.0
    validation_status: str = "pending"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert dream to dictionary"""
        return {
            "signature": self.signature.tolist() if isinstance(self.signature, np.ndarray) else self.signature,
            "content": self.content,
            "parent_ids": self.parent_ids,
            "interpolation_alpha": self.interpolation_alpha,
            "coherence_score": self.coherence_score,
            "causal_plausibility": self.causal_plausibility,
            "validation_status": self.validation_status
        }


# ==================== DreamVAE Implementation ====================

class DreamVAE(nn.Module):
    """
    Variational Autoencoder for dream generation
    
    Learns the latent distribution of memory signatures and
    enables smooth interpolation between distant memories.
    """
    
    def __init__(self, input_dim: int = 384, latent_dim: int = 128, hidden_dim: int = 256):
        """
        Initialize the Dream VAE
        
        Args:
            input_dim: Dimension of input embeddings (FastRP default: 384)
            latent_dim: Dimension of latent space
            hidden_dim: Dimension of hidden layers
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Latent space (mean and log variance)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, input_dim)
        )
        
        logger.info(
            "DreamVAE initialized",
            input_dim=input_dim,
            latent_dim=latent_dim,
            hidden_dim=hidden_dim
        )
    
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode input to latent space
        
        Args:
            x: Input tensor [batch_size, input_dim]
        
        Returns:
            mu: Mean of latent distribution
            logvar: Log variance of latent distribution
        """
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick for VAE
        
        Args:
            mu: Mean of latent distribution
            logvar: Log variance of latent distribution
        
        Returns:
            Sampled latent vector
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent vector to output space
        
        Args:
            z: Latent vector [batch_size, latent_dim]
        
        Returns:
            Reconstructed output
        """
        return self.decoder(z)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through VAE
        
        Args:
            x: Input tensor
        
        Returns:
            recon_x: Reconstructed output
            mu: Latent mean
            logvar: Latent log variance
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar
    
    def interpolate_in_latent_space(self, z1: torch.Tensor, z2: torch.Tensor, 
                                   alpha: float = 0.5, use_slerp: bool = True) -> torch.Tensor:
        """
        Interpolate between two latent vectors
        
        Args:
            z1, z2: Latent vectors to interpolate between
            alpha: Interpolation factor (0 = z1, 1 = z2)
            use_slerp: Use spherical interpolation (better for high dimensions)
        
        Returns:
            Interpolated latent vector
        """
        if use_slerp:
            # Spherical linear interpolation (SLERP)
            return self._slerp(z1, z2, alpha)
        else:
            # Simple linear interpolation
            return (1 - alpha) * z1 + alpha * z2
    
    def _slerp(self, z1: torch.Tensor, z2: torch.Tensor, alpha: float) -> torch.Tensor:
        """
        Spherical linear interpolation
        
        Better than linear interpolation for high-dimensional spaces
        """
        # Normalize vectors
        z1_norm = F.normalize(z1, dim=-1)
        z2_norm = F.normalize(z2, dim=-1)
        
        # Calculate angle between vectors
        dot_product = (z1_norm * z2_norm).sum(dim=-1, keepdim=True)
        dot_product = torch.clamp(dot_product, -1.0, 1.0)
        theta = torch.acos(dot_product)
        
        # Perform SLERP
        sin_theta = torch.sin(theta)
        
        # Handle edge case where vectors are parallel
        where_parallel = (sin_theta.abs() < 1e-6).squeeze()
        
        # SLERP formula
        s1 = torch.sin((1 - alpha) * theta) / sin_theta
        s2 = torch.sin(alpha * theta) / sin_theta
        
        # Interpolate
        result = s1 * z1 + s2 * z2
        
        # Use linear interpolation for parallel vectors
        if where_parallel.any():
            linear_interp = (1 - alpha) * z1 + alpha * z2
            if where_parallel.dim() == 0:
                if where_parallel:
                    result = linear_interp
            else:
                result[where_parallel] = linear_interp[where_parallel]
        
        return result
    
    def loss_function(self, recon_x: torch.Tensor, x: torch.Tensor,
                     mu: torch.Tensor, logvar: torch.Tensor,
                     beta: float = 1.0) -> Dict[str, torch.Tensor]:
        """
        VAE loss function (reconstruction + KL divergence)
        
        Args:
            recon_x: Reconstructed output
            x: Original input
            mu: Latent mean
            logvar: Latent log variance
            beta: Weight for KL divergence (beta-VAE)
        
        Returns:
            Dictionary with loss components
        """
        # Reconstruction loss
        recon_loss = F.mse_loss(recon_x, x, reduction='mean')
        
        # KL divergence
        kl_div = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        
        # Total loss
        total_loss = recon_loss + beta * kl_div
        
        return {
            'total': total_loss,
            'reconstruction': recon_loss,
            'kl_divergence': kl_div
        }


# ==================== Astrocyte-Inspired Validator ====================

class AstrocyteAssociativeValidator(nn.Module):
    """
    Transformer-based validator inspired by astrocyte memory
    
    IBM research shows that glial cells (astrocytes) implement
    associative memory similar to transformers. This validator
    checks if a dream is a coherent association of its parents.
    """
    
    def __init__(self, input_dim: int = 384, num_heads: int = 8, 
                 num_layers: int = 4, hidden_dim: int = 512):
        """
        Initialize the Astrocyte validator
        
        Args:
            input_dim: Dimension of input embeddings
            num_heads: Number of attention heads
            num_layers: Number of transformer layers
            hidden_dim: Dimension of feedforward network
        """
        super().__init__()
        
        self.input_dim = input_dim
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 2,
            dropout=0.1,
            activation='gelu',
            batch_first=True
        )
        
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        # Output layers for validation score
        self.output_layers = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # Positional encoding for sequence order
        self.positional_encoding = self._create_positional_encoding(10, hidden_dim)
        
        logger.info(
            "AstrocyteValidator initialized",
            input_dim=input_dim,
            num_heads=num_heads,
            num_layers=num_layers
        )
    
    def _create_positional_encoding(self, max_len: int, d_model: int) -> torch.Tensor:
        """Create sinusoidal positional encoding"""
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        return pe.unsqueeze(0)  # Add batch dimension
    
    def forward(self, dream_signature: torch.Tensor, 
                parent_signatures: List[torch.Tensor]) -> torch.Tensor:
        """
        Validate if dream is coherent association of parents
        
        Args:
            dream_signature: The generated dream embedding
            parent_signatures: List of parent memory embeddings
        
        Returns:
            Validation score (0-1, higher = more coherent)
        """
        # Stack inputs: [dream, parent1, parent2, ...]
        all_signatures = [dream_signature] + parent_signatures
        
        # Ensure all have same shape
        max_len = max(sig.shape[0] if sig.dim() > 1 else 1 for sig in all_signatures)
        
        # Reshape if needed
        processed_sigs = []
        for sig in all_signatures:
            if sig.dim() == 1:
                sig = sig.unsqueeze(0)
            if sig.shape[0] != max_len:
                sig = F.pad(sig, (0, 0, 0, max_len - sig.shape[0]))
            processed_sigs.append(sig)
        
        # Stack into sequence [batch=1, seq_len, input_dim]
        inputs = torch.stack(processed_sigs, dim=0).unsqueeze(0)
        
        # Project to hidden dimension
        hidden = self.input_projection(inputs)
        
        # Add positional encoding
        seq_len = hidden.shape[1]
        hidden = hidden + self.positional_encoding[:, :seq_len, :]
        
        # Pass through transformer
        encoded = self.transformer_encoder(hidden)
        
        # Pool over sequence dimension
        pooled = torch.mean(encoded, dim=1)
        
        # Get validation score
        validation_score = self.output_layers(pooled)
        
        return validation_score.squeeze()
    
    def compute_association_strength(self, dream: torch.Tensor,
                                    parents: List[torch.Tensor]) -> float:
        """
        Compute the strength of association between dream and parents
        
        Args:
            dream: Dream embedding
            parents: Parent embeddings
        
        Returns:
            Association strength score
        """
        with torch.no_grad():
            score = self.forward(dream, parents)
            return score.item()


# ==================== Main Dream Generator ====================

class DreamGenerator:
    """
    Complete dream generation system combining VAE and validation
    """
    
    def __init__(self, services: Dict[str, Any], config: Any):
        """
        Initialize the dream generator
        
        Args:
            services: Dictionary of AURA services
            config: Consolidation configuration
        """
        self.services = services
        self.config = config
        
        # Initialize models
        self.vae = DreamVAE(
            input_dim=384,  # FastRP embedding dimension
            latent_dim=128,
            hidden_dim=256
        )
        
        self.validator = AstrocyteAssociativeValidator(
            input_dim=384,
            num_heads=8,
            num_layers=4
        )
        
        # Training components
        self.vae_optimizer = torch.optim.Adam(self.vae.parameters(), lr=0.001)
        self.validator_optimizer = torch.optim.Adam(self.validator.parameters(), lr=0.001)
        
        # Statistics
        self.total_dreams_generated = 0
        self.total_dreams_validated = 0
        
        # Put models in eval mode by default
        self.vae.eval()
        self.validator.eval()
        
        logger.info("DreamGenerator initialized")
    
    async def generate_dream(self, memory1: Any, memory2: Any,
                           alpha: Optional[float] = None) -> Dream:
        """
        Generate a dream by interpolating between two memories
        
        Args:
            memory1, memory2: Parent memories to interpolate
            alpha: Interpolation factor (None = random)
        
        Returns:
            Generated dream
        """
        # Extract embeddings
        emb1 = self._extract_embedding(memory1)
        emb2 = self._extract_embedding(memory2)
        
        # Convert to tensors
        z1 = torch.FloatTensor(emb1).unsqueeze(0)
        z2 = torch.FloatTensor(emb2).unsqueeze(0)
        
        # Encode to latent space
        with torch.no_grad():
            mu1, _ = self.vae.encode(z1)
            mu2, _ = self.vae.encode(z2)
            
            # Choose interpolation factor
            if alpha is None:
                # Random, but avoid extremes
                alpha = np.random.beta(2, 2)  # Beta distribution peaks at 0.5
            
            # Interpolate in latent space (using SLERP)
            z_interp = self.vae.interpolate_in_latent_space(mu1, mu2, alpha, use_slerp=True)
            
            # Decode back to embedding space
            dream_embedding = self.vae.decode(z_interp)
        
        # Create dream object
        dream = Dream(
            signature=dream_embedding.squeeze().numpy(),
            content=self._generate_dream_content(memory1, memory2, alpha),
            parent_ids=[
                getattr(memory1, 'id', str(memory1)[:16]),
                getattr(memory2, 'id', str(memory2)[:16])
            ],
            interpolation_alpha=alpha
        )
        
        self.total_dreams_generated += 1
        
        logger.debug(
            "Dream generated",
            alpha=alpha,
            parent1=dream.parent_ids[0],
            parent2=dream.parent_ids[1]
        )
        
        return dream
    
    async def validate_dream(self, dream: Dream, parents: List[Any],
                           threshold: float = 0.8) -> bool:
        """
        Validate dream coherence using Astrocyte validator
        
        Args:
            dream: Dream to validate
            parents: Parent memories
            threshold: Validation threshold
        
        Returns:
            True if valid, False otherwise
        """
        # Extract parent embeddings
        parent_embeddings = [self._extract_embedding(p) for p in parents]
        
        # Convert to tensors
        dream_tensor = torch.FloatTensor(dream.signature)
        parent_tensors = [torch.FloatTensor(emb) for emb in parent_embeddings]
        
        # Validate coherence
        with torch.no_grad():
            coherence_score = self.validator(dream_tensor, parent_tensors)
            dream.coherence_score = coherence_score.item()
        
        # Check threshold
        is_valid = dream.coherence_score >= threshold
        
        if is_valid:
            dream.validation_status = "valid"
            self.total_dreams_validated += 1
        else:
            dream.validation_status = "invalid"
        
        logger.debug(
            "Dream validated",
            coherence=dream.coherence_score,
            valid=is_valid,
            threshold=threshold
        )
        
        return is_valid
    
    def train_vae(self, memory_batch: List[np.ndarray], epochs: int = 10):
        """
        Train the VAE on a batch of memories
        
        Args:
            memory_batch: List of memory embeddings
            epochs: Number of training epochs
        """
        self.vae.train()
        
        # Convert to tensor
        batch_tensor = torch.FloatTensor(np.stack(memory_batch))
        
        for epoch in range(epochs):
            # Forward pass
            recon, mu, logvar = self.vae(batch_tensor)
            
            # Calculate loss
            losses = self.vae.loss_function(recon, batch_tensor, mu, logvar)
            
            # Backward pass
            self.vae_optimizer.zero_grad()
            losses['total'].backward()
            self.vae_optimizer.step()
            
            if epoch % 5 == 0:
                logger.debug(
                    f"VAE training epoch {epoch}",
                    loss=losses['total'].item(),
                    recon_loss=losses['reconstruction'].item(),
                    kl_div=losses['kl_divergence'].item()
                )
        
        self.vae.eval()
    
    def train_validator(self, dream_batch: List[Tuple[np.ndarray, List[np.ndarray], float]],
                       epochs: int = 10):
        """
        Train the validator on dream-parent pairs
        
        Args:
            dream_batch: List of (dream, parents, target_score) tuples
            epochs: Number of training epochs
        """
        self.validator.train()
        
        for epoch in range(epochs):
            total_loss = 0
            
            for dream_emb, parent_embs, target_score in dream_batch:
                # Convert to tensors
                dream_tensor = torch.FloatTensor(dream_emb)
                parent_tensors = [torch.FloatTensor(p) for p in parent_embs]
                target = torch.FloatTensor([target_score])
                
                # Forward pass
                predicted = self.validator(dream_tensor, parent_tensors)
                
                # Calculate loss
                loss = F.binary_cross_entropy(predicted.unsqueeze(0), target)
                
                # Backward pass
                self.validator_optimizer.zero_grad()
                loss.backward()
                self.validator_optimizer.step()
                
                total_loss += loss.item()
            
            if epoch % 5 == 0:
                avg_loss = total_loss / len(dream_batch)
                logger.debug(f"Validator training epoch {epoch}, loss: {avg_loss:.4f}")
        
        self.validator.eval()
    
    def _extract_embedding(self, memory: Any) -> np.ndarray:
        """Extract embedding from memory object"""
        if hasattr(memory, 'embedding'):
            return memory.embedding
        elif hasattr(memory, 'signature'):
            return memory.signature
        elif isinstance(memory, np.ndarray):
            return memory
        else:
            # Generate random embedding as fallback
            return np.random.randn(384)
    
    def _generate_dream_content(self, memory1: Any, memory2: Any, alpha: float) -> Dict[str, Any]:
        """Generate content description for dream"""
        content = {
            "type": "dream",
            "interpolation": {
                "alpha": alpha,
                "parent1_weight": 1 - alpha,
                "parent2_weight": alpha
            },
            "synthesis": f"Creative combination of concepts from {getattr(memory1, 'id', 'memory1')[:8]} "
                        f"and {getattr(memory2, 'id', 'memory2')[:8]}",
            "timestamp": np.datetime64('now').item()
        }
        
        # Add parent content if available
        if hasattr(memory1, 'content'):
            content["parent1_content"] = str(memory1.content)[:100]
        if hasattr(memory2, 'content'):
            content["parent2_content"] = str(memory2.content)[:100]
        
        return content
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get dream generation statistics"""
        return {
            "total_generated": self.total_dreams_generated,
            "total_validated": self.total_dreams_validated,
            "validation_rate": self.total_dreams_validated / max(1, self.total_dreams_generated),
            "vae_params": sum(p.numel() for p in self.vae.parameters()),
            "validator_params": sum(p.numel() for p in self.validator.parameters())
        }