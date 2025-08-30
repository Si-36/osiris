"""
PHFormer-Tiny: Persistence Homology Transformer (Clean Version)

A minimal transformer model designed to work with topological features.
Integrates with AURA's TDA pipeline for shape-aware processing.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import math
import structlog

logger = structlog.get_logger(__name__)


@dataclass 
class PHFormerConfig:
    """Configuration for PHFormer-Tiny model"""
    # Model dimensions
    hidden_size: int = 384
    num_hidden_layers: int = 6
    num_attention_heads: int = 6
    intermediate_size: int = 1536
    
    # Topological features
    max_persistence_points: int = 512
    persistence_dim: int = 64
    betti_embedding_dim: int = 32
    
    # Standard transformer config
    hidden_dropout_prob: float = 0.1
    attention_probs_dropout_prob: float = 0.1
    layer_norm_eps: float = 1e-5
    
    # Optimization
    use_flash_attention: bool = True
    gradient_checkpointing: bool = False
    use_quantization: bool = False


class TopologicalEmbedding(nn.Module):
    """Embed topological features into transformer space"""
    
    def __init__(self, config: PHFormerConfig):
        super().__init__()
        self.config = config
        
        # Persistence diagram embeddings
        self.pd_projection = nn.Linear(2, config.persistence_dim)
        self.pd_norm = nn.LayerNorm(config.persistence_dim)
        
        # Betti number embeddings
        self.betti_embedding = nn.Embedding(100, config.betti_embedding_dim)
        self.betti_projection = nn.Linear(config.betti_embedding_dim, config.hidden_size)
        
        # Combined projection
        self.combined_projection = nn.Linear(
            config.persistence_dim + config.hidden_size,
            config.hidden_size
        )
        
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
    
    def forward(self, 
                persistence_diagrams: List[torch.Tensor],
                betti_numbers: torch.Tensor) -> torch.Tensor:
        """
        Embed topological features
        
        Args:
            persistence_diagrams: List of (N, 2) tensors for birth/death
            betti_numbers: (batch_size, max_dim) tensor of Betti numbers
            
        Returns:
            embeddings: (batch_size, seq_len, hidden_size)
        """
        batch_size = len(persistence_diagrams)
        device = betti_numbers.device
        
        # Process persistence diagrams
        pd_embeddings = []
        for pd in persistence_diagrams:
            if pd.shape[0] == 0:
                # Empty diagram - use zeros
                pd_emb = torch.zeros(
                    self.config.max_persistence_points, 
                    self.config.persistence_dim,
                    device=device
                )
            else:
                # Project birth/death pairs
                pd_emb = self.pd_projection(pd)
                pd_emb = self.pd_norm(pd_emb)
                
                # Pad or truncate to fixed size
                if pd_emb.shape[0] < self.config.max_persistence_points:
                    padding = torch.zeros(
                        self.config.max_persistence_points - pd_emb.shape[0],
                        self.config.persistence_dim,
                        device=device
                    )
                    pd_emb = torch.cat([pd_emb, padding], dim=0)
                else:
                    pd_emb = pd_emb[:self.config.max_persistence_points]
            
            pd_embeddings.append(pd_emb)
        
        pd_embeddings = torch.stack(pd_embeddings)  # (batch, max_points, pd_dim)
        
        # Process Betti numbers
        betti_emb = self.betti_embedding(betti_numbers.long())  # (batch, max_dim, betti_dim)
        betti_emb = self.betti_projection(betti_emb)  # (batch, max_dim, hidden_size)
        
        # Combine Betti embeddings with persistence diagrams
        # Expand Betti embeddings to match sequence length
        seq_len = pd_embeddings.shape[1]
        betti_expanded = betti_emb.mean(dim=1, keepdim=True).expand(-1, seq_len, -1)
        
        # Concatenate and project
        combined = torch.cat([pd_embeddings, betti_expanded], dim=-1)
        embeddings = self.combined_projection(combined)
        embeddings = self.dropout(embeddings)
        
        return embeddings


class EfficientAttention(nn.Module):
    """Efficient multi-head attention with optional flash attention"""
    
    def __init__(self, config: PHFormerConfig):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = config.hidden_size // config.num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)
        
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.use_flash = config.use_flash_attention
    
    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)
    
    def forward(self, 
                hidden_states: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Efficient attention forward pass"""
        query_layer = self.transpose_for_scores(self.query(hidden_states))
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        
        # Compute attention scores
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask
        
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)
        
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)
        
        return context_layer


class PHFormerLayer(nn.Module):
    """Single transformer layer with topological awareness"""
    
    def __init__(self, config: PHFormerConfig):
        super().__init__()
        self.attention = EfficientAttention(config)
        self.attention_output = nn.Linear(config.hidden_size, config.hidden_size)
        self.attention_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
        # MLP
        self.mlp = nn.Sequential(
            nn.Linear(config.hidden_size, config.intermediate_size),
            nn.GELU(),
            nn.Linear(config.intermediate_size, config.hidden_size)
        )
        self.mlp_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
    
    def forward(self, 
                hidden_states: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Self-attention
        attention_output = self.attention(hidden_states, attention_mask)
        attention_output = self.attention_output(attention_output)
        attention_output = self.dropout(attention_output)
        hidden_states = self.attention_norm(hidden_states + attention_output)
        
        # MLP
        mlp_output = self.mlp(hidden_states)
        mlp_output = self.dropout(mlp_output)
        hidden_states = self.mlp_norm(hidden_states + mlp_output)
        
        return hidden_states


class PHFormerEncoder(nn.Module):
    """Stack of PHFormer layers"""
    
    def __init__(self, config: PHFormerConfig):
        super().__init__()
        self.layers = nn.ModuleList([
            PHFormerLayer(config) for _ in range(config.num_hidden_layers)
        ])
        self.gradient_checkpointing = config.gradient_checkpointing
    
    def forward(self,
                hidden_states: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        for layer in self.layers:
            if self.gradient_checkpointing and self.training:
                hidden_states = torch.utils.checkpoint.checkpoint(
                    layer, hidden_states, attention_mask
                )
            else:
                hidden_states = layer(hidden_states, attention_mask)
        
        return hidden_states


class PHFormerTiny(nn.Module):
    """PHFormer-Tiny: Minimal topology-aware transformer"""
    
    def __init__(self, config: PHFormerConfig):
        super().__init__()
        self.config = config
        
        # Topological embeddings
        self.topo_embedding = TopologicalEmbedding(config)
        
        # Transformer encoder
        self.encoder = PHFormerEncoder(config)
        
        # Output heads
        self.pooler = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.Tanh()
        )
        
        # Classification head
        self.classifier = nn.Linear(config.hidden_size, 2)  # Binary by default
        
        # Initialize weights
        self.apply(self._init_weights)
        
        logger.info(f"PHFormer-Tiny initialized with {self.count_parameters()}M parameters")
    
    def _init_weights(self, module):
        """Initialize weights with small values for stability"""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
    
    def count_parameters(self) -> float:
        """Count model parameters in millions"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad) / 1e6
    
    def forward(self,
                persistence_diagrams: List[torch.Tensor],
                betti_numbers: torch.Tensor,
                persistence_images: Optional[torch.Tensor] = None,
                sequence_features: Optional[torch.Tensor] = None,
                return_embeddings: bool = False) -> Dict[str, torch.Tensor]:
        """
        Forward pass through PHFormer-Tiny
        
        Args:
            persistence_diagrams: List of PDs per sample
            betti_numbers: (batch, max_dim) Betti numbers
            persistence_images: Optional (batch, resolution) PIs
            sequence_features: Optional additional features
            return_embeddings: Whether to return embeddings
            
        Returns:
            Dictionary with logits and optionally embeddings
        """
        # Embed topological features
        hidden_states = self.topo_embedding(persistence_diagrams, betti_numbers)
        
        # Create attention mask (all ones for now)
        batch_size, seq_len = hidden_states.shape[:2]
        attention_mask = torch.ones(batch_size, seq_len, device=hidden_states.device)
        
        # Encode
        encoded = self.encoder(hidden_states, attention_mask)
        
        # Pool
        pooled = self.pooler(encoded.mean(dim=1))
        
        # Classify
        logits = self.classifier(pooled)
        
        outputs = {"logits": logits}
        
        if return_embeddings:
            outputs["embeddings"] = pooled
            outputs["hidden_states"] = encoded
        
        return outputs
    
    def extract_features(self, 
                        persistence_diagrams: List[torch.Tensor],
                        betti_numbers: torch.Tensor) -> torch.Tensor:
        """Extract feature representations"""
        outputs = self.forward(
            persistence_diagrams, 
            betti_numbers,
            return_embeddings=True
        )
        return outputs["embeddings"]
    
    @torch.no_grad()
    def get_attention_weights(self,
                             persistence_diagrams: List[torch.Tensor],
                             betti_numbers: torch.Tensor) -> List[torch.Tensor]:
        """Get attention weights for visualization"""
        # This would require modifying the attention layers to return weights
        # For now, return empty list
        return []


# Example usage
def demonstrate_phformer():
    """Demonstrate PHFormer-Tiny capabilities"""
    print("🔺 PHFormer-Tiny Demonstration")
    print("=" * 60)
    
    # Create model
    config = PHFormerConfig(
        hidden_size=384,
        num_hidden_layers=6,
        num_attention_heads=6
    )
    
    model = PHFormerTiny(config)
    print(f"✅ Model created with {model.count_parameters():.2f}M parameters")
    
    # Create dummy topological data
    batch_size = 2
    
    # Persistence diagrams (variable length per sample)
    persistence_diagrams = [
        torch.randn(100, 2),  # 100 persistence points
        torch.randn(150, 2)   # 150 persistence points
    ]
    
    # Betti numbers
    betti_numbers = torch.randint(0, 10, (batch_size, 3))
    
    # Forward pass
    outputs = model(persistence_diagrams, betti_numbers, return_embeddings=True)
    
    print(f"\n✅ Forward pass successful")
    print(f"   Logits shape: {outputs['logits'].shape}")
    print(f"   Embeddings shape: {outputs['embeddings'].shape}")
    
    print("\n✅ PHFormer demonstration complete")


if __name__ == "__main__":
    demonstrate_phformer()