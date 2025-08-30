"""
🎨 Creative Agent - Multi-Modal Creative Intelligence
===================================================

Specializes in:
- Parallel creative variations on GPU
- Topological diversity optimization
- Multi-modal reasoning and generation
- Style transfer and brainstorming
- Context-aware content creation
"""

import asyncio
import time
import json
import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, Any, List, Optional, Tuple, Set
from dataclasses import dataclass, field
import structlog
from collections import defaultdict

from .test_agents import TestAgentBase, TestAgentConfig, Tool, AgentRole
from ..lnn.gpu_optimized_lnn import GPUOptimizedLNN
from ..moe.gpu_optimized_moe import GPUOptimizedMoE

logger = structlog.get_logger(__name__)


@dataclass
class CreativeOutput:
    """Result of creative generation"""
    content: List[Dict[str, Any]]
    diversity_score: float
    quality_metrics: Dict[str, float]
    topological_signature: np.ndarray
    generation_time_ms: float = 0.0
    model_used: str = "default"
    style_attributes: Dict[str, Any] = field(default_factory=dict)


class ParallelVariationGenerator:
    """Generate creative variations in parallel on GPU"""
    
    def __init__(self, device: str = "cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        
        # Neural components for variation
        self.variation_network = torch.nn.Sequential(
            torch.nn.Linear(768, 1024),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(1024, 768)
        ).to(self.device)
        
        # Compile if possible
        if hasattr(torch, 'compile'):
            self.variation_network = torch.compile(self.variation_network)
            
    async def generate_batch(self, 
                           prompt: str,
                           num_variations: int = 50,
                           temperature: float = 0.8,
                           gpu_parallel: bool = True) -> List[Dict[str, Any]]:
        """Generate multiple variations in parallel"""
        start_time = time.perf_counter()
        
        # Encode prompt (simplified - would use actual embeddings)
        prompt_embedding = self._encode_prompt(prompt)
        
        if gpu_parallel and self.device.type == "cuda":
            # Generate all variations on GPU at once
            variations = await self._gpu_parallel_generation(
                prompt_embedding,
                num_variations,
                temperature
            )
        else:
            # CPU fallback
            variations = await self._sequential_generation(
                prompt_embedding,
                num_variations,
                temperature
            )
            
        generation_time = (time.perf_counter() - start_time) * 1000
        
        # Add metadata to each variation
        for i, var in enumerate(variations):
            var['generation_time_ms'] = generation_time / len(variations)
            var['variation_id'] = i
            var['temperature'] = temperature
            
        return variations
        
    def _encode_prompt(self, prompt: str) -> torch.Tensor:
        """Encode prompt to embedding (simplified)"""
        # In practice, would use actual language model
        # For now, create random embedding based on prompt hash
        torch.manual_seed(hash(prompt) % 2**32)
        return torch.randn(1, 768, device=self.device)
        
    async def _gpu_parallel_generation(self,
                                     prompt_embedding: torch.Tensor,
                                     num_variations: int,
                                     temperature: float) -> List[Dict[str, Any]]:
        """Generate variations in parallel on GPU"""
        # Expand prompt for batch processing
        batch_embeddings = prompt_embedding.expand(num_variations, -1)
        
        # Add noise for variation
        noise = torch.randn_like(batch_embeddings) * temperature
        varied_embeddings = batch_embeddings + noise
        
        # Pass through variation network
        with torch.no_grad():
            output_embeddings = self.variation_network(varied_embeddings)
            
        # Decode embeddings to content (simplified)
        variations = []
        for i in range(num_variations):
            # In practice, would decode using language model
            content = self._decode_embedding(output_embeddings[i])
            
            variations.append({
                "content": content,
                "embedding": output_embeddings[i].cpu().numpy(),
                "variation_strength": float(torch.norm(noise[i]).item())
            })
            
        return variations
        
    async def _sequential_generation(self,
                                   prompt_embedding: torch.Tensor,
                                   num_variations: int,
                                   temperature: float) -> List[Dict[str, Any]]:
        """Sequential generation fallback"""
        variations = []
        
        for i in range(num_variations):
            noise = torch.randn_like(prompt_embedding) * temperature
            varied_embedding = prompt_embedding + noise
            
            with torch.no_grad():
                output_embedding = self.variation_network(varied_embedding)
                
            content = self._decode_embedding(output_embedding[0])
            
            variations.append({
                "content": content,
                "embedding": output_embedding[0].cpu().numpy(),
                "variation_strength": float(torch.norm(noise).item())
            })
            
        return variations
        
    def _decode_embedding(self, embedding: torch.Tensor) -> str:
        """Decode embedding to content (simplified)"""
        # In practice, would use actual decoder
        # For now, generate template-based content
        
        # Extract features from embedding
        features = embedding.cpu().numpy()
        
        # Map to creative attributes
        creativity_score = float(np.mean(np.abs(features[:256])))
        formality_score = float(np.mean(features[256:512]))
        complexity_score = float(np.mean(features[512:]))
        
        # Generate content based on scores
        if creativity_score > 0.7:
            style = "highly creative and imaginative"
        elif creativity_score > 0.4:
            style = "balanced and thoughtful"
        else:
            style = "structured and methodical"
            
        content = f"A {style} approach to the given prompt, "
        
        if formality_score > 0.5:
            content += "presented in a formal manner "
        else:
            content += "expressed casually "
            
        if complexity_score > 0.6:
            content += "with sophisticated reasoning."
        else:
            content += "with clear simplicity."
            
        return content


class TopologicalDiversityOptimizer:
    """Optimize for topological diversity in creative outputs"""
    
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def select_diverse_subset(self,
                            variations: List[Dict[str, Any]],
                            target_diversity: float = 0.8,
                            subset_size: Optional[int] = None) -> List[Dict[str, Any]]:
        """Select diverse subset using topological distance"""
        if not variations:
            return []
            
        # Extract embeddings
        embeddings = []
        for var in variations:
            if 'embedding' in var:
                embeddings.append(var['embedding'])
            else:
                # Generate random embedding if missing
                embeddings.append(np.random.randn(768))
                
        embeddings = np.array(embeddings)
        
        # Convert to torch for GPU processing
        embeddings_tensor = torch.tensor(embeddings, device=self.device)
        
        # Compute pairwise distances
        distances = torch.cdist(embeddings_tensor, embeddings_tensor)
        
        # Greedy selection for diversity
        if subset_size is None:
            subset_size = min(len(variations), max(5, len(variations) // 3))
            
        selected_indices = self._greedy_diverse_selection(
            distances,
            subset_size,
            target_diversity
        )
        
        # Return selected variations
        return [variations[i] for i in selected_indices]
        
    def _greedy_diverse_selection(self,
                                distances: torch.Tensor,
                                k: int,
                                target_diversity: float) -> List[int]:
        """Greedy algorithm for diverse selection"""
        n = distances.shape[0]
        
        if k >= n:
            return list(range(n))
            
        selected = []
        remaining = set(range(n))
        
        # Start with the most central point
        centrality = distances.mean(dim=1)
        first = torch.argmin(centrality).item()
        selected.append(first)
        remaining.remove(first)
        
        # Iteratively add points that maximize minimum distance
        while len(selected) < k and remaining:
            max_min_dist = -1
            best_candidate = None
            
            for candidate in remaining:
                # Minimum distance to selected set
                min_dist = float('inf')
                for s in selected:
                    dist = distances[candidate, s].item()
                    if dist < min_dist:
                        min_dist = dist
                        
                if min_dist > max_min_dist:
                    max_min_dist = min_dist
                    best_candidate = candidate
                    
            if best_candidate is not None:
                selected.append(best_candidate)
                remaining.remove(best_candidate)
                
                # Check if we've achieved target diversity
                if len(selected) >= 3:
                    current_diversity = self._calculate_diversity(distances, selected)
                    if current_diversity >= target_diversity:
                        break
                        
        return selected
        
    def _calculate_diversity(self, distances: torch.Tensor, indices: List[int]) -> float:
        """Calculate diversity score of selected subset"""
        if len(indices) < 2:
            return 0.0
            
        # Extract submatrix
        subset_distances = distances[indices][:, indices]
        
        # Calculate average pairwise distance
        mask = torch.ones_like(subset_distances) - torch.eye(len(indices), device=distances.device)
        avg_distance = (subset_distances * mask).sum() / mask.sum()
        
        # Normalize by maximum possible distance
        max_distance = distances.max()
        
        return float(avg_distance / max_distance)


class MultiModalReasoner:
    """Multi-modal reasoning and enhancement"""
    
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Cross-modal attention
        self.cross_attention = torch.nn.MultiheadAttention(
            embed_dim=768,
            num_heads=8,
            batch_first=True
        ).to(self.device)
        
        # Modal fusion network
        self.fusion_network = torch.nn.Sequential(
            torch.nn.Linear(768 * 3, 1024),  # text + image + audio features
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(1024, 768)
        ).to(self.device)
        
    async def enhance(self, content_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Enhance content with multi-modal reasoning"""
        enhanced = []
        
        for content in content_list:
            # Extract features from different modalities (simplified)
            text_features = self._extract_text_features(content)
            visual_features = self._extract_visual_features(content)
            audio_features = self._extract_audio_features(content)
            
            # Cross-modal attention
            enhanced_features = await self._cross_modal_reasoning(
                text_features,
                visual_features,
                audio_features
            )
            
            # Update content with enhanced features
            enhanced_content = content.copy()
            enhanced_content['multi_modal_features'] = enhanced_features.cpu().numpy()
            enhanced_content['modality_scores'] = {
                'text': float(torch.norm(text_features).item()),
                'visual': float(torch.norm(visual_features).item()),
                'audio': float(torch.norm(audio_features).item()),
                'fusion': float(torch.norm(enhanced_features).item())
            }
            
            enhanced.append(enhanced_content)
            
        return enhanced
        
    def _extract_text_features(self, content: Dict[str, Any]) -> torch.Tensor:
        """Extract text features (simplified)"""
        # In practice, would use language model
        text = content.get('content', '')
        
        # Simple feature extraction based on text properties
        features = torch.zeros(768, device=self.device)
        
        # Length feature
        features[0] = len(text) / 1000.0
        
        # Complexity features
        features[1] = len(set(text.split())) / max(len(text.split()), 1)  # Vocabulary diversity
        
        # Random features for demonstration
        features[2:256] = torch.randn(254, device=self.device) * 0.1
        
        return features.unsqueeze(0)
        
    def _extract_visual_features(self, content: Dict[str, Any]) -> torch.Tensor:
        """Extract visual features (simplified)"""
        # In practice, would use vision model
        # For now, generate based on content
        
        features = torch.zeros(768, device=self.device)
        
        # Check for visual descriptors in content
        visual_words = ['color', 'shape', 'image', 'visual', 'picture', 'scene']
        text = content.get('content', '').lower()
        
        visual_score = sum(1 for word in visual_words if word in text)
        features[256] = visual_score / len(visual_words)
        
        # Random visual features
        features[257:512] = torch.randn(255, device=self.device) * 0.1
        
        return features.unsqueeze(0)
        
    def _extract_audio_features(self, content: Dict[str, Any]) -> torch.Tensor:
        """Extract audio features (simplified)"""
        # In practice, would use audio model
        # For now, generate based on content
        
        features = torch.zeros(768, device=self.device)
        
        # Check for audio descriptors
        audio_words = ['sound', 'music', 'voice', 'rhythm', 'tone', 'melody']
        text = content.get('content', '').lower()
        
        audio_score = sum(1 for word in audio_words if word in text)
        features[512] = audio_score / len(audio_words)
        
        # Random audio features
        features[513:768] = torch.randn(255, device=self.device) * 0.1
        
        return features.unsqueeze(0)
        
    async def _cross_modal_reasoning(self,
                                   text_features: torch.Tensor,
                                   visual_features: torch.Tensor,
                                   audio_features: torch.Tensor) -> torch.Tensor:
        """Perform cross-modal reasoning"""
        # Stack features for attention
        all_features = torch.cat([text_features, visual_features, audio_features], dim=0)
        
        # Self-attention across modalities
        with torch.no_grad():
            attended_features, _ = self.cross_attention(
                all_features,
                all_features,
                all_features
            )
            
        # Fusion
        fused = torch.cat([
            attended_features[0],  # Enhanced text
            attended_features[1],  # Enhanced visual
            attended_features[2]   # Enhanced audio
        ], dim=-1)
        
        # Final fusion network
        with torch.no_grad():
            output = self.fusion_network(fused)
            
        return output[0]


class CreativeAgent(TestAgentBase):
    """
    Specialized agent for creative content generation.
    
    Capabilities:
    - Parallel creative variations
    - Topological diversity optimization
    - Multi-modal reasoning
    - Style transfer
    - Brainstorming and ideation
    """
    
    def __init__(self, agent_id: str = "creative_agent_001", **kwargs):
        config = TestAgentConfig(
            agent_role=AgentRole.EXECUTOR,
            specialty="creative",
            target_latency_ms=200.0,  # Higher for creative processing
            **kwargs
        )
        
        super().__init__(agent_id=agent_id, config=config, **kwargs)
        
        # Initialize specialized components
        self.variation_generator = ParallelVariationGenerator()
        self.diversity_optimizer = TopologicalDiversityOptimizer()
        self.multimodal_reasoner = MultiModalReasoner()
        
        # Creative styles library
        self.style_library = {
            "formal": {"temperature": 0.3, "vocabulary": "academic", "structure": "rigid"},
            "casual": {"temperature": 0.7, "vocabulary": "conversational", "structure": "flexible"},
            "poetic": {"temperature": 0.9, "vocabulary": "metaphorical", "structure": "flowing"},
            "technical": {"temperature": 0.4, "vocabulary": "precise", "structure": "logical"},
            "humorous": {"temperature": 0.8, "vocabulary": "playful", "structure": "surprising"}
        }
        
        # Brainstorming techniques
        self.brainstorming_techniques = {
            "mind_mapping": self._brainstorm_mind_map,
            "scamper": self._brainstorm_scamper,
            "six_hats": self._brainstorm_six_hats,
            "morphological": self._brainstorm_morphological
        }
        
        # Initialize tools
        self._init_creative_tools()
        
        logger.info("Creative Agent initialized",
                   agent_id=agent_id,
                   capabilities=["variation_generation", "diversity_optimization", 
                               "multi_modal", "brainstorming"])
                   
    def _init_creative_tools(self):
        """Initialize creative-specific tools"""
        self.tools = {
            "generate_variations": Tool(
                name="generate_variations",
                description="Generate creative variations",
                func=self._tool_generate_variations
            ),
            "optimize_diversity": Tool(
                name="optimize_diversity",
                description="Optimize for creative diversity",
                func=self._tool_optimize_diversity
            ),
            "apply_style": Tool(
                name="apply_style",
                description="Apply creative style",
                func=self._tool_apply_style
            ),
            "brainstorm": Tool(
                name="brainstorm",
                description="Brainstorm ideas",
                func=self._tool_brainstorm
            ),
            "enhance_multimodal": Tool(
                name="enhance_multimodal",
                description="Enhance with multi-modal reasoning",
                func=self._tool_enhance_multimodal
            )
        }
        
    async def _handle_analyze(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle creative analysis requests"""
        content = context.get("original", {}).get("content", "")
        
        # Analyze creative aspects
        analysis = {
            "creativity_score": self._analyze_creativity(content),
            "style_profile": self._analyze_style(content),
            "emotional_tone": self._analyze_emotion(content),
            "structural_complexity": self._analyze_structure(content)
        }
        
        # Extract topological features for creative patterns
        if hasattr(content, '__len__'):
            # Create embedding for topological analysis
            embedding = torch.randn(768)  # Simplified
            
            # Store in memory
            await self.shape_memory.store(
                {
                    "type": "creative_analysis",
                    "content_snippet": content[:100] if isinstance(content, str) else str(content)[:100],
                    "analysis": analysis
                },
                embedding=embedding.numpy()
            )
            
        return analysis
        
    async def _handle_generate(self, context: Dict[str, Any]) -> CreativeOutput:
        """Handle creative generation requests"""
        start_time = time.perf_counter()
        
        prompt = context.get("original", {}).get("prompt", "")
        style = context.get("original", {}).get("style", "balanced")
        num_variations = context.get("original", {}).get("num_variations", 10)
        
        # Generate variations
        variations = await self.variation_generator.generate_batch(
            prompt,
            num_variations=num_variations * 3,  # Generate more for selection
            temperature=self.style_library.get(style, {}).get("temperature", 0.7)
        )
        
        # Optimize for diversity
        diverse_set = self.diversity_optimizer.select_diverse_subset(
            variations,
            target_diversity=0.8,
            subset_size=num_variations
        )
        
        # Multi-modal enhancement
        enhanced_content = await self.multimodal_reasoner.enhance(diverse_set)
        
        # Calculate metrics
        diversity_score = self._calculate_final_diversity(enhanced_content)
        quality_metrics = self._assess_quality(enhanced_content, prompt)
        
        # Extract topological signature
        embeddings = [c.get('multi_modal_features', c.get('embedding', np.random.randn(768))) 
                     for c in enhanced_content]
        topological_signature = np.mean(embeddings, axis=0) if embeddings else np.zeros(768)
        
        generation_time = (time.perf_counter() - start_time) * 1000
        
        return CreativeOutput(
            content=enhanced_content,
            diversity_score=diversity_score,
            quality_metrics=quality_metrics,
            topological_signature=topological_signature,
            generation_time_ms=generation_time,
            model_used="creative_ensemble",
            style_attributes=self.style_library.get(style, {})
        )
        
    def _analyze_creativity(self, content: Any) -> float:
        """Analyze creativity level"""
        if not isinstance(content, str):
            content = str(content)
            
        # Simple creativity metrics
        unique_words = len(set(content.lower().split()))
        total_words = len(content.split())
        
        # Vocabulary diversity
        diversity = unique_words / max(total_words, 1)
        
        # Check for creative indicators
        creative_words = ['imagine', 'create', 'innovate', 'unique', 'novel', 'original']
        creative_count = sum(1 for word in creative_words if word in content.lower())
        
        creativity_score = min(diversity * 5 + creative_count * 0.5, 10.0)
        
        return creativity_score
        
    def _analyze_style(self, content: Any) -> Dict[str, float]:
        """Analyze style profile"""
        if not isinstance(content, str):
            content = str(content)
            
        words = content.lower().split()
        
        # Style indicators
        formal_words = ['therefore', 'moreover', 'furthermore', 'consequently']
        casual_words = ['gonna', 'wanna', 'yeah', 'cool', 'awesome']
        technical_words = ['algorithm', 'implementation', 'architecture', 'optimize']
        
        formal_score = sum(1 for word in formal_words if word in words) / max(len(words), 1)
        casual_score = sum(1 for word in casual_words if word in words) / max(len(words), 1)
        technical_score = sum(1 for word in technical_words if word in words) / max(len(words), 1)
        
        return {
            "formal": min(formal_score * 100, 1.0),
            "casual": min(casual_score * 100, 1.0),
            "technical": min(technical_score * 100, 1.0),
            "balanced": 1.0 - max(formal_score, casual_score, technical_score)
        }
        
    def _analyze_emotion(self, content: Any) -> Dict[str, float]:
        """Analyze emotional tone"""
        if not isinstance(content, str):
            content = str(content)
            
        # Simple emotion detection
        emotions = {
            "positive": ['happy', 'joy', 'excited', 'wonderful', 'great'],
            "negative": ['sad', 'angry', 'frustrated', 'terrible', 'awful'],
            "neutral": ['okay', 'fine', 'normal', 'regular', 'standard']
        }
        
        scores = {}
        words = content.lower().split()
        
        for emotion, keywords in emotions.items():
            count = sum(1 for word in keywords if word in words)
            scores[emotion] = min(count / max(len(words), 1) * 10, 1.0)
            
        return scores
        
    def _analyze_structure(self, content: Any) -> float:
        """Analyze structural complexity"""
        if not isinstance(content, str):
            content = str(content)
            
        # Simple complexity metrics
        sentences = content.count('.') + content.count('!') + content.count('?')
        avg_sentence_length = len(content.split()) / max(sentences, 1)
        
        # Complexity score based on sentence length and structure
        if avg_sentence_length < 10:
            complexity = 0.3
        elif avg_sentence_length < 20:
            complexity = 0.6
        else:
            complexity = 0.9
            
        return complexity
        
    def _calculate_final_diversity(self, content_list: List[Dict[str, Any]]) -> float:
        """Calculate diversity score of final output"""
        if len(content_list) < 2:
            return 0.0
            
        # Extract features
        features = []
        for c in content_list:
            if 'multi_modal_features' in c:
                features.append(c['multi_modal_features'])
            elif 'embedding' in c:
                features.append(c['embedding'])
                
        if len(features) < 2:
            return 0.5
            
        # Calculate pairwise distances
        features = np.array(features)
        distances = []
        
        for i in range(len(features)):
            for j in range(i + 1, len(features)):
                dist = np.linalg.norm(features[i] - features[j])
                distances.append(dist)
                
        # Average distance as diversity metric
        avg_distance = np.mean(distances) if distances else 0
        
        # Normalize to 0-1 range
        return min(avg_distance / 10.0, 1.0)
        
    def _assess_quality(self, content_list: List[Dict[str, Any]], prompt: str) -> Dict[str, float]:
        """Assess quality of generated content"""
        quality_scores = {
            "relevance": [],
            "coherence": [],
            "originality": [],
            "engagement": []
        }
        
        for content in content_list:
            # Relevance to prompt (simplified)
            text = content.get('content', '')
            prompt_words = set(prompt.lower().split())
            content_words = set(text.lower().split())
            
            overlap = len(prompt_words & content_words)
            relevance = min(overlap / max(len(prompt_words), 1), 1.0)
            quality_scores["relevance"].append(relevance)
            
            # Coherence (based on structure)
            coherence = 0.7  # Default
            if len(text.split()) > 10:
                coherence = 0.8
            if '.' in text and ',' in text:
                coherence = 0.9
            quality_scores["coherence"].append(coherence)
            
            # Originality (based on variation strength)
            originality = content.get('variation_strength', 0.5)
            quality_scores["originality"].append(min(originality, 1.0))
            
            # Engagement (based on emotional tone)
            modality_scores = content.get('modality_scores', {})
            engagement = sum(modality_scores.values()) / max(len(modality_scores), 1)
            quality_scores["engagement"].append(min(engagement / 4, 1.0))
            
        # Average scores
        return {
            metric: np.mean(scores) if scores else 0.0
            for metric, scores in quality_scores.items()
        }
        
    # Brainstorming techniques
    async def _brainstorm_mind_map(self, topic: str) -> List[Dict[str, Any]]:
        """Mind mapping brainstorming"""
        ideas = []
        
        # Central concept
        central = {"level": 0, "content": topic, "connections": []}
        ideas.append(central)
        
        # Primary branches (simplified)
        branches = ["What", "Why", "How", "When", "Where", "Who"]
        
        for i, branch in enumerate(branches):
            idea = {
                "level": 1,
                "content": f"{branch} {topic}",
                "parent": topic,
                "connections": []
            }
            ideas.append(idea)
            
            # Sub-branches
            for j in range(2):
                sub_idea = {
                    "level": 2,
                    "content": f"Aspect {j+1} of {branch} {topic}",
                    "parent": f"{branch} {topic}"
                }
                ideas.append(sub_idea)
                
        return ideas
        
    async def _brainstorm_scamper(self, topic: str) -> List[Dict[str, Any]]:
        """SCAMPER brainstorming technique"""
        scamper = {
            "Substitute": "What can be substituted?",
            "Combine": "What can be combined?",
            "Adapt": "What can be adapted?",
            "Modify": "What can be modified or magnified?",
            "Put to other uses": "How else can this be used?",
            "Eliminate": "What can be eliminated?",
            "Reverse": "What can be reversed or rearranged?"
        }
        
        ideas = []
        for technique, question in scamper.items():
            ideas.append({
                "technique": technique,
                "question": question,
                "application": f"{question} in context of {topic}",
                "potential_ideas": [
                    f"Idea 1 for {technique}",
                    f"Idea 2 for {technique}"
                ]
            })
            
        return ideas
        
    async def _brainstorm_six_hats(self, topic: str) -> List[Dict[str, Any]]:
        """Six Thinking Hats brainstorming"""
        hats = {
            "White": "Facts and information",
            "Red": "Emotions and feelings",
            "Black": "Critical and cautious",
            "Yellow": "Positive and optimistic",
            "Green": "Creative and alternatives",
            "Blue": "Process and control"
        }
        
        ideas = []
        for color, focus in hats.items():
            ideas.append({
                "hat": color,
                "focus": focus,
                "perspective": f"{focus} view of {topic}",
                "insights": [
                    f"{color} hat insight 1",
                    f"{color} hat insight 2"
                ]
            })
            
        return ideas
        
    async def _brainstorm_morphological(self, topic: str) -> List[Dict[str, Any]]:
        """Morphological analysis brainstorming"""
        # Define parameters (simplified)
        parameters = {
            "Form": ["Physical", "Digital", "Hybrid"],
            "Function": ["Create", "Transform", "Connect"],
            "Material": ["Concrete", "Abstract", "Mixed"]
        }
        
        ideas = []
        
        # Generate combinations
        for form in parameters["Form"]:
            for function in parameters["Function"]:
                for material in parameters["Material"]:
                    ideas.append({
                        "combination": f"{form}-{function}-{material}",
                        "description": f"{topic} as {form} to {function} using {material}",
                        "novelty_score": np.random.uniform(0.5, 1.0)
                    })
                    
        return ideas
        
    # Tool implementations
    async def _tool_generate_variations(self,
                                      prompt: str,
                                      count: int = 10,
                                      temperature: float = 0.7) -> List[Dict[str, Any]]:
        """Generate variations tool"""
        variations = await self.variation_generator.generate_batch(
            prompt,
            num_variations=count,
            temperature=temperature
        )
        
        return variations
        
    async def _tool_optimize_diversity(self,
                                     variations: List[Dict[str, Any]],
                                     target_count: int = 5) -> List[Dict[str, Any]]:
        """Optimize diversity tool"""
        diverse_set = self.diversity_optimizer.select_diverse_subset(
            variations,
            target_diversity=0.8,
            subset_size=target_count
        )
        
        return diverse_set
        
    async def _tool_apply_style(self,
                              content: str,
                              style: str = "balanced") -> Dict[str, Any]:
        """Apply style tool"""
        style_params = self.style_library.get(style, self.style_library["casual"])
        
        # Generate styled variation
        styled = await self.variation_generator.generate_batch(
            content,
            num_variations=1,
            temperature=style_params["temperature"]
        )
        
        if styled:
            return {
                "original": content,
                "styled": styled[0]["content"],
                "style": style,
                "parameters": style_params
            }
        else:
            return {"error": "Style application failed"}
            
    async def _tool_brainstorm(self,
                             topic: str,
                             technique: str = "mind_mapping") -> List[Dict[str, Any]]:
        """Brainstorming tool"""
        if technique in self.brainstorming_techniques:
            ideas = await self.brainstorming_techniques[technique](topic)
        else:
            # Default to mind mapping
            ideas = await self._brainstorm_mind_map(topic)
            
        return ideas
        
    async def _tool_enhance_multimodal(self,
                                     content_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Multi-modal enhancement tool"""
        enhanced = await self.multimodal_reasoner.enhance(content_list)
        return enhanced


# Factory function
def create_creative_agent(agent_id: Optional[str] = None, **kwargs) -> CreativeAgent:
    """Create a Creative Agent instance"""
    if agent_id is None:
        agent_id = f"creative_agent_{int(time.time())}"
        
    return CreativeAgent(agent_id=agent_id, **kwargs)