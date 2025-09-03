"""
Priority Replay Buffer with Surprise-Based Ranking
===================================================

Implements a sophisticated replay buffer that prioritizes memories
based on prediction error (surprise) from the CausalPatternTracker.

Key features:
- Max-heap for O(log n) priority operations
- Batch surprise scoring for efficiency
- Memory caching to avoid repeated fetches
- Semantic distance calculation for dream pairs
- Binary spike encoding support (SESLR)
"""

import heapq
import numpy as np
from typing import List, Tuple, Any, Optional, Dict, Set
from dataclasses import dataclass, field
from datetime import datetime
import hashlib
import structlog

logger = structlog.get_logger(__name__)


@dataclass
class ReplayMemory:
    """Enhanced memory representation for replay"""
    id: str
    content: Any
    embedding: np.ndarray
    topology_signature: Optional[np.ndarray] = None
    importance: float = 0.5
    access_count: int = 0
    surprise_score: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
    source_tier: str = "episodic"
    binary_spike: Optional[bytes] = None  # SESLR encoding
    
    def __lt__(self, other):
        """For heap comparison - higher surprise is better"""
        return self.surprise_score < other.surprise_score
    
    def __hash__(self):
        """Hash for set operations"""
        return hash(self.id)


class PriorityReplayBuffer:
    """
    Advanced replay buffer with surprise-based prioritization
    
    This buffer maintains memories in a priority queue where priority
    is determined by the prediction error (surprise) from the causal
    pattern tracker. This ensures consolidation focuses on the most
    informative experiences.
    """
    
    def __init__(self, causal_tracker: Any, max_size: int = 10000,
                 enable_binary_spikes: bool = True):
        """
        Initialize the priority replay buffer
        
        Args:
            causal_tracker: CausalPatternTracker instance for surprise scoring
            max_size: Maximum buffer size
            enable_binary_spikes: Enable SESLR binary spike encoding
        """
        self.causal_tracker = causal_tracker
        self.max_size = max_size
        self.enable_binary_spikes = enable_binary_spikes
        
        # Main priority queue (max-heap using negative scores)
        self.buffer: List[Tuple[float, ReplayMemory]] = []
        
        # Memory cache for fast access
        self.memory_cache: Dict[str, ReplayMemory] = {}
        
        # Track replayed memories
        self.replayed_ids: Set[str] = set()
        
        # Statistics
        self.total_populated = 0
        self.total_evicted = 0
        self.highest_surprise = 0.0
        self.lowest_surprise = float('inf')
        
        logger.info(
            "PriorityReplayBuffer initialized",
            max_size=max_size,
            binary_spikes=enable_binary_spikes
        )
    
    async def populate(self, candidates: List[Any]) -> int:
        """
        Populate buffer with memory candidates, prioritized by surprise
        
        Args:
            candidates: List of memory candidates to add
        
        Returns:
            Number of memories added to buffer
        """
        if not candidates:
            return 0
        
        # Convert candidates to ReplayMemory objects
        replay_memories = []
        for candidate in candidates:
            replay_memory = self._convert_to_replay_memory(candidate)
            replay_memories.append(replay_memory)
        
        # Batch calculate surprise scores
        contents = [m.content for m in replay_memories]
        surprise_scores = await self._batch_calculate_surprise(contents)
        
        # Update surprise scores
        for memory, score in zip(replay_memories, surprise_scores):
            memory.surprise_score = score
            self.highest_surprise = max(self.highest_surprise, score)
            self.lowest_surprise = min(self.lowest_surprise, score)
        
        # Add to buffer with priority management
        added_count = 0
        for memory in replay_memories:
            if self._add_to_buffer(memory):
                added_count += 1
        
        self.total_populated += added_count
        
        logger.info(
            "Buffer populated",
            candidates=len(candidates),
            added=added_count,
            buffer_size=len(self.buffer),
            highest_surprise=f"{self.highest_surprise:.3f}",
            lowest_surprise=f"{self.lowest_surprise:.3f}"
        )
        
        return added_count
    
    def get_top_candidates(self, n: int) -> List[ReplayMemory]:
        """
        Get the N most surprising memories for replay
        
        Args:
            n: Number of candidates to retrieve
        
        Returns:
            List of top N memories by surprise score
        """
        # Use heapq.nlargest for efficient extraction
        top_items = heapq.nlargest(n, self.buffer)
        
        # Extract memories and mark as replayed
        top_memories = []
        for _, memory in top_items:
            top_memories.append(memory)
            self.replayed_ids.add(memory.id)
        
        logger.debug(
            "Retrieved top candidates",
            requested=n,
            returned=len(top_memories),
            avg_surprise=np.mean([m.surprise_score for m in top_memories]) if top_memories else 0
        )
        
        return top_memories
    
    def select_distant_pairs(self, count: int = 50) -> List[Tuple[ReplayMemory, ReplayMemory]]:
        """
        Select semantically distant memory pairs for dream generation
        
        This method finds pairs of memories that are:
        1. Both high-surprise (important)
        2. Semantically distant (different contexts)
        3. Not recently paired together
        
        Args:
            count: Number of pairs to select
        
        Returns:
            List of memory pairs for dreaming
        """
        # Get high-surprise memories
        high_surprise_memories = self.get_top_candidates(min(count * 3, len(self.buffer)))
        
        if len(high_surprise_memories) < 2:
            return []
        
        # Calculate pairwise distances
        pairs_with_distance = []
        
        for i in range(len(high_surprise_memories)):
            for j in range(i + 1, len(high_surprise_memories)):
                m1, m2 = high_surprise_memories[i], high_surprise_memories[j]
                
                # Calculate semantic distance
                distance = self._calculate_semantic_distance(m1, m2)
                
                # Add to candidates if sufficiently distant
                if distance > 0.5:  # Threshold for "distant"
                    pairs_with_distance.append((distance, m1, m2))
        
        # Sort by distance (descending) and take top pairs
        pairs_with_distance.sort(reverse=True)
        selected_pairs = [(m1, m2) for _, m1, m2 in pairs_with_distance[:count]]
        
        logger.info(
            "Selected distant pairs for dreaming",
            requested=count,
            selected=len(selected_pairs),
            avg_distance=np.mean([d for d, _, _ in pairs_with_distance[:count]]) if pairs_with_distance else 0
        )
        
        return selected_pairs
    
    def get_replayed_ids(self) -> List[str]:
        """Get IDs of memories that have been replayed"""
        return list(self.replayed_ids)
    
    def clear_replayed(self):
        """Clear the replayed tracking (for new cycle)"""
        self.replayed_ids.clear()
    
    # ==================== Binary Spike Encoding (SESLR) ====================
    
    def encode_to_binary_spike(self, memory: ReplayMemory) -> bytes:
        """
        Encode memory to binary spike format (SESLR protocol)
        
        This achieves 32x compression by encoding as binary spikes
        
        Args:
            memory: Memory to encode
        
        Returns:
            Binary spike representation
        """
        if not self.enable_binary_spikes:
            return b""
        
        # Convert embedding to binary spikes
        # Threshold at median to create binary representation
        embedding = memory.embedding
        threshold = np.median(embedding)
        binary_array = (embedding > threshold).astype(np.uint8)
        
        # Pack into bytes (8 bits per byte)
        packed_bytes = np.packbits(binary_array)
        
        # Add metadata header
        header = np.array([
            len(embedding),  # Original dimension
            int(threshold * 1000),  # Threshold (scaled)
            int(memory.surprise_score * 1000),  # Surprise (scaled)
            int(memory.importance * 1000)  # Importance (scaled)
        ], dtype=np.uint32)
        
        # Combine header and packed data
        binary_spike = header.tobytes() + packed_bytes.tobytes()
        
        return binary_spike
    
    def decode_from_binary_spike(self, binary_spike: bytes) -> np.ndarray:
        """
        Decode binary spike back to approximate embedding
        
        Args:
            binary_spike: Binary spike representation
        
        Returns:
            Reconstructed embedding
        """
        if not binary_spike:
            return np.array([])
        
        # Extract header (4 uint32 values = 16 bytes)
        header = np.frombuffer(binary_spike[:16], dtype=np.uint32)
        original_dim = header[0]
        threshold = header[1] / 1000.0
        
        # Extract packed data
        packed_data = np.frombuffer(binary_spike[16:], dtype=np.uint8)
        
        # Unpack bits
        binary_array = np.unpackbits(packed_data)[:original_dim]
        
        # Reconstruct approximate embedding
        # Use threshold +/- small noise for binary values
        noise = np.random.normal(0, 0.1, original_dim)
        embedding = np.where(
            binary_array,
            threshold + 0.2 + noise,  # Above threshold
            threshold - 0.2 + noise   # Below threshold
        )
        
        return embedding
    
    # ==================== Helper Methods ====================
    
    def _convert_to_replay_memory(self, candidate: Any) -> ReplayMemory:
        """Convert a candidate to ReplayMemory format"""
        # Extract fields based on candidate type
        if isinstance(candidate, ReplayMemory):
            return candidate
        
        # Convert from generic memory object
        replay_memory = ReplayMemory(
            id=getattr(candidate, 'id', self._generate_id(candidate)),
            content=getattr(candidate, 'content', candidate),
            embedding=getattr(candidate, 'embedding', np.random.randn(384)),
            topology_signature=getattr(candidate, 'topology_signature', None),
            importance=getattr(candidate, 'importance', 0.5),
            access_count=getattr(candidate, 'access_count', 0),
            timestamp=getattr(candidate, 'timestamp', datetime.now()),
            source_tier=getattr(candidate, 'tier', 'episodic')
        )
        
        # Encode to binary spike if enabled
        if self.enable_binary_spikes:
            replay_memory.binary_spike = self.encode_to_binary_spike(replay_memory)
        
        return replay_memory
    
    def _generate_id(self, content: Any) -> str:
        """Generate unique ID for content"""
        content_str = str(content)
        return hashlib.sha256(content_str.encode()).hexdigest()[:16]
    
    async def _batch_calculate_surprise(self, contents: List[Any]) -> List[float]:
        """
        Batch calculate surprise scores using CausalPatternTracker
        
        Args:
            contents: List of memory contents
        
        Returns:
            List of surprise scores
        """
        try:
            # Use causal tracker's batch prediction
            if hasattr(self.causal_tracker, 'batch_predict_surprise'):
                scores = await self.causal_tracker.batch_predict_surprise(contents)
            else:
                # Fallback to individual calculation
                scores = []
                for content in contents:
                    score = await self._calculate_single_surprise(content)
                    scores.append(score)
            
            return scores
            
        except Exception as e:
            logger.error(f"Error calculating surprise scores: {e}")
            # Return default scores
            return [0.5] * len(contents)
    
    async def _calculate_single_surprise(self, content: Any) -> float:
        """Calculate surprise score for single memory"""
        try:
            # Convert content to format expected by causal tracker
            if hasattr(content, 'to_dict'):
                content_dict = content.to_dict()
            else:
                content_dict = {"content": str(content)}
            
            # Predict and get surprise
            prediction = await self.causal_tracker.predict_outcome(content_dict)
            
            # Surprise is the prediction error
            if isinstance(prediction, dict):
                # Use failure probability as surprise (unexpected outcomes)
                surprise = prediction.get('failure_probability', 0.5)
            else:
                surprise = 0.5  # Default
            
            return surprise
            
        except Exception as e:
            logger.debug(f"Could not calculate surprise: {e}")
            return 0.5  # Default surprise
    
    def _add_to_buffer(self, memory: ReplayMemory) -> bool:
        """
        Add memory to buffer with priority management
        
        Args:
            memory: Memory to add
        
        Returns:
            True if added, False if rejected
        """
        # Check if already in buffer
        if memory.id in self.memory_cache:
            # Update surprise score if higher
            existing = self.memory_cache[memory.id]
            if memory.surprise_score > existing.surprise_score:
                existing.surprise_score = memory.surprise_score
                # Re-heapify to maintain order
                heapq.heapify(self.buffer)
            return False
        
        # Add to buffer if space available
        if len(self.buffer) < self.max_size:
            # Use negative score for max-heap behavior
            heapq.heappush(self.buffer, (-memory.surprise_score, memory))
            self.memory_cache[memory.id] = memory
            return True
        
        # Check if should replace lowest priority
        if memory.surprise_score > -self.buffer[0][0]:
            # Remove lowest priority
            _, evicted = heapq.heappop(self.buffer)
            del self.memory_cache[evicted.id]
            self.total_evicted += 1
            
            # Add new memory
            heapq.heappush(self.buffer, (-memory.surprise_score, memory))
            self.memory_cache[memory.id] = memory
            return True
        
        return False
    
    def _calculate_semantic_distance(self, m1: ReplayMemory, m2: ReplayMemory) -> float:
        """
        Calculate semantic distance between two memories
        
        Args:
            m1, m2: Memories to compare
        
        Returns:
            Distance score (0 = identical, 1 = maximally different)
        """
        # Use embeddings for distance calculation
        if m1.embedding is not None and m2.embedding is not None:
            # Cosine distance
            dot_product = np.dot(m1.embedding, m2.embedding)
            norm_product = np.linalg.norm(m1.embedding) * np.linalg.norm(m2.embedding)
            
            if norm_product > 0:
                cosine_similarity = dot_product / norm_product
                # Convert to distance (0 = similar, 1 = different)
                distance = 1.0 - (cosine_similarity + 1.0) / 2.0
            else:
                distance = 0.5
        else:
            # Fallback to random distance
            distance = np.random.uniform(0.3, 0.7)
        
        # Adjust based on tier difference
        if m1.source_tier != m2.source_tier:
            distance += 0.1  # Different tiers are more distant
        
        # Adjust based on time difference
        time_diff = abs((m1.timestamp - m2.timestamp).total_seconds())
        time_factor = min(0.2, time_diff / 86400)  # Max 0.2 for 1+ day difference
        distance += time_factor
        
        return min(1.0, distance)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get buffer statistics"""
        surprise_scores = [-score for score, _ in self.buffer]
        
        return {
            "buffer_size": len(self.buffer),
            "max_size": self.max_size,
            "total_populated": self.total_populated,
            "total_evicted": self.total_evicted,
            "replayed_count": len(self.replayed_ids),
            "highest_surprise": self.highest_surprise,
            "lowest_surprise": self.lowest_surprise,
            "avg_surprise": np.mean(surprise_scores) if surprise_scores else 0,
            "std_surprise": np.std(surprise_scores) if surprise_scores else 0,
            "fill_ratio": len(self.buffer) / self.max_size
        }
    
    def clear(self):
        """Clear the buffer completely"""
        self.buffer.clear()
        self.memory_cache.clear()
        self.replayed_ids.clear()
        self.total_populated = 0
        self.total_evicted = 0
        self.highest_surprise = 0.0
        self.lowest_surprise = float('inf')
        
        logger.info("Replay buffer cleared")