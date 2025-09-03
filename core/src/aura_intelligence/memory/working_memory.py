"""
Working Memory System - Cognitive Architecture Implementation
=============================================================

Implements a biologically-accurate working memory system based on:
- Baddeley & Hitch model with central executive
- Miller's 7±2 capacity limit
- Cowan's focus of attention (4±1 items)
- Phonological loop and visuospatial sketchpad
- Episodic buffer for integration

Key Features:
- Dynamic capacity management (5-9 items)
- Attention-based prioritization
- Active rehearsal mechanisms
- Interference detection and resolution
- Integration with long-term memory
"""

import asyncio
import time
import numpy as np
from typing import Dict, Any, List, Optional, Set, Tuple, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import deque, OrderedDict
from enum import Enum
import hashlib
import structlog

logger = structlog.get_logger(__name__)


# ==================== Data Structures ====================

class MemoryModalType(Enum):
    """Modality types for working memory items"""
    VERBAL = "verbal"              # Phonological loop
    VISUAL = "visual"              # Visuospatial sketchpad
    SPATIAL = "spatial"            # Spatial component
    EPISODIC = "episodic"          # Episodic buffer
    SEMANTIC = "semantic"          # Semantic information
    PROCEDURAL = "procedural"      # Action sequences
    MULTIMODAL = "multimodal"      # Combined modalities


class AttentionState(Enum):
    """Attention states for memory items"""
    FOCUSED = "focused"            # In focus of attention (1-4 items)
    ACTIVE = "active"              # Actively maintained (5-9 items)
    PERIPHERAL = "peripheral"      # At edge of awareness
    DECAYING = "decaying"          # Losing activation
    DORMANT = "dormant"            # No longer active


@dataclass
class WorkingMemoryItem:
    """Individual item in working memory"""
    id: str
    content: Any
    modality: MemoryModalType
    embedding: np.ndarray
    
    # Cognitive properties
    activation_level: float = 1.0          # Current activation strength
    attention_weight: float = 0.0          # Attention allocation
    rehearsal_count: int = 0               # Times rehearsed
    interference_level: float = 0.0        # Interference from other items
    
    # Temporal properties
    entry_time: datetime = field(default_factory=datetime.now)
    last_access_time: datetime = field(default_factory=datetime.now)
    last_rehearsal_time: Optional[datetime] = None
    decay_rate: float = 0.1                # Item-specific decay
    
    # Relational properties
    associations: Set[str] = field(default_factory=set)
    chunk_members: Set[str] = field(default_factory=set)
    context: Dict[str, Any] = field(default_factory=dict)
    
    # State
    attention_state: AttentionState = AttentionState.ACTIVE
    is_rehearsing: bool = False
    is_consolidated: bool = False
    
    def __hash__(self):
        return hash(self.id)
    
    def __lt__(self, other):
        """For priority queue operations"""
        return self.activation_level < other.activation_level


@dataclass
class MemoryChunk:
    """Chunked items for increased capacity (Miller's chunking)"""
    id: str
    member_ids: Set[str]
    chunk_embedding: np.ndarray
    coherence_score: float = 0.0
    creation_time: datetime = field(default_factory=datetime.now)
    activation_level: float = 1.0


@dataclass
class WorkingMemoryMetrics:
    """Metrics for monitoring working memory performance"""
    current_capacity: int = 7
    items_in_focus: int = 0
    items_active: int = 0
    items_peripheral: int = 0
    
    total_items_processed: int = 0
    total_rehearsals: int = 0
    total_consolidations: int = 0
    total_interference_resolutions: int = 0
    
    avg_activation_level: float = 0.0
    avg_attention_weight: float = 0.0
    cognitive_load: float = 0.0
    
    successful_recalls: int = 0
    failed_recalls: int = 0
    recall_accuracy: float = 0.0


# ==================== Central Executive Controller ====================

class CentralExecutive:
    """
    Central executive component managing attention and control
    
    Based on Baddeley's model, controls:
    - Attention focusing and switching
    - Dual-task coordination
    - Retrieval from long-term memory
    - Inhibition of irrelevant information
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the central executive"""
        self.config = config or {}
        
        # Attention parameters
        self.focus_capacity = self.config.get('focus_capacity', 4)  # Cowan's limit
        self.attention_resources = 1.0  # Total attention available
        self.switching_cost = 0.1  # Cost of attention switching
        
        # Current focus
        self.focused_items: OrderedDict[str, float] = OrderedDict()
        self.attention_history: deque = deque(maxlen=100)
        
        # Task management
        self.active_tasks: List[str] = []
        self.task_priorities: Dict[str, float] = {}
        
        logger.info(
            "CentralExecutive initialized",
            focus_capacity=self.focus_capacity
        )
    
    def allocate_attention(self, items: List[WorkingMemoryItem]) -> Dict[str, float]:
        """
        Allocate attention resources across items
        
        Uses a combination of:
        - Recency
        - Relevance
        - Activation level
        - Task demands
        """
        if not items:
            return {}
        
        attention_weights = {}
        
        # Calculate raw scores
        scores = []
        for item in items:
            # Recency factor
            recency = 1.0 / (1.0 + (datetime.now() - item.last_access_time).total_seconds())
            
            # Activation factor
            activation = item.activation_level
            
            # Relevance factor (based on associations)
            relevance = len(item.associations) / 10.0  # Normalize
            
            # Combined score
            score = (
                0.4 * activation +
                0.3 * recency +
                0.2 * relevance +
                0.1 * (1.0 - item.interference_level)
            )
            scores.append((item.id, score))
        
        # Sort by score
        scores.sort(key=lambda x: x[1], reverse=True)
        
        # Allocate attention with focus limit
        total_weight = 0.0
        for i, (item_id, score) in enumerate(scores):
            if i < self.focus_capacity:
                # Items in focus get more attention
                weight = score * 0.8
            else:
                # Items outside focus get less
                weight = score * 0.2 * (1.0 / (i - self.focus_capacity + 2))
            
            attention_weights[item_id] = weight
            total_weight += weight
        
        # Normalize to sum to available resources
        if total_weight > 0:
            for item_id in attention_weights:
                attention_weights[item_id] = (
                    attention_weights[item_id] / total_weight * self.attention_resources
                )
        
        # Update focused items
        self.focused_items.clear()
        for item_id, weight in list(attention_weights.items())[:self.focus_capacity]:
            self.focused_items[item_id] = weight
        
        # Record history
        self.attention_history.append({
            'timestamp': datetime.now(),
            'focused': list(self.focused_items.keys()),
            'total_items': len(items)
        })
        
        return attention_weights
    
    def switch_attention(self, from_item: str, to_item: str) -> float:
        """
        Switch attention between items
        
        Returns the switching cost
        """
        cost = self.switching_cost
        
        # Remove from focus if present
        if from_item in self.focused_items:
            del self.focused_items[from_item]
            cost *= 1.5  # Higher cost for removing from focus
        
        # Add to focus if space available
        if len(self.focused_items) < self.focus_capacity:
            self.focused_items[to_item] = 0.5  # Initial weight
        else:
            # Need to remove least important item
            if self.focused_items:
                least_important = min(self.focused_items.items(), key=lambda x: x[1])
                del self.focused_items[least_important[0]]
                self.focused_items[to_item] = 0.5
                cost *= 2.0  # Highest cost for forced replacement
        
        # Reduce available resources temporarily
        self.attention_resources = max(0.5, self.attention_resources - cost)
        
        return cost
    
    def inhibit_item(self, item_id: str):
        """Inhibit an item from entering focus"""
        if item_id in self.focused_items:
            del self.focused_items[item_id]
    
    def coordinate_dual_task(self, task1: str, task2: str) -> Tuple[float, float]:
        """
        Coordinate resources between two tasks
        
        Returns resource allocation for each task
        """
        # Simple resource sharing model
        if task1 == task2:
            return (1.0, 0.0)
        
        # Check for interference
        interference = self._calculate_task_interference(task1, task2)
        
        # Allocate based on priorities and interference
        priority1 = self.task_priorities.get(task1, 0.5)
        priority2 = self.task_priorities.get(task2, 0.5)
        
        total_priority = priority1 + priority2
        if total_priority > 0:
            allocation1 = (priority1 / total_priority) * (1.0 - interference)
            allocation2 = (priority2 / total_priority) * (1.0 - interference)
        else:
            allocation1 = allocation2 = 0.5 * (1.0 - interference)
        
        return (allocation1, allocation2)
    
    def _calculate_task_interference(self, task1: str, task2: str) -> float:
        """Calculate interference between tasks"""
        # Simplified interference model
        if 'verbal' in task1 and 'verbal' in task2:
            return 0.7  # High interference for same modality
        elif 'visual' in task1 and 'visual' in task2:
            return 0.6
        elif 'spatial' in task1 and 'spatial' in task2:
            return 0.5
        else:
            return 0.2  # Low interference for different modalities


# ==================== Rehearsal Mechanisms ====================

class RehearsalLoop:
    """
    Implements rehearsal mechanisms for maintenance
    
    Includes:
    - Articulatory rehearsal (phonological loop)
    - Visual rehearsal (visuospatial sketchpad)
    - Elaborative rehearsal (semantic processing)
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize rehearsal mechanisms"""
        self.config = config or {}
        
        # Rehearsal parameters
        self.rehearsal_rate = self.config.get('rehearsal_rate', 2.0)  # Items per second
        self.rehearsal_decay = self.config.get('rehearsal_decay', 0.1)
        self.elaboration_threshold = self.config.get('elaboration_threshold', 5)
        
        # Rehearsal queues by modality
        self.rehearsal_queues: Dict[MemoryModalType, deque] = {
            modality: deque(maxlen=10)
            for modality in MemoryModalType
        }
        
        # Active rehearsals
        self.active_rehearsals: Set[str] = set()
        self.rehearsal_counts: Dict[str, int] = {}
        
        logger.info(
            "RehearsalLoop initialized",
            rate=self.rehearsal_rate
        )
    
    async def rehearse_item(self, item: WorkingMemoryItem) -> float:
        """
        Rehearse a single item
        
        Returns the activation boost
        """
        if item.id in self.active_rehearsals:
            return 0.0  # Already rehearsing
        
        self.active_rehearsals.add(item.id)
        
        try:
            # Determine rehearsal type based on modality
            if item.modality == MemoryModalType.VERBAL:
                boost = await self._articulatory_rehearsal(item)
            elif item.modality in [MemoryModalType.VISUAL, MemoryModalType.SPATIAL]:
                boost = await self._visual_rehearsal(item)
            else:
                boost = await self._elaborative_rehearsal(item)
            
            # Update item
            item.rehearsal_count += 1
            item.last_rehearsal_time = datetime.now()
            item.activation_level = min(1.0, item.activation_level + boost)
            
            # Track rehearsal count
            self.rehearsal_counts[item.id] = self.rehearsal_counts.get(item.id, 0) + 1
            
            # Check for elaboration opportunity
            if self.rehearsal_counts[item.id] >= self.elaboration_threshold:
                await self._trigger_elaboration(item)
            
            return boost
            
        finally:
            self.active_rehearsals.discard(item.id)
    
    async def _articulatory_rehearsal(self, item: WorkingMemoryItem) -> float:
        """
        Phonological loop rehearsal
        
        Simulates subvocal articulation
        """
        # Calculate rehearsal duration based on content length
        content_length = len(str(item.content)) if item.content else 10
        duration = content_length / (self.rehearsal_rate * 10)
        
        # Simulate rehearsal time
        await asyncio.sleep(min(duration, 0.5))
        
        # Boost based on rehearsal quality
        boost = 0.2 * (1.0 - item.interference_level)
        
        # Add to rehearsal queue
        self.rehearsal_queues[MemoryModalType.VERBAL].append(item.id)
        
        return boost
    
    async def _visual_rehearsal(self, item: WorkingMemoryItem) -> float:
        """
        Visuospatial sketchpad rehearsal
        
        Simulates visual imagery maintenance
        """
        # Visual rehearsal is faster but less effective
        await asyncio.sleep(0.1)
        
        # Boost based on visual complexity
        complexity = np.std(item.embedding) if item.embedding is not None else 0.5
        boost = 0.15 * complexity * (1.0 - item.interference_level)
        
        # Add to rehearsal queue
        self.rehearsal_queues[item.modality].append(item.id)
        
        return boost
    
    async def _elaborative_rehearsal(self, item: WorkingMemoryItem) -> float:
        """
        Semantic elaboration for deeper processing
        
        Creates associations and meaning
        """
        # Elaboration takes longer but is more effective
        await asyncio.sleep(0.3)
        
        # Create semantic associations
        if len(item.associations) < 5:
            # Generate associations based on embedding similarity
            item.associations.add(f"assoc_{len(item.associations)}")
        
        # Strong boost for elaboration
        boost = 0.3 * (1.0 + len(item.associations) / 10.0)
        
        # Mark for potential consolidation
        if len(item.associations) >= 3:
            item.is_consolidated = True
        
        return boost
    
    async def _trigger_elaboration(self, item: WorkingMemoryItem):
        """Trigger deep elaborative processing"""
        logger.debug(f"Triggering elaboration for item {item.id}")
        
        # Mark for transfer to long-term memory
        item.is_consolidated = True
        
        # Boost activation significantly
        item.activation_level = min(1.0, item.activation_level + 0.3)
    
    def get_rehearsal_schedule(self, items: List[WorkingMemoryItem]) -> List[str]:
        """
        Generate optimal rehearsal schedule
        
        Prioritizes items close to decay threshold
        """
        schedule = []
        
        # Sort by decay urgency
        decay_scores = []
        for item in items:
            if item.activation_level < 0.5:  # Below threshold
                urgency = (0.5 - item.activation_level) * 2.0
                decay_scores.append((item.id, urgency))
        
        decay_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Add to schedule
        for item_id, _ in decay_scores[:5]:  # Max 5 items
            if item_id not in self.active_rehearsals:
                schedule.append(item_id)
        
        return schedule


# ==================== Main Working Memory System ====================

class WorkingMemory:
    """
    Complete working memory implementation
    
    Integrates:
    - Central executive control
    - Rehearsal mechanisms
    - Capacity management (7±2)
    - Interference detection
    - Chunking strategies
    - Long-term memory gateway
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize working memory system"""
        self.config = config or {}
        
        # Capacity parameters (Miller's 7±2)
        self.base_capacity = self.config.get('base_capacity', 7)
        self.capacity_variance = self.config.get('capacity_variance', 2)
        self.current_capacity = self.base_capacity
        
        # Storage
        self.items: Dict[str, WorkingMemoryItem] = {}
        self.chunks: Dict[str, MemoryChunk] = {}
        
        # Components
        self.central_executive = CentralExecutive(config)
        self.rehearsal_loop = RehearsalLoop(config)
        
        # Decay parameters
        self.base_decay_rate = self.config.get('base_decay_rate', 0.1)
        self.interference_factor = self.config.get('interference_factor', 0.2)
        
        # Metrics
        self.metrics = WorkingMemoryMetrics(current_capacity=self.current_capacity)
        
        # Background tasks
        self._decay_task = None
        self._rehearsal_task = None
        self._is_running = False
        
        # Callbacks for long-term memory
        self.consolidation_callback: Optional[Callable] = None
        
        logger.info(
            "WorkingMemory initialized",
            capacity=f"{self.base_capacity}±{self.capacity_variance}",
            focus_limit=self.central_executive.focus_capacity
        )
    
    async def start(self):
        """Start background processes"""
        if self._is_running:
            return
        
        self._is_running = True
        self._decay_task = asyncio.create_task(self._decay_loop())
        self._rehearsal_task = asyncio.create_task(self._rehearsal_loop())
        
        logger.info("Working memory started")
    
    async def stop(self):
        """Stop background processes"""
        self._is_running = False
        
        if self._decay_task:
            self._decay_task.cancel()
        if self._rehearsal_task:
            self._rehearsal_task.cancel()
        
        await asyncio.sleep(0.1)
        logger.info("Working memory stopped")
    
    # ==================== Core Operations ====================
    
    async def add_item(self, content: Any, modality: MemoryModalType = MemoryModalType.MULTIMODAL,
                       embedding: Optional[np.ndarray] = None) -> Optional[str]:
        """
        Add item to working memory
        
        Returns item ID if successful, None if capacity exceeded
        """
        # Check capacity
        if len(self.items) >= self.current_capacity:
            # Try to make room
            removed = await self._make_room()
            if not removed:
                logger.warning("Working memory at capacity, cannot add item")
                return None
        
        # Create item
        item_id = self._generate_id(content)
        
        if embedding is None:
            embedding = np.random.randn(384)  # Default embedding
        
        item = WorkingMemoryItem(
            id=item_id,
            content=content,
            modality=modality,
            embedding=embedding
        )
        
        # Calculate initial activation based on current load
        cognitive_load = len(self.items) / self.current_capacity
        item.activation_level = 1.0 - (cognitive_load * 0.2)
        
        # Add to storage
        self.items[item_id] = item
        
        # Allocate attention
        await self._update_attention()
        
        # Calculate interference
        self._update_interference()
        
        # Update metrics
        self.metrics.total_items_processed += 1
        self._update_metrics()
        
        logger.debug(
            f"Added item to working memory",
            item_id=item_id,
            modality=modality.value,
            activation=item.activation_level
        )
        
        return item_id
    
    async def retrieve_item(self, item_id: str) -> Optional[Any]:
        """
        Retrieve item from working memory
        
        Boosts activation on access
        """
        if item_id not in self.items:
            # Check chunks
            for chunk in self.chunks.values():
                if item_id in chunk.member_ids:
                    return await self._retrieve_from_chunk(chunk, item_id)
            
            self.metrics.failed_recalls += 1
            return None
        
        item = self.items[item_id]
        
        # Boost activation on access
        item.activation_level = min(1.0, item.activation_level + 0.1)
        item.last_access_time = datetime.now()
        
        # Move to focus if important
        if item.activation_level > 0.7:
            self.central_executive.focused_items[item_id] = item.activation_level
        
        self.metrics.successful_recalls += 1
        self._update_metrics()
        
        return item.content
    
    async def update_item(self, item_id: str, new_content: Any) -> bool:
        """Update item content"""
        if item_id not in self.items:
            return False
        
        item = self.items[item_id]
        item.content = new_content
        item.last_access_time = datetime.now()
        
        # Boost activation on update
        item.activation_level = min(1.0, item.activation_level + 0.15)
        
        # Recalculate embedding if needed
        if hasattr(new_content, 'embedding'):
            item.embedding = new_content.embedding
        
        return True
    
    async def remove_item(self, item_id: str) -> bool:
        """Remove item from working memory"""
        if item_id in self.items:
            del self.items[item_id]
            
            # Remove from focus
            self.central_executive.inhibit_item(item_id)
            
            # Update metrics
            self._update_metrics()
            
            return True
        
        return False
    
    # ==================== Chunking Operations ====================
    
    async def create_chunk(self, item_ids: List[str]) -> Optional[str]:
        """
        Create chunk from multiple items (increases effective capacity)
        
        Based on Miller's chunking strategy
        """
        if len(item_ids) < 2:
            return None
        
        # Verify all items exist
        chunk_items = []
        for item_id in item_ids:
            if item_id in self.items:
                chunk_items.append(self.items[item_id])
        
        if len(chunk_items) < 2:
            return None
        
        # Calculate chunk coherence
        coherence = self._calculate_chunk_coherence(chunk_items)
        
        if coherence < 0.5:
            logger.warning("Items too dissimilar for chunking")
            return None
        
        # Create chunk
        chunk_id = f"chunk_{len(self.chunks):03d}"
        
        # Combine embeddings
        chunk_embedding = np.mean([item.embedding for item in chunk_items], axis=0)
        
        chunk = MemoryChunk(
            id=chunk_id,
            member_ids=set(item_ids),
            chunk_embedding=chunk_embedding,
            coherence_score=coherence
        )
        
        self.chunks[chunk_id] = chunk
        
        # Update items
        for item in chunk_items:
            item.chunk_members = set(item_ids)
        
        logger.info(
            f"Created chunk",
            chunk_id=chunk_id,
            size=len(item_ids),
            coherence=coherence
        )
        
        # This effectively increases capacity
        self.current_capacity = self.base_capacity + len(self.chunks)
        
        return chunk_id
    
    def _calculate_chunk_coherence(self, items: List[WorkingMemoryItem]) -> float:
        """Calculate coherence of potential chunk"""
        if len(items) < 2:
            return 0.0
        
        # Calculate pairwise similarities
        similarities = []
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                sim = self._calculate_similarity(items[i], items[j])
                similarities.append(sim)
        
        # Average similarity is coherence
        return np.mean(similarities) if similarities else 0.0
    
    def _calculate_similarity(self, item1: WorkingMemoryItem, 
                            item2: WorkingMemoryItem) -> float:
        """Calculate similarity between items"""
        # Embedding similarity
        if item1.embedding is not None and item2.embedding is not None:
            cosine_sim = np.dot(item1.embedding, item2.embedding) / (
                np.linalg.norm(item1.embedding) * np.linalg.norm(item2.embedding)
            )
            embedding_sim = (cosine_sim + 1.0) / 2.0
        else:
            embedding_sim = 0.5
        
        # Modality similarity
        modality_sim = 1.0 if item1.modality == item2.modality else 0.3
        
        # Association overlap
        if item1.associations and item2.associations:
            overlap = len(item1.associations & item2.associations)
            total = len(item1.associations | item2.associations)
            association_sim = overlap / total if total > 0 else 0.0
        else:
            association_sim = 0.0
        
        # Weighted combination
        similarity = (
            0.5 * embedding_sim +
            0.3 * modality_sim +
            0.2 * association_sim
        )
        
        return similarity
    
    async def _retrieve_from_chunk(self, chunk: MemoryChunk, item_id: str) -> Optional[Any]:
        """Retrieve item from chunk"""
        # Retrieving from chunk activates all members
        for member_id in chunk.member_ids:
            if member_id in self.items:
                self.items[member_id].activation_level += 0.05
        
        # Return the specific item
        if item_id in self.items:
            return self.items[item_id].content
        
        return None
    
    # ==================== Attention Management ====================
    
    async def _update_attention(self):
        """Update attention allocation across items"""
        if not self.items:
            return
        
        # Get attention weights from central executive
        item_list = list(self.items.values())
        attention_weights = self.central_executive.allocate_attention(item_list)
        
        # Apply attention weights
        for item_id, weight in attention_weights.items():
            if item_id in self.items:
                item = self.items[item_id]
                item.attention_weight = weight
                
                # Update attention state
                if item_id in self.central_executive.focused_items:
                    item.attention_state = AttentionState.FOCUSED
                elif weight > 0.1:
                    item.attention_state = AttentionState.ACTIVE
                elif weight > 0.01:
                    item.attention_state = AttentionState.PERIPHERAL
                else:
                    item.attention_state = AttentionState.DECAYING
    
    def focus_on_item(self, item_id: str):
        """Explicitly focus attention on an item"""
        if item_id not in self.items:
            return
        
        # Switch attention
        current_focus = list(self.central_executive.focused_items.keys())
        if current_focus and item_id not in current_focus:
            cost = self.central_executive.switch_attention(current_focus[0], item_id)
            logger.debug(f"Attention switch cost: {cost}")
        
        # Boost item
        self.items[item_id].activation_level = min(1.0, 
                                                   self.items[item_id].activation_level + 0.2)
        self.items[item_id].attention_state = AttentionState.FOCUSED
    
    # ==================== Interference Management ====================
    
    def _update_interference(self):
        """Calculate and update interference between items"""
        items_list = list(self.items.values())
        
        for i, item1 in enumerate(items_list):
            interference = 0.0
            
            for j, item2 in enumerate(items_list):
                if i != j:
                    # Calculate pairwise interference
                    if item1.modality == item2.modality:
                        # Same modality = high interference
                        similarity = self._calculate_similarity(item1, item2)
                        interference += similarity * 0.3
                    else:
                        # Different modality = low interference
                        interference += 0.05
            
            # Normalize by number of items
            if len(items_list) > 1:
                item1.interference_level = min(1.0, interference / (len(items_list) - 1))
    
    async def resolve_interference(self, item_id1: str, item_id2: str):
        """Resolve interference between two items"""
        if item_id1 not in self.items or item_id2 not in self.items:
            return
        
        item1 = self.items[item_id1]
        item2 = self.items[item_id2]
        
        # Strategy 1: Separate in time (rehearse separately)
        if item1.modality == item2.modality:
            # Stagger rehearsal times
            item1.last_rehearsal_time = datetime.now()
            item2.last_rehearsal_time = datetime.now() + timedelta(seconds=1)
        
        # Strategy 2: Create distinction
        if self._calculate_similarity(item1, item2) > 0.8:
            # Too similar - add distinguishing features
            item1.associations.add(f"distinct_{item1.id}")
            item2.associations.add(f"distinct_{item2.id}")
        
        # Reduce interference levels
        item1.interference_level *= 0.7
        item2.interference_level *= 0.7
        
        self.metrics.total_interference_resolutions += 1
    
    # ==================== Decay and Maintenance ====================
    
    async def _decay_loop(self):
        """Background task for memory decay"""
        while self._is_running:
            try:
                await asyncio.sleep(1.0)  # Decay every second
                
                items_to_remove = []
                
                for item_id, item in self.items.items():
                    # Calculate decay based on multiple factors
                    time_since_access = (datetime.now() - item.last_access_time).total_seconds()
                    
                    # Base decay
                    decay = self.base_decay_rate * (time_since_access / 60.0)
                    
                    # Interference increases decay
                    decay *= (1.0 + item.interference_level * self.interference_factor)
                    
                    # Attention reduces decay
                    decay *= (1.0 - item.attention_weight * 0.5)
                    
                    # Apply decay
                    item.activation_level = max(0.0, item.activation_level - decay)
                    
                    # Mark for removal if too low
                    if item.activation_level < 0.1:
                        items_to_remove.append(item_id)
                        item.attention_state = AttentionState.DORMANT
                
                # Remove decayed items
                for item_id in items_to_remove:
                    # Try to consolidate before removing
                    if self.consolidation_callback and self.items[item_id].is_consolidated:
                        await self.consolidation_callback(self.items[item_id])
                        self.metrics.total_consolidations += 1
                    
                    await self.remove_item(item_id)
                    logger.debug(f"Item {item_id} decayed and removed")
                
                # Update metrics
                self._update_metrics()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Decay loop error: {e}")
                await asyncio.sleep(1.0)
    
    async def _rehearsal_loop(self):
        """Background task for automatic rehearsal"""
        while self._is_running:
            try:
                await asyncio.sleep(0.5)  # Check every 500ms
                
                # Get rehearsal schedule
                items_list = list(self.items.values())
                schedule = self.rehearsal_loop.get_rehearsal_schedule(items_list)
                
                # Rehearse scheduled items
                for item_id in schedule:
                    if item_id in self.items:
                        item = self.items[item_id]
                        boost = await self.rehearsal_loop.rehearse_item(item)
                        self.metrics.total_rehearsals += 1
                        
                        logger.debug(
                            f"Rehearsed item",
                            item_id=item_id,
                            boost=boost,
                            new_activation=item.activation_level
                        )
                
                # Update attention periodically
                await self._update_attention()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Rehearsal loop error: {e}")
                await asyncio.sleep(0.5)
    
    # ==================== Utility Methods ====================
    
    async def _make_room(self) -> bool:
        """Try to make room in working memory"""
        # Find least important item
        if not self.items:
            return True
        
        # Sort by importance (activation * (1 - interference))
        importance_scores = [
            (item_id, item.activation_level * (1.0 - item.interference_level))
            for item_id, item in self.items.items()
            if item.attention_state != AttentionState.FOCUSED
        ]
        
        if not importance_scores:
            return False  # All items are focused
        
        importance_scores.sort(key=lambda x: x[1])
        
        # Remove least important
        least_important_id = importance_scores[0][0]
        
        # Try to consolidate first
        if self.consolidation_callback and self.items[least_important_id].rehearsal_count > 3:
            await self.consolidation_callback(self.items[least_important_id])
            self.metrics.total_consolidations += 1
        
        await self.remove_item(least_important_id)
        logger.debug(f"Made room by removing {least_important_id}")
        
        return True
    
    def _generate_id(self, content: Any) -> str:
        """Generate unique ID for content"""
        content_str = str(content)[:100]  # Limit length
        timestamp = str(datetime.now().timestamp())
        hash_input = f"{content_str}_{timestamp}"
        return hashlib.sha256(hash_input.encode()).hexdigest()[:16]
    
    def _update_metrics(self):
        """Update performance metrics"""
        # Count items by state
        self.metrics.items_in_focus = len(self.central_executive.focused_items)
        self.metrics.items_active = sum(
            1 for item in self.items.values()
            if item.attention_state == AttentionState.ACTIVE
        )
        self.metrics.items_peripheral = sum(
            1 for item in self.items.values()
            if item.attention_state == AttentionState.PERIPHERAL
        )
        
        # Calculate averages
        if self.items:
            self.metrics.avg_activation_level = np.mean(
                [item.activation_level for item in self.items.values()]
            )
            self.metrics.avg_attention_weight = np.mean(
                [item.attention_weight for item in self.items.values()]
            )
        
        # Cognitive load
        self.metrics.cognitive_load = len(self.items) / self.current_capacity
        
        # Recall accuracy
        total_recalls = self.metrics.successful_recalls + self.metrics.failed_recalls
        if total_recalls > 0:
            self.metrics.recall_accuracy = self.metrics.successful_recalls / total_recalls
    
    # ==================== Public Interface ====================
    
    async def get_all_items(self) -> List[WorkingMemoryItem]:
        """Get all items in working memory"""
        return list(self.items.values())
    
    async def get_focused_items(self) -> List[WorkingMemoryItem]:
        """Get items currently in focus"""
        focused = []
        for item_id in self.central_executive.focused_items:
            if item_id in self.items:
                focused.append(self.items[item_id])
        return focused
    
    def get_capacity_usage(self) -> Tuple[int, int]:
        """Get current usage and capacity"""
        return (len(self.items), self.current_capacity)
    
    def adjust_capacity(self, delta: int):
        """Adjust working memory capacity"""
        new_capacity = self.current_capacity + delta
        
        # Enforce Miller's limits
        min_capacity = self.base_capacity - self.capacity_variance
        max_capacity = self.base_capacity + self.capacity_variance
        
        self.current_capacity = max(min_capacity, min(max_capacity, new_capacity))
        self.metrics.current_capacity = self.current_capacity
        
        logger.info(f"Adjusted capacity to {self.current_capacity}")
    
    def set_consolidation_callback(self, callback: Callable):
        """Set callback for consolidation to long-term memory"""
        self.consolidation_callback = callback
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get performance metrics"""
        return {
            'capacity': f"{len(self.items)}/{self.current_capacity}",
            'focused': self.metrics.items_in_focus,
            'active': self.metrics.items_active,
            'peripheral': self.metrics.items_peripheral,
            'avg_activation': f"{self.metrics.avg_activation_level:.3f}",
            'avg_attention': f"{self.metrics.avg_attention_weight:.3f}",
            'cognitive_load': f"{self.metrics.cognitive_load:.2%}",
            'recall_accuracy': f"{self.metrics.recall_accuracy:.2%}",
            'total_processed': self.metrics.total_items_processed,
            'total_rehearsals': self.metrics.total_rehearsals,
            'total_consolidations': self.metrics.total_consolidations,
            'chunks': len(self.chunks)
        }
    
    async def clear(self):
        """Clear all items from working memory"""
        # Consolidate important items first
        if self.consolidation_callback:
            for item in self.items.values():
                if item.is_consolidated or item.rehearsal_count > 5:
                    await self.consolidation_callback(item)
                    self.metrics.total_consolidations += 1
        
        self.items.clear()
        self.chunks.clear()
        self.central_executive.focused_items.clear()
        
        logger.info("Working memory cleared")