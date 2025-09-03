"""
Unified Cognitive Memory System - The Memory Operating System
=============================================================

Based on September 2025 research:
- Unified interface for all memory operations
- Intelligent query decomposition and planning
- Multi-store orchestration with resilience
- Semantic synthesis beyond simple RAG
- Continuous memory lifecycle management
- Real-time and offline consolidation
- Failure prevention through causal tracking

This is the BRAIN that makes all memory systems work together.
"""

import asyncio
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import time
import json
import hashlib
from collections import defaultdict, deque
from enum import Enum
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
import structlog
from concurrent.futures import ThreadPoolExecutor
import traceback

logger = structlog.get_logger(__name__)


# ==================== Data Structures ====================

class QueryType(Enum):
    """Types of queries the system can handle"""
    FACTUAL = "factual"  # "What is X?"
    EPISODIC = "episodic"  # "What happened when?"
    CAUSAL = "causal"  # "Why did X happen?"
    PREDICTIVE = "predictive"  # "What will happen if?"
    CREATIVE = "creative"  # "What if we combine X and Y?"
    ANALYTICAL = "analytical"  # "What patterns exist?"
    METACOGNITIVE = "metacognitive"  # "What do I know about X?"


@dataclass
class QueryPlan:
    """
    Structured plan for executing a query across memory stores
    """
    query_text: str
    query_type: QueryType
    
    # Store-specific sub-queries
    working_memory_query: Optional[Dict[str, Any]] = None
    episodic_query: Optional[Dict[str, Any]] = None
    semantic_query: Optional[Dict[str, Any]] = None
    
    # Routing hints
    priority_stores: List[str] = field(default_factory=list)
    max_hops: int = 3
    time_budget_ms: int = 5000
    
    # Synthesis strategy
    synthesis_mode: str = "default"  # default, causal, creative, analytical
    require_grounding: bool = True  # Must be grounded in experience
    
    # Fallback strategy
    fallback_enabled: bool = True
    fallback_stores: List[str] = field(default_factory=lambda: ["episodic"])


@dataclass
class MemoryContext:
    """
    Unified context returned from memory queries
    """
    query: str
    timestamp: datetime
    
    # Retrieved content
    working_memories: List[Any] = field(default_factory=list)
    episodes: List[Any] = field(default_factory=list)
    concepts: List[Any] = field(default_factory=list)
    causal_chains: List[Any] = field(default_factory=list)
    
    # Synthesis results
    synthesized_answer: Optional[str] = None
    confidence: float = 0.0
    grounding_strength: float = 0.0  # How well grounded in experience
    
    # Reasoning trace
    reasoning_path: List[Dict[str, Any]] = field(default_factory=list)
    
    # Performance metrics
    retrieval_time_ms: int = 0
    synthesis_time_ms: int = 0
    total_sources: int = 0


@dataclass
class MemoryTransferEvent:
    """Event for memory transfer between stores"""
    source_store: str
    target_store: str
    content: Any
    transfer_reason: str
    timestamp: datetime = field(default_factory=datetime.now)
    success: bool = False
    error: Optional[str] = None


# ==================== Query Planning Neural Network ====================

class QueryPlannerNetwork(nn.Module):
    """
    Neural network for decomposing queries into structured plans
    
    Research: "LLM-based query planner for multi-store retrieval"
    """
    
    def __init__(self, hidden_dim: int = 768):
        super().__init__()
        
        # Query type classification
        self.type_classifier = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, len(QueryType))
        )
        
        # Store routing (which stores to query)
        self.store_router = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 4)  # working, episodic, semantic, causal
        )
        
        # Synthesis mode selection
        self.synthesis_selector = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 5)  # default, causal, creative, analytical, metacognitive
        )
        
        # Complexity estimation (for time budgeting)
        self.complexity_estimator = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def forward(self, query_embedding: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Generate query plan from embedding"""
        query_type_logits = self.type_classifier(query_embedding)
        store_logits = self.store_router(query_embedding)
        synthesis_logits = self.synthesis_selector(query_embedding)
        complexity = self.complexity_estimator(query_embedding)
        
        return {
            'query_type': F.softmax(query_type_logits, dim=-1),
            'stores': torch.sigmoid(store_logits),  # Multi-label
            'synthesis': F.softmax(synthesis_logits, dim=-1),
            'complexity': complexity
        }


# ==================== Memory Lifecycle Manager ====================

class MemoryLifecycleManager:
    """
    Manages the lifecycle of memories across all stores
    
    Research: "Continuous memory management during operation"
    """
    
    def __init__(self, memory_stores: Dict[str, Any]):
        """Initialize lifecycle manager"""
        self.stores = memory_stores
        
        # Transfer queues
        self.transfer_queues = {
            'working_to_episodic': asyncio.Queue(),
            'episodic_to_semantic': asyncio.Queue(),
            'consolidation_requests': asyncio.Queue()
        }
        
        # Transfer history
        self.transfer_history = deque(maxlen=1000)
        
        # Background tasks
        self.background_tasks = []
        
        # Metrics
        self.transfer_counts = defaultdict(int)
        self.consolidation_triggers = 0
        
        logger.info("MemoryLifecycleManager initialized")
    
    async def start(self):
        """Start background lifecycle management"""
        # Start transfer workers
        self.background_tasks.append(
            asyncio.create_task(self._working_to_episodic_worker())
        )
        self.background_tasks.append(
            asyncio.create_task(self._episodic_to_semantic_worker())
        )
        self.background_tasks.append(
            asyncio.create_task(self._consolidation_worker())
        )
        
        logger.info("Memory lifecycle management started")
    
    async def stop(self):
        """Stop background tasks"""
        for task in self.background_tasks:
            task.cancel()
        await asyncio.gather(*self.background_tasks, return_exceptions=True)
        logger.info("Memory lifecycle management stopped")
    
    async def _working_to_episodic_worker(self):
        """Transfer memories from working to episodic"""
        while True:
            try:
                # Wait for transfer request
                transfer_event = await self.transfer_queues['working_to_episodic'].get()
                
                # Perform transfer
                episode = await self.stores['episodic'].receive_from_working_memory(
                    transfer_event.content
                )
                
                # Record transfer
                transfer_event.success = True
                transfer_event.target_store = 'episodic'
                self.transfer_history.append(transfer_event)
                self.transfer_counts['working_to_episodic'] += 1
                
                logger.debug(
                    "Memory transferred",
                    source="working",
                    target="episodic",
                    episode_id=episode.id if hasattr(episode, 'id') else None
                )
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Transfer error: {e}")
                await asyncio.sleep(1)
    
    async def _episodic_to_semantic_worker(self):
        """Extract knowledge from episodes to semantic"""
        while True:
            try:
                # Periodic knowledge extraction
                await asyncio.sleep(60)  # Every minute
                
                # Get consolidated episodes
                episodes = await self.stores['episodic'].prepare_for_semantic_extraction()
                
                if episodes:
                    # Extract knowledge
                    concepts = await self.stores['semantic'].extract_knowledge_from_episodes(
                        episodes
                    )
                    
                    self.transfer_counts['episodic_to_semantic'] += len(concepts)
                    
                    logger.info(
                        "Knowledge extracted",
                        episodes=len(episodes),
                        concepts=len(concepts)
                    )
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Knowledge extraction error: {e}")
                await asyncio.sleep(10)
    
    async def _consolidation_worker(self):
        """Trigger consolidation based on conditions"""
        while True:
            try:
                # Check consolidation triggers
                await asyncio.sleep(30)  # Check every 30 seconds
                
                # Trigger consolidation if needed
                if self._should_consolidate():
                    await self.stores['consolidation'].run_awake_consolidation()
                    self.consolidation_triggers += 1
                    
                    logger.info("Awake consolidation triggered")
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Consolidation error: {e}")
                await asyncio.sleep(10)
    
    def _should_consolidate(self) -> bool:
        """Determine if consolidation should run"""
        # Check various conditions
        # - High memory pressure
        # - Time since last consolidation
        # - Importance of recent memories
        # - System idle time
        
        # Simplified logic
        return self.transfer_counts['working_to_episodic'] > 100
    
    async def request_transfer(self, source: str, target: str, content: Any, reason: str):
        """Request memory transfer between stores"""
        event = MemoryTransferEvent(
            source_store=source,
            target_store=target,
            content=content,
            transfer_reason=reason
        )
        
        queue_name = f"{source}_to_{target}"
        if queue_name in self.transfer_queues:
            await self.transfer_queues[queue_name].put(event)
        else:
            logger.warning(f"Unknown transfer path: {queue_name}")


# ==================== Semantic Synthesis Engine ====================

class SemanticSynthesisEngine:
    """
    Synthesizes retrieved memories into coherent answers
    
    Research: "Beyond simple RAG - semantic interpretation of memories"
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize synthesis engine"""
        self.config = config or {}
        
        # Synthesis strategies
        self.strategies = {
            'default': self._default_synthesis,
            'causal': self._causal_synthesis,
            'creative': self._creative_synthesis,
            'analytical': self._analytical_synthesis,
            'metacognitive': self._metacognitive_synthesis
        }
        
        # Language model for synthesis (could be local or API)
        self.synthesis_model = AutoModel.from_pretrained(
            config.get('synthesis_model', 'microsoft/deberta-v3-base')
        )
        self.synthesis_tokenizer = AutoTokenizer.from_pretrained(
            config.get('synthesis_model', 'microsoft/deberta-v3-base')
        )
        
        logger.info("SemanticSynthesisEngine initialized")
    
    async def synthesize(
        self,
        retrieved_data: Dict[str, List[Any]],
        query_plan: QueryPlan
    ) -> MemoryContext:
        """
        Synthesize retrieved data into unified context
        
        This is the key differentiator from simple RAG
        """
        start_time = time.time()
        
        # Create context
        context = MemoryContext(
            query=query_plan.query_text,
            timestamp=datetime.now(),
            working_memories=retrieved_data.get('working', []),
            episodes=retrieved_data.get('episodic', []),
            concepts=retrieved_data.get('semantic', []),
            causal_chains=retrieved_data.get('causal', [])
        )
        
        # Count sources
        context.total_sources = sum(len(v) for v in retrieved_data.values())
        
        # Select synthesis strategy
        strategy = self.strategies.get(
            query_plan.synthesis_mode,
            self._default_synthesis
        )
        
        # Perform synthesis
        try:
            synthesis_result = await strategy(context, query_plan)
            context.synthesized_answer = synthesis_result['answer']
            context.confidence = synthesis_result['confidence']
            context.grounding_strength = synthesis_result['grounding']
            context.reasoning_path = synthesis_result.get('reasoning', [])
        except Exception as e:
            logger.error(f"Synthesis error: {e}")
            context.synthesized_answer = "Unable to synthesize answer"
            context.confidence = 0.0
        
        # Record timing
        context.synthesis_time_ms = int((time.time() - start_time) * 1000)
        
        return context
    
    async def _default_synthesis(
        self,
        context: MemoryContext,
        query_plan: QueryPlan
    ) -> Dict[str, Any]:
        """Default synthesis strategy - combine and summarize"""
        # Combine all retrieved information
        all_content = []
        
        # Add working memories
        for mem in context.working_memories[:3]:
            all_content.append(f"Current: {mem}")
        
        # Add episodes with context
        for episode in context.episodes[:5]:
            if hasattr(episode, 'content'):
                all_content.append(f"Experience: {episode.content}")
        
        # Add semantic concepts
        for concept in context.concepts[:5]:
            if hasattr(concept, 'definition'):
                all_content.append(f"Knowledge: {concept.label} - {concept.definition}")
        
        # Simple synthesis
        if all_content:
            answer = f"Based on {len(all_content)} memories: " + "; ".join(all_content[:3])
            confidence = min(1.0, len(all_content) / 10.0)
            grounding = min(1.0, len(context.episodes) / 5.0)
        else:
            answer = "No relevant memories found"
            confidence = 0.0
            grounding = 0.0
        
        return {
            'answer': answer,
            'confidence': confidence,
            'grounding': grounding,
            'reasoning': [{'step': 'default', 'content': all_content}]
        }
    
    async def _causal_synthesis(
        self,
        context: MemoryContext,
        query_plan: QueryPlan
    ) -> Dict[str, Any]:
        """Causal synthesis - explain why something happened"""
        reasoning_steps = []
        
        # Step 1: Identify causal chains
        causal_explanations = []
        for chain in context.causal_chains:
            if hasattr(chain, 'events'):
                explanation = " → ".join([str(e) for e in chain.events])
                causal_explanations.append(explanation)
                reasoning_steps.append({
                    'step': 'causal_chain',
                    'content': explanation
                })
        
        # Step 2: Find supporting episodes
        supporting_episodes = []
        for episode in context.episodes:
            if hasattr(episode, 'causal_context'):
                supporting_episodes.append(episode)
                reasoning_steps.append({
                    'step': 'supporting_episode',
                    'content': f"Episode {episode.id} supports causal link"
                })
        
        # Step 3: Use semantic knowledge to explain
        explanatory_concepts = []
        for concept in context.concepts:
            if hasattr(concept, 'properties') and 'causes' in concept.properties:
                explanatory_concepts.append(concept)
                reasoning_steps.append({
                    'step': 'explanatory_concept',
                    'content': f"{concept.label} explains mechanism"
                })
        
        # Synthesize causal explanation
        if causal_explanations:
            answer = f"Causal explanation: {causal_explanations[0]}"
            if explanatory_concepts:
                answer += f" This occurs because {explanatory_concepts[0].label}."
            confidence = 0.8
            grounding = len(supporting_episodes) / max(1, len(context.episodes))
        else:
            answer = "No clear causal relationship found"
            confidence = 0.2
            grounding = 0.0
        
        return {
            'answer': answer,
            'confidence': confidence,
            'grounding': grounding,
            'reasoning': reasoning_steps
        }
    
    async def _creative_synthesis(
        self,
        context: MemoryContext,
        query_plan: QueryPlan
    ) -> Dict[str, Any]:
        """Creative synthesis - generate novel combinations"""
        # Combine concepts in new ways
        if len(context.concepts) >= 2:
            concept1 = context.concepts[0]
            concept2 = context.concepts[1]
            
            # Generate creative combination
            answer = f"Creative insight: Combining {concept1.label} with {concept2.label} "
            answer += f"could lead to new possibilities based on their shared properties."
            
            # Look for bridging episodes
            bridging_episodes = []
            for episode in context.episodes:
                # Check if episode relates to both concepts
                if hasattr(episode, 'abstract_tags'):
                    tags = episode.abstract_tags
                    if any(c.label in tags for c in [concept1, concept2]):
                        bridging_episodes.append(episode)
            
            if bridging_episodes:
                answer += f" This is supported by {len(bridging_episodes)} related experiences."
            
            confidence = 0.6
            grounding = len(bridging_episodes) / max(1, len(context.episodes))
        else:
            answer = "Insufficient concepts for creative synthesis"
            confidence = 0.1
            grounding = 0.0
        
        return {
            'answer': answer,
            'confidence': confidence,
            'grounding': grounding,
            'reasoning': [{'step': 'creative_combination', 'content': answer}]
        }
    
    async def _analytical_synthesis(
        self,
        context: MemoryContext,
        query_plan: QueryPlan
    ) -> Dict[str, Any]:
        """Analytical synthesis - identify patterns and trends"""
        patterns = []
        
        # Analyze episode patterns
        if context.episodes:
            # Temporal patterns
            timestamps = [ep.timestamp for ep in context.episodes if hasattr(ep, 'timestamp')]
            if timestamps:
                time_span = max(timestamps) - min(timestamps)
                patterns.append(f"Events span {time_span} time units")
            
            # Emotional patterns
            emotions = [ep.emotional_state for ep in context.episodes 
                       if hasattr(ep, 'emotional_state')]
            if emotions:
                avg_valence = np.mean([e.valence for e in emotions])
                patterns.append(f"Average emotional valence: {avg_valence:.2f}")
        
        # Analyze concept relationships
        if context.concepts:
            # Find common properties
            all_properties = defaultdict(int)
            for concept in context.concepts:
                if hasattr(concept, 'properties'):
                    for prop in concept.properties:
                        all_properties[prop] += 1
            
            common_props = [p for p, count in all_properties.items() if count > 1]
            if common_props:
                patterns.append(f"Common properties: {', '.join(common_props[:3])}")
        
        # Synthesize analysis
        if patterns:
            answer = f"Analysis reveals {len(patterns)} patterns: " + "; ".join(patterns)
            confidence = min(0.9, len(patterns) / 5.0)
            grounding = min(1.0, (len(context.episodes) + len(context.concepts)) / 10.0)
        else:
            answer = "No clear patterns identified"
            confidence = 0.3
            grounding = 0.0
        
        return {
            'answer': answer,
            'confidence': confidence,
            'grounding': grounding,
            'reasoning': [{'step': 'pattern', 'content': p} for p in patterns]
        }
    
    async def _metacognitive_synthesis(
        self,
        context: MemoryContext,
        query_plan: QueryPlan
    ) -> Dict[str, Any]:
        """Metacognitive synthesis - reasoning about own knowledge"""
        knowledge_assessment = []
        
        # Assess knowledge coverage
        if context.concepts:
            knowledge_assessment.append(
                f"I have {len(context.concepts)} concepts related to this topic"
            )
            
            # Check grounding strength
            well_grounded = [c for c in context.concepts 
                            if hasattr(c, 'grounding_strength') and c.grounding_strength > 0.7]
            knowledge_assessment.append(
                f"{len(well_grounded)} concepts are well-grounded in experience"
            )
        
        # Assess experiential knowledge
        if context.episodes:
            knowledge_assessment.append(
                f"I have {len(context.episodes)} relevant experiences"
            )
            
            # Check consolidation state
            consolidated = [e for e in context.episodes
                          if hasattr(e, 'consolidation_state') and e.consolidation_state == 'deep']
            knowledge_assessment.append(
                f"{len(consolidated)} memories are deeply consolidated"
            )
        
        # Identify knowledge gaps
        if not context.concepts:
            knowledge_assessment.append("I lack semantic knowledge about this topic")
        if not context.episodes:
            knowledge_assessment.append("I have no direct experience with this")
        
        # Synthesize metacognitive assessment
        answer = "Self-assessment: " + "; ".join(knowledge_assessment)
        
        # Calculate confidence based on knowledge quality
        confidence = min(1.0, (len(context.concepts) + len(context.episodes)) / 20.0)
        grounding = min(1.0, len(context.episodes) / 10.0)
        
        return {
            'answer': answer,
            'confidence': confidence,
            'grounding': grounding,
            'reasoning': [{'step': 'assessment', 'content': a} for a in knowledge_assessment]
        }


# ==================== Main Unified Cognitive Memory System ====================

class UnifiedCognitiveMemory:
    """
    The Memory Operating System - Central orchestrator for all memory systems
    
    This is the BRAIN that coordinates:
    - Working Memory (immediate, 7±2 items)
    - Episodic Memory (autobiographical timeline)
    - Semantic Memory (knowledge graph)
    - Memory Consolidation (learning cycles)
    - Causal Tracking (failure prevention)
    - Shape Memory (topological signatures)
    - Hierarchical Routing (intelligent access)
    
    Research: "Unified cognitive architecture for continuous learning agents"
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the complete memory system"""
        self.config = config or {}
        
        # ========== Initialize All Memory Stores ==========
        from .working_memory import WorkingMemory
        from .episodic_memory import EpisodicMemory
        from .semantic_memory import SemanticMemory
        from .consolidation.orchestrator import SleepConsolidation
        from .routing.hierarchical_router_2025 import HierarchicalMemoryRouter2025
        from .shape_memory_v2 import ShapeMemoryV2
        from .core.causal_tracker import CausalPatternTracker
        from .core.topology_adapter import TopologyMemoryAdapter
        
        # Core memory stores
        self.working_memory = WorkingMemory(config.get('working', {}))
        self.episodic_memory = EpisodicMemory(config.get('episodic', {}))
        self.semantic_memory = SemanticMemory(config.get('semantic', {}))
        
        # Supporting systems
        self.consolidation = SleepConsolidation({
            'working_memory': self.working_memory,
            'episodic_memory': self.episodic_memory,
            'semantic_memory': self.semantic_memory
        })
        
        self.router = HierarchicalMemoryRouter2025(config.get('router', {}))
        self.shape_memory = ShapeMemoryV2(config.get('shape', {}))
        self.causal_tracker = CausalPatternTracker()
        self.topology_adapter = TopologyMemoryAdapter()
        
        # Store references for easy access
        self.stores = {
            'working': self.working_memory,
            'episodic': self.episodic_memory,
            'semantic': self.semantic_memory,
            'consolidation': self.consolidation,
            'router': self.router,
            'shape': self.shape_memory,
            'causal': self.causal_tracker,
            'topology': self.topology_adapter
        }
        
        # ========== Initialize Management Components ==========
        
        # Lifecycle manager
        self.lifecycle_manager = MemoryLifecycleManager(self.stores)
        
        # Query planner
        self.query_planner = self._init_query_planner()
        
        # Synthesis engine
        self.synthesis_engine = SemanticSynthesisEngine(config.get('synthesis', {}))
        
        # Embedding model for queries
        self.encoder = SentenceTransformer(
            'sentence-transformers/all-mpnet-base-v2',
            device='cuda' if config.get('use_cuda', False) else 'cpu'
        )
        
        # Thread pool for parallel operations
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # ========== Metrics and Monitoring ==========
        self.metrics = {
            'total_queries': 0,
            'successful_queries': 0,
            'failed_queries': 0,
            'total_writes': 0,
            'consolidation_cycles': 0,
            'memory_transfers': defaultdict(int)
        }
        
        # Query cache for performance
        self.query_cache = {}
        self.cache_size = config.get('cache_size', 100)
        
        # Circadian rhythm manager (for sleep cycles)
        self.last_sleep_cycle = datetime.now()
        self.sleep_interval = timedelta(hours=config.get('sleep_interval_hours', 8))
        
        # Start background processes
        self._background_task = None
        
        logger.info(
            "UnifiedCognitiveMemory initialized",
            stores=list(self.stores.keys()),
            sleep_interval=self.sleep_interval
        )
    
    def _init_query_planner(self) -> QueryPlannerNetwork:
        """Initialize or load query planner model"""
        planner = QueryPlannerNetwork()
        
        # Try to load pre-trained weights
        model_path = self.config.get('planner_model_path')
        if model_path:
            try:
                planner.load_state_dict(torch.load(model_path))
                logger.info(f"Loaded query planner from {model_path}")
            except:
                logger.warning("Could not load query planner weights, using random init")
        
        planner.eval()
        return planner
    
    # ==================== Lifecycle Management ====================
    
    async def start(self):
        """Start the memory system and all background processes"""
        logger.info("Starting UnifiedCognitiveMemory...")
        
        # Start individual memory systems
        await self.episodic_memory.start()
        
        # Start lifecycle manager
        await self.lifecycle_manager.start()
        
        # Start background monitoring
        self._background_task = asyncio.create_task(self._background_monitor())
        
        # Set up working memory overflow callback
        self.working_memory.consolidation_callback = self._handle_working_overflow
        
        logger.info("UnifiedCognitiveMemory started successfully")
    
    async def stop(self):
        """Stop all background processes"""
        logger.info("Stopping UnifiedCognitiveMemory...")
        
        # Stop background task
        if self._background_task:
            self._background_task.cancel()
            await asyncio.gather(self._background_task, return_exceptions=True)
        
        # Stop lifecycle manager
        await self.lifecycle_manager.stop()
        
        # Stop individual systems
        await self.episodic_memory.stop()
        await self.semantic_memory.close()
        
        logger.info("UnifiedCognitiveMemory stopped")
    
    async def _background_monitor(self):
        """Background monitoring and maintenance"""
        while True:
            try:
                await asyncio.sleep(60)  # Check every minute
                
                # Check if sleep cycle needed
                if await self._should_sleep():
                    await self.run_sleep_cycle()
                
                # Clean query cache
                if len(self.query_cache) > self.cache_size:
                    # Remove oldest entries
                    to_remove = len(self.query_cache) - self.cache_size
                    for key in list(self.query_cache.keys())[:to_remove]:
                        del self.query_cache[key]
                
                # Log metrics
                logger.info(
                    "Memory system metrics",
                    **self.metrics
                )
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Background monitor error: {e}")
                await asyncio.sleep(10)
    
    async def _should_sleep(self) -> bool:
        """Determine if sleep cycle should run"""
        time_since_sleep = datetime.now() - self.last_sleep_cycle
        
        # Check various conditions
        should_sleep = (
            time_since_sleep > self.sleep_interval or
            self.lifecycle_manager.transfer_counts['working_to_episodic'] > 1000 or
            self.metrics['failed_queries'] > 10
        )
        
        return should_sleep
    
    # ==================== Write Path (Experience → Memory) ====================
    
    async def process_experience(
        self,
        content: Any,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Primary WRITE path - process new experience into memory
        
        Flow: Input → Working Memory → Episodic → Semantic
        """
        start_time = time.time()
        result = {'success': False, 'error': None}
        
        try:
            # Step 1: Extract topological signature
            topology = await self.topology_adapter.extract_topology(content)
            
            # Step 2: Check for failure patterns
            if isinstance(content, dict):
                # Create causal event
                event = self.causal_tracker.CausalEvent(
                    event_id=hashlib.md5(str(content).encode()).hexdigest()[:8],
                    event_type=content.get('type', 'unknown'),
                    topology_signature=topology.get('signature', np.zeros(128)),
                    timestamp=datetime.now(),
                    agent_id=content.get('agent_id', 'default'),
                    context=context or {},
                    embeddings=topology.get('embedding', np.zeros(384)),
                    confidence=1.0
                )
                
                # Track for failure prediction
                pattern_id = await self.causal_tracker.track_event(event)
                if pattern_id:
                    logger.warning(
                        "Failure pattern detected",
                        pattern_id=pattern_id,
                        event_type=event.event_type
                    )
                    result['warning'] = f"Failure pattern {pattern_id} detected"
            
            # Step 3: Add to working memory
            working_item = await self.working_memory.add(
                content=content,
                importance=context.get('importance', 0.5) if context else 0.5
            )
            
            # Step 4: Store topological signature in shape memory
            await self.shape_memory.store(
                memory_id=working_item.id,
                topology=topology,
                metadata={
                    'source': 'working_memory',
                    'timestamp': datetime.now().isoformat()
                }
            )
            
            # Step 5: Check for immediate consolidation needs
            if working_item.importance >= 0.9:
                # High importance - immediate episodic storage
                episode = await self.episodic_memory.receive_from_working_memory(working_item)
                result['episode_id'] = episode.id
                
                # Trigger rapid consolidation
                await self.consolidation.run_awake_consolidation()
                logger.info(
                    "High-importance memory triggered rapid consolidation",
                    item_id=working_item.id,
                    importance=working_item.importance
                )
            
            # Update metrics
            self.metrics['total_writes'] += 1
            
            result['success'] = True
            result['working_memory_id'] = working_item.id
            result['processing_time_ms'] = int((time.time() - start_time) * 1000)
            
        except Exception as e:
            logger.error(f"Experience processing error: {e}")
            result['error'] = str(e)
        
        return result
    
    async def _handle_working_overflow(self, items: List[Any]):
        """Handle overflow from working memory"""
        logger.info(f"Working memory overflow: {len(items)} items")
        
        for item in items:
            await self.lifecycle_manager.request_transfer(
                source='working',
                target='episodic',
                content=item,
                reason='working_memory_overflow'
            )
    
    # ==================== Read Path (Query → Retrieval → Synthesis) ====================
    
    async def query(
        self,
        query_text: str,
        context: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None
    ) -> MemoryContext:
        """
        Primary READ path - unified interface for memory retrieval
        
        Flow: Query → Plan → Route → Retrieve → Synthesize
        """
        start_time = time.time()
        timeout = timeout or self.config.get('default_query_timeout', 5.0)
        
        # Check cache
        cache_key = hashlib.md5(f"{query_text}{context}".encode()).hexdigest()
        if cache_key in self.query_cache:
            logger.debug(f"Query cache hit for: {query_text[:50]}")
            cached = self.query_cache[cache_key]
            cached.timestamp = datetime.now()  # Update timestamp
            return cached
        
        try:
            # Step 1: Create query plan
            query_plan = await self._create_query_plan(query_text, context)
            
            # Step 2: Route and retrieve from appropriate stores
            retrieved_data = await asyncio.wait_for(
                self._execute_retrieval(query_plan),
                timeout=timeout
            )
            
            # Step 3: Synthesize results
            memory_context = await self.synthesis_engine.synthesize(
                retrieved_data,
                query_plan
            )
            
            # Add retrieval timing
            memory_context.retrieval_time_ms = int((time.time() - start_time) * 1000)
            
            # Update metrics
            self.metrics['total_queries'] += 1
            self.metrics['successful_queries'] += 1
            
            # Cache result
            self.query_cache[cache_key] = memory_context
            
            logger.debug(
                "Query completed",
                query=query_text[:50],
                sources=memory_context.total_sources,
                confidence=memory_context.confidence,
                time_ms=memory_context.retrieval_time_ms
            )
            
            return memory_context
            
        except asyncio.TimeoutError:
            logger.warning(f"Query timeout: {query_text[:50]}")
            
            # Try fallback retrieval
            if query_plan.fallback_enabled:
                return await self._fallback_query(query_text)
            
            # Return empty context
            self.metrics['failed_queries'] += 1
            return MemoryContext(
                query=query_text,
                timestamp=datetime.now(),
                synthesized_answer="Query timed out",
                confidence=0.0
            )
            
        except Exception as e:
            logger.error(f"Query error: {e}\n{traceback.format_exc()}")
            self.metrics['failed_queries'] += 1
            
            return MemoryContext(
                query=query_text,
                timestamp=datetime.now(),
                synthesized_answer=f"Query failed: {str(e)}",
                confidence=0.0
            )
    
    async def _create_query_plan(
        self,
        query_text: str,
        context: Optional[Dict[str, Any]] = None
    ) -> QueryPlan:
        """Create structured plan for query execution"""
        # Generate query embedding
        query_embedding = await asyncio.get_event_loop().run_in_executor(
            self.executor,
            self.encoder.encode,
            query_text,
            True  # Normalize
        )
        
        # Use neural planner
        with torch.no_grad():
            query_tensor = torch.tensor(query_embedding).unsqueeze(0)
            plan_outputs = self.query_planner(query_tensor)
        
        # Determine query type
        query_type_idx = torch.argmax(plan_outputs['query_type']).item()
        query_type = list(QueryType)[query_type_idx]
        
        # Determine which stores to query
        store_probs = plan_outputs['stores'].squeeze()
        stores_to_query = []
        if store_probs[0] > 0.5:
            stores_to_query.append('working')
        if store_probs[1] > 0.5:
            stores_to_query.append('episodic')
        if store_probs[2] > 0.5:
            stores_to_query.append('semantic')
        if store_probs[3] > 0.5:
            stores_to_query.append('causal')
        
        # Determine synthesis mode
        synthesis_idx = torch.argmax(plan_outputs['synthesis']).item()
        synthesis_modes = ['default', 'causal', 'creative', 'analytical', 'metacognitive']
        synthesis_mode = synthesis_modes[synthesis_idx]
        
        # Estimate complexity for time budgeting
        complexity = plan_outputs['complexity'].item()
        time_budget = int(2000 + complexity * 8000)  # 2-10 seconds
        
        # Create plan
        plan = QueryPlan(
            query_text=query_text,
            query_type=query_type,
            priority_stores=stores_to_query,
            synthesis_mode=synthesis_mode,
            time_budget_ms=time_budget
        )
        
        # Add store-specific queries
        if 'working' in stores_to_query:
            plan.working_memory_query = {
                'query': query_text,
                'k': 3
            }
        
        if 'episodic' in stores_to_query:
            plan.episodic_query = {
                'query': query_text,
                'k': 10,
                'use_mmr': True
            }
            
            # Add time filter if context specifies
            if context and 'time_range' in context:
                plan.episodic_query['time_filter'] = context['time_range']
        
        if 'semantic' in stores_to_query:
            plan.semantic_query = {
                'query': query_text,
                'k': 10,
                'max_hops': 3,
                'use_reasoning': True
            }
        
        logger.debug(
            "Query plan created",
            query_type=query_type.value,
            stores=stores_to_query,
            synthesis=synthesis_mode,
            time_budget_ms=time_budget
        )
        
        return plan
    
    async def _execute_retrieval(self, query_plan: QueryPlan) -> Dict[str, List[Any]]:
        """Execute parallel retrieval from multiple stores"""
        tasks = {}
        
        # Create retrieval tasks
        if query_plan.working_memory_query:
            tasks['working'] = self.working_memory.retrieve(
                query_plan.working_memory_query['query'],
                k=query_plan.working_memory_query.get('k', 3)
            )
        
        if query_plan.episodic_query:
            tasks['episodic'] = self.episodic_memory.retrieve(
                **query_plan.episodic_query
            )
        
        if query_plan.semantic_query:
            tasks['semantic'] = self.semantic_memory.query(
                **query_plan.semantic_query
            )
        
        # Check causal patterns if needed
        if query_plan.query_type == QueryType.CAUSAL:
            # Get recent events for causal analysis
            recent_events = []  # Would fetch from episodic
            if recent_events:
                failure_prob, pattern_id = self.causal_tracker.get_failure_prediction(
                    recent_events
                )
                if failure_prob > 0.5:
                    tasks['causal'] = asyncio.create_task(
                        self._get_causal_explanation(pattern_id)
                    )
        
        # Execute all tasks in parallel
        if tasks:
            results = await asyncio.gather(
                *[tasks[key] for key in tasks.keys()],
                return_exceptions=True
            )
            
            # Map results back to store names
            retrieved_data = {}
            for i, key in enumerate(tasks.keys()):
                if not isinstance(results[i], Exception):
                    retrieved_data[key] = results[i] if isinstance(results[i], list) else [results[i]]
                else:
                    logger.warning(f"Retrieval error from {key}: {results[i]}")
                    retrieved_data[key] = []
        else:
            retrieved_data = {}
        
        return retrieved_data
    
    async def _get_causal_explanation(self, pattern_id: str) -> List[Dict]:
        """Get causal explanation for a pattern"""
        if pattern_id in self.causal_tracker.patterns:
            pattern = self.causal_tracker.patterns[pattern_id]
            return [{
                'pattern_id': pattern_id,
                'events': pattern.event_sequence,
                'outcome': pattern.outcome,
                'confidence': pattern.compute_confidence()
            }]
        return []
    
    async def _fallback_query(self, query_text: str) -> MemoryContext:
        """Simplified fallback query when main path fails"""
        try:
            # Simple episodic search
            episodes = await asyncio.wait_for(
                self.episodic_memory.retrieve(query_text, k=5),
                timeout=2.0
            )
            
            context = MemoryContext(
                query=query_text,
                timestamp=datetime.now(),
                episodes=episodes,
                synthesized_answer=f"Found {len(episodes)} related memories (fallback mode)",
                confidence=0.3
            )
            
            return context
            
        except Exception as e:
            logger.error(f"Fallback query failed: {e}")
            
            return MemoryContext(
                query=query_text,
                timestamp=datetime.now(),
                synthesized_answer="No memories available",
                confidence=0.0
            )
    
    # ==================== Learning Path (Consolidation) ====================
    
    async def run_sleep_cycle(self):
        """
        Run full sleep consolidation cycle
        
        Research: "Sleep-like offline consolidation"
        """
        logger.info("Starting sleep cycle...")
        start_time = time.time()
        
        try:
            # Run complete sleep cycle
            results = await self.consolidation.run_full_cycle()
            
            # Update last sleep time
            self.last_sleep_cycle = datetime.now()
            
            # Update metrics
            self.metrics['consolidation_cycles'] += 1
            
            # Extract knowledge from consolidated episodes
            if 'consolidated_episodes' in results:
                concepts = await self.semantic_memory.extract_knowledge_from_episodes(
                    results['consolidated_episodes']
                )
                logger.info(f"Extracted {len(concepts)} concepts during sleep")
            
            duration = time.time() - start_time
            logger.info(
                "Sleep cycle completed",
                duration_seconds=duration,
                **results
            )
            
        except Exception as e:
            logger.error(f"Sleep cycle error: {e}")
    
    async def run_awake_consolidation(self):
        """
        Run rapid consolidation during wake
        
        Research: "Sharp-wave ripples during brief idle periods"
        """
        logger.debug("Running awake consolidation...")
        
        try:
            # Get high-importance recent memories
            recent = await self.episodic_memory.get_recent(limit=20)
            important = [m for m in recent if m.total_importance >= 0.8]
            
            if important:
                # Rapid replay
                for memory in important:
                    memory.replay_count += 1
                    memory.consolidation_state = 'rapid'
                
                logger.info(f"Rapid consolidation of {len(important)} memories")
            
        except Exception as e:
            logger.error(f"Awake consolidation error: {e}")
    
    # ==================== Utility Methods ====================
    
    async def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive system statistics"""
        stats = {
            'system_metrics': self.metrics,
            'working_memory': {
                'current_items': len(self.working_memory.items),
                'total_processed': self.working_memory.total_items_processed
            },
            'episodic_memory': await self.episodic_memory.get_statistics(),
            'semantic_memory': await self.semantic_memory.get_statistics(),
            'lifecycle': {
                'transfers': dict(self.lifecycle_manager.transfer_counts),
                'consolidation_triggers': self.lifecycle_manager.consolidation_triggers
            },
            'causal_tracker': self.causal_tracker.get_statistics(),
            'cache_size': len(self.query_cache),
            'last_sleep': self.last_sleep_cycle.isoformat()
        }
        
        return stats
    
    async def health_check(self) -> Dict[str, bool]:
        """Check health of all memory systems"""
        health = {}
        
        # Check each store
        try:
            # Working memory
            test_item = await self.working_memory.add("health_check", 0.1)
            health['working_memory'] = test_item is not None
        except:
            health['working_memory'] = False
        
        try:
            # Episodic memory
            episodes = await self.episodic_memory.get_recent(limit=1)
            health['episodic_memory'] = True
        except:
            health['episodic_memory'] = False
        
        try:
            # Semantic memory
            stats = await self.semantic_memory.get_statistics()
            health['semantic_memory'] = stats is not None
        except:
            health['semantic_memory'] = False
        
        health['overall'] = all(health.values())
        
        return health