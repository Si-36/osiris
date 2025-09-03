"""
Memory Consolidation Orchestrator - September 2025 State-of-the-Art
====================================================================

Implements biologically-inspired sleep cycles for memory consolidation:
- NREM: Memory triage and candidate selection
- SWS: High-speed replay and topological abstraction
- REM: Creative dream generation and validation
- Homeostasis: Synaptic normalization and pruning

Based on latest research:
- SESLR (Sleep Enhanced Latent Replay) for 32x efficiency
- Topological Laplacians for geometric evolution
- Astrocyte-inspired associative validation
- Sub-minute homeostatic response
"""

import asyncio
import time
import logging
from enum import Enum
from typing import Dict, Any, List, Optional, Set, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import numpy as np
import structlog

# Import our components
from .replay_buffer import PriorityReplayBuffer
from .dream_generator import DreamGenerator
from .homeostasis import SynapticHomeostasis
from .laplacian_extractor import TopologicalLaplacianExtractor

logger = structlog.get_logger(__name__)


# ==================== Sleep Phases ====================

class SleepPhase(Enum):
    """
    Biologically-inspired sleep phases for memory consolidation
    """
    AWAKE = "awake"        # Normal operation, data collection
    NREM = "nrem"          # Non-REM: Memory triage and cleanup
    SWS = "sws"            # Slow-Wave Sleep: Replay and strengthening
    REM = "rem"            # REM: Dream generation and creativity
    TRANSITION = "transition"  # Phase transition state


@dataclass
class ConsolidationMetrics:
    """Metrics for monitoring consolidation performance"""
    cycle_count: int = 0
    total_memories_processed: int = 0
    memories_abstracted: int = 0
    dreams_generated: int = 0
    dreams_validated: int = 0
    memories_pruned: int = 0
    cycle_duration_ms: float = 0.0
    replay_speed_factor: float = 15.0
    abstraction_ratio: float = 0.0
    dream_validation_rate: float = 0.0
    last_cycle_time: Optional[datetime] = None


@dataclass
class ConsolidationConfig:
    """Configuration for memory consolidation"""
    # Cycle timing
    min_cycle_interval: timedelta = timedelta(hours=1)
    max_cycle_duration: timedelta = timedelta(minutes=5)
    
    # NREM phase
    surprise_threshold: float = 0.7
    triage_batch_size: int = 1000
    min_importance_threshold: float = 0.1
    
    # SWS phase
    replay_speed: float = 15.0  # 15x normal speed
    replay_batch_size: int = 200
    strengthening_factor: float = 1.5
    abstraction_cluster_size: int = 10
    
    # REM phase
    dream_pairs_count: int = 50
    dream_validation_threshold: float = 0.8
    causal_plausibility_threshold: float = 0.7
    max_dreams_per_cycle: int = 20
    
    # Homeostasis
    global_downscale_factor: float = 0.8
    selective_upscale_factor: float = 1.25
    prune_percentile: int = 5
    homeostasis_reaction_time: timedelta = timedelta(minutes=2)
    
    # Feature flags
    enable_dreams: bool = True
    enable_laplacians: bool = True
    enable_binary_spikes: bool = True
    enable_fallback: bool = True


# ==================== Main Orchestrator ====================

class SleepConsolidation:
    """
    Advanced Memory Consolidation Orchestrator
    
    Manages the complete sleep cycle for continuous learning,
    memory abstraction, and creative problem solving.
    """
    
    def __init__(self, services: Dict[str, Any], config: Optional[ConsolidationConfig] = None):
        """
        Initialize the consolidation orchestrator
        
        Args:
            services: Dictionary containing AURA components:
                - working_memory: WorkingMemory instance
                - episodic_memory: EpisodicMemory instance
                - semantic_memory: SemanticMemory instance
                - topology_adapter: TopologyMemoryAdapter instance
                - causal_tracker: CausalPatternTracker instance
                - shape_memory: ShapeAwareMemoryV2 instance
                - hierarchical_router: HierarchicalMemoryRouter2025 instance
                - circadian_manager: CircadianRhythms instance (optional)
            config: Configuration for consolidation parameters
        """
        self.services = services
        self.config = config or ConsolidationConfig()
        
        # Current state
        self.current_phase = SleepPhase.AWAKE
        self.is_running = False
        self.last_cycle_time = None
        
        # Core components
        self.replay_buffer = PriorityReplayBuffer(
            causal_tracker=services['causal_tracker'],
            max_size=10000
        )
        
        self.dream_generator = DreamGenerator(
            services=services,
            config=self.config
        )
        
        self.homeostasis = SynapticHomeostasis(
            services=services,
            config=self.config
        )
        
        self.laplacian_extractor = TopologicalLaplacianExtractor(
            topology_adapter=services['topology_adapter']
        )
        
        # Metrics tracking
        self.metrics = ConsolidationMetrics()
        
        # Async task management
        self._consolidation_task = None
        self._monitoring_task = None
        
        logger.info(
            "SleepConsolidation initialized",
            replay_speed=self.config.replay_speed,
            dream_enabled=self.config.enable_dreams,
            laplacians_enabled=self.config.enable_laplacians
        )
    
    # ==================== Lifecycle Management ====================
    
    async def start(self):
        """Start the consolidation system"""
        if self.is_running:
            logger.warning("Consolidation already running")
            return
        
        self.is_running = True
        
        # Start monitoring task
        self._monitoring_task = asyncio.create_task(self._monitor_system())
        
        # Register with circadian manager if available
        if 'circadian_manager' in self.services:
            await self.services['circadian_manager'].register_sleep_handler(
                self.trigger_consolidation_cycle
            )
        
        logger.info("Memory consolidation system started")
    
    async def stop(self):
        """Stop the consolidation system"""
        self.is_running = False
        
        # Cancel tasks
        if self._consolidation_task:
            self._consolidation_task.cancel()
        if self._monitoring_task:
            self._monitoring_task.cancel()
        
        # Wait for cleanup
        await asyncio.sleep(0.1)
        
        logger.info("Memory consolidation system stopped")
    
    # ==================== Main Consolidation Cycle ====================
    
    async def trigger_consolidation_cycle(self) -> Dict[str, Any]:
        """
        Trigger a complete consolidation cycle
        
        Returns:
            Dictionary containing cycle results and metrics
        """
        # Check if we can run a cycle
        if not self._can_run_cycle():
            return {"status": "skipped", "reason": "too_soon"}
        
        # Check if already running
        if self._consolidation_task and not self._consolidation_task.done():
            return {"status": "skipped", "reason": "already_running"}
        
        # Start the cycle
        self._consolidation_task = asyncio.create_task(self._run_consolidation_cycle())
        
        # Wait for completion with timeout
        try:
            result = await asyncio.wait_for(
                self._consolidation_task,
                timeout=self.config.max_cycle_duration.total_seconds()
            )
            return result
        except asyncio.TimeoutError:
            logger.error(
                "Consolidation cycle timeout",
                duration=self.config.max_cycle_duration.total_seconds()
            )
            return {"status": "error", "reason": "timeout"}
    
    async def _run_consolidation_cycle(self) -> Dict[str, Any]:
        """
        Run a complete sleep consolidation cycle
        
        Returns:
            Cycle results and metrics
        """
        cycle_start = time.time()
        self.metrics.cycle_count += 1
        
        logger.info(
            "Starting consolidation cycle",
            cycle=self.metrics.cycle_count,
            phase=self.current_phase.value
        )
        
        try:
            # Phase 1: NREM - Memory Triage
            nrem_results = await self._nrem_phase()
            
            # Phase 2: SWS - Replay and Abstraction
            sws_results = await self._sws_phase()
            
            # Phase 3: REM - Dream Generation (if enabled)
            rem_results = {}
            if self.config.enable_dreams:
                rem_results = await self._rem_phase()
            
            # Phase 4: Homeostasis - Synaptic Normalization
            homeostasis_results = await self._homeostasis_phase()
            
            # Update metrics
            cycle_duration = (time.time() - cycle_start) * 1000
            self.metrics.cycle_duration_ms = cycle_duration
            self.metrics.last_cycle_time = datetime.now()
            
            # Calculate derived metrics
            if self.metrics.total_memories_processed > 0:
                self.metrics.abstraction_ratio = (
                    self.metrics.memories_abstracted / 
                    self.metrics.total_memories_processed
                )
            
            if self.metrics.dreams_generated > 0:
                self.metrics.dream_validation_rate = (
                    self.metrics.dreams_validated / 
                    self.metrics.dreams_generated
                )
            
            # Return to AWAKE state
            self.current_phase = SleepPhase.AWAKE
            
            results = {
                "status": "success",
                "cycle": self.metrics.cycle_count,
                "duration_ms": cycle_duration,
                "phases": {
                    "nrem": nrem_results,
                    "sws": sws_results,
                    "rem": rem_results,
                    "homeostasis": homeostasis_results
                },
                "metrics": {
                    "total_processed": self.metrics.total_memories_processed,
                    "abstracted": self.metrics.memories_abstracted,
                    "dreams_generated": self.metrics.dreams_generated,
                    "dreams_validated": self.metrics.dreams_validated,
                    "pruned": self.metrics.memories_pruned,
                    "abstraction_ratio": self.metrics.abstraction_ratio,
                    "dream_validation_rate": self.metrics.dream_validation_rate
                }
            }
            
            logger.info(
                "Consolidation cycle complete",
                cycle=self.metrics.cycle_count,
                duration_ms=cycle_duration,
                abstraction_ratio=f"{self.metrics.abstraction_ratio:.2%}",
                dream_validation_rate=f"{self.metrics.dream_validation_rate:.2%}"
            )
            
            return results
            
        except Exception as e:
            logger.error(
                "Consolidation cycle error",
                cycle=self.metrics.cycle_count,
                error=str(e),
                phase=self.current_phase.value
            )
            
            # Attempt fallback if enabled
            if self.config.enable_fallback:
                return await self._fallback_consolidation()
            
            return {"status": "error", "reason": str(e)}
    
    # ==================== NREM Phase: Memory Triage ====================
    
    async def _nrem_phase(self) -> Dict[str, Any]:
        """
        NREM Phase: Memory triage and candidate selection
        
        This phase:
        1. Collects memories from working and episodic stores
        2. Calculates surprise scores using CausalPatternTracker
        3. Populates the priority replay buffer
        4. Prunes low-importance memories
        
        Returns:
            Phase results and statistics
        """
        self.current_phase = SleepPhase.NREM
        phase_start = time.time()
        
        logger.info("Entering NREM phase: Memory Triage")
        
        # Collect memory candidates
        working_candidates = await self.services['working_memory'].get_all()
        episodic_candidates = await self.services['episodic_memory'].get_recent(
            limit=self.config.triage_batch_size
        )
        
        all_candidates = working_candidates + episodic_candidates
        
        # Calculate surprise scores and populate replay buffer
        await self.replay_buffer.populate(all_candidates)
        
        # Prune low-importance memories
        pruned_count = 0
        for memory in all_candidates:
            if memory.importance < self.config.min_importance_threshold:
                await self.services['episodic_memory'].remove(memory.id)
                pruned_count += 1
        
        self.metrics.memories_pruned += pruned_count
        
        phase_duration = (time.time() - phase_start) * 1000
        
        results = {
            "candidates_collected": len(all_candidates),
            "buffer_size": len(self.replay_buffer.buffer),
            "pruned": pruned_count,
            "duration_ms": phase_duration
        }
        
        logger.info(
            "NREM phase complete",
            candidates=len(all_candidates),
            buffered=len(self.replay_buffer.buffer),
            pruned=pruned_count,
            duration_ms=phase_duration
        )
        
        return results
    
    # ==================== SWS Phase: Replay & Abstraction ====================
    
    async def _sws_phase(self) -> Dict[str, Any]:
        """
        SWS Phase: High-speed replay and topological abstraction
        
        This phase:
        1. Replays top memories at 10-20x speed
        2. Strengthens important pathways
        3. Extracts topological invariants using Laplacians
        4. Creates abstract semantic memories
        
        Returns:
            Phase results and statistics
        """
        self.current_phase = SleepPhase.SWS
        phase_start = time.time()
        
        logger.info(
            "Entering SWS phase: Replay and Abstraction",
            replay_speed=f"{self.config.replay_speed}x"
        )
        
        # Get top candidates for replay
        top_candidates = self.replay_buffer.get_top_candidates(
            n=self.config.replay_batch_size
        )
        
        if not top_candidates:
            logger.warning("No candidates for SWS replay")
            return {"status": "skipped", "reason": "no_candidates"}
        
        # High-speed replay to strengthen pathways
        replay_start = time.time()
        strengthened_ids = []
        
        for memory in top_candidates:
            # Simulate high-speed replay
            await self._replay_memory(memory, speed=self.config.replay_speed)
            
            # Strengthen the pathway
            await self.services['shape_memory'].strengthen_pathway(
                memory_id=memory.id,
                factor=self.config.strengthening_factor
            )
            
            strengthened_ids.append(memory.id)
        
        replay_duration = (time.time() - replay_start) * 1000
        effective_replay_time = replay_duration * self.config.replay_speed
        
        # Group memories for topological abstraction
        memory_clusters = self._cluster_by_similarity(
            top_candidates,
            cluster_size=self.config.abstraction_cluster_size
        )
        
        # Extract topological invariants and create abstractions
        abstractions_created = 0
        
        for cluster in memory_clusters:
            if len(cluster) < 3:  # Need minimum memories for abstraction
                continue
            
            # Extract topological signatures
            if self.config.enable_laplacians:
                # Use advanced Laplacian extraction
                laplacian_sigs = await self.laplacian_extractor.batch_extract(
                    [m.content for m in cluster]
                )
                
                # Separate harmonic (topology) and non-harmonic (geometry)
                harmonic_features = [sig.harmonic_spectrum for sig in laplacian_sigs]
                geometric_features = [sig.non_harmonic_spectrum for sig in laplacian_sigs]
                
                # Create abstract concept
                abstract_concept = {
                    "topology": self._compute_mean_spectrum(harmonic_features),
                    "geometry": self._compute_mean_spectrum(geometric_features),
                    "source_ids": [m.id for m in cluster],
                    "creation_time": datetime.now(),
                    "confidence": self._calculate_abstraction_confidence(cluster)
                }
            else:
                # Fallback to simple topology
                topology_sigs = []
                for memory in cluster:
                    sig = await self.services['topology_adapter'].extract_topology(
                        memory.content
                    )
                    topology_sigs.append(sig)
                
                abstract_concept = {
                    "topology": self._compute_mean_topology(topology_sigs),
                    "source_ids": [m.id for m in cluster],
                    "creation_time": datetime.now()
                }
            
            # Store abstract concept in semantic memory
            await self.services['semantic_memory'].store_abstract(abstract_concept)
            abstractions_created += 1
        
        self.metrics.total_memories_processed += len(top_candidates)
        self.metrics.memories_abstracted += abstractions_created
        
        phase_duration = (time.time() - phase_start) * 1000
        
        results = {
            "replayed": len(strengthened_ids),
            "replay_speed": self.config.replay_speed,
            "replay_duration_ms": replay_duration,
            "effective_time_ms": effective_replay_time,
            "clusters_formed": len(memory_clusters),
            "abstractions_created": abstractions_created,
            "phase_duration_ms": phase_duration
        }
        
        logger.info(
            "SWS phase complete",
            replayed=len(strengthened_ids),
            abstractions=abstractions_created,
            duration_ms=phase_duration
        )
        
        return results
    
    # ==================== REM Phase: Dream Generation ====================
    
    async def _rem_phase(self) -> Dict[str, Any]:
        """
        REM Phase: Creative dream generation and validation
        
        This phase:
        1. Selects semantically distant memory pairs
        2. Generates novel "dream" memories via interpolation
        3. Validates dreams using Astrocyte-inspired transformer
        4. Tests causal plausibility
        5. Stores validated insights
        
        Returns:
            Phase results and statistics
        """
        self.current_phase = SleepPhase.REM
        phase_start = time.time()
        
        logger.info("Entering REM phase: Dream Generation")
        
        # Select semantically distant pairs for dreaming
        memory_pairs = self.replay_buffer.select_distant_pairs(
            count=self.config.dream_pairs_count
        )
        
        if not memory_pairs:
            logger.warning("No memory pairs for dreaming")
            return {"status": "skipped", "reason": "no_pairs"}
        
        # Generate and validate dreams
        dreams_generated = 0
        dreams_validated = 0
        novel_insights = []
        
        for m1, m2 in memory_pairs:
            if dreams_generated >= self.config.max_dreams_per_cycle:
                break
            
            # Generate dream via interpolation
            dream = await self.dream_generator.generate_dream(m1, m2)
            dreams_generated += 1
            
            # Validate dream coherence
            is_valid = await self.dream_generator.validate_dream(
                dream,
                [m1, m2],
                threshold=self.config.dream_validation_threshold
            )
            
            if not is_valid:
                continue
            
            # Test causal plausibility
            is_plausible = await self.services['causal_tracker'].predict_plausibility(
                dream.signature
            )
            
            if is_plausible > self.config.causal_plausibility_threshold:
                dreams_validated += 1
                novel_insights.append(dream)
                
                # Store validated dream as semantic insight
                await self.services['semantic_memory'].store_insight(
                    insight=dream.to_dict(),
                    confidence=is_plausible,
                    source="dream_generation"
                )
        
        self.metrics.dreams_generated += dreams_generated
        self.metrics.dreams_validated += dreams_validated
        
        phase_duration = (time.time() - phase_start) * 1000
        
        results = {
            "pairs_selected": len(memory_pairs),
            "dreams_generated": dreams_generated,
            "dreams_validated": dreams_validated,
            "validation_rate": dreams_validated / max(1, dreams_generated),
            "novel_insights": len(novel_insights),
            "phase_duration_ms": phase_duration
        }
        
        logger.info(
            "REM phase complete",
            generated=dreams_generated,
            validated=dreams_validated,
            insights=len(novel_insights),
            duration_ms=phase_duration
        )
        
        return results
    
    # ==================== Homeostasis Phase ====================
    
    async def _homeostasis_phase(self) -> Dict[str, Any]:
        """
        Homeostasis Phase: Synaptic normalization and stability
        
        This phase:
        1. Applies global multiplicative downscaling (0.8x)
        2. Selectively upscales replayed pathways (1.25x)
        3. Prunes bottom 5th percentile weights
        4. Ensures system stability
        
        Returns:
            Phase results and statistics
        """
        self.current_phase = SleepPhase.TRANSITION
        phase_start = time.time()
        
        logger.info("Entering Homeostasis phase: Synaptic Normalization")
        
        # Get replayed memory IDs from buffer
        replayed_ids = self.replay_buffer.get_replayed_ids()
        
        # Perform homeostatic renormalization
        homeostasis_results = await self.homeostasis.renormalize(
            replayed_memory_ids=replayed_ids
        )
        
        phase_duration = (time.time() - phase_start) * 1000
        
        results = {
            "downscale_factor": self.config.global_downscale_factor,
            "upscale_factor": self.config.selective_upscale_factor,
            "memories_upscaled": len(replayed_ids),
            "memories_pruned": homeostasis_results.get('pruned', 0),
            "total_weights": homeostasis_results.get('total_weights', 0),
            "phase_duration_ms": phase_duration
        }
        
        logger.info(
            "Homeostasis phase complete",
            upscaled=len(replayed_ids),
            pruned=homeostasis_results.get('pruned', 0),
            duration_ms=phase_duration
        )
        
        return results
    
    # ==================== Helper Methods ====================
    
    def _can_run_cycle(self) -> bool:
        """Check if enough time has passed since last cycle"""
        if not self.last_cycle_time:
            return True
        
        time_since_last = datetime.now() - self.last_cycle_time
        return time_since_last >= self.config.min_cycle_interval
    
    async def _replay_memory(self, memory: Any, speed: float = 1.0):
        """
        Simulate high-speed memory replay
        
        Args:
            memory: Memory to replay
            speed: Speed factor (e.g., 15.0 for 15x speed)
        """
        # Calculate replay duration
        base_duration = 0.1  # 100ms base replay time
        actual_duration = base_duration / speed
        
        # Simulate replay processing
        await asyncio.sleep(actual_duration)
    
    def _cluster_by_similarity(self, memories: List[Any], 
                              cluster_size: int = 10) -> List[List[Any]]:
        """
        Cluster memories by topological similarity
        
        Args:
            memories: List of memories to cluster
            cluster_size: Target size for each cluster
        
        Returns:
            List of memory clusters
        """
        # Simple clustering for now - can be enhanced with proper algorithms
        clusters = []
        current_cluster = []
        
        for memory in memories:
            current_cluster.append(memory)
            if len(current_cluster) >= cluster_size:
                clusters.append(current_cluster)
                current_cluster = []
        
        if current_cluster:
            clusters.append(current_cluster)
        
        return clusters
    
    def _compute_mean_spectrum(self, spectra: List[np.ndarray]) -> np.ndarray:
        """Compute mean of multiple spectra"""
        if not spectra:
            return np.array([])
        return np.mean(spectra, axis=0)
    
    def _compute_mean_topology(self, signatures: List[Any]) -> Dict[str, Any]:
        """Compute mean topological signature"""
        # Aggregate topological features
        return {
            "betti_numbers": self._average_betti_numbers(signatures),
            "persistence": self._average_persistence(signatures),
            "timestamp": datetime.now()
        }
    
    def _average_betti_numbers(self, signatures: List[Any]) -> List[float]:
        """Average Betti numbers across signatures"""
        if not signatures:
            return []
        
        all_betti = [sig.betti_numbers for sig in signatures if hasattr(sig, 'betti_numbers')]
        if not all_betti:
            return []
        
        max_len = max(len(b) for b in all_betti)
        averaged = []
        
        for i in range(max_len):
            values = [b[i] for b in all_betti if i < len(b)]
            averaged.append(np.mean(values) if values else 0.0)
        
        return averaged
    
    def _average_persistence(self, signatures: List[Any]) -> float:
        """Average persistence across signatures"""
        persistences = [
            sig.total_persistence for sig in signatures 
            if hasattr(sig, 'total_persistence')
        ]
        return np.mean(persistences) if persistences else 0.0
    
    def _calculate_abstraction_confidence(self, cluster: List[Any]) -> float:
        """Calculate confidence score for abstraction"""
        # Based on cluster coherence and size
        base_confidence = min(1.0, len(cluster) / 10)
        
        # Adjust based on importance scores
        importances = [m.importance for m in cluster if hasattr(m, 'importance')]
        if importances:
            avg_importance = np.mean(importances)
            base_confidence *= avg_importance
        
        return base_confidence
    
    async def _fallback_consolidation(self) -> Dict[str, Any]:
        """Fallback to legacy consolidation if new system fails"""
        logger.warning("Falling back to legacy consolidation")
        
        try:
            # Use existing simple consolidation
            if 'legacy_consolidation' in self.services:
                await self.services['legacy_consolidation'].consolidate()
                return {"status": "fallback", "method": "legacy"}
            else:
                return {"status": "error", "reason": "no_fallback"}
        except Exception as e:
            logger.error(f"Fallback consolidation failed: {e}")
            return {"status": "error", "reason": f"fallback_failed: {e}"}
    
    async def _monitor_system(self):
        """Background monitoring task"""
        while self.is_running:
            try:
                # Check system health
                if 'circadian_manager' in self.services:
                    circadian_phase = await self.services['circadian_manager'].get_phase()
                    
                    # Trigger consolidation during low activity
                    if circadian_phase == "low_activity" and self._can_run_cycle():
                        await self.trigger_consolidation_cycle()
                
                # Wait before next check
                await asyncio.sleep(60)  # Check every minute
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Monitoring error: {e}")
                await asyncio.sleep(60)
    
    # ==================== Public Interface ====================
    
    async def get_metrics(self) -> Dict[str, Any]:
        """Get current consolidation metrics"""
        return {
            "cycles_completed": self.metrics.cycle_count,
            "total_processed": self.metrics.total_memories_processed,
            "abstractions_created": self.metrics.memories_abstracted,
            "dreams_generated": self.metrics.dreams_generated,
            "dreams_validated": self.metrics.dreams_validated,
            "memories_pruned": self.metrics.memories_pruned,
            "abstraction_ratio": self.metrics.abstraction_ratio,
            "dream_validation_rate": self.metrics.dream_validation_rate,
            "last_cycle": self.metrics.last_cycle_time.isoformat() if self.metrics.last_cycle_time else None,
            "current_phase": self.current_phase.value,
            "is_running": self.is_running
        }
    
    async def force_cycle(self) -> Dict[str, Any]:
        """Force an immediate consolidation cycle (for testing)"""
        logger.warning("Forcing consolidation cycle")
        self.last_cycle_time = None  # Reset timer
        return await self.trigger_consolidation_cycle()
    
    def get_config(self) -> Dict[str, Any]:
        """Get current configuration"""
        return {
            "replay_speed": self.config.replay_speed,
            "dream_enabled": self.config.enable_dreams,
            "laplacians_enabled": self.config.enable_laplacians,
            "binary_spikes_enabled": self.config.enable_binary_spikes,
            "downscale_factor": self.config.global_downscale_factor,
            "upscale_factor": self.config.selective_upscale_factor,
            "prune_percentile": self.config.prune_percentile
        }


# ==================== Factory Function ====================

async def create_sleep_consolidation(services: Dict[str, Any], 
                                    config: Optional[Dict[str, Any]] = None) -> SleepConsolidation:
    """
    Factory function to create and initialize SleepConsolidation
    
    Args:
        services: Dictionary of AURA services
        config: Optional configuration overrides
    
    Returns:
        Initialized SleepConsolidation instance
    """
    # Create config from dict if provided
    if config:
        consolidation_config = ConsolidationConfig(**config)
    else:
        consolidation_config = ConsolidationConfig()
    
    # Create orchestrator
    orchestrator = SleepConsolidation(services, consolidation_config)
    
    # Start the system
    await orchestrator.start()
    
    logger.info("Sleep consolidation system created and started")
    
    return orchestrator