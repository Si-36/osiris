"""
Synaptic Homeostasis Manager - Real-Time Weight Normalization
=============================================================

Implements synaptic homeostasis based on Tononi & Cirelli (2025):
- Global multiplicative downscaling (0.8x) to prevent saturation
- Selective upscaling (1.25x) for replayed pathways
- Dynamic pruning of bottom 5th percentile
- Sub-minute reaction time for stability

This is critical for:
- Preventing catastrophic forgetting
- Maintaining network stability
- Continuous learning capacity
- Memory efficiency
"""

import asyncio
import time
import numpy as np
from typing import Dict, Any, List, Set, Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import deque
import structlog

logger = structlog.get_logger(__name__)


@dataclass
class HomeostasisMetrics:
    """Metrics for homeostasis monitoring"""
    total_normalizations: int = 0
    total_weights_processed: int = 0
    total_pruned: int = 0
    avg_weight_before: float = 0.0
    avg_weight_after: float = 0.0
    max_weight: float = 0.0
    min_weight: float = 0.0
    stability_score: float = 1.0
    last_normalization: Optional[datetime] = None
    reaction_times: deque = field(default_factory=lambda: deque(maxlen=100))


@dataclass
class WeightSnapshot:
    """Snapshot of memory weights at a point in time"""
    timestamp: datetime
    weights: Dict[str, float]
    total_weight: float
    mean_weight: float
    std_weight: float
    percentiles: Dict[int, float]


class SynapticHomeostasis:
    """
    Advanced synaptic homeostasis manager
    
    Maintains network stability through continuous weight normalization,
    implementing the latest research on synaptic scaling and homeostatic
    plasticity.
    """
    
    def __init__(self, services: Dict[str, Any], config: Any):
        """
        Initialize the homeostasis manager
        
        Args:
            services: Dictionary of AURA services
            config: Consolidation configuration
        """
        self.services = services
        self.config = config
        
        # Weight management
        self.current_weights: Dict[str, float] = {}
        self.weight_history: deque = deque(maxlen=1000)
        self.replayed_weights: Set[str] = set()
        
        # Stability monitoring
        self.stability_window = deque(maxlen=60)  # 1 minute of measurements
        self.last_reaction_time = datetime.now()
        self.is_stable = True
        
        # Metrics
        self.metrics = HomeostasisMetrics()
        
        # Background monitoring task
        self._monitor_task = None
        self._is_running = False
        
        logger.info(
            "SynapticHomeostasis initialized",
            downscale_factor=config.global_downscale_factor,
            upscale_factor=config.selective_upscale_factor,
            prune_percentile=config.prune_percentile,
            reaction_time=config.homeostasis_reaction_time.total_seconds()
        )
    
    async def start_monitoring(self):
        """Start continuous homeostasis monitoring"""
        if self._is_running:
            return
        
        self._is_running = True
        self._monitor_task = asyncio.create_task(self._continuous_monitoring())
        logger.info("Homeostasis monitoring started")
    
    async def stop_monitoring(self):
        """Stop monitoring"""
        self._is_running = False
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
        logger.info("Homeostasis monitoring stopped")
    
    async def renormalize(self, replayed_memory_ids: List[str]) -> Dict[str, Any]:
        """
        Perform complete homeostatic renormalization
        
        This is the main entry point called at the end of each sleep cycle.
        
        Args:
            replayed_memory_ids: IDs of memories that were replayed
        
        Returns:
            Normalization results and statistics
        """
        start_time = time.time()
        self.replayed_weights = set(replayed_memory_ids)
        
        logger.info(
            "Starting synaptic renormalization",
            replayed_count=len(replayed_memory_ids)
        )
        
        # Take weight snapshot before normalization
        before_snapshot = await self._take_weight_snapshot()
        
        # Step 1: Fetch all current weights
        all_weights = await self._fetch_all_weights()
        initial_count = len(all_weights)
        
        if not all_weights:
            logger.warning("No weights to normalize")
            return {"status": "skipped", "reason": "no_weights"}
        
        # Step 2: Apply global multiplicative downscaling
        downscaled_weights = self._apply_global_downscaling(all_weights)
        
        # Step 3: Apply selective upscaling to replayed memories
        upscaled_weights = self._apply_selective_upscaling(
            downscaled_weights, 
            replayed_memory_ids
        )
        
        # Step 4: Calculate pruning threshold
        prune_threshold = self._calculate_prune_threshold(upscaled_weights)
        
        # Step 5: Prune weak connections
        pruned_weights, pruned_ids = self._prune_weak_connections(
            upscaled_weights,
            prune_threshold
        )
        
        # Step 6: Ensure stability (additional normalization if needed)
        stable_weights = await self._ensure_stability(pruned_weights)
        
        # Step 7: Commit normalized weights
        await self._commit_weights(stable_weights)
        
        # Take weight snapshot after normalization
        after_snapshot = await self._take_weight_snapshot()
        
        # Update metrics
        self._update_metrics(before_snapshot, after_snapshot, len(pruned_ids))
        
        # Calculate reaction time
        reaction_time = (time.time() - start_time) * 1000
        self.metrics.reaction_times.append(reaction_time)
        
        results = {
            "status": "success",
            "initial_weights": initial_count,
            "final_weights": len(stable_weights),
            "pruned": len(pruned_ids),
            "downscale_factor": self.config.global_downscale_factor,
            "upscale_factor": self.config.selective_upscale_factor,
            "prune_threshold": prune_threshold,
            "reaction_time_ms": reaction_time,
            "stability_score": self.metrics.stability_score,
            "avg_weight_change": abs(
                after_snapshot.mean_weight - before_snapshot.mean_weight
            )
        }
        
        logger.info(
            "Synaptic renormalization complete",
            pruned=len(pruned_ids),
            final_weights=len(stable_weights),
            reaction_ms=reaction_time,
            stability=self.metrics.stability_score
        )
        
        return results
    
    def _apply_global_downscaling(self, weights: Dict[str, float]) -> Dict[str, float]:
        """
        Apply global multiplicative downscaling to all weights
        
        This prevents runaway potentiation and maintains stability.
        
        Args:
            weights: Current weight dictionary
        
        Returns:
            Downscaled weights
        """
        factor = self.config.global_downscale_factor
        
        downscaled = {}
        for memory_id, weight in weights.items():
            # Apply multiplicative scaling
            new_weight = weight * factor
            
            # Ensure minimum weight (prevent complete elimination)
            new_weight = max(new_weight, 0.001)
            
            downscaled[memory_id] = new_weight
        
        logger.debug(
            "Global downscaling applied",
            factor=factor,
            count=len(downscaled),
            avg_before=np.mean(list(weights.values())),
            avg_after=np.mean(list(downscaled.values()))
        )
        
        return downscaled
    
    def _apply_selective_upscaling(self, weights: Dict[str, float], 
                                  replayed_ids: List[str]) -> Dict[str, float]:
        """
        Selectively upscale weights of replayed memories
        
        This compensates for global downscaling on important pathways.
        
        Args:
            weights: Downscaled weights
            replayed_ids: IDs of replayed memories
        
        Returns:
            Weights with selective upscaling applied
        """
        factor = self.config.selective_upscale_factor
        replayed_set = set(replayed_ids)
        
        upscaled = weights.copy()
        upscaled_count = 0
        
        for memory_id in replayed_set:
            if memory_id in upscaled:
                # Apply compensatory upscaling
                upscaled[memory_id] *= factor
                upscaled_count += 1
        
        logger.debug(
            "Selective upscaling applied",
            factor=factor,
            replayed=len(replayed_ids),
            upscaled=upscaled_count
        )
        
        return upscaled
    
    def _calculate_prune_threshold(self, weights: Dict[str, float]) -> float:
        """
        Calculate dynamic pruning threshold
        
        Uses percentile-based threshold for robustness.
        
        Args:
            weights: Current weights
        
        Returns:
            Pruning threshold value
        """
        if not weights:
            return 0.0
        
        weight_values = list(weights.values())
        percentile = self.config.prune_percentile
        
        # Calculate threshold at specified percentile
        threshold = np.percentile(weight_values, percentile)
        
        # Ensure minimum threshold
        min_threshold = 0.01
        threshold = max(threshold, min_threshold)
        
        logger.debug(
            f"Prune threshold at {percentile}th percentile",
            threshold=threshold,
            min_weight=min(weight_values),
            max_weight=max(weight_values)
        )
        
        return threshold
    
    def _prune_weak_connections(self, weights: Dict[str, float], 
                               threshold: float) -> tuple[Dict[str, float], List[str]]:
        """
        Prune connections below threshold
        
        Implements "use it or lose it" principle.
        
        Args:
            weights: Current weights
            threshold: Pruning threshold
        
        Returns:
            Tuple of (pruned weights dict, list of pruned IDs)
        """
        pruned_weights = {}
        pruned_ids = []
        
        for memory_id, weight in weights.items():
            if weight >= threshold:
                pruned_weights[memory_id] = weight
            else:
                pruned_ids.append(memory_id)
        
        # Don't prune recently replayed memories (protection)
        protected_count = 0
        for memory_id in pruned_ids[:]:
            if memory_id in self.replayed_weights:
                # Restore protected memory
                pruned_weights[memory_id] = max(threshold, weights[memory_id])
                pruned_ids.remove(memory_id)
                protected_count += 1
        
        logger.debug(
            "Weak connections pruned",
            threshold=threshold,
            pruned=len(pruned_ids),
            protected=protected_count,
            remaining=len(pruned_weights)
        )
        
        return pruned_weights, pruned_ids
    
    async def _ensure_stability(self, weights: Dict[str, float]) -> Dict[str, float]:
        """
        Ensure network stability through additional normalization
        
        Checks for instability indicators and applies corrections.
        
        Args:
            weights: Current weights
        
        Returns:
            Stabilized weights
        """
        if not weights:
            return weights
        
        weight_values = list(weights.values())
        mean_weight = np.mean(weight_values)
        std_weight = np.std(weight_values)
        max_weight = max(weight_values)
        
        # Check stability criteria
        stable = True
        
        # Criterion 1: Maximum weight shouldn't be too extreme
        if max_weight > mean_weight + 5 * std_weight:
            stable = False
            logger.warning(
                "Instability detected: extreme weight",
                max_weight=max_weight,
                mean=mean_weight,
                std=std_weight
            )
        
        # Criterion 2: Coefficient of variation shouldn't be too high
        cv = std_weight / mean_weight if mean_weight > 0 else 0
        if cv > 2.0:
            stable = False
            logger.warning(
                "Instability detected: high variance",
                cv=cv
            )
        
        # Apply stabilization if needed
        if not stable:
            stabilized = self._apply_stabilization(weights, mean_weight, std_weight)
            self.is_stable = False
            return stabilized
        
        self.is_stable = True
        return weights
    
    def _apply_stabilization(self, weights: Dict[str, float], 
                            mean: float, std: float) -> Dict[str, float]:
        """
        Apply stabilization to weights
        
        Clips extreme values and renormalizes.
        
        Args:
            weights: Unstable weights
            mean: Mean weight
            std: Standard deviation
        
        Returns:
            Stabilized weights
        """
        # Clip extreme values
        upper_bound = mean + 3 * std
        lower_bound = max(0.001, mean - 3 * std)
        
        stabilized = {}
        clipped_count = 0
        
        for memory_id, weight in weights.items():
            if weight > upper_bound:
                stabilized[memory_id] = upper_bound
                clipped_count += 1
            elif weight < lower_bound:
                stabilized[memory_id] = lower_bound
                clipped_count += 1
            else:
                stabilized[memory_id] = weight
        
        # Renormalize to maintain total weight
        original_sum = sum(weights.values())
        stabilized_sum = sum(stabilized.values())
        
        if stabilized_sum > 0:
            scale_factor = original_sum / stabilized_sum
            for memory_id in stabilized:
                stabilized[memory_id] *= scale_factor
        
        logger.info(
            "Stabilization applied",
            clipped=clipped_count,
            upper_bound=upper_bound,
            lower_bound=lower_bound
        )
        
        return stabilized
    
    async def _fetch_all_weights(self) -> Dict[str, float]:
        """
        Fetch all memory weights from ShapeMemoryV2
        
        Returns:
            Dictionary of memory_id -> weight
        """
        try:
            if 'shape_memory' in self.services:
                weights = await self.services['shape_memory'].get_all_weights()
                self.current_weights = weights
                return weights
            else:
                # Fallback to mock weights for testing
                return self._generate_mock_weights()
                
        except Exception as e:
            logger.error(f"Error fetching weights: {e}")
            return self.current_weights  # Use cached weights
    
    async def _commit_weights(self, weights: Dict[str, float]):
        """
        Commit normalized weights back to memory stores
        
        Args:
            weights: Normalized weights to commit
        """
        try:
            if 'shape_memory' in self.services:
                await self.services['shape_memory'].batch_update_weights(weights)
                
                # Also update tier manager if available
                if 'tier_manager' in self.services:
                    await self.services['tier_manager'].update_importance_scores(weights)
            
            # Cache the weights
            self.current_weights = weights
            
            logger.debug(f"Committed {len(weights)} normalized weights")
            
        except Exception as e:
            logger.error(f"Error committing weights: {e}")
            raise
    
    async def _take_weight_snapshot(self) -> WeightSnapshot:
        """
        Take a snapshot of current weight distribution
        
        Returns:
            WeightSnapshot object
        """
        weights = self.current_weights or await self._fetch_all_weights()
        
        if not weights:
            # Return empty snapshot
            return WeightSnapshot(
                timestamp=datetime.now(),
                weights={},
                total_weight=0.0,
                mean_weight=0.0,
                std_weight=0.0,
                percentiles={}
            )
        
        weight_values = list(weights.values())
        
        snapshot = WeightSnapshot(
            timestamp=datetime.now(),
            weights=weights.copy(),
            total_weight=sum(weight_values),
            mean_weight=np.mean(weight_values),
            std_weight=np.std(weight_values),
            percentiles={
                5: np.percentile(weight_values, 5),
                25: np.percentile(weight_values, 25),
                50: np.percentile(weight_values, 50),
                75: np.percentile(weight_values, 75),
                95: np.percentile(weight_values, 95)
            }
        )
        
        # Store in history
        self.weight_history.append(snapshot)
        
        return snapshot
    
    def _update_metrics(self, before: WeightSnapshot, after: WeightSnapshot, 
                       pruned_count: int):
        """
        Update homeostasis metrics
        
        Args:
            before: Snapshot before normalization
            after: Snapshot after normalization
            pruned_count: Number of pruned connections
        """
        self.metrics.total_normalizations += 1
        self.metrics.total_weights_processed += len(before.weights)
        self.metrics.total_pruned += pruned_count
        
        self.metrics.avg_weight_before = before.mean_weight
        self.metrics.avg_weight_after = after.mean_weight
        
        if after.weights:
            weight_values = list(after.weights.values())
            self.metrics.max_weight = max(weight_values)
            self.metrics.min_weight = min(weight_values)
        
        # Calculate stability score
        if before.mean_weight > 0:
            change_ratio = abs(after.mean_weight - before.mean_weight) / before.mean_weight
            self.metrics.stability_score = max(0.0, 1.0 - change_ratio)
        
        self.metrics.last_normalization = datetime.now()
    
    async def _continuous_monitoring(self):
        """
        Background task for continuous homeostasis monitoring
        
        Reacts to rapid changes within the configured reaction window.
        """
        while self._is_running:
            try:
                # Check every 10 seconds
                await asyncio.sleep(10)
                
                # Take weight snapshot
                snapshot = await self._take_weight_snapshot()
                
                # Add to stability window
                self.stability_window.append(snapshot)
                
                # Check if intervention needed
                if self._needs_intervention():
                    logger.warning("Rapid weight change detected, triggering homeostasis")
                    
                    # Get recently active memories (approximation)
                    recent_ids = list(self.current_weights.keys())[:100]
                    
                    # Trigger renormalization
                    await self.renormalize(recent_ids)
                    
                    # Update reaction time
                    self.last_reaction_time = datetime.now()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Monitoring error: {e}")
                await asyncio.sleep(10)
    
    def _needs_intervention(self) -> bool:
        """
        Check if homeostatic intervention is needed
        
        Returns:
            True if intervention needed
        """
        if len(self.stability_window) < 2:
            return False
        
        # Check time since last intervention
        time_since_last = datetime.now() - self.last_reaction_time
        if time_since_last < self.config.homeostasis_reaction_time:
            return False  # Too soon
        
        # Compare recent snapshots
        recent = list(self.stability_window)[-2:]
        before, after = recent[0], recent[1]
        
        # Check for rapid weight increase
        if after.mean_weight > before.mean_weight * 1.2:  # 20% increase
            return True
        
        # Check for variance explosion
        if after.std_weight > before.std_weight * 1.5:  # 50% increase in variance
            return True
        
        # Check for extreme weights
        if after.percentiles.get(95, 0) > before.percentiles.get(95, 0) * 1.3:
            return True
        
        return False
    
    def _generate_mock_weights(self) -> Dict[str, float]:
        """Generate mock weights for testing"""
        num_weights = 1000
        weights = {}
        
        for i in range(num_weights):
            memory_id = f"mem_{i:04d}"
            # Generate weights with log-normal distribution (realistic)
            weight = np.random.lognormal(mean=0.0, sigma=0.5)
            weights[memory_id] = min(weight, 10.0)  # Cap at 10
        
        return weights
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get homeostasis statistics"""
        avg_reaction_time = np.mean(self.metrics.reaction_times) if self.metrics.reaction_times else 0
        
        return {
            "total_normalizations": self.metrics.total_normalizations,
            "total_weights_processed": self.metrics.total_weights_processed,
            "total_pruned": self.metrics.total_pruned,
            "avg_weight_before": self.metrics.avg_weight_before,
            "avg_weight_after": self.metrics.avg_weight_after,
            "weight_range": (self.metrics.min_weight, self.metrics.max_weight),
            "stability_score": self.metrics.stability_score,
            "is_stable": self.is_stable,
            "avg_reaction_time_ms": avg_reaction_time,
            "last_normalization": self.metrics.last_normalization.isoformat() if self.metrics.last_normalization else None
        }
    
    async def emergency_stabilization(self):
        """
        Emergency stabilization for critical instability
        
        Applies aggressive normalization to restore stability.
        """
        logger.warning("Emergency stabilization triggered")
        
        # Fetch current weights
        weights = await self._fetch_all_weights()
        
        if not weights:
            return
        
        # Calculate target mean (conservative)
        target_mean = 1.0
        current_mean = np.mean(list(weights.values()))
        
        # Scale all weights to target
        scale_factor = target_mean / current_mean if current_mean > 0 else 1.0
        
        stabilized = {}
        for memory_id, weight in weights.items():
            new_weight = weight * scale_factor
            # Clip to reasonable range
            new_weight = np.clip(new_weight, 0.1, 5.0)
            stabilized[memory_id] = new_weight
        
        # Commit stabilized weights
        await self._commit_weights(stabilized)
        
        logger.info(
            "Emergency stabilization complete",
            scale_factor=scale_factor,
            weights_processed=len(stabilized)
        )