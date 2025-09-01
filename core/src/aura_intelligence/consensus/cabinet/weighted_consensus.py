"""
🏛️ Cabinet Weighted Consensus - Dynamic Performance-Based Voting
================================================================

Revolutionary consensus approach where nodes receive different voting
weights based on their responsiveness and reliability. The fastest
t+1 nodes form a "cabinet" with highest decision power.

Based on latest 2025 research on heterogeneous consensus systems.
"""

import asyncio
from typing import Dict, List, Set, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import time
import structlog
import numpy as np
from collections import defaultdict, deque
from enum import Enum

logger = structlog.get_logger(__name__)


class WeightingScheme(Enum):
    """Weight assignment strategies"""
    RESPONSIVENESS = "responsiveness"      # Based on response time
    RELIABILITY = "reliability"            # Based on uptime/success
    HYBRID = "hybrid"                      # Combination of factors
    CAPABILITY = "capability"              # Based on node resources
    LOCALITY = "locality"                  # Geographic/network proximity


@dataclass
class NodeMetrics:
    """Performance metrics for a node"""
    node_id: str
    response_times: deque = field(default_factory=lambda: deque(maxlen=100))
    success_count: int = 0
    failure_count: int = 0
    last_seen: float = field(default_factory=time.time)
    latency_p50: float = 0.0
    latency_p95: float = 0.0
    reliability_score: float = 1.0
    capability_score: float = 1.0
    
    def update_latency_stats(self):
        """Update latency percentiles"""
        if self.response_times:
            times = sorted(self.response_times)
            self.latency_p50 = np.percentile(times, 50)
            self.latency_p95 = np.percentile(times, 95)
            
    def update_reliability(self):
        """Update reliability score"""
        total = self.success_count + self.failure_count
        if total > 0:
            self.reliability_score = self.success_count / total


@dataclass
class CabinetMember:
    """Member of the decision cabinet"""
    node_id: str
    weight: float
    metrics: NodeMetrics
    joined_at: float = field(default_factory=time.time)


@dataclass
class WeightedVote:
    """Vote with dynamic weight"""
    node_id: str
    value: Any
    weight: float
    timestamp: float = field(default_factory=time.time)
    signature: Optional[str] = None


class ResponsivenessTracker:
    """Track node responsiveness over time"""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.node_metrics: Dict[str, NodeMetrics] = {}
        self.global_stats = {
            "median_latency": 0.0,
            "p95_latency": 0.0,
            "active_nodes": 0
        }
        
    def record_response(self, node_id: str, response_time: float, success: bool = True):
        """Record a response from a node"""
        if node_id not in self.node_metrics:
            self.node_metrics[node_id] = NodeMetrics(node_id=node_id)
            
        metrics = self.node_metrics[node_id]
        metrics.response_times.append(response_time)
        metrics.last_seen = time.time()
        
        if success:
            metrics.success_count += 1
        else:
            metrics.failure_count += 1
            
        # Update statistics
        metrics.update_latency_stats()
        metrics.update_reliability()
        
    def get_node_score(self, node_id: str, scheme: WeightingScheme) -> float:
        """Get performance score for a node"""
        if node_id not in self.node_metrics:
            return 0.0
            
        metrics = self.node_metrics[node_id]
        
        if scheme == WeightingScheme.RESPONSIVENESS:
            # Lower latency = higher score
            if metrics.latency_p50 > 0:
                return 1.0 / (1.0 + metrics.latency_p50)
            return 0.5
            
        elif scheme == WeightingScheme.RELIABILITY:
            return metrics.reliability_score
            
        elif scheme == WeightingScheme.HYBRID:
            # Combine responsiveness and reliability
            resp_score = self.get_node_score(node_id, WeightingScheme.RESPONSIVENESS)
            rel_score = self.get_node_score(node_id, WeightingScheme.RELIABILITY)
            return 0.6 * rel_score + 0.4 * resp_score
            
        elif scheme == WeightingScheme.CAPABILITY:
            return metrics.capability_score
            
        return 1.0  # Default equal weight
        
    def get_top_nodes(self, n: int, scheme: WeightingScheme) -> List[Tuple[str, float]]:
        """Get top N nodes by performance"""
        scores = []
        
        for node_id in self.node_metrics:
            score = self.get_node_score(node_id, scheme)
            scores.append((node_id, score))
            
        # Sort by score descending
        scores.sort(key=lambda x: x[1], reverse=True)
        
        return scores[:n]
        
    def update_global_stats(self):
        """Update global statistics"""
        all_latencies = []
        active_count = 0
        
        current_time = time.time()
        for metrics in self.node_metrics.values():
            # Consider active if seen in last 30 seconds
            if current_time - metrics.last_seen < 30:
                active_count += 1
                all_latencies.extend(metrics.response_times)
                
        if all_latencies:
            self.global_stats["median_latency"] = np.median(all_latencies)
            self.global_stats["p95_latency"] = np.percentile(all_latencies, 95)
            
        self.global_stats["active_nodes"] = active_count


class CabinetWeightedConsensus:
    """
    Cabinet-style dynamically weighted consensus.
    The fastest/most reliable nodes get higher voting power.
    """
    
    def __init__(self, 
                 failure_threshold_t: int,
                 weighting_scheme: WeightingScheme = WeightingScheme.HYBRID,
                 cabinet_size: Optional[int] = None):
        
        self.t = failure_threshold_t
        self.weighting_scheme = weighting_scheme
        self.cabinet_size = cabinet_size or (failure_threshold_t + 1)
        
        # Performance tracking
        self.tracker = ResponsivenessTracker()
        
        # Cabinet management
        self.current_cabinet: List[CabinetMember] = []
        self.cabinet_epoch = 0
        self.cabinet_duration = 60.0  # Recompute cabinet every minute
        self.last_cabinet_update = 0
        
        # Voting state
        self.active_votes: Dict[str, List[WeightedVote]] = {}
        
        # Weight configuration
        self.min_weight = 0.1  # Minimum weight for any node
        self.cabinet_weight_multiplier = 3.0  # Cabinet members get 3x weight
        
        # Metrics
        self.total_decisions = 0
        self.cabinet_decisions = 0
        self.average_decision_time = 0.0
        
        logger.info(f"Cabinet consensus initialized with t={failure_threshold_t}")
        
    async def record_node_performance(self, 
                                    node_id: str, 
                                    response_time: float, 
                                    success: bool = True):
        """Record performance metrics for a node"""
        self.tracker.record_response(node_id, response_time, success)
        
        # Check if cabinet needs update
        if time.time() - self.last_cabinet_update > self.cabinet_duration:
            await self._update_cabinet()
            
    async def _update_cabinet(self):
        """Update cabinet membership based on performance"""
        logger.info(f"Updating cabinet for epoch {self.cabinet_epoch + 1}")
        
        # Get top performing nodes
        top_nodes = self.tracker.get_top_nodes(
            self.cabinet_size, 
            self.weighting_scheme
        )
        
        # Create new cabinet
        new_cabinet = []
        for node_id, score in top_nodes:
            metrics = self.tracker.node_metrics[node_id]
            member = CabinetMember(
                node_id=node_id,
                weight=score * self.cabinet_weight_multiplier,
                metrics=metrics
            )
            new_cabinet.append(member)
            
        self.current_cabinet = new_cabinet
        self.cabinet_epoch += 1
        self.last_cabinet_update = time.time()
        
        # Update global stats
        self.tracker.update_global_stats()
        
        logger.info(f"Cabinet updated with {len(new_cabinet)} members",
                   members=[m.node_id for m in new_cabinet])
        
    def get_node_weight(self, node_id: str) -> float:
        """Get current weight for a node"""
        # Check if node is in cabinet
        for member in self.current_cabinet:
            if member.node_id == node_id:
                return member.weight
                
        # Non-cabinet member gets base weight
        base_score = self.tracker.get_node_score(node_id, self.weighting_scheme)
        return max(self.min_weight, base_score)
        
    async def weighted_consensus(self, 
                               proposal_id: str,
                               proposal: Any,
                               voters: List[str]) -> Tuple[bool, Any, Dict[str, Any]]:
        """
        Execute weighted consensus on a proposal.
        
        Returns: (consensus_reached, decision, stats)
        """
        start_time = time.time()
        
        # Initialize vote tracking
        self.active_votes[proposal_id] = []
        
        # Collect votes (in production, this would be async network calls)
        votes = await self._collect_votes(proposal_id, proposal, voters)
        
        # Calculate weighted decision
        decision, stats = self._calculate_weighted_decision(votes)
        
        # Update metrics
        decision_time = time.time() - start_time
        self.total_decisions += 1
        self.average_decision_time = (
            (self.average_decision_time * (self.total_decisions - 1) + decision_time) /
            self.total_decisions
        )
        
        # Check if cabinet made the decision
        cabinet_voters = {m.node_id for m in self.current_cabinet}
        cabinet_votes = [v for v in votes if v.node_id in cabinet_voters]
        if len(cabinet_votes) >= self.t + 1:
            self.cabinet_decisions += 1
            
        # Clean up
        del self.active_votes[proposal_id]
        
        return decision is not None, decision, stats
        
    async def _collect_votes(self, 
                           proposal_id: str,
                           proposal: Any,
                           voters: List[str]) -> List[WeightedVote]:
        """Collect votes from nodes"""
        votes = []
        
        # Simulate vote collection with varying response times
        for voter in voters:
            # Simulate network latency
            latency = np.random.exponential(0.1)  # 100ms average
            await asyncio.sleep(latency)
            
            # Record response time
            await self.record_node_performance(voter, latency, True)
            
            # Create vote (in production, nodes would compute their decision)
            vote = WeightedVote(
                node_id=voter,
                value=proposal,  # Simplified - all agree
                weight=self.get_node_weight(voter),
                signature=f"sig_{voter}_{proposal_id}"
            )
            
            votes.append(vote)
            self.active_votes[proposal_id].append(vote)
            
            # Check if we have enough weight to decide early
            total_weight = sum(v.weight for v in votes)
            if total_weight >= self._get_decision_threshold():
                logger.debug(f"Early decision possible with {len(votes)} votes")
                break
                
        return votes
        
    def _calculate_weighted_decision(self, 
                                   votes: List[WeightedVote]) -> Tuple[Optional[Any], Dict[str, Any]]:
        """Calculate decision based on weighted votes"""
        # Group votes by value
        value_weights = defaultdict(float)
        value_voters = defaultdict(list)
        
        total_weight = 0.0
        for vote in votes:
            value_weights[str(vote.value)] += vote.weight
            value_voters[str(vote.value)].append(vote.node_id)
            total_weight += vote.weight
            
        # Find value with highest weight
        if value_weights:
            winning_value = max(value_weights, key=value_weights.get)
            winning_weight = value_weights[winning_value]
            
            # Check if threshold met
            threshold = self._get_decision_threshold()
            if winning_weight >= threshold:
                # Parse back from string (simplified)
                decision = votes[0].value  # Use first vote's actual value
                
                stats = {
                    "total_votes": len(votes),
                    "total_weight": total_weight,
                    "winning_weight": winning_weight,
                    "threshold": threshold,
                    "cabinet_voters": len([v for v in votes if v.node_id in 
                                         {m.node_id for m in self.current_cabinet}]),
                    "consensus_strength": winning_weight / total_weight
                }
                
                return decision, stats
                
        return None, {"total_votes": len(votes), "total_weight": total_weight}
        
    def _get_decision_threshold(self) -> float:
        """Get weighted threshold for decision"""
        # Need equivalent of 2t+1 weight
        # With cabinet having 3x weight, threshold adjusts dynamically
        
        if self.current_cabinet:
            # Cabinet can decide with t+1 members
            cabinet_weight = sum(m.weight for m in self.current_cabinet[:self.t + 1])
            return cabinet_weight * 0.9  # 90% of cabinet weight needed
        else:
            # Fallback to simple majority
            return (2 * self.t + 1) * 1.0  # Assuming weight 1.0 per node
            
    def get_consensus_info(self) -> Dict[str, Any]:
        """Get consensus statistics"""
        cabinet_ratio = self.cabinet_decisions / max(1, self.total_decisions)
        
        return {
            "weighting_scheme": self.weighting_scheme.value,
            "cabinet_size": self.cabinet_size,
            "cabinet_epoch": self.cabinet_epoch,
            "current_cabinet": [m.node_id for m in self.current_cabinet],
            "total_decisions": self.total_decisions,
            "cabinet_decision_ratio": f"{cabinet_ratio:.2%}",
            "average_decision_time": f"{self.average_decision_time:.3f}s",
            "global_stats": self.tracker.global_stats,
            "node_count": len(self.tracker.node_metrics)
        }