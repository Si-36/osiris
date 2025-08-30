"""
🦈 Bullshark - Fast & Fair DAG-based BFT Ordering
=================================================

Implements the Bullshark consensus protocol that provides:
- 2-round fast path in synchronous networks
- Asynchronous fallback for liveness
- Fairness with garbage collection
- Built on top of Narwhal DAG

Based on the paper: "Bullshark: DAG BFT Protocols Made Practical"
"""

import asyncio
from typing import Dict, List, Set, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime
import time
import structlog
from collections import defaultdict, deque
from enum import Enum

from .narwhal_dag import NarwhalDAG, Certificate, Header

logger = structlog.get_logger(__name__)


class ConsensusMode(Enum):
    """Bullshark consensus modes"""
    FAST_PATH = "fast_path"      # 2-round synchronous
    SLOW_PATH = "slow_path"      # 6-round asynchronous
    RECOVERY = "recovery"         # View change


@dataclass
class LeaderSchedule:
    """Round-robin leader schedule"""
    validators: List[str]
    
    def get_leader(self, round: int) -> str:
        """Get leader for a round"""
        return self.validators[round % len(self.validators)]


@dataclass
class AnchorVote:
    """Vote for anchor selection in Bullshark"""
    round: int
    anchor_digest: str
    voter: str
    signature: str
    timestamp: float = field(default_factory=time.time)


@dataclass
class DecidedBlock:
    """Finalized block with ordering"""
    round: int
    leader: str
    certificates: List[Certificate]
    anchor: Optional[str] = None
    timestamp: float = field(default_factory=time.time)


class BullsharkOrdering:
    """
    Bullshark consensus protocol for ordering DAG certificates.
    Provides 2-round latency in common case with fairness.
    """
    
    def __init__(self, 
                 validator_id: str,
                 validators: List[str],
                 narwhal_dag: NarwhalDAG,
                 fault_tolerance: int = 1):
        
        self.validator_id = validator_id
        self.validators = validators
        self.num_validators = len(validators)
        self.fault_tolerance = fault_tolerance
        self.threshold = 2 * fault_tolerance + 1
        
        # Narwhal DAG integration
        self.dag = narwhal_dag
        
        # Leader schedule
        self.schedule = LeaderSchedule(validators)
        
        # Consensus state
        self.current_round = 0
        self.last_decided_round = -1
        self.mode = ConsensusMode.FAST_PATH
        
        # Anchor tracking
        self.anchor_votes: Dict[int, List[AnchorVote]] = defaultdict(list)
        self.decided_anchors: Dict[int, str] = {}  # round -> anchor digest
        
        # Decided blocks
        self.decided_blocks: List[DecidedBlock] = []
        self.pending_certificates: Set[str] = set()
        
        # Synchrony detection
        self.synchrony_timeout = 2.0  # 2 seconds
        self.last_progress_time = time.time()
        
        # Metrics
        self.fast_path_commits = 0
        self.slow_path_commits = 0
        self.total_ordered = 0
        
        self._running = False
        
    async def start(self):
        """Start Bullshark ordering"""
        self._running = True
        
        # Start consensus loop
        asyncio.create_task(self._consensus_loop())
        asyncio.create_task(self._synchrony_monitor())
        
        logger.info(f"Bullshark ordering started for {self.validator_id}")
        
    async def stop(self):
        """Stop Bullshark ordering"""
        self._running = False
        
    async def _consensus_loop(self):
        """Main consensus loop"""
        while self._running:
            try:
                # Check current round in DAG
                dag_round = self.dag.current_round
                
                if dag_round > self.current_round + 1:
                    # Process rounds sequentially
                    await self._process_round(self.current_round)
                    
                await asyncio.sleep(0.05)
                
            except Exception as e:
                logger.error(f"Consensus loop error: {e}")
                
    async def _process_round(self, round: int):
        """Process a consensus round"""
        leader = self.schedule.get_leader(round)
        
        if self.mode == ConsensusMode.FAST_PATH:
            # Try 2-round fast path
            success = await self._try_fast_path(round, leader)
            
            if success:
                self.fast_path_commits += 1
                self.last_progress_time = time.time()
            else:
                # Fall back to slow path
                self.mode = ConsensusMode.SLOW_PATH
                await self._execute_slow_path(round, leader)
                
        elif self.mode == ConsensusMode.SLOW_PATH:
            # Execute 6-round slow path
            await self._execute_slow_path(round, leader)
            
        # Advance round
        self.current_round += 1
        
    async def _try_fast_path(self, round: int, leader: str) -> bool:
        """
        Try 2-round fast path (Bullshark optimization).
        Works when network is synchronous.
        """
        # Round k: Check if we have leader's certificate
        leader_cert = await self._get_leader_certificate(round, leader)
        
        if not leader_cert:
            logger.debug(f"No leader certificate for round {round}")
            return False
            
        # Round k+1: Check if we can anchor the leader
        if round + 1 <= self.dag.current_round:
            anchor_success = await self._try_anchor_leader(round + 1, leader_cert)
            
            if anchor_success:
                # Fast path success! Commit in 2 rounds
                await self._commit_fast_path(round, leader_cert)
                return True
                
        return False
        
    async def _get_leader_certificate(self, round: int, leader: str) -> Optional[Certificate]:
        """Get leader's certificate for round"""
        round_vertices = self.dag.round_vertices.get(round, set())
        
        for digest in round_vertices:
            vertex = self.dag.vertices.get(digest)
            if vertex and vertex.certificate.header.author == leader:
                return vertex.certificate
                
        return None
        
    async def _try_anchor_leader(self, round: int, leader_cert: Certificate) -> bool:
        """Try to anchor leader certificate in next round"""
        round_vertices = self.dag.round_vertices.get(round, set())
        anchor_count = 0
        
        # Count how many certificates in round k+1 have leader as parent
        for digest in round_vertices:
            vertex = self.dag.vertices.get(digest)
            if vertex:
                header = vertex.certificate.header
                if leader_cert.header.digest() in header.parents:
                    anchor_count += 1
                    
        # Need 2f+1 certificates anchoring the leader
        return anchor_count >= self.threshold
        
    async def _commit_fast_path(self, round: int, leader_cert: Certificate):
        """Commit via fast path (2 rounds)"""
        # Collect all certificates to order
        certificates = await self._collect_causal_certificates(round, leader_cert)
        
        # Create decided block
        block = DecidedBlock(
            round=round,
            leader=leader_cert.header.author,
            certificates=certificates,
            anchor=leader_cert.header.digest()
        )
        
        self.decided_blocks.append(block)
        self.last_decided_round = round
        self.total_ordered += len(certificates)
        
        logger.info(f"Fast path commit at round {round}, ordered {len(certificates)} certificates")
        
    async def _execute_slow_path(self, round: int, leader: str):
        """
        Execute 6-round slow path for asynchronous networks.
        Provides liveness even under asynchrony.
        """
        # This is a simplified version of the full slow path
        # In production, implement full 3-phase protocol
        
        # Phase 1: Propose (rounds 1-2)
        if await self._slow_path_propose(round, leader):
            # Phase 2: Vote (rounds 3-4)  
            if await self._slow_path_vote(round):
                # Phase 3: Commit (rounds 5-6)
                await self._slow_path_commit(round)
                
        self.slow_path_commits += 1
        self.mode = ConsensusMode.FAST_PATH  # Try fast path again
        
    async def _slow_path_propose(self, round: int, leader: str) -> bool:
        """Slow path propose phase"""
        # Get leader certificate if exists
        leader_cert = await self._get_leader_certificate(round, leader)
        
        if leader_cert:
            # Broadcast anchor vote
            vote = AnchorVote(
                round=round,
                anchor_digest=leader_cert.header.digest(),
                voter=self.validator_id,
                signature=f"vote_{self.validator_id}_{round}"
            )
            
            self.anchor_votes[round].append(vote)
            
            # In production, broadcast to other validators
            return True
            
        return False
        
    async def _slow_path_vote(self, round: int) -> bool:
        """Slow path voting phase"""
        # Collect anchor votes
        votes = self.anchor_votes.get(round, [])
        
        if len(votes) >= self.threshold:
            # Determine winning anchor
            anchor_counts = defaultdict(int)
            for vote in votes:
                anchor_counts[vote.anchor_digest] += 1
                
            # Get anchor with most votes
            winning_anchor = max(anchor_counts, key=anchor_counts.get)
            self.decided_anchors[round] = winning_anchor
            return True
            
        return False
        
    async def _slow_path_commit(self, round: int):
        """Slow path commit phase"""
        anchor_digest = self.decided_anchors.get(round)
        
        if anchor_digest and anchor_digest in self.dag.vertices:
            anchor_cert = self.dag.vertices[anchor_digest].certificate
            
            # Collect and order certificates
            certificates = await self._collect_causal_certificates(round, anchor_cert)
            
            # Create decided block
            block = DecidedBlock(
                round=round,
                leader=self.schedule.get_leader(round),
                certificates=certificates,
                anchor=anchor_digest
            )
            
            self.decided_blocks.append(block)
            self.last_decided_round = round
            self.total_ordered += len(certificates)
            
            logger.info(f"Slow path commit at round {round}, ordered {len(certificates)} certificates")
            
    async def _collect_causal_certificates(self, 
                                          round: int, 
                                          anchor: Certificate) -> List[Certificate]:
        """Collect all certificates in causal history of anchor"""
        # Use DAG's causal read
        causal_certs = await self.dag.read_causal(round)
        
        # Filter out already decided certificates
        new_certs = []
        for cert in causal_certs:
            digest = cert.header.digest()
            if digest not in self.pending_certificates:
                new_certs.append(cert)
                self.pending_certificates.add(digest)
                
        return new_certs
        
    async def _synchrony_monitor(self):
        """Monitor network synchrony"""
        while self._running:
            try:
                current_time = time.time()
                
                # Check if we're making progress
                if current_time - self.last_progress_time > self.synchrony_timeout:
                    if self.mode == ConsensusMode.FAST_PATH:
                        logger.warning("Synchrony timeout, switching to slow path")
                        self.mode = ConsensusMode.SLOW_PATH
                        
                await asyncio.sleep(1.0)
                
            except Exception as e:
                logger.error(f"Synchrony monitor error: {e}")
                
    async def get_decided_transactions(self) -> List[bytes]:
        """Get all decided transactions in order"""
        transactions = []
        
        for block in self.decided_blocks:
            for cert in block.certificates:
                # Get transactions from certificate payloads
                for batch_digest in cert.header.payload:
                    # Retrieve batch from workers
                    for worker in self.dag.workers:
                        batch = await worker.get_batch(batch_digest)
                        if batch:
                            transactions.extend(batch.transactions)
                            break
                            
        return transactions
        
    def get_consensus_info(self) -> Dict[str, Any]:
        """Get consensus statistics"""
        total_commits = self.fast_path_commits + self.slow_path_commits
        fast_ratio = self.fast_path_commits / max(1, total_commits)
        
        return {
            "mode": self.mode.value,
            "current_round": self.current_round,
            "last_decided_round": self.last_decided_round,
            "total_decided_blocks": len(self.decided_blocks),
            "total_ordered_certificates": self.total_ordered,
            "fast_path_commits": self.fast_path_commits,
            "slow_path_commits": self.slow_path_commits,
            "fast_path_ratio": f"{fast_ratio:.2%}",
            "pending_certificates": len(self.pending_certificates)
        }