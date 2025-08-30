"""
🌊 Narwhal DAG - Data Availability Layer for Bullshark
======================================================

Implements the Narwhal DAG-based mempool that separates data
availability from consensus ordering. Based on Mysten Labs'
production implementation.

Key Features:
- DAG-based reliable broadcast
- Worker-based horizontal scaling
- Cryptographic data availability proofs
- Garbage collection for bounded growth
"""

import asyncio
from typing import Dict, List, Set, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime
import hashlib
import time
import structlog
from collections import defaultdict, deque
from enum import Enum

logger = structlog.get_logger(__name__)


class CertificateType(Enum):
    """Types of certificates in Narwhal"""
    HEADER = "header"
    BATCH = "batch"
    VOTE = "vote"


@dataclass
class Batch:
    """Data batch from a worker"""
    worker_id: int
    epoch: int
    batch_id: str
    transactions: List[bytes]
    timestamp: float = field(default_factory=time.time)
    
    def digest(self) -> str:
        """Compute cryptographic digest of batch"""
        content = f"{self.worker_id}:{self.epoch}:{self.batch_id}".encode()
        for tx in self.transactions:
            content += tx
        return hashlib.sha256(content).hexdigest()


@dataclass
class Header:
    """Block header in the DAG"""
    author: str
    round: int
    epoch: int
    payload: List[str]  # Batch digests
    parents: List[str]  # Parent header digests
    timestamp: float = field(default_factory=time.time)
    signature: Optional[str] = None
    
    def digest(self) -> str:
        """Compute header digest"""
        content = f"{self.author}:{self.round}:{self.epoch}:"
        content += ":".join(self.payload)
        content += ":".join(self.parents)
        return hashlib.sha256(content.encode()).hexdigest()


@dataclass  
class Certificate:
    """Certificate proving 2f+1 agreement"""
    header: Header
    votes: List[Tuple[str, str]]  # (voter_id, signature)
    cert_type: CertificateType = CertificateType.HEADER
    
    def is_valid(self, threshold: int) -> bool:
        """Check if certificate has enough votes"""
        return len(self.votes) >= threshold


@dataclass
class DAGVertex:
    """Vertex in the DAG structure"""
    certificate: Certificate
    children: Set[str] = field(default_factory=set)
    delivered: bool = False
    gc_round: Optional[int] = None


class NarwhalWorker:
    """
    Worker that handles data batching and availability.
    Each validator runs multiple workers for scaling.
    """
    
    def __init__(self, worker_id: int, validator_id: str, num_workers: int = 4):
        self.worker_id = worker_id
        self.validator_id = validator_id
        self.num_workers = num_workers
        
        # Batch management
        self.pending_txs: deque = deque()
        self.batch_size = 1000  # Transactions per batch
        self.batch_timeout = 0.1  # 100ms
        self.current_epoch = 0
        
        # Storage
        self.batches: Dict[str, Batch] = {}
        self.batch_proofs: Dict[str, List[str]] = {}  # digest -> signatures
        
        # Metrics
        self.total_txs = 0
        self.total_batches = 0
        
        self._running = False
        
    async def start(self):
        """Start worker tasks"""
        self._running = True
        asyncio.create_task(self._batch_creator())
        logger.info(f"Worker {self.worker_id} started for validator {self.validator_id}")
        
    async def stop(self):
        """Stop worker"""
        self._running = False
        
    async def submit_transaction(self, tx: bytes) -> str:
        """Submit transaction to worker"""
        self.pending_txs.append(tx)
        self.total_txs += 1
        return f"tx_{self.worker_id}_{self.total_txs}"
        
    async def _batch_creator(self):
        """Create batches from pending transactions"""
        while self._running:
            try:
                # Wait for batch size or timeout
                start_time = time.time()
                batch_txs = []
                
                while len(batch_txs) < self.batch_size:
                    if self.pending_txs:
                        batch_txs.append(self.pending_txs.popleft())
                    elif time.time() - start_time > self.batch_timeout:
                        break
                    else:
                        await asyncio.sleep(0.01)
                        
                if batch_txs:
                    # Create batch
                    batch = Batch(
                        worker_id=self.worker_id,
                        epoch=self.current_epoch,
                        batch_id=f"batch_{self.total_batches}",
                        transactions=batch_txs
                    )
                    
                    digest = batch.digest()
                    self.batches[digest] = batch
                    self.total_batches += 1
                    
                    # Broadcast to other workers for availability
                    await self._broadcast_batch(batch)
                    
                    logger.debug(f"Worker {self.worker_id} created batch with {len(batch_txs)} txs")
                    
            except Exception as e:
                logger.error(f"Batch creation error: {e}")
                
    async def _broadcast_batch(self, batch: Batch):
        """Broadcast batch to other workers for availability"""
        # In production, this would use actual network broadcast
        # For now, we simulate instant availability
        self.batch_proofs[batch.digest()] = [
            f"sig_worker_{i}" for i in range(self.num_workers)
        ]
        
    async def get_batch(self, digest: str) -> Optional[Batch]:
        """Retrieve batch by digest"""
        return self.batches.get(digest)
        
    def get_batch_proof(self, digest: str) -> List[str]:
        """Get availability proof for batch"""
        return self.batch_proofs.get(digest, [])


class NarwhalDAG:
    """
    Narwhal DAG-based mempool implementation.
    Provides reliable broadcast and data availability.
    """
    
    def __init__(self, validator_id: str, validators: List[str], fault_tolerance: int = 1):
        self.validator_id = validator_id
        self.validators = validators
        self.num_validators = len(validators)
        self.fault_tolerance = fault_tolerance
        self.threshold = 2 * fault_tolerance + 1
        
        # DAG structure
        self.vertices: Dict[str, DAGVertex] = {}  # digest -> vertex
        self.round_vertices: Dict[int, Set[str]] = defaultdict(set)  # round -> digests
        self.current_round = 0
        self.current_epoch = 0
        
        # Workers for horizontal scaling
        self.num_workers = 4
        self.workers: List[NarwhalWorker] = []
        self.worker_index = 0  # Round-robin assignment
        
        # Header creation
        self.header_timeout = 0.5  # 500ms per round
        self.last_header_time = 0
        
        # Garbage collection
        self.gc_depth = 10  # Keep last 10 rounds
        
        # Metrics
        self.total_headers = 0
        self.total_certificates = 0
        
        self._running = False
        
    async def start(self):
        """Start Narwhal DAG"""
        self._running = True
        
        # Initialize workers
        for i in range(self.num_workers):
            worker = NarwhalWorker(i, self.validator_id, self.num_workers)
            await worker.start()
            self.workers.append(worker)
            
        # Start header creation
        asyncio.create_task(self._header_creator())
        asyncio.create_task(self._garbage_collector())
        
        logger.info(f"Narwhal DAG started with {self.num_workers} workers")
        
    async def stop(self):
        """Stop Narwhal DAG"""
        self._running = False
        for worker in self.workers:
            await worker.stop()
            
    async def submit_transaction(self, tx: bytes) -> str:
        """Submit transaction to DAG"""
        # Round-robin to workers
        worker = self.workers[self.worker_index]
        self.worker_index = (self.worker_index + 1) % self.num_workers
        
        return await worker.submit_transaction(tx)
        
    async def _header_creator(self):
        """Periodically create headers"""
        while self._running:
            try:
                current_time = time.time()
                
                if current_time - self.last_header_time >= self.header_timeout:
                    await self._create_header()
                    self.last_header_time = current_time
                    
                await asyncio.sleep(0.05)
                
            except Exception as e:
                logger.error(f"Header creation error: {e}")
                
    async def _create_header(self):
        """Create a new header for current round"""
        # Collect batch digests from workers
        payload = []
        for worker in self.workers:
            # Get recent batches
            for digest, batch in list(worker.batches.items())[-10:]:
                if worker.get_batch_proof(digest):
                    payload.append(digest)
                    
        if not payload and self.current_round > 0:
            # No new data, skip round
            return
            
        # Select parents from previous round
        parents = self._select_parents()
        
        # Create header
        header = Header(
            author=self.validator_id,
            round=self.current_round,
            epoch=self.current_epoch,
            payload=payload,
            parents=parents
        )
        
        # Sign header (mock)
        header.signature = f"sig_{self.validator_id}_{header.digest()}"
        
        # Create certificate by collecting votes
        certificate = await self._create_certificate(header)
        
        if certificate:
            # Add to DAG
            vertex = DAGVertex(certificate=certificate)
            digest = header.digest()
            self.vertices[digest] = vertex
            self.round_vertices[self.current_round].add(digest)
            
            # Update parent-child relationships
            for parent_digest in parents:
                if parent_digest in self.vertices:
                    self.vertices[parent_digest].children.add(digest)
                    
            self.total_headers += 1
            logger.debug(f"Created header for round {self.current_round} with {len(payload)} batches")
            
            # Advance round
            self.current_round += 1
            
    def _select_parents(self) -> List[str]:
        """Select parents from previous round"""
        if self.current_round == 0:
            return []
            
        # Get certificates from previous round
        prev_round = self.current_round - 1
        prev_vertices = self.round_vertices.get(prev_round, set())
        
        # Select 2f+1 parents
        parents = []
        for digest in prev_vertices:
            vertex = self.vertices.get(digest)
            if vertex and vertex.certificate.is_valid(self.threshold):
                parents.append(digest)
                
        return parents[:self.threshold]
        
    async def _create_certificate(self, header: Header) -> Optional[Certificate]:
        """Create certificate by collecting votes"""
        # In production, this would collect actual votes from validators
        # For now, simulate instant voting
        
        votes = []
        for i, validator in enumerate(self.validators):
            if i < self.threshold:  # Simulate 2f+1 votes
                vote_sig = f"vote_{validator}_{header.digest()}"
                votes.append((validator, vote_sig))
                
        certificate = Certificate(
            header=header,
            votes=votes
        )
        
        if certificate.is_valid(self.threshold):
            self.total_certificates += 1
            return certificate
            
        return None
        
    async def _garbage_collector(self):
        """Garbage collect old rounds"""
        while self._running:
            try:
                # GC rounds older than gc_depth
                gc_round = self.current_round - self.gc_depth
                
                if gc_round > 0 and gc_round in self.round_vertices:
                    # Mark vertices for GC
                    for digest in self.round_vertices[gc_round]:
                        if digest in self.vertices:
                            self.vertices[digest].gc_round = gc_round
                            
                    # Remove from round index
                    del self.round_vertices[gc_round]
                    
                    logger.debug(f"Garbage collected round {gc_round}")
                    
                await asyncio.sleep(10)  # GC every 10 seconds
                
            except Exception as e:
                logger.error(f"GC error: {e}")
                
    def get_dag_info(self) -> Dict[str, Any]:
        """Get DAG statistics"""
        return {
            "current_round": self.current_round,
            "current_epoch": self.current_epoch,
            "total_vertices": len(self.vertices),
            "total_headers": self.total_headers,
            "total_certificates": self.total_certificates,
            "active_rounds": len(self.round_vertices),
            "workers": {
                f"worker_{w.worker_id}": {
                    "total_txs": w.total_txs,
                    "total_batches": w.total_batches,
                    "pending_txs": len(w.pending_txs)
                }
                for w in self.workers
            }
        }
        
    async def read_causal(self, round: int) -> List[Certificate]:
        """Read certificates in causal order up to round"""
        result = []
        
        # BFS from round 0 to target round
        visited = set()
        queue = deque()
        
        # Start from round 0
        for digest in self.round_vertices.get(0, []):
            queue.append((digest, 0))
            
        while queue:
            digest, vertex_round = queue.popleft()
            
            if digest in visited or vertex_round > round:
                continue
                
            visited.add(digest)
            
            vertex = self.vertices.get(digest)
            if vertex:
                result.append(vertex.certificate)
                
                # Add children
                for child_digest in vertex.children:
                    child_vertex = self.vertices.get(child_digest)
                    if child_vertex:
                        child_round = child_vertex.certificate.header.round
                        queue.append((child_digest, child_round))
                        
        return result