"""
Causal State Persistence Manager - The Future of AI Memory
========================================================
Replaces pickle with PostgreSQL + DuckDB while adding:
- Causal tracking (WHY things happened)
- Speculative branches (multiple futures)
- GPU memory tier (100GB cache)
- Compute-on-retrieval (think while remembering)
"""

import asyncio
import asyncpg
import duckdb
import numpy as np
import torch
import json
import zstandard as zstd
import hashlib
import uuid
import time
from typing import Dict, Any, Optional, List, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
import structlog
from pathlib import Path
import pickle  # For backward compatibility only

logger = structlog.get_logger()

class StateType(Enum):
    """Types of state we persist"""
    COMPONENT_STATE = "component_state"
    LIQUID_NETWORK = "liquid_network"
    MEMORY_TIER = "memory_tier"
    METABOLIC_BUDGET = "metabolic_budget"
    SYSTEM_CONFIG = "system_config"
    AGENT_MEMORY = "agent_memory"
    NEURAL_CHECKPOINT = "neural_checkpoint"
    TDA_CACHE = "tda_cache"
    SWARM_STATE = "swarm_state"

@dataclass
class CausalContext:
    """Tracks causality for every state change"""
    causes: List[str] = field(default_factory=list)  # What caused this
    effects: List[str] = field(default_factory=list)  # What this will cause
    counterfactuals: Dict[str, Any] = field(default_factory=dict)  # What could have been
    confidence: float = 1.0
    energy_cost: float = 0.0
    decision_path: List[Dict[str, Any]] = field(default_factory=list)

@dataclass 
class StateSnapshot:
    """Enhanced snapshot with causality and versioning"""
    state_id: str
    state_type: StateType
    component_id: str
    timestamp: float
    version: int
    data: Dict[str, Any]
    compressed_data: Optional[bytes] = None
    causal_context: Optional[CausalContext] = None
    checksum: str = ""
    gpu_cached: bool = False
    branch_id: Optional[str] = None  # For speculative branches

class GPUMemoryTier:
    """Use GPU memory as ultra-fast persistence tier"""
    def __init__(self, size_gb: int = 80):
        self.size_bytes = size_gb * 1024 * 1024 * 1024
        self.cache = {}
        self.gpu_tensors = {}
        self.access_counts = {}
        self.last_access = {}
        
    async def store(self, key: str, data: Any) -> bool:
        """Store in GPU memory with automatic eviction"""
        try:
            # Convert to tensor if possible
            if isinstance(data, np.ndarray):
                tensor = torch.from_numpy(data).cuda()
            elif isinstance(data, dict):
                # Store as GPU tensor for fast access
                serialized = pickle.dumps(data)
                tensor = torch.frombuffer(serialized, dtype=torch.uint8).cuda()
            else:
                return False
                
            self.gpu_tensors[key] = tensor
            self.access_counts[key] = 0
            self.last_access[key] = time.time()
            
            # Evict if needed
            await self._evict_if_needed()
            return True
        except Exception as e:
            logger.warning(f"GPU cache store failed: {e}")
            return False
    
    async def retrieve(self, key: str) -> Optional[Any]:
        """Retrieve from GPU with access tracking"""
        if key in self.gpu_tensors:
            self.access_counts[key] += 1
            self.last_access[key] = time.time()
            
            tensor = self.gpu_tensors[key]
            if tensor.dtype == torch.uint8:
                # Deserialize dict
                return pickle.loads(tensor.cpu().numpy().tobytes())
            else:
                # Return as numpy
                return tensor.cpu().numpy()
        return None
    
    async def _evict_if_needed(self):
        """LRU eviction when GPU memory is full"""
        # Simple check - in production use pynvml for accurate memory
        if len(self.gpu_tensors) > 1000:  # Simplified limit
            # Find least recently used
            lru_key = min(self.last_access, key=self.last_access.get)
            del self.gpu_tensors[lru_key]
            del self.access_counts[lru_key]
            del self.last_access[lru_key]

class CausalPersistenceManager:
    """Next-gen persistence with causality, speculation, and GPU tiers"""
    
    def __init__(self, 
                 postgres_url: str = "postgresql://localhost/aura",
                 duckdb_path: str = "aura_analytics.db",
                 legacy_path: str = "/tmp/aura_state",
                 gpu_cache_gb: int = 80):
        
        # Storage backends
        self.postgres_url = postgres_url
        self.duckdb_path = duckdb_path
        self.legacy_path = Path(legacy_path)
        self.legacy_path.mkdir(exist_ok=True)
        
        # Initialize components
        self.pg_pool: Optional[asyncpg.Pool] = None
        self.duck_conn: Optional[duckdb.DuckDBPyConnection] = None
        self.gpu_tier = GPUMemoryTier(gpu_cache_gb)
        
        # Caches and state
        self.memory_cache = {}  # L1 cache
        self.version_counter = {}  # Track versions per component
        self.active_branches = {}  # Speculative branches
        self.causal_graph = {}  # Track cause-effect relationships
        
        # Backward compatibility
        self.legacy_mode = False
        self._legacy_manager = None
        
        # Compression
        self.compressor = zstd.ZstdCompressor(level=3)
        self.decompressor = zstd.ZstdDecompressor()
        
        logger.info("Causal Persistence Manager initialized")
    
    async def initialize(self):
        """Set up database connections and tables"""
        try:
            # PostgreSQL for reliable state storage
            self.pg_pool = await asyncpg.create_pool(
                self.postgres_url,
                min_size=5,
                max_size=20,
                command_timeout=60
            )
            
            # Create tables with causality support
            async with self.pg_pool.acquire() as conn:
                await conn.execute("""
                    CREATE TABLE IF NOT EXISTS states (
                        state_id UUID PRIMARY KEY,
                        state_type TEXT NOT NULL,
                        component_id TEXT NOT NULL,
                        version INTEGER NOT NULL,
                        branch_id TEXT,
                        timestamp TIMESTAMPTZ NOT NULL,
                        data JSONB,
                        compressed_data BYTEA,
                        causes TEXT[],
                        effects TEXT[],
                        counterfactuals JSONB,
                        confidence FLOAT,
                        energy_cost FLOAT,
                        checksum TEXT,
                        gpu_cached BOOLEAN DEFAULT FALSE,
                        created_at TIMESTAMPTZ DEFAULT NOW(),
                        updated_at TIMESTAMPTZ DEFAULT NOW(),
                        
                        -- Indexes for fast queries
                        INDEX idx_component (component_id, version DESC),
                        INDEX idx_type (state_type, timestamp DESC),
                        INDEX idx_branch (branch_id) WHERE branch_id IS NOT NULL,
                        INDEX idx_causes USING GIN (causes),
                        INDEX idx_effects USING GIN (effects)
                    );
                    
                    -- Causal edges table
                    CREATE TABLE IF NOT EXISTS causal_edges (
                        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                        from_state UUID REFERENCES states(state_id),
                        to_state UUID REFERENCES states(state_id),
                        edge_type TEXT,
                        weight FLOAT DEFAULT 1.0,
                        metadata JSONB,
                        created_at TIMESTAMPTZ DEFAULT NOW(),
                        
                        INDEX idx_from (from_state),
                        INDEX idx_to (to_state)
                    );
                    
                    -- Version tracking
                    CREATE TABLE IF NOT EXISTS version_history (
                        component_id TEXT,
                        version INTEGER,
                        branch_id TEXT,
                        merged_from TEXT,
                        created_at TIMESTAMPTZ DEFAULT NOW(),
                        PRIMARY KEY (component_id, version)
                    );
                """)
            
            # DuckDB for analytics and causal queries
            self.duck_conn = duckdb.connect(self.duckdb_path)
            self.duck_conn.execute("""
                CREATE TABLE IF NOT EXISTS causal_analytics (
                    state_id VARCHAR PRIMARY KEY,
                    component_id VARCHAR,
                    state_type VARCHAR,
                    timestamp TIMESTAMP,
                    causes VARCHAR[],
                    effects VARCHAR[],
                    decision_confidence DOUBLE,
                    compute_cost DOUBLE,
                    counterfactual_count INTEGER,
                    causal_depth INTEGER,
                    information_gain DOUBLE
                );
                
                -- Indexes for causal analysis
                CREATE INDEX IF NOT EXISTS idx_causal_component 
                ON causal_analytics(component_id, timestamp);
                
                CREATE INDEX IF NOT EXISTS idx_causal_confidence
                ON causal_analytics(decision_confidence);
            """)
            
            logger.info("Persistence backend initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize persistence: {e}")
            logger.warning("Falling back to legacy pickle mode")
            self.legacy_mode = True
            await self._init_legacy_mode()
    
    async def _init_legacy_mode(self):
        """Initialize backward-compatible pickle-based storage"""
        try:
            # Import the old state manager
            from .state_manager import StatePersistenceManager
            self._legacy_manager = StatePersistenceManager(str(self.legacy_path))
            logger.warning("Running in legacy pickle mode - upgrade PostgreSQL ASAP!")
        except Exception as e:
            logger.error(f"Failed to init legacy mode: {e}")
    
    async def save_state(self, 
                        state_type: StateType, 
                        component_id: str, 
                        state_data: Dict[str, Any],
                        causal_context: Optional[CausalContext] = None,
                        branch_id: Optional[str] = None) -> str:
        """Save state with causality tracking and GPU caching"""
        
        # Generate state ID
        state_id = str(uuid.uuid4())
        
        # Get version
        version_key = f"{component_id}:{branch_id or 'main'}"
        self.version_counter[version_key] = self.version_counter.get(version_key, 0) + 1
        version = self.version_counter[version_key]
        
        # Create snapshot
        snapshot = StateSnapshot(
            state_id=state_id,
            state_type=state_type,
            component_id=component_id,
            timestamp=time.time(),
            version=version,
            data=state_data,
            causal_context=causal_context,
            checksum=self._calculate_checksum(state_data),
            branch_id=branch_id
        )
        
        # Try GPU cache first (L0)
        cache_key = f"{state_type.value}:{component_id}:{version}"
        gpu_cached = await self.gpu_tier.store(cache_key, state_data)
        snapshot.gpu_cached = gpu_cached
        
        # Memory cache (L1)
        self.memory_cache[cache_key] = snapshot
        
        # Compress for storage
        snapshot.compressed_data = self.compressor.compress(
            json.dumps(state_data).encode()
        )
        
        # Persist to database
        if not self.legacy_mode:
            await self._save_to_postgres(snapshot)
            await self._save_to_duckdb(snapshot)
            await self._update_causal_graph(snapshot)
        else:
            # Fallback to legacy
            await self._save_legacy(state_type, component_id, state_data)
        
        logger.debug(f"State saved: {component_id} v{version} (GPU: {gpu_cached})")
        return state_id
    
    async def _save_to_postgres(self, snapshot: StateSnapshot):
        """Save to PostgreSQL with full causality"""
        async with self.pg_pool.acquire() as conn:
            ctx = snapshot.causal_context
            await conn.execute("""
                INSERT INTO states (
                    state_id, state_type, component_id, version, branch_id,
                    timestamp, compressed_data, checksum, gpu_cached,
                    causes, effects, counterfactuals, confidence, energy_cost
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14)
            """, 
                uuid.UUID(snapshot.state_id),
                snapshot.state_type.value,
                snapshot.component_id,
                snapshot.version,
                snapshot.branch_id,
                datetime.fromtimestamp(snapshot.timestamp, tz=timezone.utc),
                snapshot.compressed_data,
                snapshot.checksum,
                snapshot.gpu_cached,
                ctx.causes if ctx else [],
                ctx.effects if ctx else [],
                json.dumps(ctx.counterfactuals) if ctx else None,
                ctx.confidence if ctx else 1.0,
                ctx.energy_cost if ctx else 0.0
            )
    
    async def _save_to_duckdb(self, snapshot: StateSnapshot):
        """Save analytics data to DuckDB"""
        ctx = snapshot.causal_context
        causal_depth = len(ctx.decision_path) if ctx else 0
        
        self.duck_conn.execute("""
            INSERT INTO causal_analytics VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            snapshot.state_id,
            snapshot.component_id,
            snapshot.state_type.value,
            datetime.fromtimestamp(snapshot.timestamp),
            ctx.causes if ctx else [],
            ctx.effects if ctx else [],
            ctx.confidence if ctx else 1.0,
            ctx.energy_cost if ctx else 0.0,
            len(ctx.counterfactuals) if ctx else 0,
            causal_depth,
            0.0  # Information gain - calculated later
        ))
    
    async def load_state(self,
                        state_type: StateType,
                        component_id: str,
                        version: Optional[int] = None,
                        branch_id: Optional[str] = None,
                        compute_on_retrieval: Optional[callable] = None) -> Optional[Dict[str, Any]]:
        """Load state with compute-on-retrieval capability"""
        
        # Check GPU cache first (nanoseconds)
        cache_key = f"{state_type.value}:{component_id}:{version or 'latest'}"
        gpu_data = await self.gpu_tier.retrieve(cache_key)
        if gpu_data is not None:
            # Apply compute-on-retrieval if provided
            if compute_on_retrieval:
                gpu_data = await compute_on_retrieval(gpu_data)
            return gpu_data
        
        # Check memory cache (microseconds)
        if cache_key in self.memory_cache:
            snapshot = self.memory_cache[cache_key]
            data = snapshot.data
            if compute_on_retrieval:
                data = await compute_on_retrieval(data)
            # Promote to GPU
            await self.gpu_tier.store(cache_key, data)
            return data
        
        # Load from database (milliseconds)
        if not self.legacy_mode:
            data = await self._load_from_postgres(
                state_type, component_id, version, branch_id
            )
            if data and compute_on_retrieval:
                data = await compute_on_retrieval(data)
            # Cache for next time
            if data:
                await self.gpu_tier.store(cache_key, data)
            return data
        else:
            # Legacy fallback
            return await self._load_legacy(state_type, component_id)
    
    async def _load_from_postgres(self,
                                  state_type: StateType,
                                  component_id: str,
                                  version: Optional[int],
                                  branch_id: Optional[str]) -> Optional[Dict[str, Any]]:
        """Load from PostgreSQL with version support"""
        async with self.pg_pool.acquire() as conn:
            if version is None:
                # Get latest version
                row = await conn.fetchrow("""
                    SELECT compressed_data, data 
                    FROM states 
                    WHERE state_type = $1 AND component_id = $2 
                          AND ($3::TEXT IS NULL OR branch_id = $3)
                    ORDER BY version DESC 
                    LIMIT 1
                """, state_type.value, component_id, branch_id)
            else:
                # Get specific version
                row = await conn.fetchrow("""
                    SELECT compressed_data, data 
                    FROM states 
                    WHERE state_type = $1 AND component_id = $2 
                          AND version = $3
                          AND ($4::TEXT IS NULL OR branch_id = $4)
                """, state_type.value, component_id, version, branch_id)
            
            if row:
                if row['compressed_data']:
                    # Decompress
                    decompressed = self.decompressor.decompress(row['compressed_data'])
                    return json.loads(decompressed.decode())
                elif row['data']:
                    return dict(row['data'])
        
        return None
    
    async def create_branch(self, component_id: str, branch_name: str) -> str:
        """Create a speculative branch for exploration"""
        branch_id = f"branch_{branch_name}_{uuid.uuid4().hex[:8]}"
        
        # Copy current state to branch
        current_state = await self.load_state(
            StateType.COMPONENT_STATE, component_id
        )
        
        if current_state:
            await self.save_state(
                StateType.COMPONENT_STATE,
                component_id,
                current_state,
                branch_id=branch_id
            )
        
        self.active_branches[component_id] = self.active_branches.get(component_id, [])
        self.active_branches[component_id].append(branch_id)
        
        logger.info(f"Created branch {branch_id} for {component_id}")
        return branch_id
    
    async def merge_branch(self, component_id: str, branch_id: str, strategy: str = "best"):
        """Merge a speculative branch back to main"""
        if strategy == "best":
            # Compare branch performance metrics
            branch_metrics = await self._get_branch_metrics(component_id, branch_id)
            main_metrics = await self._get_branch_metrics(component_id, None)
            
            if branch_metrics.get("information_gain", 0) > main_metrics.get("information_gain", 0):
                # Branch is better - merge it
                branch_state = await self.load_state(
                    StateType.COMPONENT_STATE,
                    component_id,
                    branch_id=branch_id
                )
                
                if branch_state:
                    await self.save_state(
                        StateType.COMPONENT_STATE,
                        component_id,
                        branch_state,
                        causal_context=CausalContext(
                            causes=[f"merged_from:{branch_id}"],
                            confidence=branch_metrics.get("confidence", 1.0)
                        )
                    )
                    logger.info(f"Merged branch {branch_id} into main")
    
    async def get_causal_chain(self, state_id: str) -> List[Dict[str, Any]]:
        """Trace the complete causal chain for a state"""
        chain = []
        
        if not self.legacy_mode:
            async with self.pg_pool.acquire() as conn:
                # Recursive CTE to follow causal links
                rows = await conn.fetch("""
                    WITH RECURSIVE causal_chain AS (
                        -- Base case
                        SELECT state_id, causes, effects, confidence, energy_cost, 0 as depth
                        FROM states
                        WHERE state_id = $1
                        
                        UNION ALL
                        
                        -- Recursive case
                        SELECT s.state_id, s.causes, s.effects, s.confidence, s.energy_cost, cc.depth + 1
                        FROM states s
                        JOIN causal_chain cc ON s.state_id = ANY(cc.causes::uuid[])
                        WHERE cc.depth < 10  -- Limit depth
                    )
                    SELECT * FROM causal_chain ORDER BY depth
                """, uuid.UUID(state_id))
                
                for row in rows:
                    chain.append({
                        "state_id": str(row["state_id"]),
                        "causes": row["causes"],
                        "effects": row["effects"],
                        "confidence": row["confidence"],
                        "energy_cost": row["energy_cost"],
                        "depth": row["depth"]
                    })
        
        return chain
    
    async def get_counterfactuals(self, state_id: str) -> Dict[str, Any]:
        """Get what could have happened but didn't"""
        if not self.legacy_mode:
            async with self.pg_pool.acquire() as conn:
                row = await conn.fetchrow("""
                    SELECT counterfactuals FROM states WHERE state_id = $1
                """, uuid.UUID(state_id))
                
                if row and row["counterfactuals"]:
                    return json.loads(row["counterfactuals"])
        
        return {}
    
    async def _update_causal_graph(self, snapshot: StateSnapshot):
        """Update the causal relationship graph"""
        if snapshot.causal_context and snapshot.causal_context.causes:
            for cause in snapshot.causal_context.causes:
                # Track causal edge
                if cause in self.causal_graph:
                    self.causal_graph[cause].append(snapshot.state_id)
                else:
                    self.causal_graph[cause] = [snapshot.state_id]
    
    async def _get_branch_metrics(self, component_id: str, branch_id: Optional[str]) -> Dict[str, float]:
        """Calculate performance metrics for a branch"""
        # Use DuckDB for fast analytics
        result = self.duck_conn.execute("""
            SELECT 
                AVG(decision_confidence) as avg_confidence,
                SUM(compute_cost) as total_cost,
                AVG(information_gain) as avg_info_gain,
                COUNT(*) as state_count
            FROM causal_analytics
            WHERE component_id = ?
                  AND (? IS NULL OR state_id IN (
                      SELECT state_id::VARCHAR FROM states WHERE branch_id = ?
                  ))
        """, (component_id, branch_id, branch_id)).fetchone()
        
        return {
            "confidence": result[0] or 0.0,
            "compute_cost": result[1] or 0.0,
            "information_gain": result[2] or 0.0,
            "state_count": result[3] or 0
        }
    
    def _calculate_checksum(self, data: Dict[str, Any]) -> str:
        """Calculate checksum for integrity"""
        data_str = json.dumps(data, sort_keys=True)
        return hashlib.sha256(data_str.encode()).hexdigest()
    
    # Backward compatibility methods
    async def _save_legacy(self, state_type: StateType, component_id: str, state_data: Dict[str, Any]):
        """Save using legacy pickle manager"""
        if self._legacy_manager:
            await self._legacy_manager.save_state(state_type, component_id, state_data)
    
    async def _load_legacy(self, state_type: StateType, component_id: str) -> Optional[Dict[str, Any]]:
        """Load using legacy pickle manager"""
        if self._legacy_manager:
            return await self._legacy_manager.load_state(state_type, component_id)
        return None
    
    async def migrate_from_legacy(self):
        """Migrate all pickle files to new persistence"""
        logger.info("Starting migration from pickle files...")
        
        if not self._legacy_manager:
            await self._init_legacy_mode()
        
        migrated = 0
        for file_path in self.legacy_path.glob("*.pkl"):
            try:
                with open(file_path, 'rb') as f:
                    snapshot = pickle.load(f)
                
                # Convert old format to new
                if hasattr(snapshot, 'state_type') and hasattr(snapshot, 'component_id'):
                    await self.save_state(
                        snapshot.state_type,
                        snapshot.component_id,
                        snapshot.data,
                        causal_context=CausalContext(
                            causes=["migrated_from_pickle"],
                            confidence=1.0
                        )
                    )
                    migrated += 1
            except Exception as e:
                logger.warning(f"Failed to migrate {file_path}: {e}")
        
        logger.info(f"Migration complete: {migrated} states migrated")
    
    async def close(self):
        """Clean shutdown"""
        if self.pg_pool:
            await self.pg_pool.close()
        if self.duck_conn:
            self.duck_conn.close()
        logger.info("Persistence manager closed")

# Global instance management
_causal_manager: Optional[CausalPersistenceManager] = None

async def get_causal_manager() -> CausalPersistenceManager:
    """Get or create the global causal persistence manager"""
    global _causal_manager
    if _causal_manager is None:
        _causal_manager = CausalPersistenceManager()
        await _causal_manager.initialize()
    return _causal_manager

# Backward compatible interface
async def get_state_manager():
    """Legacy interface for compatibility"""
    manager = await get_causal_manager()
    if manager.legacy_mode:
        return manager._legacy_manager
    return manager