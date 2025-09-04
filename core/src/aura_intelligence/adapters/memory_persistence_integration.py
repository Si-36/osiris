"""
Memory System Persistence Integration
====================================
Integrates UnifiedMemoryInterface with causal persistence
"""

import asyncio
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime
import structlog

from ..persistence.causal_state_manager import (
    get_causal_manager,
    StateType,
    CausalContext
)
from ..persistence.memory_native import (
    MemoryNativeAI,
    MemoryProcessor,
    GPUMemoryPool
)

logger = structlog.get_logger(__name__)

class PersistentMemoryInterface:
    """Enhanced memory interface with causal persistence"""
    
    def __init__(self, memory_id: str = "unified_memory"):
        self.memory_id = memory_id
        self._persistence_manager = None
        self._memory_native = MemoryNativeAI()
        self._access_history = []
        self._memory_evolution = []
        
    async def _ensure_persistence(self):
        """Ensure persistence manager is initialized"""
        if self._persistence_manager is None:
            self._persistence_manager = await get_causal_manager()
    
    async def store_memory(self,
                         key: str,
                         value: Any,
                         memory_type: str = "episodic",
                         embedding: Optional[np.ndarray] = None,
                         metadata: Optional[Dict[str, Any]] = None) -> str:
        """Store memory with causal tracking"""
        await self._ensure_persistence()
        
        # Track access patterns
        self._access_history.append({
            "key": key,
            "type": "write",
            "memory_type": memory_type,
            "timestamp": datetime.now()
        })
        
        # Extract causes
        causes = self._extract_memory_causes(key, memory_type, metadata)
        effects = self._predict_memory_effects(value, memory_type)
        
        # Create causal context
        causal_context = CausalContext(
            causes=causes,
            effects=effects,
            confidence=metadata.get("confidence", 0.8) if metadata else 0.8,
            energy_cost=0.01  # Memory operations are cheap
        )
        
        # Prepare memory data
        memory_data = {
            "key": key,
            "value": value,
            "memory_type": memory_type,
            "metadata": metadata or {},
            "timestamp": datetime.now().isoformat(),
            "access_count": 0,
            "last_accessed": None
        }
        
        # Convert embedding to list for storage
        if embedding is not None:
            embedding_list = embedding.tolist() if isinstance(embedding, np.ndarray) else embedding
        else:
            embedding_list = None
        
        # Save to persistence
        state_id = await self._persistence_manager.save_state(
            StateType.MEMORY_STATE,
            f"{self.memory_id}_{key}",
            memory_data,
            causal_context=causal_context,
            embedding=embedding_list
        )
        
        # Also store in memory-native system
        await self._memory_native.store_memory(
            key, value, memory_type, embedding
        )
        
        logger.info("Stored memory with causality",
                   key=key,
                   memory_type=memory_type,
                   state_id=state_id)
        
        return state_id
    
    async def retrieve_memory(self,
                            key: Optional[str] = None,
                            query_embedding: Optional[np.ndarray] = None,
                            memory_type: Optional[str] = None,
                            top_k: int = 5,
                            compute_on_retrieval: Optional[callable] = None) -> List[Dict[str, Any]]:
        """Retrieve memory with optional computation"""
        await self._ensure_persistence()
        
        # Track access
        self._access_history.append({
            "key": key,
            "type": "read",
            "memory_type": memory_type,
            "timestamp": datetime.now()
        })
        
        results = []
        
        if key:
            # Direct key retrieval
            memory = await self._persistence_manager.load_state(
                StateType.MEMORY_STATE,
                f"{self.memory_id}_{key}",
                compute_on_retrieval=compute_on_retrieval
            )
            if memory:
                # Update access metadata
                memory["access_count"] = memory.get("access_count", 0) + 1
                memory["last_accessed"] = datetime.now().isoformat()
                
                # Save updated access info
                await self._persistence_manager.save_state(
                    StateType.MEMORY_STATE,
                    f"{self.memory_id}_{key}",
                    memory
                )
                
                results.append(memory)
        
        elif query_embedding is not None:
            # Vector similarity search
            similar_memories = await self._persistence_manager.search_by_embedding(
                StateType.MEMORY_STATE,
                query_embedding.tolist() if isinstance(query_embedding, np.ndarray) else query_embedding,
                top_k=top_k,
                component_id_prefix=self.memory_id
            )
            
            for memory_info in similar_memories:
                memory = memory_info["data"]
                if compute_on_retrieval:
                    memory = compute_on_retrieval(memory)
                results.append(memory)
        
        return results
    
    async def evolve_memory(self,
                          key: str,
                          evolution_fn: callable,
                          create_branch: bool = False) -> str:
        """Evolve memory through computation"""
        await self._ensure_persistence()
        
        # Create experimental branch if requested
        branch_id = None
        if create_branch:
            branch_id = await self._persistence_manager.create_branch(
                f"{self.memory_id}_{key}",
                f"evolution_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
        
        # Load and evolve
        memory = await self._persistence_manager.load_state(
            StateType.MEMORY_STATE,
            f"{self.memory_id}_{key}",
            branch_id=branch_id,
            compute_on_retrieval=evolution_fn
        )
        
        if memory:
            # Track evolution
            self._memory_evolution.append({
                "key": key,
                "timestamp": datetime.now(),
                "branch_id": branch_id,
                "before_value": memory.get("value"),
                "evolution_applied": str(evolution_fn)
            })
            
            # Create causal context for evolution
            causal_context = CausalContext(
                causes=["memory_evolution", f"function_{evolution_fn.__name__}"],
                effects=["memory_transformed", "knowledge_updated"],
                confidence=0.85
            )
            
            # Save evolved state
            state_id = await self._persistence_manager.save_state(
                StateType.MEMORY_STATE,
                f"{self.memory_id}_{key}",
                memory,
                causal_context=causal_context,
                branch_id=branch_id
            )
            
            logger.info("Evolved memory",
                       key=key,
                       branch=create_branch,
                       state_id=state_id)
            
            return state_id
        
        return None
    
    async def consolidate_memories(self,
                                 memory_type: str = "episodic",
                                 consolidation_threshold: float = 0.7) -> Dict[str, Any]:
        """Consolidate similar memories with causal tracking"""
        await self._ensure_persistence()
        
        # Get all memories of type
        all_memories = await self._persistence_manager.list_states(
            StateType.MEMORY_STATE,
            component_id_prefix=f"{self.memory_id}_"
        )
        
        # Filter by type
        typed_memories = [
            m for m in all_memories 
            if m.get("data", {}).get("memory_type") == memory_type
        ]
        
        # Group similar memories (simplified - in practice use embeddings)
        consolidated = {}
        consolidation_count = 0
        
        for memory in typed_memories:
            # Here you would use embedding similarity
            # For now, just demonstrate the pattern
            key = memory["data"]["key"]
            if key not in consolidated:
                consolidated[key] = memory["data"]
            else:
                # Merge memories
                consolidated[key]["value"] = f"{consolidated[key]['value']} | {memory['data']['value']}"
                consolidation_count += 1
        
        # Save consolidation event
        causal_context = CausalContext(
            causes=["memory_consolidation", f"threshold_{consolidation_threshold}"],
            effects=[f"consolidated_{consolidation_count}_memories"],
            confidence=0.9
        )
        
        consolidation_report = {
            "memory_type": memory_type,
            "original_count": len(typed_memories),
            "consolidated_count": len(consolidated),
            "threshold": consolidation_threshold,
            "timestamp": datetime.now().isoformat()
        }
        
        await self._persistence_manager.save_state(
            StateType.COMPONENT_STATE,
            f"{self.memory_id}_consolidation",
            consolidation_report,
            causal_context=causal_context
        )
        
        return consolidation_report
    
    async def get_memory_timeline(self,
                                key: str) -> List[Dict[str, Any]]:
        """Get the causal timeline of a memory"""
        await self._ensure_persistence()
        
        # Get all states for this memory
        states = await self._persistence_manager.get_state_history(
            StateType.MEMORY_STATE,
            f"{self.memory_id}_{key}"
        )
        
        timeline = []
        for state in states:
            timeline.append({
                "timestamp": state.get("timestamp"),
                "value": state.get("data", {}).get("value"),
                "causes": state.get("causal_context", {}).get("causes", []),
                "effects": state.get("causal_context", {}).get("effects", []),
                "access_count": state.get("data", {}).get("access_count", 0)
            })
        
        return timeline
    
    async def analyze_memory_patterns(self) -> Dict[str, Any]:
        """Analyze memory access patterns with causality"""
        await self._ensure_persistence()
        
        # Analyze access history
        total_accesses = len(self._access_history)
        read_count = sum(1 for a in self._access_history if a["type"] == "read")
        write_count = sum(1 for a in self._access_history if a["type"] == "write")
        
        # Memory type distribution
        type_distribution = {}
        for access in self._access_history:
            mem_type = access.get("memory_type", "unknown")
            type_distribution[mem_type] = type_distribution.get(mem_type, 0) + 1
        
        # Hot keys (frequently accessed)
        key_frequency = {}
        for access in self._access_history:
            if access.get("key"):
                key_frequency[access["key"]] = key_frequency.get(access["key"], 0) + 1
        
        hot_keys = sorted(key_frequency.items(), key=lambda x: x[1], reverse=True)[:5]
        
        analysis = {
            "total_accesses": total_accesses,
            "read_write_ratio": read_count / write_count if write_count > 0 else float('inf'),
            "memory_type_distribution": type_distribution,
            "hot_keys": hot_keys,
            "evolution_count": len(self._memory_evolution),
            "analysis_timestamp": datetime.now().isoformat()
        }
        
        # Save analysis
        causal_context = CausalContext(
            causes=["pattern_analysis_requested"],
            effects=["insights_generated", "optimization_opportunities_identified"],
            confidence=1.0
        )
        
        await self._persistence_manager.save_state(
            StateType.COMPONENT_STATE,
            f"{self.memory_id}_analysis",
            analysis,
            causal_context=causal_context
        )
        
        return analysis
    
    def _extract_memory_causes(self,
                             key: str,
                             memory_type: str,
                             metadata: Optional[Dict[str, Any]]) -> List[str]:
        """Extract what caused this memory storage"""
        causes = [f"memory_store_{memory_type}"]
        
        if metadata:
            if metadata.get("source"):
                causes.append(f"source_{metadata['source']}")
            if metadata.get("importance", 0) > 0.8:
                causes.append("high_importance")
            if metadata.get("emotional_valence"):
                causes.append(f"emotion_{metadata['emotional_valence']}")
        
        # Check access patterns
        recent_accesses = [a for a in self._access_history[-10:] if a.get("key") == key]
        if len(recent_accesses) > 3:
            causes.append("frequent_access_pattern")
        
        return causes
    
    def _predict_memory_effects(self,
                              value: Any,
                              memory_type: str) -> List[str]:
        """Predict effects of storing this memory"""
        effects = [f"{memory_type}_memory_updated"]
        
        # Value-based effects
        if isinstance(value, str) and len(value) > 1000:
            effects.append("large_memory_stored")
        
        if memory_type == "semantic":
            effects.append("knowledge_base_expanded")
        elif memory_type == "episodic":
            effects.append("experience_recorded")
        elif memory_type == "working":
            effects.append("context_updated")
        
        return effects


async def integrate_memory_with_persistence():
    """Integration helper for existing memory systems"""
    # Create persistent memory interface
    persistent_memory = PersistentMemoryInterface()
    
    # Example: Migrate existing memories
    logger.info("Memory persistence integration ready")
    
    return persistent_memory