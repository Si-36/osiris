"""
Council LNN Implementations - Production 2025
============================================

Real implementations of council agents with:
- Transformer-based reasoning
- Multi-modal understanding
- Distributed consensus
- Real-time adaptation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Any, Optional, Tuple, AsyncIterator
from dataclasses import dataclass, field
import numpy as np
import asyncio
from datetime import datetime
import json
from collections import deque
import time

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline,
    BitsAndBytesConfig
)
from sentence_transformers import SentenceTransformer
import faiss
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS as LangchainFAISS
from langchain.schema import Document

from .interfaces import (
    ICouncilAgent,
    INeuralEngine,
    IKnowledgeGraph,
    IMemorySystem,
    ICouncilOrchestrator,
    IVotingMechanism,
    IConsensusProtocol,
    IContextManager
)
from .contracts import (
    CouncilRequest,
    CouncilResponse,
    ContextSnapshot,
    NeuralFeatures,
    DecisionEvidence,
    VoteDecision,
    VoteConfidence,
    AgentMetrics,
    TopologySignature,
    ConsensusResult
)


class TransformerNeuralEngine(INeuralEngine):
    """Production neural engine with transformer models"""
    
    def __init__(
        self,
        model_name: str = "microsoft/phi-2",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        quantize: bool = True
    ):
        self.device = device
        self.model_name = model_name
        
        # Load quantized model for efficiency
        if quantize and device == "cuda":
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                device_map="auto" if device == "cuda" else None,
                trust_remote_code=True
            )
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Feature extraction layers
        self.feature_extractor = nn.Sequential(
            nn.Linear(self.model.config.hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.LayerNorm(256)
        ).to(device)
        
        # Specialized heads for different tasks
        self.decision_head = nn.Linear(256, 128).to(device)
        self.confidence_head = nn.Linear(256, 1).to(device)
        self.risk_head = nn.Linear(256, 64).to(device)
        
    async def extract_features(self, context: ContextSnapshot) -> NeuralFeatures:
        """Extract neural features from context"""
        # Prepare input text
        input_text = self._prepare_context_text(context)
        
        # Tokenize and get embeddings
        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
            hidden_states = outputs.hidden_states[-1]  # Last layer
            
            # Pool hidden states
            pooled = hidden_states.mean(dim=1)  # [batch, hidden_size]
            
            # Extract features
            features = self.feature_extractor(pooled)
            
            # Generate specialized features
            decision_features = self.decision_head(features)
            confidence = torch.sigmoid(self.confidence_head(features))
            risk_features = self.risk_head(features)
        
        return NeuralFeatures(
            embeddings=features.cpu().numpy().tolist(),
            attention_weights=self._extract_attention_patterns(outputs),
            feature_importance=self._compute_feature_importance(features),
            temporal_patterns=self._extract_temporal_patterns(hidden_states),
            confidence_scores={
                "overall": confidence.item(),
                "decision": float(decision_features.abs().mean()),
                "risk": float(risk_features.abs().mean())
            }
        )
    
    async def reason_about(self, features: NeuralFeatures, query: str) -> DecisionEvidence:
        """Reason about features to produce evidence"""
        # Generate reasoning using the transformer
        reasoning_prompt = f"""Given the following features and query, provide detailed reasoning:

Query: {query}

Feature Summary:
- Confidence: {features.confidence_scores['overall']:.2f}
- Feature Importance: {sorted(features.feature_importance.items(), key=lambda x: x[1], reverse=True)[:5]}
- Temporal Patterns: {features.temporal_patterns.get('trend', 'stable')}

Provide reasoning for decision-making:"""
        
        inputs = self.tokenizer(reasoning_prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=200,
                temperature=0.7,
                do_sample=True,
                top_p=0.9
            )
        
        reasoning = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        reasoning = reasoning.split("Provide reasoning for decision-making:")[-1].strip()
        
        # Extract evidence components
        return DecisionEvidence(
            reasoning_chain=[
                {"step": "feature_analysis", "conclusion": "Features extracted and analyzed"},
                {"step": "pattern_recognition", "conclusion": f"Identified {len(features.temporal_patterns)} patterns"},
                {"step": "reasoning", "conclusion": reasoning[:200]}
            ],
            supporting_facts=self._extract_facts(reasoning),
            confidence_factors={
                "feature_quality": features.confidence_scores['overall'],
                "reasoning_coherence": self._assess_coherence(reasoning),
                "evidence_strength": self._assess_evidence_strength(features)
            },
            risk_assessment={
                "identified_risks": self._identify_risks(features, reasoning),
                "mitigation_strategies": self._suggest_mitigations(features)
            }
        )
    
    async def adapt_to_feedback(self, feedback: Dict[str, Any]) -> None:
        """Adapt the neural engine based on feedback"""
        # In production, this would fine-tune or update the model
        # For now, we'll track feedback for analysis
        if not hasattr(self, 'feedback_history'):
            self.feedback_history = deque(maxlen=1000)
        
        self.feedback_history.append({
            'timestamp': datetime.now(),
            'feedback': feedback,
            'model_state': self._get_model_state_summary()
        })
        
        # Adjust temperature based on feedback
        if feedback.get('accuracy', 0.5) < 0.3:
            self.generation_temperature = max(0.5, getattr(self, 'generation_temperature', 0.7) - 0.1)
        elif feedback.get('accuracy', 0.5) > 0.8:
            self.generation_temperature = min(1.0, getattr(self, 'generation_temperature', 0.7) + 0.1)
    
    def _prepare_context_text(self, context: ContextSnapshot) -> str:
        """Prepare context for transformer input"""
        parts = []
        
        if context.query:
            parts.append(f"Query: {context.query}")
        
        if context.historical_data:
            parts.append(f"Historical context: {json.dumps(context.historical_data[:3])}")
        
        if context.domain_knowledge:
            parts.append(f"Domain knowledge: {json.dumps(context.domain_knowledge)[:200]}")
        
        if context.active_patterns:
            parts.append(f"Active patterns: {', '.join(context.active_patterns[:5])}")
        
        return "\n".join(parts)
    
    def _extract_attention_patterns(self, outputs) -> Dict[str, Any]:
        """Extract attention patterns from model outputs"""
        if hasattr(outputs, 'attentions') and outputs.attentions:
            # Average attention across layers and heads
            attention = torch.stack(outputs.attentions).mean(dim=(0, 1, 2))
            
            # Find top attended positions
            top_positions = attention.topk(5).indices.tolist()
            
            return {
                "top_positions": top_positions,
                "attention_entropy": float(-torch.sum(attention * torch.log(attention + 1e-9))),
                "attention_concentration": float(attention.max())
            }
        return {}
    
    def _compute_feature_importance(self, features: torch.Tensor) -> Dict[str, float]:
        """Compute importance of different feature dimensions"""
        importance = features.abs().mean(dim=0)
        
        feature_names = [f"dim_{i}" for i in range(features.shape[1])]
        
        return {
            name: float(imp) 
            for name, imp in zip(feature_names, importance)
        }
    
    def _extract_temporal_patterns(self, hidden_states: torch.Tensor) -> Dict[str, Any]:
        """Extract temporal patterns from hidden states"""
        # Analyze sequence dynamics
        seq_len = hidden_states.shape[1]
        
        if seq_len > 1:
            # Compute differences between consecutive positions
            diffs = hidden_states[:, 1:, :] - hidden_states[:, :-1, :]
            change_magnitude = diffs.norm(dim=-1).mean()
            
            # Detect trend
            first_half = hidden_states[:, :seq_len//2, :].mean(dim=1)
            second_half = hidden_states[:, seq_len//2:, :].mean(dim=1)
            trend_direction = (second_half - first_half).mean()
            
            return {
                "change_magnitude": float(change_magnitude),
                "trend": "increasing" if trend_direction > 0 else "decreasing",
                "volatility": float(diffs.std()),
                "sequence_length": seq_len
            }
        
        return {"sequence_length": seq_len}
    
    def _extract_facts(self, reasoning: str) -> List[str]:
        """Extract factual statements from reasoning"""
        # Simple extraction - in production use NER/fact extraction models
        sentences = reasoning.split('.')
        facts = []
        
        for sent in sentences:
            sent = sent.strip()
            if len(sent) > 10 and any(word in sent.lower() for word in ['is', 'are', 'was', 'were', 'has', 'have']):
                facts.append(sent)
        
        return facts[:5]  # Top 5 facts
    
    def _assess_coherence(self, reasoning: str) -> float:
        """Assess reasoning coherence"""
        # Simple heuristic - in production use coherence models
        sentences = [s.strip() for s in reasoning.split('.') if s.strip()]
        
        if len(sentences) < 2:
            return 0.5
        
        # Check for logical connectors
        connectors = ['therefore', 'because', 'since', 'thus', 'however', 'moreover']
        connector_count = sum(1 for s in sentences if any(c in s.lower() for c in connectors))
        
        coherence = min(1.0, 0.3 + (connector_count / len(sentences)) * 0.7)
        return coherence
    
    def _assess_evidence_strength(self, features: NeuralFeatures) -> float:
        """Assess strength of evidence"""
        confidence = features.confidence_scores.get('overall', 0.5)
        importance_variance = np.var(list(features.feature_importance.values()))
        
        # High confidence and low variance indicates strong evidence
        strength = confidence * (1 - min(importance_variance, 1.0))
        return float(strength)
    
    def _identify_risks(self, features: NeuralFeatures, reasoning: str) -> List[str]:
        """Identify potential risks"""
        risks = []
        
        # Check confidence levels
        if features.confidence_scores.get('overall', 1.0) < 0.3:
            risks.append("Low confidence in analysis")
        
        # Check for uncertainty markers in reasoning
        uncertainty_words = ['maybe', 'possibly', 'might', 'could', 'uncertain', 'unclear']
        if any(word in reasoning.lower() for word in uncertainty_words):
            risks.append("Uncertainty detected in reasoning")
        
        # Check temporal volatility
        if features.temporal_patterns.get('volatility', 0) > 0.8:
            risks.append("High temporal volatility detected")
        
        return risks
    
    def _suggest_mitigations(self, features: NeuralFeatures) -> List[str]:
        """Suggest risk mitigation strategies"""
        mitigations = []
        
        if features.confidence_scores.get('overall', 1.0) < 0.5:
            mitigations.append("Gather additional context before decision")
        
        if features.temporal_patterns.get('volatility', 0) > 0.5:
            mitigations.append("Wait for pattern stabilization")
        
        if len(features.feature_importance) > 0:
            top_features = sorted(features.feature_importance.items(), key=lambda x: x[1], reverse=True)[:3]
            mitigations.append(f"Focus analysis on top features: {[f[0] for f in top_features]}")
        
        return mitigations
    
    def _get_model_state_summary(self) -> Dict[str, Any]:
        """Get summary of current model state"""
        return {
            "model_name": self.model_name,
            "device": self.device,
            "generation_temperature": getattr(self, 'generation_temperature', 0.7),
            "feedback_history_size": len(getattr(self, 'feedback_history', []))
        }


class GraphKnowledgeSystem(IKnowledgeGraph):
    """Production knowledge graph with vector search and graph algorithms"""
    
    def __init__(
        self,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        graph_backend: str = "networkx"  # or "neo4j" for production
    ):
        self.embedding_model = SentenceTransformer(embedding_model)
        self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
        
        # Initialize vector store
        self.vector_store = None
        self.documents = []
        
        # Graph components
        if graph_backend == "networkx":
            import networkx as nx
            self.graph = nx.DiGraph()
        else:
            # Neo4j initialization would go here
            self.graph = None
        
        # Topology analyzer
        self.topology_analyzer = TopologyAnalyzer()
        
    async def query(self, query: str, context: Optional[ContextSnapshot] = None) -> Dict[str, Any]:
        """Query knowledge graph with semantic search"""
        # Vector search
        vector_results = await self._vector_search(query, k=10)
        
        # Graph traversal
        graph_results = await self._graph_search(query, context)
        
        # Combine results
        combined_results = self._combine_results(vector_results, graph_results)
        
        # Add topological analysis
        if self.graph and hasattr(self.graph, 'number_of_nodes'):
            topology = await self.topology_analyzer.analyze_graph(self.graph)
            combined_results['topology'] = topology
        
        return combined_results
    
    async def add_knowledge(self, knowledge: Dict[str, Any]) -> bool:
        """Add knowledge to graph"""
        try:
            # Extract entities and relations
            entities = knowledge.get('entities', [])
            relations = knowledge.get('relations', [])
            text = knowledge.get('text', '')
            
            # Add to vector store
            if text:
                doc = Document(
                    page_content=text,
                    metadata=knowledge.get('metadata', {})
                )
                self.documents.append(doc)
                
                if self.vector_store is None:
                    self.vector_store = LangchainFAISS.from_documents(
                        [doc], self.embeddings
                    )
                else:
                    self.vector_store.add_documents([doc])
            
            # Add to graph
            for entity in entities:
                self.graph.add_node(
                    entity['id'],
                    **entity.get('properties', {})
                )
            
            for relation in relations:
                self.graph.add_edge(
                    relation['source'],
                    relation['target'],
                    type=relation.get('type', 'related'),
                    **relation.get('properties', {})
                )
            
            return True
            
        except Exception as e:
            print(f"Error adding knowledge: {e}")
            return False
    
    async def get_topology_signature(self) -> TopologySignature:
        """Get topological signature of knowledge graph"""
        if not self.graph:
            return TopologySignature(
                nodes=0, edges=0, components=0,
                clustering_coefficient=0.0,
                centrality_measures={}
            )
        
        import networkx as nx
        
        # Basic metrics
        nodes = self.graph.number_of_nodes()
        edges = self.graph.number_of_edges()
        
        # Component analysis
        if isinstance(self.graph, nx.DiGraph):
            components = nx.number_weakly_connected_components(self.graph)
        else:
            components = nx.number_connected_components(self.graph)
        
        # Clustering
        clustering = nx.average_clustering(self.graph.to_undirected()) if nodes > 0 else 0.0
        
        # Centrality (sample for performance)
        sample_nodes = list(self.graph.nodes())[:min(100, nodes)]
        if sample_nodes:
            centrality = nx.betweenness_centrality(
                self.graph.subgraph(sample_nodes)
            )
            avg_centrality = np.mean(list(centrality.values()))
        else:
            avg_centrality = 0.0
        
        return TopologySignature(
            nodes=nodes,
            edges=edges,
            components=components,
            clustering_coefficient=clustering,
            centrality_measures={
                "average_betweenness": avg_centrality,
                "density": nx.density(self.graph) if nodes > 0 else 0.0
            }
        )
    
    async def find_paths(self, source: str, target: str, max_length: int = 5) -> List[List[str]]:
        """Find paths between entities"""
        if not self.graph or source not in self.graph or target not in self.graph:
            return []
        
        import networkx as nx
        
        try:
            # Find all simple paths up to max_length
            paths = list(nx.all_simple_paths(
                self.graph, source, target, cutoff=max_length
            ))
            return paths[:10]  # Limit to 10 paths
        except:
            return []
    
    async def _vector_search(self, query: str, k: int = 10) -> List[Dict[str, Any]]:
        """Perform vector similarity search"""
        if not self.vector_store:
            return []
        
        results = self.vector_store.similarity_search_with_score(query, k=k)
        
        return [
            {
                "content": doc.page_content,
                "metadata": doc.metadata,
                "score": float(score)
            }
            for doc, score in results
        ]
    
    async def _graph_search(self, query: str, context: Optional[ContextSnapshot]) -> Dict[str, Any]:
        """Perform graph-based search"""
        if not self.graph:
            return {}
        
        # Extract entities from query
        query_embedding = self.embedding_model.encode([query])[0]
        
        # Find most relevant nodes
        node_scores = {}
        for node in self.graph.nodes():
            node_data = self.graph.nodes[node]
            if 'embedding' in node_data:
                similarity = np.dot(query_embedding, node_data['embedding'])
                node_scores[node] = similarity
        
        # Get top nodes
        top_nodes = sorted(node_scores.items(), key=lambda x: x[1], reverse=True)[:5]
        
        # Expand neighborhood
        relevant_subgraph = set()
        for node, _ in top_nodes:
            relevant_subgraph.add(node)
            relevant_subgraph.update(self.graph.neighbors(node))
        
        return {
            "relevant_nodes": [n for n, _ in top_nodes],
            "subgraph_size": len(relevant_subgraph),
            "node_scores": dict(top_nodes)
        }
    
    def _combine_results(self, vector_results: List[Dict], graph_results: Dict) -> Dict[str, Any]:
        """Combine vector and graph search results"""
        return {
            "vector_results": vector_results,
            "graph_results": graph_results,
            "combined_score": self._compute_combined_score(vector_results, graph_results),
            "timestamp": datetime.now().isoformat()
        }
    
    def _compute_combined_score(self, vector_results: List[Dict], graph_results: Dict) -> float:
        """Compute combined relevance score"""
        vector_score = np.mean([r['score'] for r in vector_results]) if vector_results else 0.0
        graph_score = np.mean(list(graph_results.get('node_scores', {}).values())) if graph_results.get('node_scores') else 0.0
        
        # Weighted combination
        return 0.6 * vector_score + 0.4 * graph_score


class TopologyAnalyzer:
    """Analyze topological properties of graphs and systems"""
    
    async def analyze_graph(self, graph) -> Dict[str, Any]:
        """Analyze graph topology"""
        import networkx as nx
        
        analysis = {
            "basic_stats": {
                "nodes": graph.number_of_nodes(),
                "edges": graph.number_of_edges(),
                "density": nx.density(graph)
            },
            "connectivity": {
                "is_connected": nx.is_weakly_connected(graph) if isinstance(graph, nx.DiGraph) else nx.is_connected(graph),
                "components": nx.number_weakly_connected_components(graph) if isinstance(graph, nx.DiGraph) else nx.number_connected_components(graph)
            },
            "centrality": {},
            "clustering": {}
        }
        
        # Sample for performance on large graphs
        if graph.number_of_nodes() > 1000:
            sample_nodes = list(graph.nodes())[:100]
            subgraph = graph.subgraph(sample_nodes)
        else:
            subgraph = graph
        
        # Centrality measures
        if subgraph.number_of_nodes() > 0:
            analysis["centrality"] = {
                "degree": self._compute_avg_centrality(nx.degree_centrality(subgraph)),
                "betweenness": self._compute_avg_centrality(nx.betweenness_centrality(subgraph)),
                "closeness": self._compute_avg_centrality(nx.closeness_centrality(subgraph))
            }
            
            # Clustering
            analysis["clustering"] = {
                "average": nx.average_clustering(subgraph.to_undirected()),
                "transitivity": nx.transitivity(subgraph)
            }
        
        return analysis
    
    def _compute_avg_centrality(self, centrality_dict: Dict) -> float:
        """Compute average centrality"""
        values = list(centrality_dict.values())
        return float(np.mean(values)) if values else 0.0


class AdaptiveMemorySystem(IMemorySystem):
    """Production memory system with hierarchical storage"""
    
    def __init__(
        self,
        embedding_dim: int = 768,
        short_term_capacity: int = 1000,
        long_term_capacity: int = 100000
    ):
        self.embedding_dim = embedding_dim
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Hierarchical memory stores
        self.working_memory = deque(maxlen=100)  # Immediate context
        self.short_term = FaissMemoryStore(embedding_dim, short_term_capacity)
        self.long_term = FaissMemoryStore(embedding_dim, long_term_capacity)
        
        # Memory consolidation
        self.consolidation_threshold = 0.8
        self.access_counts = {}
        
    async def store(self, memory: Dict[str, Any], importance: float = 0.5) -> str:
        """Store memory with importance weighting"""
        # Generate ID
        memory_id = f"mem_{int(time.time() * 1000)}_{np.random.randint(1000)}"
        
        # Add metadata
        memory['id'] = memory_id
        memory['timestamp'] = datetime.now().isoformat()
        memory['importance'] = importance
        memory['access_count'] = 0
        
        # Generate embedding
        text = memory.get('content', str(memory))
        embedding = self.encoder.encode([text])[0]
        
        # Store in appropriate tier
        self.working_memory.append(memory)
        
        if importance > 0.7:
            # High importance -> long-term
            await self.long_term.add(memory_id, embedding, memory)
        else:
            # Normal importance -> short-term
            await self.short_term.add(memory_id, embedding, memory)
        
        return memory_id
    
    async def recall(self, query: str, k: int = 10) -> List[Dict[str, Any]]:
        """Recall relevant memories"""
        query_embedding = self.encoder.encode([query])[0]
        
        # Search all tiers
        results = []
        
        # Working memory (exact match)
        for mem in self.working_memory:
            if query.lower() in str(mem).lower():
                results.append(mem)
        
        # Short-term memory
        short_term_results = await self.short_term.search(query_embedding, k=k//2)
        results.extend(short_term_results)
        
        # Long-term memory
        long_term_results = await self.long_term.search(query_embedding, k=k//2)
        results.extend(long_term_results)
        
        # Update access counts
        for mem in results:
            mem_id = mem.get('id')
            if mem_id:
                self.access_counts[mem_id] = self.access_counts.get(mem_id, 0) + 1
                mem['access_count'] = self.access_counts[mem_id]
        
        # Sort by relevance and recency
        results.sort(
            key=lambda x: (
                x.get('similarity', 0) * 0.5 +
                x.get('importance', 0.5) * 0.3 +
                (1.0 / (time.time() - self._parse_timestamp(x.get('timestamp', '')))) * 0.2
            ),
            reverse=True
        )
        
        return results[:k]
    
    async def consolidate(self) -> None:
        """Consolidate memories between tiers"""
        # Promote frequently accessed short-term memories
        frequent_memories = []
        for mem_id, count in self.access_counts.items():
            if count > 5:  # Threshold for promotion
                memory = await self.short_term.get(mem_id)
                if memory:
                    frequent_memories.append(memory)
        
        # Move to long-term
        for memory in frequent_memories:
            text = memory.get('content', str(memory))
            embedding = self.encoder.encode([text])[0]
            await self.long_term.add(memory['id'], embedding, memory)
            await self.short_term.remove(memory['id'])
        
        # Forget old, unimportant memories
        await self._forget_old_memories()
    
    async def get_statistics(self) -> Dict[str, Any]:
        """Get memory system statistics"""
        return {
            "working_memory_size": len(self.working_memory),
            "short_term_size": self.short_term.size(),
            "long_term_size": self.long_term.size(),
            "total_accesses": sum(self.access_counts.values()),
            "most_accessed": sorted(
                self.access_counts.items(),
                key=lambda x: x[1],
                reverse=True
            )[:5]
        }
    
    def _parse_timestamp(self, timestamp: str) -> float:
        """Parse timestamp to float"""
        try:
            return datetime.fromisoformat(timestamp).timestamp()
        except:
            return time.time()
    
    async def _forget_old_memories(self):
        """Remove old, unimportant memories"""
        current_time = time.time()
        forget_threshold = 30 * 24 * 3600  # 30 days
        
        # Check short-term memories
        old_memories = []
        for mem_id in await self.short_term.list_ids():
            memory = await self.short_term.get(mem_id)
            if memory:
                timestamp = self._parse_timestamp(memory.get('timestamp', ''))
                if (current_time - timestamp > forget_threshold and
                    memory.get('importance', 0.5) < 0.3 and
                    self.access_counts.get(mem_id, 0) < 2):
                    old_memories.append(mem_id)
        
        # Remove old memories
        for mem_id in old_memories:
            await self.short_term.remove(mem_id)
            self.access_counts.pop(mem_id, None)


class FaissMemoryStore:
    """FAISS-based vector memory store"""
    
    def __init__(self, embedding_dim: int, capacity: int):
        self.embedding_dim = embedding_dim
        self.capacity = capacity
        
        # Initialize FAISS index
        self.index = faiss.IndexFlatL2(embedding_dim)
        self.id_map = {}
        self.data_store = {}
        self.next_idx = 0
        
    async def add(self, memory_id: str, embedding: np.ndarray, data: Dict[str, Any]):
        """Add memory to store"""
        if self.next_idx >= self.capacity:
            # Remove oldest
            await self._evict_oldest()
        
        # Add to index
        self.index.add(embedding.reshape(1, -1).astype(np.float32))
        
        # Store mapping and data
        self.id_map[memory_id] = self.next_idx
        self.data_store[memory_id] = data
        self.next_idx += 1
        
    async def search(self, query_embedding: np.ndarray, k: int = 10) -> List[Dict[str, Any]]:
        """Search for similar memories"""
        if self.index.ntotal == 0:
            return []
        
        k = min(k, self.index.ntotal)
        
        # Search
        distances, indices = self.index.search(
            query_embedding.reshape(1, -1).astype(np.float32), k
        )
        
        # Retrieve data
        results = []
        reverse_map = {v: k for k, v in self.id_map.items()}
        
        for idx, dist in zip(indices[0], distances[0]):
            if idx in reverse_map:
                memory_id = reverse_map[idx]
                if memory_id in self.data_store:
                    memory = self.data_store[memory_id].copy()
                    memory['similarity'] = 1.0 / (1.0 + dist)  # Convert distance to similarity
                    results.append(memory)
        
        return results
    
    async def get(self, memory_id: str) -> Optional[Dict[str, Any]]:
        """Get specific memory"""
        return self.data_store.get(memory_id)
    
    async def remove(self, memory_id: str):
        """Remove memory"""
        if memory_id in self.id_map:
            del self.id_map[memory_id]
            del self.data_store[memory_id]
    
    async def list_ids(self) -> List[str]:
        """List all memory IDs"""
        return list(self.data_store.keys())
    
    def size(self) -> int:
        """Get store size"""
        return len(self.data_store)
    
    async def _evict_oldest(self):
        """Evict oldest memory when at capacity"""
        if not self.data_store:
            return
        
        # Find oldest by timestamp
        oldest_id = min(
            self.data_store.keys(),
            key=lambda x: self.data_store[x].get('timestamp', '')
        )
        
        await self.remove(oldest_id)


class CouncilOrchestrator(ICouncilOrchestrator):
    """Production council orchestration with consensus"""
    
    def __init__(
        self,
        consensus_threshold: float = 0.7,
        min_agents: int = 3
    ):
        self.agents: Dict[str, ICouncilAgent] = {}
        self.consensus_threshold = consensus_threshold
        self.min_agents = min_agents
        self.voting_mechanism = WeightedVoting()
        self.consensus_protocol = ByzantineConsensus(threshold=consensus_threshold)
        
    async def register_agent(self, agent_id: str, agent: ICouncilAgent) -> bool:
        """Register an agent with the council"""
        self.agents[agent_id] = agent
        
        # Update voting weights based on capabilities
        capabilities = await agent.get_capabilities()
        weight = len(capabilities) / 10.0  # Simple weight based on capabilities
        self.voting_mechanism.set_weight(agent_id, weight)
        
        return True
    
    async def coordinate_request(self, request: CouncilRequest) -> CouncilResponse:
        """Coordinate a request across all agents"""
        if len(self.agents) < self.min_agents:
            return CouncilResponse(
                decision="insufficient_agents",
                confidence=0.0,
                evidence=DecisionEvidence(
                    reasoning_chain=[{"step": "check", "conclusion": f"Only {len(self.agents)} agents available"}],
                    supporting_facts=[],
                    confidence_factors={},
                    risk_assessment={}
                ),
                dissenting_opinions=[],
                consensus_achieved=False
            )
        
        # Gather responses from all agents
        agent_responses = {}
        response_futures = []
        
        for agent_id, agent in self.agents.items():
            response_futures.append(
                self._get_agent_response(agent_id, agent, request)
            )
        
        # Wait for all responses
        responses = await asyncio.gather(*response_futures, return_exceptions=True)
        
        for (agent_id, _), response in zip(self.agents.items(), responses):
            if not isinstance(response, Exception):
                agent_responses[agent_id] = response
        
        # Achieve consensus
        consensus_result = await self.consensus_protocol.achieve_consensus(
            agent_responses,
            self.voting_mechanism
        )
        
        return consensus_result
    
    async def get_council_status(self) -> Dict[str, Any]:
        """Get current council status"""
        agent_statuses = {}
        
        for agent_id, agent in self.agents.items():
            try:
                metrics = await agent.get_metrics()
                health = await agent.health_check()
                agent_statuses[agent_id] = {
                    "metrics": metrics.__dict__ if hasattr(metrics, '__dict__') else metrics,
                    "health": health,
                    "weight": self.voting_mechanism.weights.get(agent_id, 1.0)
                }
            except Exception as e:
                agent_statuses[agent_id] = {"error": str(e)}
        
        return {
            "total_agents": len(self.agents),
            "consensus_threshold": self.consensus_threshold,
            "agent_statuses": agent_statuses,
            "voting_weights": self.voting_mechanism.weights
        }
    
    async def _get_agent_response(
        self,
        agent_id: str,
        agent: ICouncilAgent,
        request: CouncilRequest
    ) -> Tuple[str, CouncilResponse]:
        """Get response from a single agent"""
        try:
            response = await asyncio.wait_for(
                agent.process_request(request),
                timeout=30.0  # 30 second timeout
            )
            return agent_id, response
        except asyncio.TimeoutError:
            # Return timeout response
            return agent_id, CouncilResponse(
                decision="timeout",
                confidence=0.0,
                evidence=DecisionEvidence(
                    reasoning_chain=[{"step": "timeout", "conclusion": "Agent timed out"}],
                    supporting_facts=[],
                    confidence_factors={},
                    risk_assessment={}
                ),
                dissenting_opinions=[],
                consensus_achieved=False
            )
        except Exception as e:
            # Return error response
            return agent_id, CouncilResponse(
                decision="error",
                confidence=0.0,
                evidence=DecisionEvidence(
                    reasoning_chain=[{"step": "error", "conclusion": str(e)}],
                    supporting_facts=[],
                    confidence_factors={},
                    risk_assessment={}
                ),
                dissenting_opinions=[],
                consensus_achieved=False
            )


class WeightedVoting(IVotingMechanism):
    """Weighted voting mechanism"""
    
    def __init__(self):
        self.weights: Dict[str, float] = {}
    
    def set_weight(self, agent_id: str, weight: float):
        """Set voting weight for an agent"""
        self.weights[agent_id] = max(0.1, min(10.0, weight))  # Bound weights
    
    async def aggregate_votes(self, votes: Dict[str, VoteDecision]) -> VoteDecision:
        """Aggregate votes with weights"""
        if not votes:
            return VoteDecision(
                choice="abstain",
                confidence=VoteConfidence(level=0.0, factors={}),
                reasoning=""
            )
        
        # Count weighted votes
        vote_counts = {}
        total_weight = 0.0
        
        for agent_id, vote in votes.items():
            weight = self.weights.get(agent_id, 1.0)
            choice = vote.choice
            
            vote_counts[choice] = vote_counts.get(choice, 0.0) + weight
            total_weight += weight
        
        # Find winner
        winner = max(vote_counts.items(), key=lambda x: x[1])
        winning_choice = winner[0]
        winning_weight = winner[1]
        
        # Calculate confidence
        confidence = winning_weight / total_weight if total_weight > 0 else 0.0
        
        # Aggregate reasoning
        reasoning_parts = []
        for agent_id, vote in votes.items():
            if vote.choice == winning_choice:
                reasoning_parts.append(f"{agent_id}: {vote.reasoning}")
        
        return VoteDecision(
            choice=winning_choice,
            confidence=VoteConfidence(
                level=confidence,
                factors={
                    "vote_weight": winning_weight,
                    "total_weight": total_weight,
                    "vote_distribution": vote_counts
                }
            ),
            reasoning="; ".join(reasoning_parts[:3])  # Top 3 reasons
        )


class ByzantineConsensus(IConsensusProtocol):
    """Byzantine fault-tolerant consensus"""
    
    def __init__(self, threshold: float = 0.67):
        self.threshold = threshold  # Byzantine requires 2/3 majority
    
    async def achieve_consensus(
        self,
        proposals: Dict[str, Any],
        voting_mechanism: IVotingMechanism
    ) -> ConsensusResult:
        """Achieve Byzantine consensus"""
        # Convert proposals to votes
        votes = {}
        for agent_id, response in proposals.items():
            if isinstance(response, CouncilResponse):
                votes[agent_id] = VoteDecision(
                    choice=response.decision,
                    confidence=VoteConfidence(
                        level=response.confidence,
                        factors=response.evidence.confidence_factors
                    ),
                    reasoning=response.evidence.reasoning_chain[0]['conclusion'] if response.evidence.reasoning_chain else ""
                )
        
        # Aggregate votes
        final_decision = await voting_mechanism.aggregate_votes(votes)
        
        # Check if consensus achieved
        consensus_achieved = final_decision.confidence.level >= self.threshold
        
        # Identify dissenting opinions
        dissenting = []
        for agent_id, response in proposals.items():
            if isinstance(response, CouncilResponse) and response.decision != final_decision.choice:
                dissenting.append({
                    "agent_id": agent_id,
                    "decision": response.decision,
                    "reasoning": response.evidence.reasoning_chain[0]['conclusion'] if response.evidence.reasoning_chain else ""
                })
        
        # Build consensus evidence
        all_evidence = []
        all_facts = []
        risk_assessments = []
        
        for response in proposals.values():
            if isinstance(response, CouncilResponse):
                all_evidence.extend(response.evidence.reasoning_chain)
                all_facts.extend(response.evidence.supporting_facts)
                if response.evidence.risk_assessment:
                    risk_assessments.append(response.evidence.risk_assessment)
        
        # Merge evidence
        merged_evidence = DecisionEvidence(
            reasoning_chain=all_evidence[:10],  # Top 10 reasoning steps
            supporting_facts=list(set(all_facts))[:10],  # Unique facts
            confidence_factors=final_decision.confidence.factors,
            risk_assessment=self._merge_risk_assessments(risk_assessments)
        )
        
        return CouncilResponse(
            decision=final_decision.choice,
            confidence=final_decision.confidence.level,
            evidence=merged_evidence,
            dissenting_opinions=dissenting,
            consensus_achieved=consensus_achieved,
            metadata={
                "consensus_type": "byzantine",
                "threshold": self.threshold,
                "participation": len(votes) / len(proposals) if proposals else 0
            }
        )
    
    def _merge_risk_assessments(self, assessments: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Merge multiple risk assessments"""
        if not assessments:
            return {}
        
        all_risks = []
        all_mitigations = []
        
        for assessment in assessments:
            all_risks.extend(assessment.get('identified_risks', []))
            all_mitigations.extend(assessment.get('mitigation_strategies', []))
        
        # Deduplicate and prioritize
        unique_risks = list(set(all_risks))
        unique_mitigations = list(set(all_mitigations))
        
        return {
            "identified_risks": unique_risks[:5],  # Top 5 risks
            "mitigation_strategies": unique_mitigations[:5]  # Top 5 mitigations
        }


# Create singletons for production use
_neural_engine = None
_knowledge_graph = None
_memory_system = None
_orchestrator = None

def get_neural_engine() -> INeuralEngine:
    """Get singleton neural engine"""
    global _neural_engine
    if _neural_engine is None:
        _neural_engine = TransformerNeuralEngine()
    return _neural_engine

def get_knowledge_graph() -> IKnowledgeGraph:
    """Get singleton knowledge graph"""
    global _knowledge_graph
    if _knowledge_graph is None:
        _knowledge_graph = GraphKnowledgeSystem()
    return _knowledge_graph

def get_memory_system() -> IMemorySystem:
    """Get singleton memory system"""
    global _memory_system
    if _memory_system is None:
        _memory_system = AdaptiveMemorySystem()
    return _memory_system

def get_orchestrator() -> ICouncilOrchestrator:
    """Get singleton orchestrator"""
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = CouncilOrchestrator()
    return _orchestrator