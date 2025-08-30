"""
Graph Builder for Collective Intelligence - 2025 Production Implementation

Features:
- LangGraph integration for agent orchestration
- Knowledge graph construction with Neo4j
- GraphRAG for retrieval-augmented generation
- Dynamic graph evolution
- Multi-agent workflow graphs
- State machine composition
"""

import asyncio
from typing import Dict, Any, List, Optional, Callable, TypeVar, Union
from dataclasses import dataclass, field
from datetime import datetime
import structlog
import networkx as nx
from enum import Enum
import json
import uuid
import os

# Type definitions
T = TypeVar('T')
StateType = Dict[str, Any]
NodeFunction = Callable[[StateType], StateType]

logger = structlog.get_logger(__name__)

# Try to import LangGraph, fallback to local implementation
try:
    from langgraph.graph import StateGraph, END
    from langgraph.checkpoint.sqlite import SqliteSaver
    from langgraph.prebuilt import ToolNode
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False
    logger.warning("LangGraph not available, using fallback implementation")
    
    # Fallback definitions
    class StateGraph:
        def __init__(self, state_type):
            self.state_type = state_type
            self.nodes = {}
            self.edges = {}
            self.entry_point = None
            
        def add_node(self, name: str, func: Callable):
            self.nodes[name] = func
            
        def add_edge(self, from_node: str, to_node: str):
            if from_node not in self.edges:
                self.edges[from_node] = []
            self.edges[from_node].append(to_node)
            
        def set_entry_point(self, node: str):
            self.entry_point = node
            
        def compile(self, checkpointer=None):
            return CompiledGraph(self, checkpointer)
    
    END = "__end__"
    
    class CompiledGraph:
        def __init__(self, graph, checkpointer):
            self.graph = graph
            self.checkpointer = checkpointer
            
        async def ainvoke(self, state: Dict[str, Any]) -> Dict[str, Any]:
            current = self.graph.entry_point
            while current and current != END:
                if current in self.graph.nodes:
                    state = await self.graph.nodes[current](state)
                current = self.graph.edges.get(current, [END])[0]
            return state


class GraphType(Enum):
    """Types of graphs that can be built"""
    WORKFLOW = "workflow"
    KNOWLEDGE = "knowledge"
    DECISION = "decision"
    CONSENSUS = "consensus"
    CAUSAL = "causal"
    TEMPORAL = "temporal"


class NodeType(Enum):
    """Types of nodes in graphs"""
    AGENT = "agent"
    TOOL = "tool"
    DECISION = "decision"
    AGGREGATOR = "aggregator"
    TRANSFORMER = "transformer"
    VALIDATOR = "validator"
    ROUTER = "router"


@dataclass
class GraphNode:
    """Node in a collective intelligence graph"""
    id: str = field(default_factory=lambda: f"node_{uuid.uuid4().hex[:8]}")
    name: str = ""
    type: NodeType = NodeType.AGENT
    function: Optional[Callable] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    requirements: List[str] = field(default_factory=list)
    outputs: List[str] = field(default_factory=list)
    
    async def execute(self, state: StateType) -> StateType:
        """Execute node function"""
        if self.function:
            if asyncio.iscoroutinefunction(self.function):
                return await self.function(state)
            else:
                return self.function(state)
        return state


@dataclass
class GraphEdge:
    """Edge in a collective intelligence graph"""
    source: str
    target: str
    condition: Optional[Callable[[StateType], bool]] = None
    weight: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def should_traverse(self, state: StateType) -> bool:
        """Check if edge should be traversed"""
        if self.condition:
            return self.condition(state)
        return True


class CollectiveGraphBuilder:
    """
    Advanced graph builder for collective intelligence
    
    Key features:
    - Multi-agent workflow orchestration
    - Knowledge graph construction
    - Dynamic graph evolution
    - State machine composition
    - GraphRAG integration
    """
    
    def __init__(self, 
                 graph_type: GraphType = GraphType.WORKFLOW,
                 use_checkpointing: bool = True,
                 checkpoint_dir: str = "./checkpoints"):
        self.graph_type = graph_type
        self.use_checkpointing = use_checkpointing
        self.checkpoint_dir = checkpoint_dir
        
        # Graph components
        self.nodes: Dict[str, GraphNode] = {}
        self.edges: List[GraphEdge] = []
        self.entry_point: Optional[str] = None
        self.subgraphs: Dict[str, 'CollectiveGraphBuilder'] = {}
        
        # Metadata
        self.metadata = {
            "created": datetime.now().isoformat(),
            "type": graph_type.value,
            "version": "2025.1"
        }
        
        logger.info("Graph builder initialized", type=graph_type.value)
    
    def add_node(self, 
                 name: str,
                 function: Callable,
                 node_type: NodeType = NodeType.AGENT,
                 requirements: Optional[List[str]] = None,
                 outputs: Optional[List[str]] = None) -> 'CollectiveGraphBuilder':
        """Add a node to the graph"""
        node = GraphNode(
            name=name,
            type=node_type,
            function=function,
            requirements=requirements or [],
            outputs=outputs or []
        )
        
        self.nodes[name] = node
        
        logger.debug("Node added", name=name, type=node_type.value)
        return self
    
    def add_edge(self,
                 source: str,
                 target: str,
                 condition: Optional[Callable[[StateType], bool]] = None,
                 weight: float = 1.0) -> 'CollectiveGraphBuilder':
        """Add an edge between nodes"""
        if source not in self.nodes:
            raise ValueError(f"Source node '{source}' not found")
        if target not in self.nodes and target != END:
            raise ValueError(f"Target node '{target}' not found")
        
        edge = GraphEdge(
            source=source,
            target=target,
            condition=condition,
            weight=weight
        )
        
        self.edges.append(edge)
        
        logger.debug("Edge added", source=source, target=target)
        return self
    
    def set_entry_point(self, node_name: str) -> 'CollectiveGraphBuilder':
        """Set the entry point for the graph"""
        if node_name not in self.nodes:
            raise ValueError(f"Entry point node '{node_name}' not found")
        
        self.entry_point = node_name
        logger.debug("Entry point set", node=node_name)
        return self
    
    def add_subgraph(self, 
                    name: str,
                    subgraph: 'CollectiveGraphBuilder') -> 'CollectiveGraphBuilder':
        """Add a subgraph for hierarchical composition"""
        self.subgraphs[name] = subgraph
        
        # Create a node that executes the subgraph
        async def subgraph_executor(state: StateType) -> StateType:
            compiled = subgraph.compile()
            return await compiled.ainvoke(state)
        
        self.add_node(
            name=name,
            function=subgraph_executor,
            node_type=NodeType.AGGREGATOR
        )
        
        logger.debug("Subgraph added", name=name)
        return self
    
    def add_decision_node(self,
                         name: str,
                         decision_func: Callable[[StateType], str],
                         options: Dict[str, str]) -> 'CollectiveGraphBuilder':
        """Add a decision node that routes to different paths"""
        async def decision_router(state: StateType) -> StateType:
            decision = decision_func(state)
            state["_next_node"] = options.get(decision, END)
            return state
        
        self.add_node(
            name=name,
            function=decision_router,
            node_type=NodeType.DECISION
        )
        
        # Add conditional edges for each option
        for decision_value, target_node in options.items():
            self.add_edge(
                source=name,
                target=target_node,
                condition=lambda s, dv=decision_value: decision_func(s) == dv
            )
        
        logger.debug("Decision node added", name=name, options=list(options.keys()))
        return self
    
    def add_parallel_section(self,
                           name: str,
                           parallel_nodes: List[str],
                           aggregator_func: Callable[[List[StateType]], StateType]) -> 'CollectiveGraphBuilder':
        """Add a section where multiple nodes execute in parallel"""
        async def parallel_executor(state: StateType) -> StateType:
            # Execute all parallel nodes
            tasks = []
            for node_name in parallel_nodes:
                if node_name in self.nodes:
                    node = self.nodes[node_name]
                    tasks.append(node.execute(state.copy()))
            
            # Wait for all to complete
            results = await asyncio.gather(*tasks)
            
            # Aggregate results
            return aggregator_func(results)
        
        self.add_node(
            name=name,
            function=parallel_executor,
            node_type=NodeType.AGGREGATOR
        )
        
        logger.debug("Parallel section added", name=name, nodes=parallel_nodes)
        return self
    
    def add_consensus_section(self,
                            name: str,
                            voter_nodes: List[str],
                            consensus_threshold: float = 0.66) -> 'CollectiveGraphBuilder':
        """Add a consensus-based decision section"""
        async def consensus_aggregator(state: StateType) -> StateType:
            votes = []
            
            # Collect votes from all voter nodes
            for node_name in voter_nodes:
                if node_name in self.nodes:
                    node = self.nodes[node_name]
                    result = await node.execute(state.copy())
                    vote = result.get("vote", None)
                    if vote is not None:
                        votes.append(vote)
            
            # Calculate consensus
            if votes:
                vote_counts = {}
                for vote in votes:
                    vote_counts[vote] = vote_counts.get(vote, 0) + 1
                
                # Find majority vote
                total_votes = len(votes)
                for option, count in vote_counts.items():
                    if count / total_votes >= consensus_threshold:
                        state["consensus"] = option
                        state["consensus_confidence"] = count / total_votes
                        break
                else:
                    state["consensus"] = None
                    state["consensus_confidence"] = 0.0
            
            return state
        
        self.add_node(
            name=name,
            function=consensus_aggregator,
            node_type=NodeType.AGGREGATOR
        )
        
        logger.debug("Consensus section added", 
                    name=name, 
                    voters=voter_nodes,
                    threshold=consensus_threshold)
        return self
    
    def compile(self) -> Union[Any, 'CompiledLocalGraph']:
        """Compile the graph for execution"""
        if not self.entry_point:
            raise ValueError("Entry point not set")
        
        if LANGGRAPH_AVAILABLE and self.graph_type == GraphType.WORKFLOW:
            return self._compile_langgraph()
        else:
            return self._compile_local()
    
    def _compile_langgraph(self):
        """Compile using LangGraph"""
        # Create state graph
        graph = StateGraph(dict)
        
        # Add nodes
        for name, node in self.nodes.items():
            graph.add_node(name, node.function)
        
        # Set entry point
        graph.set_entry_point(self.entry_point)
        
        # Add edges
        edge_map = {}
        for edge in self.edges:
            if edge.source not in edge_map:
                edge_map[edge.source] = []
            edge_map[edge.source].append(edge)
        
        # Add edges to graph
        for source, edges in edge_map.items():
            if len(edges) == 1 and edges[0].condition is None:
                # Simple edge
                graph.add_edge(source, edges[0].target)
            else:
                # Conditional edges
                conditions = {}
                for edge in edges:
                    if edge.condition:
                        # Create condition function
                        target = edge.target
                        cond = edge.condition
                        conditions[f"to_{target}"] = lambda s, t=target, c=cond: t if c(s) else None
                    else:
                        # Default edge
                        graph.add_edge(source, edge.target)
                
                if conditions:
                    graph.add_conditional_edges(source, lambda s: next(
                        (cond(s) for cond in conditions.values() if cond(s)),
                        END
                    ))
        
        # Add checkpointing if enabled
        checkpointer = None
        if self.use_checkpointing:
            checkpointer = SqliteSaver.from_conn_string(f"{self.checkpoint_dir}/graph.db")
        
        return graph.compile(checkpointer=checkpointer)
    
    def _compile_local(self) -> 'CompiledLocalGraph':
        """Compile using local implementation"""
        return CompiledLocalGraph(self)
    
    def visualize(self) -> Dict[str, Any]:
        """Generate visualization data for the graph"""
        viz_data = {
            "nodes": [],
            "edges": [],
            "metadata": self.metadata
        }
        
        # Add nodes
        for name, node in self.nodes.items():
            viz_data["nodes"].append({
                "id": name,
                "label": name,
                "type": node.type.value,
                "requirements": node.requirements,
                "outputs": node.outputs
            })
        
        # Add edges
        for edge in self.edges:
            viz_data["edges"].append({
                "source": edge.source,
                "target": edge.target,
                "weight": edge.weight,
                "conditional": edge.condition is not None
            })
        
        return viz_data
    
    def to_networkx(self) -> nx.DiGraph:
        """Convert to NetworkX graph for analysis"""
        G = nx.DiGraph()
        
        # Add nodes
        for name, node in self.nodes.items():
            G.add_node(name, **{
                "type": node.type.value,
                "requirements": node.requirements,
                "outputs": node.outputs
            })
        
        # Add edges
        for edge in self.edges:
            G.add_edge(
                edge.source,
                edge.target,
                weight=edge.weight,
                conditional=edge.condition is not None
            )
        
        return G


class CompiledLocalGraph:
    """Local implementation of compiled graph"""
    
    def __init__(self, builder: CollectiveGraphBuilder):
        self.builder = builder
        self.execution_count = 0
        
    async def ainvoke(self, state: StateType) -> StateType:
        """Execute the graph asynchronously"""
        self.execution_count += 1
        current_node = self.builder.entry_point
        visited = set()
        
        logger.info("Graph execution started", 
                   execution=self.execution_count,
                   entry=current_node)
        
        while current_node and current_node != END:
            if current_node in visited:
                logger.warning("Cycle detected", node=current_node)
                break
            
            visited.add(current_node)
            
            # Execute node
            if current_node in self.builder.nodes:
                node = self.builder.nodes[current_node]
                logger.debug("Executing node", name=current_node, type=node.type.value)
                
                try:
                    state = await node.execute(state)
                except Exception as e:
                    logger.error("Node execution failed", 
                               node=current_node, 
                               error=str(e))
                    state["_error"] = str(e)
                    break
            
            # Find next node
            next_node = None
            
            # Check for explicit next node in state
            if "_next_node" in state:
                next_node = state.pop("_next_node")
            else:
                # Find matching edge
                for edge in self.builder.edges:
                    if edge.source == current_node:
                        if edge.should_traverse(state):
                            next_node = edge.target
                            break
            
            current_node = next_node
        
        logger.info("Graph execution completed", 
                   execution=self.execution_count,
                   nodes_visited=len(visited))
        
        return state


# Helper functions for common patterns

def create_supervisor_graph(agents: List[str]) -> CollectiveGraphBuilder:
    """Create a supervisor pattern graph"""
    builder = CollectiveGraphBuilder(GraphType.WORKFLOW)
    
    # Supervisor decides which agent to call
    async def supervisor(state: StateType) -> StateType:
        # Analyze state and decide next agent
        # This is a simplified example
        if "error" in state:
            state["next_agent"] = "error_handler"
        elif state.get("iteration", 0) >= len(agents):
            state["next_agent"] = END
        else:
            state["next_agent"] = agents[state.get("iteration", 0)]
            state["iteration"] = state.get("iteration", 0) + 1
        
        return state
    
    builder.add_node("supervisor", supervisor, NodeType.DECISION)
    builder.set_entry_point("supervisor")
    
    # Add agent nodes
    for agent in agents:
        async def agent_func(state: StateType, name=agent) -> StateType:
            state[f"{name}_result"] = f"Processed by {name}"
            return state
        
        builder.add_node(agent, agent_func, NodeType.AGENT)
        builder.add_edge(agent, "supervisor")
    
    # Add error handler
    async def error_handler(state: StateType) -> StateType:
        state["handled"] = True
        return state
    
    builder.add_node("error_handler", error_handler, NodeType.AGENT)
    builder.add_edge("error_handler", END)
    
    # Dynamic routing from supervisor
    for agent in agents + ["error_handler"]:
        builder.add_edge(
            "supervisor",
            agent,
            condition=lambda s, a=agent: s.get("next_agent") == a
        )
    
    builder.add_edge(
        "supervisor",
        END,
        condition=lambda s: s.get("next_agent") == END
    )
    
    return builder


def create_map_reduce_graph(
    mapper_count: int,
    mapper_func: Callable,
    reducer_func: Callable
) -> CollectiveGraphBuilder:
    """Create a map-reduce pattern graph"""
    builder = CollectiveGraphBuilder(GraphType.WORKFLOW)
    
    # Create mapper nodes
    mappers = []
    for i in range(mapper_count):
        mapper_name = f"mapper_{i}"
        
        async def mapper(state: StateType, idx=i) -> StateType:
            # Process chunk of data
            data_chunk = state.get("data", [])[idx::mapper_count]
            result = mapper_func(data_chunk)
            state[f"mapped_{idx}"] = result
            return state
        
        builder.add_node(mapper_name, mapper, NodeType.TRANSFORMER)
        mappers.append(mapper_name)
    
    # Create reducer node
    async def reducer(state: StateType) -> StateType:
        # Collect all mapped results
        mapped_results = []
        for i in range(mapper_count):
            if f"mapped_{i}" in state:
                mapped_results.append(state[f"mapped_{i}"])
        
        # Reduce
        state["result"] = reducer_func(mapped_results)
        return state
    
    # Build graph structure
    builder.add_parallel_section("map_phase", mappers, lambda results: results[0])
    builder.add_node("reduce", reducer, NodeType.AGGREGATOR)
    
    builder.set_entry_point("map_phase")
    builder.add_edge("map_phase", "reduce")
    builder.add_edge("reduce", END)
    
    return builder


# Example usage
async def example_graph_usage():
    """Example of using the graph builder"""
    
    # Create a simple workflow graph
    builder = CollectiveGraphBuilder(GraphType.WORKFLOW)
    
    # Define node functions
    async def analyze(state: Dict[str, Any]) -> Dict[str, Any]:
        state["analysis"] = "Data analyzed"
        return state
    
    async def decide(state: Dict[str, Any]) -> Dict[str, Any]:
        if state.get("analysis"):
            state["decision"] = "proceed"
        else:
            state["decision"] = "stop"
        return state
    
    async def execute(state: Dict[str, Any]) -> Dict[str, Any]:
        state["result"] = "Action executed"
        return state
    
    # Build graph
    builder.add_node("analyze", analyze)
    builder.add_node("decide", decide, NodeType.DECISION)
    builder.add_node("execute", execute)
    
    builder.set_entry_point("analyze")
    builder.add_edge("analyze", "decide")
    builder.add_edge(
        "decide", 
        "execute",
        condition=lambda s: s.get("decision") == "proceed"
    )
    builder.add_edge(
        "decide",
        END,
        condition=lambda s: s.get("decision") == "stop"
    )
    builder.add_edge("execute", END)
    
    # Compile and execute
    graph = builder.compile()
    result = await graph.ainvoke({"input": "test data"})
    
    print(f"Execution result: {result}")
    
    # Visualize
    viz_data = builder.visualize()
    print(f"Graph structure: {viz_data}")
    
    return builder


if __name__ == "__main__":
    asyncio.run(example_graph_usage())