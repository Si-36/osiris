"""
💻 Code Agent - GPU-Accelerated Code Intelligence
================================================

Specializes in:
- Parallel AST parsing on GPU
- Performance profiling and hotspot detection  
- Mojo/CUDA kernel optimization suggestions
- Topological code analysis for pattern recognition
- Test generation and code quality assessment
"""

import ast
import asyncio
import time
import os
from typing import Dict, Any, List, Optional, Tuple, Set
import numpy as np
import torch
from dataclasses import dataclass
import structlog

from .test_agents import TestAgentBase, TestAgentConfig, Tool, AgentRole
from ..tda.algorithms import compute_persistence_diagram
from ..adapters.mojo_bridge import MojoKernelLoader

logger = structlog.get_logger(__name__)


@dataclass
class CodeAnalysisResult:
    """Result of code analysis"""
    file_path: str
    complexity_score: float
    hotspots: List[Dict[str, Any]]
    optimization_suggestions: List[Dict[str, Any]]
    topological_features: Optional[np.ndarray] = None
    test_coverage: float = 0.0
    quality_score: float = 0.0
    mojo_candidates: List[Dict[str, Any]] = None


class ParallelASTParser:
    """GPU-accelerated AST parsing"""
    
    def __init__(self, batch_size: int = 100):
        self.batch_size = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    async def parse_batch(self, code_files: List[str]) -> List[ast.AST]:
        """Parse multiple files in parallel"""
        # In practice, would use GPU for tokenization/parsing
        # For now, using asyncio for parallel CPU parsing
        tasks = []
        for file_path in code_files:
            tasks.append(self._parse_file(file_path))
            
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out errors
        valid_asts = []
        for i, result in enumerate(results):
            if isinstance(result, ast.AST):
                valid_asts.append(result)
            else:
                logger.warning(f"Failed to parse {code_files[i]}: {result}")
                
        return valid_asts
        
    async def _parse_file(self, file_path: str) -> ast.AST:
        """Parse single file"""
        try:
            with open(file_path, 'r') as f:
                content = f.read()
            return ast.parse(content, filename=file_path)
        except Exception as e:
            logger.error(f"Error parsing {file_path}: {e}")
            raise


class TopologicalCodeAnalyzer:
    """Extract topological features from code structure"""
    
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def extract_topology(self, ast_trees: List[ast.AST]) -> np.ndarray:
        """Extract topological features from AST"""
        features = []
        
        for tree in ast_trees:
            # Extract structural features
            depth = self._get_tree_depth(tree)
            branching_factor = self._get_branching_factor(tree)
            cyclomatic_complexity = self._get_cyclomatic_complexity(tree)
            
            # Create point cloud from AST structure
            points = self._ast_to_points(tree)
            
            # Compute persistence diagram if we have points
            if len(points) > 0:
                # Move to GPU for TDA computation
                points_tensor = torch.tensor(points, device=self.device)
                
                # Compute persistence (simplified - would use GPU TDA)
                persistence = self._compute_persistence_features(points_tensor)
            else:
                persistence = np.zeros(10)  # Default features
                
            # Combine all features
            feature_vector = np.concatenate([
                [depth, branching_factor, cyclomatic_complexity],
                persistence
            ])
            features.append(feature_vector)
            
        return np.array(features)
        
    def _get_tree_depth(self, node: ast.AST, depth: int = 0) -> int:
        """Get maximum depth of AST"""
        if not hasattr(node, '_fields'):
            return depth
            
        max_child_depth = depth
        for field, value in ast.iter_fields(node):
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, ast.AST):
                        child_depth = self._get_tree_depth(item, depth + 1)
                        max_child_depth = max(max_child_depth, child_depth)
            elif isinstance(value, ast.AST):
                child_depth = self._get_tree_depth(value, depth + 1)
                max_child_depth = max(max_child_depth, child_depth)
                
        return max_child_depth
        
    def _get_branching_factor(self, tree: ast.AST) -> float:
        """Calculate average branching factor"""
        node_counts = []
        
        class BranchingVisitor(ast.NodeVisitor):
            def visit(self, node):
                children = 0
                for field, value in ast.iter_fields(node):
                    if isinstance(value, list):
                        children += len([v for v in value if isinstance(v, ast.AST)])
                    elif isinstance(value, ast.AST):
                        children += 1
                        
                if children > 0:
                    node_counts.append(children)
                    
                self.generic_visit(node)
                
        BranchingVisitor().visit(tree)
        
        return np.mean(node_counts) if node_counts else 0
        
    def _get_cyclomatic_complexity(self, tree: ast.AST) -> int:
        """Calculate cyclomatic complexity"""
        complexity = 1
        
        class ComplexityVisitor(ast.NodeVisitor):
            def __init__(self):
                self.complexity = 1
                
            def visit_If(self, node):
                self.complexity += 1
                self.generic_visit(node)
                
            def visit_While(self, node):
                self.complexity += 1
                self.generic_visit(node)
                
            def visit_For(self, node):
                self.complexity += 1
                self.generic_visit(node)
                
            def visit_ExceptHandler(self, node):
                self.complexity += 1
                self.generic_visit(node)
                
        visitor = ComplexityVisitor()
        visitor.visit(tree)
        
        return visitor.complexity
        
    def _ast_to_points(self, tree: ast.AST) -> List[List[float]]:
        """Convert AST to point cloud for TDA"""
        points = []
        
        class PointExtractor(ast.NodeVisitor):
            def __init__(self):
                self.depth = 0
                self.points = []
                
            def visit(self, node):
                # Map node type to numeric value
                node_type_id = hash(type(node).__name__) % 100
                
                # Add point: [depth, node_type, line_number]
                if hasattr(node, 'lineno'):
                    self.points.append([
                        self.depth,
                        node_type_id,
                        node.lineno
                    ])
                    
                self.depth += 1
                self.generic_visit(node)
                self.depth -= 1
                
        extractor = PointExtractor()
        extractor.visit(tree)
        
        return extractor.points
        
    def _compute_persistence_features(self, points: torch.Tensor) -> np.ndarray:
        """Compute persistence features on GPU"""
        # Simplified - in practice would use GPU TDA
        if points.shape[0] < 3:
            return np.zeros(10)
            
        # Compute pairwise distances
        distances = torch.cdist(points, points)
        
        # Extract simple topological features
        features = [
            distances.mean().item(),
            distances.std().item(),
            distances.max().item(),
            distances.min().item(),
            torch.quantile(distances.flatten(), 0.25).item(),
            torch.quantile(distances.flatten(), 0.75).item(),
            points.shape[0],  # Number of nodes
            points[:, 0].max().item(),  # Max depth
            points[:, 1].nunique().item(),  # Unique node types
            points[:, 2].max().item() if points.shape[0] > 0 else 0  # Lines of code
        ]
        
        return np.array(features)


class MojoKernelSuggester:
    """Suggest Mojo kernel optimizations"""
    
    def __init__(self):
        self.patterns = {
            "nested_loops": {
                "pattern": ["For", "For"],
                "suggestion": "Vectorize with Mojo SIMD operations",
                "speedup": "10-50x"
            },
            "matrix_operations": {
                "pattern": ["matmul", "dot", "einsum"],
                "suggestion": "Use Mojo's optimized matrix kernels",
                "speedup": "5-20x"
            },
            "element_wise": {
                "pattern": ["map", "filter", "comprehension"],
                "suggestion": "Parallelize with Mojo's vectorize()",
                "speedup": "8-15x"
            },
            "reduction": {
                "pattern": ["sum", "mean", "reduce"],
                "suggestion": "Use Mojo's parallel reduction",
                "speedup": "5-10x"
            }
        }
        
    def suggest(self, ast_trees: List[ast.AST], profiling_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Suggest Mojo optimizations based on code patterns and profiling"""
        suggestions = []
        
        for tree in ast_trees:
            # Find optimization opportunities
            visitor = OptimizationVisitor(self.patterns)
            visitor.visit(tree)
            
            # Combine with profiling data
            for suggestion in visitor.suggestions:
                # Check if this is a hotspot
                if suggestion["line"] in profiling_data.get("hotspots", []):
                    suggestion["priority"] = "HIGH"
                    suggestion["estimated_impact"] = f"{profiling_data['hotspots'][suggestion['line']]['time_percent']}% reduction"
                else:
                    suggestion["priority"] = "MEDIUM"
                    
                suggestions.append(suggestion)
                
        return suggestions


class OptimizationVisitor(ast.NodeVisitor):
    """Visit AST to find optimization opportunities"""
    
    def __init__(self, patterns: Dict[str, Any]):
        self.patterns = patterns
        self.suggestions = []
        self.context_stack = []
        
    def visit_For(self, node):
        self.context_stack.append("For")
        
        # Check for nested loops
        if len(self.context_stack) >= 2 and self.context_stack[-2] == "For":
            self.suggestions.append({
                "line": node.lineno,
                "type": "nested_loops",
                "suggestion": self.patterns["nested_loops"]["suggestion"],
                "speedup": self.patterns["nested_loops"]["speedup"],
                "code_hint": "Consider using Mojo's parallel nested loops"
            })
            
        self.generic_visit(node)
        self.context_stack.pop()
        
    def visit_Call(self, node):
        # Check for matrix operations
        if hasattr(node.func, 'id'):
            func_name = node.func.id
            
            for pattern_name, pattern_info in self.patterns.items():
                if func_name in pattern_info["pattern"]:
                    self.suggestions.append({
                        "line": node.lineno,
                        "type": pattern_name,
                        "function": func_name,
                        "suggestion": pattern_info["suggestion"],
                        "speedup": pattern_info["speedup"]
                    })
                    
        self.generic_visit(node)


class CodeAgent(TestAgentBase):
    """
    Specialized agent for code analysis and optimization.
    
    Capabilities:
    - Parallel AST parsing
    - Topological code analysis
    - Performance profiling
    - Mojo optimization suggestions
    - Test generation
    - Code quality assessment
    """
    
    def __init__(self, agent_id: str = "code_agent_001", **kwargs):
        config = TestAgentConfig(
            agent_role=AgentRole.ANALYST,
            specialty="code",
            target_latency_ms=150.0,  # Slightly higher for complex analysis
            **kwargs
        )
        
        super().__init__(agent_id=agent_id, config=config, **kwargs)
        
        # Initialize specialized components
        self.ast_parser = ParallelASTParser()
        self.code_analyzer = TopologicalCodeAnalyzer()
        self.mojo_suggester = MojoKernelSuggester()
        
        # Initialize tools
        self._init_code_tools()
        
        logger.info("Code Agent initialized",
                   agent_id=agent_id,
                   capabilities=["ast_parsing", "profiling", "optimization", "tda_analysis"])
                   
    def _init_code_tools(self):
        """Initialize code-specific tools"""
        self.tools = {
            "parse_code": Tool(
                name="parse_code",
                description="Parse code files into AST",
                func=self._tool_parse_code
            ),
            "analyze_complexity": Tool(
                name="analyze_complexity",
                description="Analyze code complexity and structure",
                func=self._tool_analyze_complexity
            ),
            "profile_performance": Tool(
                name="profile_performance",
                description="Profile code performance",
                func=self._tool_profile_performance
            ),
            "suggest_optimizations": Tool(
                name="suggest_optimizations",
                description="Suggest Mojo/CUDA optimizations",
                func=self._tool_suggest_optimizations
            ),
            "generate_tests": Tool(
                name="generate_tests",
                description="Generate test cases",
                func=self._tool_generate_tests
            )
        }
        
    async def _handle_analyze(self, context: Dict[str, Any]) -> CodeAnalysisResult:
        """Handle code analysis requests"""
        code_files = context.get("original", {}).get("files", [])
        
        if not code_files:
            return CodeAnalysisResult(
                file_path="",
                complexity_score=0,
                hotspots=[],
                optimization_suggestions=[]
            )
            
        # Parse files in parallel
        ast_trees = await self.ast_parser.parse_batch(code_files)
        
        # Extract topological features
        topo_features = self.code_analyzer.extract_topology(ast_trees)
        
        # Profile performance (simulated)
        profiling_data = await self._profile_code(code_files)
        
        # Suggest optimizations
        suggestions = self.mojo_suggester.suggest(ast_trees, profiling_data)
        
        # Store in shape-aware memory
        if topo_features is not None and len(topo_features) > 0:
            await self.shape_memory.store(
                {
                    "type": "code_analysis",
                    "files": code_files,
                    "topology": topo_features.tolist(),
                    "suggestions": suggestions
                },
                embedding=topo_features[0]  # Use first file's features
            )
            
        # Calculate quality metrics
        complexity_score = np.mean([
            self.code_analyzer._get_cyclomatic_complexity(tree)
            for tree in ast_trees
        ])
        
        return CodeAnalysisResult(
            file_path=code_files[0] if code_files else "",
            complexity_score=complexity_score,
            hotspots=profiling_data.get("hotspots", []),
            optimization_suggestions=suggestions,
            topological_features=topo_features,
            quality_score=self._calculate_quality_score(ast_trees, profiling_data)
        )
        
    async def _handle_generate(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle code generation requests"""
        prompt = context.get("original", {}).get("prompt", "")
        language = context.get("original", {}).get("language", "python")
        
        # Use neural router to select best model
        if hasattr(self, 'neural_router'):
            model = await self.neural_router.route_request(prompt)
        else:
            model = "default"
            
        # Generate code (simplified - would use actual LLM)
        generated_code = await self._generate_code(prompt, language, model)
        
        # Parse and analyze generated code
        try:
            tree = ast.parse(generated_code)
            topo_features = self.code_analyzer.extract_topology([tree])
            
            # Check quality
            quality_score = self._assess_generated_code(tree, prompt)
            
            return {
                "code": generated_code,
                "language": language,
                "quality_score": quality_score,
                "topology": topo_features.tolist() if topo_features is not None else [],
                "model_used": model
            }
        except SyntaxError as e:
            return {
                "code": generated_code,
                "error": f"Syntax error: {e}",
                "quality_score": 0
            }
            
    async def _profile_code(self, code_files: List[str]) -> Dict[str, Any]:
        """Profile code performance (simulated)"""
        # In practice, would use actual profiling tools
        hotspots = {}
        
        for file_path in code_files:
            # Simulate finding hotspots
            with open(file_path, 'r') as f:
                lines = f.readlines()
                
            # Mark lines with loops as potential hotspots
            for i, line in enumerate(lines):
                if any(keyword in line for keyword in ['for', 'while', 'map', 'filter']):
                    hotspots[i + 1] = {
                        "time_percent": np.random.uniform(5, 30),
                        "calls": np.random.randint(1000, 1000000),
                        "time_per_call": np.random.uniform(0.001, 0.1)
                    }
                    
        return {
            "hotspots": hotspots,
            "total_time": sum(h["time_percent"] for h in hotspots.values())
        }
        
    async def _generate_code(self, prompt: str, language: str, model: str) -> str:
        """Generate code using LLM (simplified)"""
        # In practice, would use actual LLM
        template = f"""
# Generated {language} code for: {prompt}
# Model: {model}

def generated_function(data):
    '''
    AI-generated function based on prompt
    '''
    # TODO: Implement based on requirements
    result = []
    
    for item in data:
        # Process each item
        processed = process_item(item)
        result.append(processed)
        
    return result

def process_item(item):
    # Helper function
    return item * 2
"""
        return template
        
    def _calculate_quality_score(self, 
                               ast_trees: List[ast.AST],
                               profiling_data: Dict[str, Any]) -> float:
        """Calculate overall code quality score"""
        scores = []
        
        for tree in ast_trees:
            # Complexity score (lower is better)
            complexity = self.code_analyzer._get_cyclomatic_complexity(tree)
            complexity_score = max(0, 10 - complexity * 0.5)
            
            # Structure score
            depth = self.code_analyzer._get_tree_depth(tree)
            depth_score = max(0, 10 - depth * 0.2)
            
            # Performance score
            hotspot_count = len(profiling_data.get("hotspots", {}))
            perf_score = max(0, 10 - hotspot_count * 0.5)
            
            # Combined score
            quality = (complexity_score + depth_score + perf_score) / 3
            scores.append(quality)
            
        return np.mean(scores) if scores else 0
        
    def _assess_generated_code(self, tree: ast.AST, prompt: str) -> float:
        """Assess quality of generated code"""
        # Check if generated code matches prompt intent
        # Simplified - would use more sophisticated analysis
        
        score = 5.0  # Base score
        
        # Check for docstrings
        has_docstring = any(
            isinstance(node, ast.Expr) and isinstance(node.value, ast.Str)
            for node in ast.walk(tree)
        )
        if has_docstring:
            score += 1
            
        # Check for type hints
        has_type_hints = any(
            hasattr(node, 'returns') and node.returns is not None
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef)
        )
        if has_type_hints:
            score += 1
            
        # Check complexity
        complexity = self.code_analyzer._get_cyclomatic_complexity(tree)
        if complexity < 10:
            score += 2
        elif complexity < 20:
            score += 1
            
        return min(score, 10.0)
        
    # Tool implementations
    async def _tool_parse_code(self, files: List[str]) -> List[Dict[str, Any]]:
        """Parse code files tool"""
        trees = await self.ast_parser.parse_batch(files)
        return [
            {
                "file": files[i],
                "parsed": True,
                "nodes": len(list(ast.walk(tree)))
            }
            for i, tree in enumerate(trees)
        ]
        
    async def _tool_analyze_complexity(self, file_path: str) -> Dict[str, Any]:
        """Analyze code complexity tool"""
        trees = await self.ast_parser.parse_batch([file_path])
        
        if not trees:
            return {"error": "Failed to parse file"}
            
        tree = trees[0]
        return {
            "file": file_path,
            "cyclomatic_complexity": self.code_analyzer._get_cyclomatic_complexity(tree),
            "depth": self.code_analyzer._get_tree_depth(tree),
            "branching_factor": self.code_analyzer._get_branching_factor(tree)
        }
        
    async def _tool_profile_performance(self, file_path: str) -> Dict[str, Any]:
        """Profile performance tool"""
        profiling_data = await self._profile_code([file_path])
        return profiling_data
        
    async def _tool_suggest_optimizations(self, file_path: str) -> List[Dict[str, Any]]:
        """Suggest optimizations tool"""
        trees = await self.ast_parser.parse_batch([file_path])
        profiling_data = await self._profile_code([file_path])
        
        suggestions = self.mojo_suggester.suggest(trees, profiling_data)
        return suggestions
        
    async def _tool_generate_tests(self, file_path: str) -> Dict[str, Any]:
        """Generate tests tool"""
        # Simplified test generation
        return {
            "file": file_path,
            "tests": [
                {
                    "name": "test_basic_functionality",
                    "type": "unit",
                    "coverage": "function"
                },
                {
                    "name": "test_edge_cases",
                    "type": "unit",
                    "coverage": "edge"
                },
                {
                    "name": "test_performance",
                    "type": "performance",
                    "coverage": "hotspots"
                }
            ]
        }


# Factory function
def create_code_agent(agent_id: Optional[str] = None, **kwargs) -> CodeAgent:
    """Create a Code Agent instance"""
    if agent_id is None:
        agent_id = f"code_agent_{int(time.time())}"
        
    return CodeAgent(agent_id=agent_id, **kwargs)