"""
Enhanced Code Agent with Causal Persistence
==========================================
Code agent that remembers why it made decisions
"""

from typing import Dict, Any, List, Optional
import time
import os
import ast
import structlog

from .test_code_agent import AdvancedCodeAgent, CodeAnalysisResult
from .persistence_mixin import PersistenceMixin

logger = structlog.get_logger(__name__)

class EnhancedCodeAgent(PersistenceMixin, AdvancedCodeAgent):
    """Code agent with causal memory"""
    
    def __init__(self, agent_id: str = "enhanced_code_agent", **kwargs):
        self.agent_id = agent_id
        super().__init__(**kwargs)
        logger.info(f"Enhanced Code Agent initialized with persistence", agent_id=agent_id)
    
    async def analyze_code(self, 
                          file_path: str,
                          context: Optional[Dict[str, Any]] = None) -> CodeAnalysisResult:
        """Analyze code with causal tracking"""
        # Load previous analysis if exists
        memory = await self.load_memory(
            compute_fn=lambda m: self._enhance_with_code_context(m, file_path)
        )
        
        # Perform analysis
        start_time = time.time()
        result = await super().analyze_code(file_path, context)
        analysis_time = time.time() - start_time
        
        # Save decision with causality
        await self.save_decision(
            decision="code_analysis",
            context={
                "file_path": file_path,
                "complexity_score": result.complexity_score,
                "hotspots_found": len(result.hotspots),
                "optimization_suggestions": len(result.optimization_suggestions),
                "analysis_time": analysis_time,
                "previous_analysis": memory is not None,
                "quality_score": result.quality_score,
                "mojo_candidates": len(result.mojo_candidates or [])
            },
            confidence=result.quality_score
        )
        
        # If significant issues found, create experimental branch
        if result.complexity_score > 0.8 or len(result.hotspots) > 5:
            branch_id = await self.create_experiment_branch(
                f"refactor_{os.path.basename(file_path)}"
            )
            logger.info(f"Created refactoring branch due to high complexity",
                       branch_id=branch_id,
                       complexity=result.complexity_score)
        
        return result
    
    async def suggest_optimizations(self, 
                                   code: str,
                                   performance_data: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Suggest optimizations with causal reasoning"""
        # Track why we're suggesting optimizations
        causes = []
        if performance_data:
            if performance_data.get("latency_ms", 0) > 100:
                causes.append("high_latency")
            if performance_data.get("cpu_usage", 0) > 0.8:
                causes.append("high_cpu_usage")
            if performance_data.get("memory_mb", 0) > 1000:
                causes.append("high_memory_usage")
        
        # Get suggestions
        suggestions = await super().suggest_optimizations(code, performance_data)
        
        # Save with causality
        await self.save_decision(
            decision="optimization_suggestions",
            context={
                "num_suggestions": len(suggestions),
                "performance_data": performance_data,
                "causes": causes,
                "code_length": len(code),
                "suggestion_types": [s.get("type", "unknown") for s in suggestions]
            },
            confidence=0.8 if suggestions else 0.3
        )
        
        return suggestions
    
    async def review_with_memory(self, 
                                code: str,
                                review_type: str = "general") -> Dict[str, Any]:
        """Review code using historical context"""
        # Load memory of similar code reviews
        memory = await self.load_memory(
            compute_fn=lambda m: self._filter_similar_reviews(m, review_type)
        )
        
        # Perform review
        review_result = {
            "review_type": review_type,
            "issues_found": [],
            "suggestions": [],
            "quality_score": 0.0,
            "historical_context": memory is not None
        }
        
        # Use memory to inform review
        if memory and "previous_issues" in memory:
            # Check if previous issues still exist
            for issue in memory["previous_issues"]:
                if self._check_issue_exists(code, issue):
                    review_result["issues_found"].append({
                        **issue,
                        "recurring": True,
                        "first_seen": memory.get("timestamp")
                    })
        
        # Standard review process
        ast_tree = ast.parse(code)
        review_result["issues_found"].extend(self._analyze_ast_issues(ast_tree))
        review_result["quality_score"] = self._calculate_quality_score(review_result)
        
        # Save review decision
        await self.save_decision(
            decision=f"code_review_{review_type}",
            context={
                "issues_found": len(review_result["issues_found"]),
                "recurring_issues": sum(1 for i in review_result["issues_found"] if i.get("recurring")),
                "quality_score": review_result["quality_score"],
                "used_memory": memory is not None,
                "code_size": len(code)
            },
            confidence=review_result["quality_score"]
        )
        
        return review_result
    
    async def learn_from_feedback(self, 
                                 decision_id: str,
                                 feedback: Dict[str, Any]) -> None:
        """Learn from feedback on previous decisions"""
        # Get the causal chain for the decision
        chain = await self.get_decision_chain(decision_id)
        
        if chain:
            # Analyze what led to this decision
            causes = set()
            for link in chain:
                causes.update(link.get("causes", []))
            
            # Save learning
            await self.save_decision(
                decision="feedback_learning",
                context={
                    "original_decision": decision_id,
                    "feedback": feedback,
                    "causal_factors": list(causes),
                    "feedback_score": feedback.get("score", 0),
                    "should_adjust": feedback.get("score", 0) < 0.5
                },
                confidence=0.9
            )
            
            logger.info("Learned from feedback",
                       decision_id=decision_id,
                       feedback_score=feedback.get("score", 0),
                       causal_factors=len(causes))
    
    def _enhance_with_code_context(self, memory: Optional[Dict[str, Any]], file_path: str) -> Optional[Dict[str, Any]]:
        """Enhance memory with current code context"""
        if not memory:
            return None
        
        # Add current context
        memory["current_file"] = file_path
        memory["file_similarity"] = self._calculate_file_similarity(
            memory.get("file_path", ""),
            file_path
        )
        
        return memory
    
    def _filter_similar_reviews(self, memory: Optional[Dict[str, Any]], review_type: str) -> Optional[Dict[str, Any]]:
        """Filter memory for similar code reviews"""
        if not memory or memory.get("decision") != f"code_review_{review_type}":
            return None
        
        return memory
    
    def _check_issue_exists(self, code: str, issue: Dict[str, Any]) -> bool:
        """Check if an issue still exists in code"""
        # Simple check - in practice would be more sophisticated
        return issue.get("pattern", "") in code
    
    def _analyze_ast_issues(self, ast_tree: ast.AST) -> List[Dict[str, Any]]:
        """Analyze AST for issues"""
        issues = []
        
        # Check for common issues
        for node in ast.walk(ast_tree):
            if isinstance(node, ast.FunctionDef):
                # Check function complexity
                if self._calculate_cyclomatic_complexity(node) > 10:
                    issues.append({
                        "type": "high_complexity",
                        "location": f"function:{node.name}",
                        "severity": "medium",
                        "pattern": f"def {node.name}"
                    })
        
        return issues
    
    def _calculate_cyclomatic_complexity(self, node: ast.FunctionDef) -> int:
        """Calculate cyclomatic complexity of a function"""
        complexity = 1  # Base complexity
        
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For)):
                complexity += 1
            elif isinstance(child, ast.BoolOp):
                complexity += len(child.values) - 1
        
        return complexity
    
    def _calculate_quality_score(self, review_result: Dict[str, Any]) -> float:
        """Calculate code quality score"""
        issues = review_result["issues_found"]
        
        if not issues:
            return 1.0
        
        # Deduct points for issues
        score = 1.0
        for issue in issues:
            severity_penalty = {
                "critical": 0.3,
                "high": 0.2,
                "medium": 0.1,
                "low": 0.05
            }
            score -= severity_penalty.get(issue.get("severity", "low"), 0.05)
        
        # Extra penalty for recurring issues
        recurring = sum(1 for i in issues if i.get("recurring"))
        score -= recurring * 0.1
        
        return max(0.0, score)
    
    def _calculate_file_similarity(self, file1: str, file2: str) -> float:
        """Calculate similarity between file paths"""
        if file1 == file2:
            return 1.0
        
        # Check if same directory
        if os.path.dirname(file1) == os.path.dirname(file2):
            return 0.5
        
        # Check if same extension
        if os.path.splitext(file1)[1] == os.path.splitext(file2)[1]:
            return 0.3
        
        return 0.0


def create_enhanced_code_agent(agent_id: str = "enhanced_code_agent") -> EnhancedCodeAgent:
    """Create an enhanced code agent with persistence"""
    return EnhancedCodeAgent(agent_id=agent_id)