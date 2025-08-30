"""
Conditional Flows - Dynamic Branching Based on Metrics
=====================================================
Implements runtime branching based on free-energy SLOs
Based on Bytebase 2025 recommendations
"""

import asyncio
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass, field
import time
from enum import Enum
import operator

# AURA imports
from ...components.registry import get_registry
from ...observability.prometheus_client import MetricsCollector

import logging
logger = logging.getLogger(__name__)


class ConditionOperator(Enum):
    """Comparison operators for conditions"""
    EQ = "eq"
    NE = "ne"
    GT = "gt"
    GE = "ge"
    LT = "lt"
    LE = "le"
    IN = "in"
    NOT_IN = "not_in"
    CONTAINS = "contains"
    REGEX = "regex"


@dataclass
class BranchCondition:
    """Single condition for branching"""
    metric_name: str
    operator: ConditionOperator
    threshold: Union[float, str, List[Any]]
    
    # Optional aggregation
    aggregation: str = "last"  # last, mean, max, min, sum
    window_seconds: int = 60
    
    def evaluate(self, metrics: Dict[str, Any]) -> bool:
        """Evaluate condition against metrics"""
        value = metrics.get(self.metric_name)
        
        if value is None:
            logger.warning(f"Metric {self.metric_name} not found")
            return False
            
        # Apply operator
        op_map = {
            ConditionOperator.EQ: operator.eq,
            ConditionOperator.NE: operator.ne,
            ConditionOperator.GT: operator.gt,
            ConditionOperator.GE: operator.ge,
            ConditionOperator.LT: operator.lt,
            ConditionOperator.LE: operator.le,
        }
        
        if self.operator in op_map:
            return op_map[self.operator](value, self.threshold)
        elif self.operator == ConditionOperator.IN:
            return value in self.threshold
        elif self.operator == ConditionOperator.NOT_IN:
            return value not in self.threshold
        elif self.operator == ConditionOperator.CONTAINS:
            return self.threshold in str(value)
        elif self.operator == ConditionOperator.REGEX:
            import re
            return bool(re.match(self.threshold, str(value)))
        else:
            return False


@dataclass
class ConditionalBranch:
    """Branch definition with conditions"""
    branch_id: str
    conditions: List[BranchCondition]
    target: str  # Target phase/node
    
    # Logical operator for multiple conditions
    logic: str = "and"  # and, or
    
    # Optional metadata
    priority: int = 0
    description: str = ""
    
    def should_branch(self, metrics: Dict[str, Any]) -> bool:
        """Check if should take this branch"""
        results = [cond.evaluate(metrics) for cond in self.conditions]
        
        if self.logic == "and":
            return all(results)
        elif self.logic == "or":
            return any(results)
        else:
            return False


class ConditionalFlow:
    """
    Manages conditional workflow branching
    Integrates with Prometheus metrics for real-time decisions
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or self._default_config()
        
        # Branch registry
        self.branches: Dict[str, List[ConditionalBranch]] = {}
        
        # Metrics collector
        self.metrics_collector = MetricsCollector()
        
        # Branch history
        self.branch_history = []
        
        # Dynamic rules from feature flags
        self.dynamic_rules: Dict[str, Callable] = {}
        
        logger.info("Conditional Flow manager initialized")
        
    def _default_config(self) -> Dict[str, Any]:
        return {
            'max_branches_per_node': 10,
            'default_branch': 'continue',
            'metrics_window': 300,  # 5 minutes
            'enable_dynamic_rules': True
        }
        
    def register_branch(self, 
                       source_node: str,
                       branch: ConditionalBranch):
        """Register a conditional branch"""
        if source_node not in self.branches:
            self.branches[source_node] = []
            
        # Check limit
        if len(self.branches[source_node]) >= self.config['max_branches_per_node']:
            raise ValueError(f"Too many branches for node {source_node}")
            
        self.branches[source_node].append(branch)
        
        # Sort by priority
        self.branches[source_node].sort(key=lambda b: b.priority, reverse=True)
        
        logger.debug(f"Registered branch {branch.branch_id} for {source_node}")
        
    def register_dynamic_rule(self,
                            rule_id: str,
                            evaluator: Callable[[Dict[str, Any]], Optional[str]]):
        """Register dynamic branching rule"""
        self.dynamic_rules[rule_id] = evaluator
        logger.info(f"Registered dynamic rule {rule_id}")
        
    async def evaluate_branches(self,
                              source_node: str,
                              context: Dict[str, Any]) -> str:
        """
        Evaluate conditional branches for a node
        Returns target node/phase
        """
        # Collect current metrics
        metrics = await self._collect_metrics(context)
        
        # Check dynamic rules first
        if self.config['enable_dynamic_rules']:
            for rule_id, evaluator in self.dynamic_rules.items():
                try:
                    target = evaluator(metrics)
                    if target:
                        self._record_branch(source_node, target, rule_id, metrics)
                        return target
                except Exception as e:
                    logger.error(f"Dynamic rule {rule_id} failed: {e}")
                    
        # Check registered branches
        if source_node in self.branches:
            for branch in self.branches[source_node]:
                if branch.should_branch(metrics):
                    self._record_branch(source_node, branch.target, 
                                      branch.branch_id, metrics)
                    return branch.target
                    
        # Default branch
        default = self.config['default_branch']
        self._record_branch(source_node, default, 'default', metrics)
        return default
        
    async def _collect_metrics(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Collect metrics for branching decisions"""
        metrics = {}
        
        # Get workflow metrics
        if 'workflow_id' in context:
            workflow_metrics = await self.metrics_collector.get_workflow_metrics(
                context['workflow_id']
            )
            metrics.update(workflow_metrics)
            
        # Get phase results
        if 'phase_results' in context:
            for phase, result in context['phase_results'].items():
                if isinstance(result, dict):
                    # Extract key metrics
                    metrics[f'{phase}_success'] = result.get('success', False)
                    metrics[f'{phase}_duration_ms'] = result.get('duration_ms', 0)
                    metrics[f'{phase}_confidence'] = result.get('confidence', 0)
                    
                    # Extract custom metrics
                    if 'metrics' in result:
                        for k, v in result['metrics'].items():
                            metrics[f'{phase}_{k}'] = v
                            
        # Get system metrics
        system_metrics = await self._get_system_metrics()
        metrics.update(system_metrics)
        
        # Add context metadata
        metrics['timestamp'] = time.time()
        metrics['priority'] = context.get('priority', 'normal')
        
        return metrics
        
    async def _get_system_metrics(self) -> Dict[str, Any]:
        """Get system-level metrics"""
        # Would integrate with actual monitoring
        return {
            'cpu_usage': 45.0,
            'memory_usage': 60.0,
            'active_workflows': 12,
            'queue_depth': 5
        }
        
    def _record_branch(self,
                      source: str,
                      target: str,
                      branch_id: str,
                      metrics: Dict[str, Any]):
        """Record branching decision"""
        record = {
            'timestamp': time.time(),
            'source': source,
            'target': target,
            'branch_id': branch_id,
            'metrics_snapshot': metrics.copy()
        }
        
        self.branch_history.append(record)
        
        # Emit metrics
        self.metrics_collector.increment(
            'workflow_branches_total',
            labels={'source': source, 'target': target, 'branch': branch_id}
        )
        
    def create_free_energy_branch(self,
                                threshold: float = 1.0) -> ConditionalBranch:
        """Create branch based on free energy threshold"""
        return ConditionalBranch(
            branch_id="high_free_energy",
            conditions=[
                BranchCondition(
                    metric_name="inference_free_energy",
                    operator=ConditionOperator.GT,
                    threshold=threshold
                )
            ],
            target="fallback_inference",
            description="Branch to fallback when free energy is high"
        )
        
    def create_consensus_timeout_branch(self,
                                      timeout_ms: float = 100.0) -> ConditionalBranch:
        """Create branch for consensus timeout"""
        return ConditionalBranch(
            branch_id="consensus_timeout", 
            conditions=[
                BranchCondition(
                    metric_name="consensus_duration_ms",
                    operator=ConditionOperator.GT,
                    threshold=timeout_ms
                )
            ],
            target="fast_consensus",
            description="Use fast consensus when slow"
        )
        
    def create_load_based_branch(self,
                               cpu_threshold: float = 80.0) -> ConditionalBranch:
        """Create branch based on system load"""
        return ConditionalBranch(
            branch_id="high_load",
            conditions=[
                BranchCondition(
                    metric_name="cpu_usage",
                    operator=ConditionOperator.GT,
                    threshold=cpu_threshold
                ),
                BranchCondition(
                    metric_name="queue_depth",
                    operator=ConditionOperator.GT,
                    threshold=10
                )
            ],
            logic="or",
            target="lightweight_processing",
            description="Use lightweight processing under high load"
        )
        
    def create_confidence_branch(self,
                               min_confidence: float = 0.8) -> ConditionalBranch:
        """Create branch based on confidence levels"""
        return ConditionalBranch(
            branch_id="low_confidence",
            conditions=[
                BranchCondition(
                    metric_name="inference_confidence",
                    operator=ConditionOperator.LT,
                    threshold=min_confidence
                )
            ],
            target="enhanced_analysis",
            description="Use enhanced analysis for low confidence"
        )
        
    def get_branch_analytics(self, 
                           time_window: Optional[int] = None) -> Dict[str, Any]:
        """Get analytics on branching patterns"""
        if time_window:
            cutoff = time.time() - time_window
            history = [r for r in self.branch_history if r['timestamp'] > cutoff]
        else:
            history = self.branch_history
            
        if not history:
            return {'total_branches': 0}
            
        # Analyze branching patterns
        branch_counts = {}
        path_counts = {}
        
        for record in history:
            # Count branches
            branch_id = record['branch_id']
            branch_counts[branch_id] = branch_counts.get(branch_id, 0) + 1
            
            # Count paths
            path = f"{record['source']}->{record['target']}"
            path_counts[path] = path_counts.get(path, 0) + 1
            
        # Find most common patterns
        most_common_branch = max(branch_counts.items(), key=lambda x: x[1])
        most_common_path = max(path_counts.items(), key=lambda x: x[1])
        
        return {
            'total_branches': len(history),
            'unique_branches': len(branch_counts),
            'unique_paths': len(path_counts),
            'most_common_branch': most_common_branch,
            'most_common_path': most_common_path,
            'branch_distribution': branch_counts,
            'path_distribution': path_counts
        }


# Example usage patterns
def create_adaptive_workflow_branches():
    """Create standard adaptive workflow branches"""
    flow = ConditionalFlow()
    
    # After perception phase
    flow.register_branch(
        "perception",
        ConditionalBranch(
            branch_id="skip_inference",
            conditions=[
                BranchCondition(
                    metric_name="perception_anomaly_score",
                    operator=ConditionOperator.LT,
                    threshold=0.1
                )
            ],
            target="action",  # Skip to action if very low anomaly
            priority=10
        )
    )
    
    # After inference phase  
    flow.register_branch(
        "inference",
        flow.create_free_energy_branch(threshold=2.0)
    )
    
    flow.register_branch(
        "inference",
        flow.create_confidence_branch(min_confidence=0.7)
    )
    
    # After consensus phase
    flow.register_branch(
        "consensus", 
        flow.create_consensus_timeout_branch(timeout_ms=150.0)
    )
    
    # System-wide load management
    flow.register_dynamic_rule(
        "system_overload",
        lambda m: "graceful_degradation" if m.get('cpu_usage', 0) > 90 else None
    )
    
    return flow