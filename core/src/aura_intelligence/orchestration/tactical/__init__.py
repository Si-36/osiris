"""
Tactical Orchestration Layer
============================
Temporal workflows, pipeline registry, and conditional flows
Based on Temporal 1.26 Active-Active patterns
"""

from .temporal_workflows import (
    CognitiveWorkflow,
    PerceptionPhase,
    InferencePhase,
    ConsensusPhase,
    ActionPhase,
    WorkflowResult
)
from .pipeline_registry import PipelineRegistry, Pipeline, PipelineVersion
from .conditional_flows import ConditionalFlow, BranchCondition
from .experiment_manager import ExperimentManager, Experiment

__all__ = [
    # Temporal workflows
    'CognitiveWorkflow',
    'PerceptionPhase',
    'InferencePhase', 
    'ConsensusPhase',
    'ActionPhase',
    'WorkflowResult',
    
    # Pipeline management
    'PipelineRegistry',
    'Pipeline',
    'PipelineVersion',
    
    # Conditional flows
    'ConditionalFlow',
    'BranchCondition',
    
    # Experiments
    'ExperimentManager',
    'Experiment'
]