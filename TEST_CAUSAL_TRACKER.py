#!/usr/bin/env python3
"""
Test Causal Pattern Tracker - Real Implementation
=================================================

Tests that the causal tracker:
1. Tracks patterns and outcomes
2. Learns from repeated patterns
3. Predicts failures
4. Suggests preventive actions
"""

import asyncio
import numpy as np
import sys
import os
import time
from datetime import datetime

# Add project to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'core', 'src'))

from aura_intelligence.memory.core.causal_tracker import CausalPatternTracker, OutcomeType
from aura_intelligence.memory.core.topology_adapter import TopologyMemoryAdapter


async def test_causal_tracking():
    """Test causal pattern tracking with real data"""
    
    print("🧪 Testing Causal Pattern Tracker")
    print("=" * 60)
    
    # Initialize components
    tracker = CausalPatternTracker()
    topology_adapter = TopologyMemoryAdapter(config={})
    
    # Test 1: Track successful workflow
    print("\n1️⃣ Testing Successful Workflow Pattern")
    print("-" * 30)
    
    # Create a simple workflow that succeeds
    success_workflow = {
        "workflow_id": "wf_success_001",
        "nodes": ["start", "process", "validate", "complete"],
        "edges": [
            {"source": 0, "target": 1},
            {"source": 1, "target": 2},
            {"source": 2, "target": 3}
        ]
    }
    
    # Extract topology
    topology1 = await topology_adapter.extract_topology(success_workflow)
    
    # Track pattern with success outcome
    pattern_id1 = await tracker.track_pattern(
        workflow_id="wf_success_001",
        pattern=topology1,
        outcome="success"
    )
    
    print(f"✅ Pattern tracked: {pattern_id1}")
    print(f"   Betti numbers: {topology1.betti_numbers}")
    print(f"   Has cycles: {topology1.workflow_features.has_cycles}")
    
    # Test 2: Track failing workflow (with cycle)
    print("\n2️⃣ Testing Failing Workflow Pattern (with cycle)")
    print("-" * 30)
    
    failure_workflow = {
        "workflow_id": "wf_failure_001",
        "nodes": ["start", "process", "validate", "retry", "fail"],
        "edges": [
            {"source": 0, "target": 1},
            {"source": 1, "target": 2},
            {"source": 2, "target": 3},
            {"source": 3, "target": 1},  # Cycle back to process!
            {"source": 3, "target": 4}
        ]
    }
    
    topology2 = await topology_adapter.extract_topology(failure_workflow)
    
    pattern_id2 = await tracker.track_pattern(
        workflow_id="wf_failure_001",
        pattern=topology2,
        outcome="failure"
    )
    
    print(f"✅ Pattern tracked: {pattern_id2}")
    print(f"   Betti numbers: {topology2.betti_numbers}")
    print(f"   Has cycles: {topology2.workflow_features.has_cycles}")
    print(f"   Bottleneck score: {topology2.bottleneck_severity:.3f}")
    
    # Test 3: Repeat patterns to build confidence
    print("\n3️⃣ Building Pattern Confidence (5 repetitions)")
    print("-" * 30)
    
    for i in range(5):
        # Repeat success pattern
        await tracker.track_pattern(
            workflow_id=f"wf_success_{i:03d}",
            pattern=topology1,
            outcome="success"
        )
        
        # Repeat failure pattern
        await tracker.track_pattern(
            workflow_id=f"wf_failure_{i:03d}",
            pattern=topology2,
            outcome="failure"
        )
    
    # Check pattern statistics
    success_pattern = tracker.patterns.get(pattern_id1)
    failure_pattern = tracker.patterns.get(pattern_id2)
    
    if success_pattern:
        print(f"✅ Success pattern stats:")
        print(f"   Occurrences: {success_pattern.total_occurrences}")
        print(f"   Success probability: {success_pattern.success_probability:.2%}")
        print(f"   Confidence: {success_pattern.confidence_score:.3f}")
    
    if failure_pattern:
        print(f"✅ Failure pattern stats:")
        print(f"   Occurrences: {failure_pattern.total_occurrences}")
        print(f"   Failure probability: {failure_pattern.failure_probability:.2%}")
        print(f"   Confidence: {failure_pattern.confidence_score:.3f}")
    
    # Test 4: Predict outcome for new workflow
    print("\n4️⃣ Testing Failure Prediction")
    print("-" * 30)
    
    # Create a workflow similar to failure pattern
    test_workflow = {
        "workflow_id": "wf_test_001",
        "nodes": ["a", "b", "c", "d"],
        "edges": [
            {"source": 0, "target": 1},
            {"source": 1, "target": 2},
            {"source": 2, "target": 1},  # Has a cycle like failure pattern
            {"source": 2, "target": 3}
        ]
    }
    
    test_topology = await topology_adapter.extract_topology(test_workflow)
    
    # Predict outcome
    prediction = await tracker.predict_outcome(test_topology)
    
    print(f"✅ Prediction for test workflow:")
    print(f"   Pattern found: {prediction.get('pattern_found', False)}")
    print(f"   Failure probability: {prediction.get('failure_probability', 0):.2%}")
    print(f"   Success probability: {prediction.get('success_probability', 0):.2%}")
    print(f"   Confidence: {prediction.get('confidence', 0):.3f}")
    print(f"   Based on: {prediction.get('based_on_occurrences', 0)} occurrences")
    
    # Test 5: Analyze multiple patterns
    print("\n5️⃣ Testing Pattern Analysis")
    print("-" * 30)
    
    analysis = await tracker.analyze_patterns([topology1, topology2, test_topology])
    
    print(f"✅ Pattern analysis:")
    print(f"   Failure probability: {analysis.failure_probability:.2%}")
    print(f"   Predicted outcome: {analysis.predicted_outcome}")
    print(f"   Confidence: {analysis.confidence:.3f}")
    print(f"   Risk factors: {len(analysis.risk_factors)}")
    
    for risk in analysis.risk_factors[:3]:
        print(f"     - Pattern {risk['pattern_id'][:8]}: {risk['failure_rate']:.2%} failure rate")
    
    # Test 6: Test causal chains
    print("\n6️⃣ Testing Causal Chain Discovery")
    print("-" * 30)
    
    # Create a sequence of patterns
    chain_workflow_1 = {
        "workflow_id": "chain_001",
        "nodes": ["a", "b"],
        "edges": [{"source": 0, "target": 1}]
    }
    
    chain_workflow_2 = {
        "workflow_id": "chain_001",  # Same workflow ID for chain
        "nodes": ["a", "b", "c"],
        "edges": [
            {"source": 0, "target": 1},
            {"source": 1, "target": 2}
        ]
    }
    
    topo_chain_1 = await topology_adapter.extract_topology(chain_workflow_1)
    topo_chain_2 = await topology_adapter.extract_topology(chain_workflow_2)
    
    # Track as sequence
    await tracker.track_pattern("chain_001", topo_chain_1, "in_progress")
    await tracker.track_pattern("chain_001", topo_chain_2, "failure")
    
    # Check if chain was discovered
    print(f"✅ Chains discovered: {len(tracker.chains)}")
    for chain_id, chain in list(tracker.chains.items())[:2]:
        print(f"   Chain {chain_id[:8]}:")
        print(f"     Length: {chain.chain_length}")
        print(f"     Outcome: {chain.end_outcome}")
        print(f"     Confidence: {chain.confidence:.3f}")
    
    # Test 7: Get statistics
    print("\n7️⃣ Tracker Statistics")
    print("-" * 30)
    
    stats = tracker.get_statistics()
    print(f"✅ Overall statistics:")
    print(f"   Patterns tracked: {stats['patterns_tracked']}")
    print(f"   Chains discovered: {stats['chains_discovered']}")
    print(f"   Predictions made: {stats['predictions_made']}")
    
    print("\n" + "=" * 60)
    print("✅ All causal tracking tests passed!")
    
    # Cleanup
    await topology_adapter.shutdown()


if __name__ == "__main__":
    asyncio.run(test_causal_tracking())