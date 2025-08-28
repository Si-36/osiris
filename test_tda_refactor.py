"""
Test the refactored TDA implementation
"""

import asyncio
import numpy as np
import time
from aura_intelligence.tda import (
    AgentTopologyAnalyzer,
    RealtimeTopologyMonitor,
    EventAdapter,
    EventType,
    SystemEvent,
    compute_persistence,
    diagram_entropy,
    create_monitor
)


async def test_workflow_analysis():
    """Test workflow topology analysis."""
    print("\n=== Testing Workflow Analysis ===")
    
    analyzer = AgentTopologyAnalyzer()
    
    # Create sample workflow
    workflow_data = {
        "agents": [
            {"id": "agent_1", "type": "data_processor"},
            {"id": "agent_2", "type": "analyzer"},
            {"id": "agent_3", "type": "validator"},
            {"id": "agent_4", "type": "reporter"}
        ],
        "dependencies": [
            {"source": "agent_1", "target": "agent_2", "weight": 1.0},
            {"source": "agent_2", "target": "agent_3", "weight": 1.0},
            {"source": "agent_3", "target": "agent_4", "weight": 1.0},
            {"source": "agent_2", "target": "agent_4", "weight": 0.5}  # Bypass
        ]
    }
    
    # Analyze
    features = await analyzer.analyze_workflow("workflow_1", workflow_data)
    
    print(f"Workflow Analysis Results:")
    print(f"  Agents: {features.num_agents}")
    print(f"  Edges: {features.num_edges}")
    print(f"  Has Cycles: {features.has_cycles}")
    print(f"  Critical Path: {features.critical_path_agents}")
    print(f"  Bottlenecks: {features.bottleneck_agents}")
    print(f"  Bottleneck Score: {features.bottleneck_score:.3f}")
    print(f"  Failure Risk: {features.failure_risk:.3f}")
    print(f"  Recommendations: {features.recommendations}")
    
    # Test with cycle
    print("\n--- Testing with Cycle ---")
    workflow_data["dependencies"].append(
        {"source": "agent_4", "target": "agent_1", "weight": 0.3}
    )
    
    features = await analyzer.analyze_workflow("workflow_2", workflow_data)
    print(f"Has Cycles: {features.has_cycles}")
    print(f"Bottleneck Score: {features.bottleneck_score:.3f}")
    print(f"Recommendations: {features.recommendations}")


async def test_communication_analysis():
    """Test communication topology analysis."""
    print("\n=== Testing Communication Analysis ===")
    
    analyzer = AgentTopologyAnalyzer()
    
    # Create communication data
    comm_data = {
        "agents": [
            {"id": f"agent_{i}"} for i in range(10)
        ],
        "messages": [
            # Hub pattern - agent_0 talks to everyone
            *[{"source": "agent_0", "target": f"agent_{i}"} for i in range(1, 6)],
            # Some inter-agent communication
            {"source": "agent_1", "target": "agent_2"},
            {"source": "agent_2", "target": "agent_3"},
            {"source": "agent_3", "target": "agent_4"},
            # Isolated component
            {"source": "agent_7", "target": "agent_8"},
            {"source": "agent_8", "target": "agent_9"},
        ]
    }
    
    features = await analyzer.analyze_communications(comm_data)
    
    print(f"Communication Analysis Results:")
    print(f"  Total Agents: {features.total_agents}")
    print(f"  Active Connections: {features.active_connections}")
    print(f"  Network Density: {features.network_density:.3f}")
    print(f"  Components: {features.num_components}")
    print(f"  Isolated Agents: {features.isolated_agents}")
    print(f"  Hub Agents: {features.hub_agents}")
    print(f"  Overall Health: {features.overall_health:.3f}")


async def test_realtime_monitor():
    """Test real-time topology monitoring."""
    print("\n=== Testing Real-time Monitor ===")
    
    # Create monitor with small windows for testing
    config = {
        "window_size": 5,  # 5 seconds
        "slide_interval": 2  # 2 seconds
    }
    
    monitor = await create_monitor(config)
    adapter = EventAdapter()
    
    # Simulate some events
    events = [
        # Start agents
        adapter.from_agent_lifecycle("agent_1", "start"),
        adapter.from_agent_lifecycle("agent_2", "start"),
        adapter.from_agent_lifecycle("agent_3", "start"),
        
        # Task assignments
        adapter.from_task_event(
            "task_1", "agent_1", "agent_2", "workflow_1", "assigned"
        ),
        adapter.from_task_event(
            "task_2", "agent_2", "agent_3", "workflow_1", "assigned"
        ),
        
        # Messages
        adapter.from_message("agent_1", "agent_2"),
        adapter.from_message("agent_2", "agent_3"),
        
        # Task completion
        adapter.from_task_event(
            "task_1", "agent_1", "agent_2", "workflow_1", "completed"
        ),
    ]
    
    # Process events
    for event in events:
        queued = await monitor.process_event(event)
        print(f"Event {event.event_type.value} - Queued: {queued}")
        await asyncio.sleep(0.1)
    
    # Wait for window to complete
    print("\nWaiting for window processing...")
    await asyncio.sleep(6)
    
    # Force flush
    await monitor.flush_window(time.time())
    
    # Check metrics
    metrics = monitor.get_metrics()
    print(f"\nMonitor Metrics:")
    print(f"  Processed Events: {metrics['processed_events']}")
    print(f"  Published Features: {metrics['published_features']}")
    
    # Stop monitor
    await monitor.stop()


async def test_persistence_algorithms():
    """Test core TDA algorithms."""
    print("\n=== Testing TDA Algorithms ===")
    
    # Create sample point cloud
    points = np.random.rand(10, 2)
    
    # Compute persistence
    diagrams = compute_persistence(points, max_dimension=1)
    
    print(f"Computed {len(diagrams)} persistence diagrams")
    
    # Compute entropy
    for i, diag in enumerate(diagrams):
        if not diag.is_empty():
            entropy = diagram_entropy(diag)
            print(f"  Dimension {i} entropy: {entropy:.3f}")
            print(f"  Total persistence: {diag.total_persistence:.3f}")


async def test_legacy_compatibility():
    """Test legacy import compatibility."""
    print("\n=== Testing Legacy Compatibility ===")
    
    # Test deprecation warnings
    import warnings
    
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        
        # Try legacy import
        from aura_intelligence.tda import UnifiedTDAEngine2025
        
        # Check warning
        assert len(w) == 1
        assert "deprecated" in str(w[0].message)
        print(f"✓ Deprecation warning works: {w[0].message}")
        
        # Verify it returns the new class
        assert UnifiedTDAEngine2025 == AgentTopologyAnalyzer
        print("✓ Legacy mapping works correctly")


async def main():
    """Run all tests."""
    print("TDA Refactor Test Suite")
    print("=" * 50)
    
    await test_workflow_analysis()
    await test_communication_analysis()
    await test_persistence_algorithms()
    await test_realtime_monitor()
    await test_legacy_compatibility()
    
    print("\n✅ All tests completed!")


if __name__ == "__main__":
    asyncio.run(main())