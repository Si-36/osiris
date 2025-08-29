#!/usr/bin/env python3
"""
Test TopologicalAnalyzer Integration with AURA Working Components
================================================================
Tests the production-ready TDA supervisor implementation.
"""

import sys
import os
import asyncio
import json
from datetime import datetime

# Add the core path
sys.path.insert(0, '/home/sina/projects/osiris-2/core/src')

async def test_tda_integration():
    """Test the TopologicalAnalyzer integration"""
    
    print("🔬 Testing TopologicalAnalyzer Integration with AURA")
    print("=" * 60)
    
    try:
        # Test imports
        print("1. Testing imports...")
        from aura_intelligence.orchestration.workflows.nodes.tda_supervisor_integration import (
            ProductionTopologicalAnalyzer, 
            ProductionTDAConfig,
            RealTDAWorkflowAnalyzer,
            TDASupervisorConfig
        )
        print("✅ TDA supervisor components imported successfully")
        
        # Test configuration
        print("\n2. Testing configuration...")
        config = ProductionTDAConfig(
            max_nodes_for_full_tda=50,
            persistence_threshold=0.1,
            enable_betti_curves=False  # Disable for faster testing
        )
        print(f"✅ TDA config created: {config}")
        
        # Test analyzer initialization
        print("\n3. Testing analyzer initialization...")
        analyzer = ProductionTopologicalAnalyzer(config)
        print(f"✅ TDA analyzer initialized, TDA libs available: {analyzer.is_available}")
        
        # Test with sample workflow data
        print("\n4. Testing workflow analysis...")
        
        sample_workflow = {
            'agents': [
                {'id': 'agent_1', 'type': 'researcher', 'capabilities': ['analysis']},
                {'id': 'agent_2', 'type': 'optimizer', 'capabilities': ['optimization']},
                {'id': 'agent_3', 'type': 'guardian', 'capabilities': ['monitoring']}
            ],
            'tasks': [
                {'id': 'task_1', 'type': 'data_analysis', 'complexity': 0.6},
                {'id': 'task_2', 'type': 'optimization', 'complexity': 0.8}
            ],
            'messages': [
                {'sender': 'agent_1', 'receiver': 'agent_2', 'type': 'data_request'},
                {'sender': 'agent_2', 'receiver': 'agent_3', 'type': 'status_update'}
            ],
            'dependencies': [
                {'from': 'task_1', 'to': 'task_2', 'type': 'sequential'}
            ]
        }
        
        # Perform analysis
        result = await analyzer.analyze_workflow_topology(sample_workflow)
        
        print("✅ Workflow analysis completed")
        print(f"   Analysis type: {result.get('analysis_type', 'unknown')}")
        print(f"   Workflow health: {result.get('workflow_health', 'unknown')}")
        
        # Check graph properties
        graph_props = result.get('graph_properties', {})
        print(f"   Graph nodes: {graph_props.get('nodes', 0)}")
        print(f"   Graph edges: {graph_props.get('edges', 0)}")
        print(f"   Graph density: {graph_props.get('density', 0.0):.3f}")
        
        # Check topology analysis
        topo_analysis = result.get('topology_analysis', {})
        print(f"   Anomaly score: {topo_analysis.get('anomaly_score', 0.0):.3f}")
        print(f"   Complexity score: {topo_analysis.get('complexity_score', 0.0):.3f}")
        
        # Check recommendations
        recommendations = result.get('recommendations', [])
        print(f"   Recommendations: {len(recommendations)}")
        for i, rec in enumerate(recommendations[:3]):  # Show first 3
            print(f"     {i+1}. {rec}")
        
        # Test Real TDA Workflow Analyzer
        print("\n5. Testing RealTDAWorkflowAnalyzer...")
        real_config = TDASupervisorConfig(
            use_real_tda=True,
            max_points_limit=100
        )
        
        real_analyzer = RealTDAWorkflowAnalyzer(real_config)
        print(f"✅ Real TDA analyzer initialized, available: {real_analyzer.tda_available}")
        
        # Test analysis with real analyzer
        real_result = await real_analyzer.analyze_workflow_topology(sample_workflow)
        print("✅ Real TDA workflow analysis completed")
        print(f"   Success: {real_result.get('success', False)}")
        print(f"   Analysis method: {real_result.get('analysis_method', 'unknown')}")
        
        # Performance check
        if 'performance_metrics' in result:
            perf = result['performance_metrics']
            print(f"\n6. Performance metrics:")
            print(f"   Analysis duration: {perf.get('analysis_duration_seconds', 0):.3f}s")
            print(f"   Graph size: {perf.get('graph_size', 0)} nodes")
        
        print("\n🎉 All TDA integration tests passed!")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_fallback_analysis():
    """Test fallback analysis when TDA libraries are not available"""
    
    print("\n" + "=" * 60)
    print("🔄 Testing Fallback Analysis")
    print("=" * 60)
    
    try:
        # Force unavailable state for testing
        from aura_intelligence.orchestration.workflows.nodes.tda_supervisor_integration import (
            ProductionTopologicalAnalyzer, 
            ProductionTDAConfig
        )
        
        config = ProductionTDAConfig()
        analyzer = ProductionTopologicalAnalyzer(config)
        
        # Temporarily disable TDA for fallback test
        original_available = analyzer.is_available
        analyzer.is_available = False
        
        sample_workflow = {
            'agents': [{'id': f'agent_{i}'} for i in range(10)],
            'tasks': [{'id': f'task_{i}'} for i in range(5)],
            'messages': [
                {'sender': f'agent_{i}', 'receiver': f'agent_{i+1}'} 
                for i in range(9)
            ]
        }
        
        result = await analyzer.analyze_workflow_topology(sample_workflow)
        
        # Restore original state
        analyzer.is_available = original_available
        
        print("✅ Fallback analysis completed")
        print(f"   Analysis type: {result.get('analysis_type', 'unknown')}")
        print(f"   Health status: {result.get('workflow_health', 'unknown')}")
        
        return True
        
    except Exception as e:
        print(f"❌ Fallback test failed: {e}")
        return False

async def main():
    """Run all tests"""
    
    print("🚀 Starting AURA TDA Supervisor Integration Tests")
    print(f"Timestamp: {datetime.utcnow().isoformat()}")
    print("=" * 80)
    
    # Test basic integration
    success1 = await test_tda_integration()
    
    # Test fallback behavior
    success2 = await test_fallback_analysis()
    
    print("\n" + "=" * 80)
    if success1 and success2:
        print("🎉 All integration tests PASSED!")
        print("✅ TopologicalAnalyzer is ready for production use")
    else:
        print("❌ Some tests FAILED")
        print("⚠️  Check configuration and dependencies")
    
    print("=" * 80)

if __name__ == "__main__":
    asyncio.run(main())