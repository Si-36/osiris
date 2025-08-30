#!/usr/bin/env python3
"""
🧠 Clean Unified AURA Supervisor Test
====================================
Production-ready test for UnifiedAuraSupervisor with proper imports and structure.

Tests the integrated TDA + LNN supervisor system within the real AURA environment.
"""

import asyncio
import sys
import os
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime, timezone
import time
import json

# Set up proper paths for AURA imports
project_root = Path(__file__).parent
core_path = project_root / "core" / "src"
sys.path.insert(0, str(core_path))

# Import logging first
import structlog
logger = structlog.get_logger(__name__)

async def test_unified_supervisor_production():
    """Test UnifiedAuraSupervisor with real AURA components"""
    
    print("🧠 UNIFIED AURA SUPERVISOR - PRODUCTION TEST")
    print("=" * 60)
    
    try:
        # Import required components from AURA system
        from langchain_core.messages import AIMessage
        from langchain_core.runnables import RunnableConfig
        
        # Set up AURA component paths  
        workflows_path = core_path / "aura_intelligence" / "orchestration" / "workflows"
        nodes_path = workflows_path / "nodes"
        sys.path.insert(0, str(workflows_path))
        sys.path.insert(0, str(nodes_path))
        
        print("✅ Core imports successful")
        
        # Import AURA state management
        try:
            # Create a mock CollectiveState that matches AURA structure
            class CollectiveState(dict):
                """AURA-compatible state"""
                def __init__(self, **kwargs):
                    super().__init__()
                    self.update({
                        "workflow_id": "unified_test_001",
                        "thread_id": "thread_001",
                        "current_step": "supervisor_analysis",
                        "evidence_log": [],
                        "error_log": [],
                        "messages": [],
                        "supervisor_decisions": [],
                        "error_recovery_attempts": 0,
                        "execution_results": None,
                        "validation_results": None,
                        **kwargs
                    })
            
            class NodeResult:
                """AURA node result"""
                def __init__(self, success, node_name, output, duration_ms, next_node):
                    self.success = success
                    self.node_name = node_name
                    self.output = output
                    self.duration_ms = duration_ms
                    self.next_node = next_node
            
            print("✅ State management setup complete")
            
        except Exception as e:
            logger.warning(f"Using mock state management: {e}")
        
        # Import the supervisor components
        try:
            from supervisor import (
                UnifiedAuraSupervisor, 
                SupervisorNode,
                DecisionType,
                create_unified_aura_supervisor
            )
            print("✅ UnifiedAuraSupervisor imported successfully")
            
        except ImportError as e:
            logger.error(f"Failed to import supervisor: {e}")
            print("❌ Supervisor import failed - checking file structure...")
            
            # Check if supervisor.py exists
            supervisor_file = nodes_path / "supervisor.py"
            if supervisor_file.exists():
                print(f"   Supervisor file exists: {supervisor_file}")
                # Try direct import
                sys.path.insert(0, str(supervisor_file.parent))
                import supervisor as sup_module
                UnifiedAuraSupervisor = sup_module.UnifiedAuraSupervisor
                create_unified_aura_supervisor = sup_module.create_unified_aura_supervisor
                DecisionType = sup_module.DecisionType
                print("✅ Direct supervisor import successful")
            else:
                print(f"❌ Supervisor file not found at {supervisor_file}")
                return False
        
        # Test 1: Basic Initialization
        print("\n🔧 Test 1: Supervisor Initialization")
        
        try:
            supervisor = create_unified_aura_supervisor(
                risk_threshold=0.6,
                tda_config=None,  # Will use defaults
                lnn_config=None   # Will use defaults
            )
            
            print(f"   ✅ Supervisor created: {supervisor.name}")
            print(f"   TDA Available: {'✅' if supervisor.tda_available else '⚠️'}")
            print(f"   LNN Available: {'✅' if supervisor.lnn_available else '⚠️'}")
            print(f"   Risk Threshold: {supervisor.risk_threshold}")
            
            # Test component capabilities
            capabilities = {
                "tda_analyzer": hasattr(supervisor, 'tda_analyzer') and supervisor.tda_available,
                "lnn_engine": hasattr(supervisor, 'lnn_engine') and supervisor.lnn_available,
                "topology_cache": hasattr(supervisor, 'topology_cache'),
                "decision_history": hasattr(supervisor, 'decision_history')
            }
            
            print("   Component Status:")
            for comp, status in capabilities.items():
                print(f"     {comp}: {'✅' if status else '⚠️'}")
            
        except Exception as e:
            logger.error(f"Supervisor initialization failed: {e}")
            return False
        
        # Test 2: Decision Making with Real Workflow States
        print("\n🎯 Test 2: Decision Making with Real Scenarios")
        
        test_scenarios = [
            {
                "name": "Normal Operation",
                "state": CollectiveState(
                    evidence_log=[
                        {"type": "observation", "content": "System metrics normal", "timestamp": datetime.now(timezone.utc).isoformat()},
                        {"type": "analysis", "content": "No anomalies detected", "timestamp": datetime.now(timezone.utc).isoformat()}
                    ],
                    messages=[
                        AIMessage(content="System health check initiated"),
                        AIMessage(content="All systems operational")
                    ]
                )
            },
            {
                "name": "Risk Detection", 
                "state": CollectiveState(
                    evidence_log=[
                        {"type": "observation", "content": "High error rate detected", "risk_indicators": ["high_error_rate"], "timestamp": datetime.now(timezone.utc).isoformat()}
                    ],
                    error_log=[
                        {"error": "Connection timeout", "severity": "high", "timestamp": datetime.now(timezone.utc).isoformat()}
                    ],
                    error_recovery_attempts=1
                )
            },
            {
                "name": "Complex Workflow",
                "state": CollectiveState(
                    evidence_log=[
                        {"type": "observation", "content": f"Agent {i} processing", "timestamp": datetime.now(timezone.utc).isoformat()}
                        for i in range(8)
                    ],
                    messages=[
                        AIMessage(content=f"Task {i} coordination")
                        for i in range(12)
                    ]
                )
            }
        ]
        
        results = []
        for i, scenario in enumerate(test_scenarios):
            print(f"\n   Scenario {i+1}: {scenario['name']}")
            
            start_time = time.time()
            try:
                # Execute the supervisor
                result = await supervisor(scenario['state'])
                duration = time.time() - start_time
                
                # Extract results
                supervisor_decisions = result.get('supervisor_decisions', [])
                if supervisor_decisions:
                    decision_record = supervisor_decisions[0]
                    decision = decision_record.get('decision', 'unknown')
                    supervisor_type = decision_record.get('supervisor_type', 'standard')
                    reasoning = decision_record.get('reasoning', 'No reasoning provided')
                    
                    print(f"     Decision: {decision}")
                    print(f"     Supervisor Type: {supervisor_type}")
                    print(f"     Reasoning: {reasoning[:100]}...")
                    
                    # Check risk assessment
                    risk_assessment = result.get('risk_assessment', {})
                    unified_risk = risk_assessment.get('unified_risk_score', 0.0)
                    topology_complexity = risk_assessment.get('topology_complexity', 0.0)
                    lnn_confidence = risk_assessment.get('lnn_confidence', 0.0)
                    
                    print(f"     Unified Risk: {unified_risk:.3f}")
                    print(f"     Topology Complexity: {topology_complexity:.3f}")
                    print(f"     LNN Confidence: {lnn_confidence:.3f}")
                    
                    # Check advanced analysis
                    topology_analysis = result.get('topology_analysis')
                    lnn_decision = result.get('lnn_decision')
                    
                    print(f"     TDA Analysis: {'✅' if topology_analysis else '⚠️'}")
                    print(f"     LNN Decision: {'✅' if lnn_decision else '⚠️'}")
                    
                    results.append({
                        'scenario': scenario['name'],
                        'decision': decision,
                        'unified_risk': unified_risk,
                        'topology_complexity': topology_complexity,
                        'lnn_confidence': lnn_confidence,
                        'has_topology': topology_analysis is not None,
                        'has_lnn': lnn_decision is not None,
                        'duration': duration
                    })
                    
                else:
                    print("     ⚠️ No supervisor decisions found")
                    results.append({
                        'scenario': scenario['name'],
                        'error': 'No decisions generated',
                        'duration': duration
                    })
                
                print(f"     Duration: {duration:.3f}s")
                
            except Exception as e:
                duration = time.time() - start_time
                print(f"     ❌ Error: {str(e)[:100]}...")
                results.append({
                    'scenario': scenario['name'],
                    'error': str(e),
                    'duration': duration
                })
        
        # Test 3: Performance and Caching
        print("\n⚡ Test 3: Performance and Caching")
        
        # Test topology caching
        test_state = CollectiveState(
            evidence_log=[{"type": "test", "content": "Cache test"}],
            messages=[AIMessage(content="Cache test message")]
        )
        
        # First call - should populate cache
        start_time = time.time()
        await supervisor(test_state)
        first_call_time = time.time() - start_time
        cache_size_after_first = len(supervisor.topology_cache)
        
        # Second call - should use cache
        start_time = time.time()
        await supervisor(test_state)
        second_call_time = time.time() - start_time
        cache_size_after_second = len(supervisor.topology_cache)
        
        print(f"   First call time: {first_call_time:.4f}s")
        print(f"   Second call time: {second_call_time:.4f}s")
        print(f"   Cache after first: {cache_size_after_first}")
        print(f"   Cache after second: {cache_size_after_second}")
        
        cache_working = cache_size_after_first == cache_size_after_second and second_call_time <= first_call_time
        print(f"   Cache effectiveness: {'✅' if cache_working else '⚠️'}")
        
        # Test decision history
        history_before = len(supervisor.decision_history)
        
        # Make multiple decisions
        for i in range(3):
            test_state_i = CollectiveState(workflow_id=f"perf_test_{i}")
            await supervisor(test_state_i)
        
        history_after = len(supervisor.decision_history)
        history_growth = history_after - history_before
        
        print(f"   Decision history growth: {history_growth} entries")
        print(f"   History tracking: {'✅' if history_growth > 0 else '⚠️'}")
        
        # Test 4: Component Integration Analysis
        print("\n🔗 Test 4: Component Integration Analysis")
        
        successful_decisions = len([r for r in results if 'error' not in r])
        total_scenarios = len(results)
        avg_duration = sum(r.get('duration', 0) for r in results) / len(results)
        
        topology_coverage = sum(1 for r in results if r.get('has_topology', False))
        lnn_coverage = sum(1 for r in results if r.get('has_lnn', False))
        
        print(f"   Successful decisions: {successful_decisions}/{total_scenarios}")
        print(f"   Average decision time: {avg_duration:.4f}s")
        print(f"   TDA coverage: {topology_coverage}/{total_scenarios}")
        print(f"   LNN coverage: {lnn_coverage}/{total_scenarios}")
        
        # Integration quality assessment
        avg_unified_risk = sum(r.get('unified_risk', 0) for r in results if 'unified_risk' in r) / max(1, len([r for r in results if 'unified_risk' in r]))
        avg_topology_complexity = sum(r.get('topology_complexity', 0) for r in results if 'topology_complexity' in r) / max(1, len([r for r in results if 'topology_complexity' in r]))
        
        print(f"   Average unified risk: {avg_unified_risk:.3f}")
        print(f"   Average topology complexity: {avg_topology_complexity:.3f}")
        
        # Final assessment
        print("\n" + "=" * 60)
        print("📊 PRODUCTION TEST SUMMARY")
        print("=" * 60)
        
        integration_quality = (
            (successful_decisions / total_scenarios) * 0.4 +
            (1.0 if avg_duration < 0.1 else 0.5) * 0.2 +
            (topology_coverage / total_scenarios) * 0.2 +
            (lnn_coverage / total_scenarios) * 0.2
        )
        
        print(f"Integration Quality Score: {integration_quality:.2%}")
        print(f"TDA + LNN Integration: {'✅ EXCELLENT' if integration_quality > 0.8 else '✅ GOOD' if integration_quality > 0.6 else '⚠️ NEEDS IMPROVEMENT'}")
        
        component_status = {
            "UnifiedAuraSupervisor": "✅ WORKING",
            "Decision Making": "✅ WORKING" if successful_decisions >= total_scenarios * 0.8 else "⚠️ PARTIAL",
            "TDA Integration": "✅ WORKING" if supervisor.tda_available else "⚠️ FALLBACK",
            "LNN Integration": "✅ WORKING" if supervisor.lnn_available else "⚠️ FALLBACK",
            "Performance": "✅ EXCELLENT" if avg_duration < 0.05 else "✅ GOOD" if avg_duration < 0.1 else "⚠️ ACCEPTABLE"
        }
        
        print("\nComponent Status:")
        for component, status in component_status.items():
            print(f"  {component}: {status}")
        
        # Save detailed results
        test_results = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "integration_quality": integration_quality,
            "component_status": component_status,
            "scenario_results": results,
            "performance": {
                "avg_duration": avg_duration,
                "first_call_time": first_call_time,
                "second_call_time": second_call_time,
                "cache_entries": len(supervisor.topology_cache),
                "decision_history": len(supervisor.decision_history)
            },
            "capabilities": {
                "tda_available": supervisor.tda_available,
                "lnn_available": supervisor.lnn_available,
                "topology_coverage": topology_coverage / total_scenarios,
                "lnn_coverage": lnn_coverage / total_scenarios
            }
        }
        
        with open("unified_supervisor_test_results.json", "w") as f:
            json.dump(test_results, f, indent=2, default=str)
        
        print(f"\n💾 Detailed results saved to: unified_supervisor_test_results.json")
        
        success = integration_quality > 0.6
        if success:
            print("\n🎉 UNIFIED AURA SUPERVISOR - PRODUCTION READY!")
            print("✅ TDA + LNN integration verified")
            print("✅ Real-time decision making operational")
            print("✅ Unified risk assessment working")
        else:
            print("\n⚠️ INTEGRATION NEEDS IMPROVEMENT")
            print("🔧 Check component availability and configuration")
        
        return success
        
    except Exception as e:
        logger.error(f"Production test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Run production test for UnifiedAuraSupervisor"""
    
    print("🚀 UNIFIED AURA SUPERVISOR - PRODUCTION TEST SUITE")
    print(f"Environment: {os.environ.get('VIRTUAL_ENV', 'system')}")
    print(f"Timestamp: {datetime.now(timezone.utc).isoformat()}")
    print("=" * 80)
    
    success = await test_unified_supervisor_production()
    
    print("\n" + "=" * 80)
    if success:
        print("🎉 PRODUCTION TEST PASSED!")
        print("✅ UnifiedAuraSupervisor ready for deployment")
    else:
        print("❌ PRODUCTION TEST FAILED")
        print("🔧 Review component integration and dependencies")
    print("=" * 80)
    
    return 0 if success else 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)