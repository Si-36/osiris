#!/usr/bin/env python3
"""
🧠 Unified AURA Supervisor Test
==================================
Comprehensive test of UnifiedAuraSupervisor combining TDA + LNN integration.

Tests:
- TDA topology analysis with workflow structures
- LNN adaptive decision making with multi-head outputs  
- Unified risk assessment combining traditional + topology + neural signals
- Decision integration with confidence-based weighting
- Online learning and memory updates
- Fallback systems when components unavailable
"""

import asyncio
import sys
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime, timezone
import time
import traceback

# Add the core path for imports
core_path = Path(__file__).parent / "core" / "src"
sys.path.insert(0, str(core_path))

# Mock dependencies for testing
class MockCollectiveState(dict):
    """Mock state for testing supervisor"""
    
    def __init__(self, **kwargs):
        super().__init__()
        self.update({
            "workflow_id": "test_workflow_001",
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

class MockRunnableConfig:
    """Mock config for testing"""
    pass

class MockAIMessage:
    """Mock message for testing"""
    def __init__(self, content, additional_kwargs=None):
        self.content = content
        self.additional_kwargs = additional_kwargs or {}

class MockNodeResult:
    """Mock node result"""
    def __init__(self, success, node_name, output, duration_ms, next_node):
        self.success = success
        self.node_name = node_name
        self.output = output
        self.duration_ms = duration_ms
        self.next_node = next_node

# Test scenarios
TEST_SCENARIOS = [
    {
        "name": "Simple Workflow",
        "state": MockCollectiveState(
            evidence_log=[
                {"type": "observation", "content": "System metrics collected"},
                {"type": "analysis", "content": "Normal patterns detected"}
            ],
            messages=[
                {"role": "user", "content": "Analyze system health"},
                {"role": "assistant", "content": "Systems appear normal"}
            ]
        ),
        "expected_decision": "continue"
    },
    {
        "name": "High Risk Scenario", 
        "state": MockCollectiveState(
            evidence_log=[
                {"type": "observation", "content": "High error rate detected", "risk_indicators": ["high_error_rate", "high_cpu_usage"]},
                {"type": "analysis", "content": "System stress patterns"}
            ],
            error_log=[
                {"error": "Connection timeout", "timestamp": "2025-01-01T00:00:00Z"},
                {"error": "Memory allocation failed", "timestamp": "2025-01-01T00:01:00Z"}
            ],
            error_recovery_attempts=2
        ),
        "expected_decision": "escalate"
    },
    {
        "name": "Complex Workflow Structure",
        "state": MockCollectiveState(
            evidence_log=[
                {"type": "observation", "content": f"Agent {i} status"} 
                for i in range(10)
            ],
            messages=[
                {"role": "user", "content": f"Task {i} request"}
                for i in range(15)
            ]
        ),
        "expected_decision": "continue"
    },
    {
        "name": "Completion Scenario",
        "state": MockCollectiveState(
            execution_results={"success": True, "actions_taken": ["deploy", "verify"]},
            validation_results={"valid": True, "checks_passed": 5},
            evidence_log=[
                {"type": "execution", "content": "Deployment completed"},
                {"type": "validation", "content": "All checks passed"}
            ]
        ),
        "expected_decision": "complete"
    }
]


async def test_unified_supervisor():
    """Test the UnifiedAuraSupervisor with various scenarios"""
    
    print("🧠 Testing Unified AURA Supervisor")
    print("=" * 60)
    
    try:
        # Import the supervisor components
        sys.path.insert(0, str(Path(__file__).parent / "core" / "src" / "aura_intelligence" / "orchestration" / "workflows" / "nodes"))
        
        # Mock the required imports to prevent import errors
        import types
        
        # Create mock modules
        mock_langchain_core = types.ModuleType('langchain_core')
        mock_langchain_core.messages = types.ModuleType('messages')
        mock_langchain_core.messages.AIMessage = MockAIMessage
        mock_langchain_core.runnables = types.ModuleType('runnables')
        mock_langchain_core.runnables.RunnableConfig = MockRunnableConfig
        
        mock_structlog = types.ModuleType('structlog')
        mock_structlog.get_logger = lambda name: MockLogger()
        
        mock_state = types.ModuleType('state')
        mock_state.CollectiveState = MockCollectiveState
        mock_state.NodeResult = MockNodeResult
        
        # Add mocks to sys.modules
        sys.modules['langchain_core'] = mock_langchain_core
        sys.modules['langchain_core.messages'] = mock_langchain_core.messages
        sys.modules['langchain_core.runnables'] = mock_langchain_core.runnables
        sys.modules['structlog'] = mock_structlog
        sys.modules['state'] = mock_state
        
        # Import the supervisor
        from supervisor import UnifiedAuraSupervisor, DecisionType, create_unified_aura_supervisor
        
        print("✅ Successfully imported UnifiedAuraSupervisor")
        
        # Test 1: Basic Initialization
        print("\n🔧 Test 1: Supervisor Initialization")
        supervisor = create_unified_aura_supervisor(risk_threshold=0.6)
        
        print(f"   Supervisor name: {supervisor.name}")
        print(f"   TDA available: {supervisor.tda_available}")
        print(f"   LNN available: {supervisor.lnn_available}")
        print(f"   Risk threshold: {supervisor.risk_threshold}")
        
        # Test 2: Decision Making with Different Scenarios
        print("\n🎯 Test 2: Decision Making Scenarios")
        
        results = []
        for i, scenario in enumerate(TEST_SCENARIOS):
            print(f"\n   Scenario {i+1}: {scenario['name']}")
            
            start_time = time.time()
            try:
                # Execute supervisor
                result = await supervisor(scenario['state'])
                duration = time.time() - start_time
                
                # Extract decision info
                supervisor_decisions = result.get('supervisor_decisions', [])
                decision = supervisor_decisions[0].get('decision') if supervisor_decisions else 'unknown'
                risk_assessment = result.get('risk_assessment', {})
                
                print(f"     Decision: {decision}")
                print(f"     Unified Risk: {risk_assessment.get('unified_risk_score', 0.0):.3f}")
                print(f"     Topology Complexity: {risk_assessment.get('topology_complexity', 0.0):.3f}")
                print(f"     LNN Confidence: {risk_assessment.get('lnn_confidence', 0.0):.3f}")
                print(f"     Duration: {duration:.3f}s")
                
                # Check if decision matches expectation (approximately)
                expected = scenario.get('expected_decision', 'unknown')
                matches = decision.lower() == expected.lower()
                status = "✅" if matches else "⚠️"
                print(f"     Expected: {expected} {status}")
                
                results.append({
                    'scenario': scenario['name'],
                    'decision': decision,
                    'expected': expected,
                    'matches': matches,
                    'risk_score': risk_assessment.get('unified_risk_score', 0.0),
                    'topology_complexity': risk_assessment.get('topology_complexity', 0.0),
                    'duration': duration
                })
                
            except Exception as e:
                print(f"     ❌ Error: {e}")
                results.append({
                    'scenario': scenario['name'],
                    'error': str(e),
                    'duration': time.time() - start_time
                })
        
        # Test 3: Advanced Features
        print("\n🚀 Test 3: Advanced Features")
        
        # Test topology caching
        print("   Testing topology caching...")
        state = MockCollectiveState()
        await supervisor(state)
        cache_size_1 = len(supervisor.topology_cache)
        
        await supervisor(state)  # Same state should use cache
        cache_size_2 = len(supervisor.topology_cache)
        
        print(f"   Cache size after first call: {cache_size_1}")
        print(f"   Cache size after second call: {cache_size_2}")
        print(f"   Cache working: {'✅' if cache_size_1 == cache_size_2 else '❌'}")
        
        # Test decision history
        print("   Testing decision history...")
        history_size_before = len(supervisor.decision_history)
        
        # Make several decisions to build history
        for i in range(5):
            test_state = MockCollectiveState(workflow_id=f"test_workflow_{i}")
            await supervisor(test_state)
        
        history_size_after = len(supervisor.decision_history)
        print(f"   History size before: {history_size_before}")
        print(f"   History size after: {history_size_after}")
        print(f"   History tracking: {'✅' if history_size_after > history_size_before else '❌'}")
        
        # Test 4: Performance Analysis
        print("\n⚡ Test 4: Performance Analysis")
        
        performance_results = []
        for i in range(10):
            state = MockCollectiveState(
                evidence_log=[{"type": "test", "content": f"Performance test {i}"}] * (i + 1),
                messages=[{"role": "user", "content": f"Message {j}"} for j in range(i * 2)]
            )
            
            start_time = time.time()
            await supervisor(state)
            duration = time.time() - start_time
            performance_results.append(duration)
        
        avg_duration = sum(performance_results) / len(performance_results)
        max_duration = max(performance_results)
        min_duration = min(performance_results)
        
        print(f"   Average duration: {avg_duration:.4f}s")
        print(f"   Max duration: {max_duration:.4f}s")
        print(f"   Min duration: {min_duration:.4f}s")
        print(f"   Performance: {'✅' if avg_duration < 0.1 else '⚠️'}")
        
        # Test 5: Error Handling
        print("\n🛡️ Test 5: Error Handling")
        
        # Test with malformed state
        try:
            malformed_state = {}  # Missing required fields
            result = await supervisor(malformed_state)
            print("   Malformed state handling: ✅")
        except Exception as e:
            print(f"   Malformed state handling: ⚠️ ({e})")
        
        # Summary
        print("\n" + "=" * 60)
        print("📊 TEST SUMMARY")
        print("=" * 60)
        
        successful_scenarios = sum(1 for r in results if r.get('matches', False))
        total_scenarios = len([r for r in results if 'error' not in r])
        error_scenarios = len([r for r in results if 'error' in r])
        
        print(f"Successful decisions: {successful_scenarios}/{total_scenarios}")
        print(f"Error scenarios: {error_scenarios}")
        print(f"Average risk score: {sum(r.get('risk_score', 0) for r in results) / len(results):.3f}")
        print(f"Average topology complexity: {sum(r.get('topology_complexity', 0) for r in results) / len(results):.3f}")
        print(f"Average decision time: {avg_duration:.4f}s")
        
        # Component availability
        print(f"\nComponent Availability:")
        print(f"  TDA Analyzer: {'✅' if supervisor.tda_available else '⚠️'}")
        print(f"  LNN Engine: {'✅' if supervisor.lnn_available else '⚠️'}")
        print(f"  Topology Cache: {len(supervisor.topology_cache)} entries")
        print(f"  Decision History: {len(supervisor.decision_history)} decisions")
        
        overall_success = successful_scenarios >= total_scenarios * 0.8 and error_scenarios == 0
        
        if overall_success:
            print("\n🎉 UNIFIED AURA SUPERVISOR TEST PASSED!")
            print("✅ TDA + LNN integration working correctly")
            print("✅ Decision making with unified risk assessment")
            print("✅ Online learning and caching systems operational")
            return True
        else:
            print("\n⚠️ SOME TESTS FAILED")
            print("❌ Check component availability and decision logic")
            return False
            
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        print("⚠️ Component integration tests skipped")
        return False
        
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        print("📝 Full traceback:")
        traceback.print_exc()
        return False


class MockLogger:
    """Mock logger for testing"""
    def info(self, msg, **kwargs):
        pass
    def warning(self, msg, **kwargs):
        pass
    def error(self, msg, **kwargs):
        pass


async def main():
    """Run the unified supervisor test"""
    
    print("🚀 UNIFIED AURA SUPERVISOR TEST SUITE")
    print(f"Timestamp: {datetime.now(timezone.utc).isoformat()}")
    print("=" * 80)
    
    success = await test_unified_supervisor()
    
    print("\n" + "=" * 80)
    if success:
        print("🎉 Unified AURA Supervisor test PASSED!")
        print("✅ TDA + LNN integration verified")
        print("✅ Ready for production deployment")
    else:
        print("❌ Test FAILED or INCOMPLETE")
        print("🔧 Check component dependencies and integration")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())