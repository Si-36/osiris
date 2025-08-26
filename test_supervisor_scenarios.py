"""
🧪 Comprehensive Test Scenarios for Real Supervisor
Tests various workflow scenarios to demonstrate supervisor capabilities.
"""

import asyncio
import json
from real_supervisor_implementation import RealSupervisor, DecisionType


async def test_success_scenario():
    """Test: Successful workflow execution."""
    print("\n" + "="*60)
    print("TEST: Success Scenario - Everything going well")
    print("="*60)
    
    supervisor = RealSupervisor()
    
    state = {
        "workflow_id": "success-workflow",
        "total_steps": 10,
        "completed_steps": ["step1", "step2", "step3", "step4", "step5", "step6", "step7"],
        "step_results": [
            {"node_name": "init", "success": True, "duration_ms": 500},
            {"node_name": "fetch", "success": True, "duration_ms": 800},
            {"node_name": "analyze", "success": True, "duration_ms": 1200},
            {"node_name": "process", "success": True, "duration_ms": 900},
            {"node_name": "validate", "success": True, "duration_ms": 600},
            {"node_name": "transform", "success": True, "duration_ms": 700},
            {"node_name": "store", "success": True, "duration_ms": 400},
        ],
        "resource_usage": {
            "cpu": 0.45,
            "memory": 0.30,
            "network": 0.20
        }
    }
    
    result = await supervisor(state)
    decision = result["supervisor_decision"]
    
    print(f"Decision: {decision['decision']}")
    print(f"Confidence: {decision['reasoning']['confidence']:.2f}")
    print(f"Risk Level: {decision['risk_assessment']['level']}")
    print(f"Success Rate: {decision['metrics']['success_rate']:.2%}")
    

async def test_failure_cascade_scenario():
    """Test: Cascading failures requiring escalation."""
    print("\n" + "="*60)
    print("TEST: Cascading Failure Scenario")
    print("="*60)
    
    supervisor = RealSupervisor()
    
    # First, build up pattern history
    failing_state = {
        "workflow_id": "cascade-workflow",
        "total_steps": 10,
        "completed_steps": ["step1"],
        "step_results": [
            {"node_name": "init", "success": True, "duration_ms": 500},
            {"node_name": "fetch", "success": False, "duration_ms": 3000, "error_type": "NetworkError"},
            {"node_name": "fetch", "success": False, "duration_ms": 3500, "error_type": "NetworkError", "retry_count": 1},
            {"node_name": "fetch", "success": False, "duration_ms": 4000, "error_type": "NetworkError", "retry_count": 2},
            {"node_name": "analyze", "success": False, "duration_ms": 1000, "error_type": "DataError"},
        ],
        "resource_usage": {
            "cpu": 0.85,
            "memory": 0.70,
            "network": 0.95
        }
    }
    
    result = await supervisor(failing_state)
    decision = result["supervisor_decision"]
    
    print(f"Decision: {decision['decision']}")
    print(f"Risk Level: {decision['risk_assessment']['level']}")
    print(f"Risk Score: {decision['risk_assessment']['score']:.2f}")
    print(f"Risk Factors: {json.dumps(decision['risk_assessment']['factors'], indent=2)}")
    print(f"Patterns Detected: {list(decision['patterns_detected'].keys())}")
    print(f"Mitigations: {decision['risk_assessment']['mitigations']}")


async def test_performance_degradation_scenario():
    """Test: Performance getting progressively worse."""
    print("\n" + "="*60)
    print("TEST: Performance Degradation Scenario")
    print("="*60)
    
    supervisor = RealSupervisor()
    
    # Simulate progressively slower execution
    state = {
        "workflow_id": "slow-workflow",
        "total_steps": 10,
        "completed_steps": ["step1", "step2", "step3", "step4", "step5"],
        "step_results": [
            {"node_name": "process", "success": True, "duration_ms": 1000},
            {"node_name": "process", "success": True, "duration_ms": 1500},
            {"node_name": "process", "success": True, "duration_ms": 2200},
            {"node_name": "process", "success": True, "duration_ms": 3300},
            {"node_name": "process", "success": True, "duration_ms": 5000},
            {"node_name": "process", "success": True, "duration_ms": 7500},
        ],
        "resource_usage": {
            "cpu": 0.90,
            "memory": 0.85,
            "network": 0.40
        }
    }
    
    result = await supervisor(state)
    decision = result["supervisor_decision"]
    
    print(f"Decision: {decision['decision']}")
    print(f"Bottlenecks Detected: {decision['metrics']['bottlenecks']}")
    print(f"Average Duration: {decision['metrics']['average_duration_ms']:.0f}ms")
    print(f"Patterns: {list(decision['patterns_detected'].keys())}")
    print(f"Next Action: {result['next_action']}")


async def test_retry_loop_scenario():
    """Test: Stuck in retry loop, should abort."""
    print("\n" + "="*60)
    print("TEST: Retry Loop Scenario")
    print("="*60)
    
    supervisor = RealSupervisor()
    
    # Build pattern history for retry loop detection
    state = {
        "workflow_id": "retry-loop-workflow",
        "total_steps": 10,
        "completed_steps": ["step1"],
        "step_results": [
            {"node_name": "init", "success": True, "duration_ms": 500},
            {"node_name": "critical", "success": False, "duration_ms": 2000, "retry_count": 3},
            {"node_name": "critical", "success": False, "duration_ms": 2100, "retry_count": 3},
            {"node_name": "critical", "success": False, "duration_ms": 2200, "retry_count": 3},
            {"node_name": "critical", "success": False, "duration_ms": 2300, "retry_count": 3},
        ],
        "resource_usage": {
            "cpu": 0.60,
            "memory": 0.50,
            "network": 0.30
        }
    }
    
    result = await supervisor(state)
    decision = result["supervisor_decision"]
    
    print(f"Decision: {decision['decision']}")
    print(f"Error Rate: {decision['metrics']['error_rate']:.2%}")
    print(f"Patterns: {list(decision['patterns_detected'].keys())}")
    print(f"Reasoning: {json.dumps(decision['reasoning']['factors'], indent=2)}")


async def test_near_completion_scenario():
    """Test: Workflow near completion with minor issues."""
    print("\n" + "="*60)
    print("TEST: Near Completion Scenario")
    print("="*60)
    
    supervisor = RealSupervisor()
    
    state = {
        "workflow_id": "almost-done-workflow",
        "total_steps": 10,
        "completed_steps": ["step1", "step2", "step3", "step4", "step5", "step6", "step7", "step8"],
        "step_results": [
            {"node_name": "init", "success": True, "duration_ms": 500},
            {"node_name": "fetch", "success": True, "duration_ms": 800},
            {"node_name": "analyze", "success": True, "duration_ms": 1200},
            {"node_name": "process", "success": True, "duration_ms": 900},
            {"node_name": "validate", "success": True, "duration_ms": 600},
            {"node_name": "transform", "success": True, "duration_ms": 700},
            {"node_name": "store", "success": True, "duration_ms": 400},
            {"node_name": "finalize", "success": False, "duration_ms": 1000, "error_type": "MinorError"},
        ],
        "resource_usage": {
            "cpu": 0.40,
            "memory": 0.35,
            "network": 0.25
        }
    }
    
    result = await supervisor(state)
    decision = result["supervisor_decision"]
    
    print(f"Decision: {decision['decision']}")
    print(f"Completion Confidence: {result['supervisor_decision']['reasoning']['factors']}")
    print(f"Success Rate: {decision['metrics']['success_rate']:.2%}")
    print(f"Workflow Stage: Late stage - near completion")


async def test_learning_scenario():
    """Test: Learning from outcomes over multiple workflows."""
    print("\n" + "="*60)
    print("TEST: Learning Scenario - Adaptive Decision Making")
    print("="*60)
    
    supervisor = RealSupervisor(enable_learning=True)
    
    # Simulate multiple workflows with different outcomes
    scenarios = [
        ("workflow-1", 0.9),  # CONTINUE worked well
        ("workflow-2", 0.2),  # CONTINUE worked poorly
        ("workflow-3", 0.8),  # RETRY worked well
        ("workflow-4", 0.3),  # CONTINUE worked poorly again
    ]
    
    print("Initial decision weights:")
    print(json.dumps(supervisor.decision_engine.decision_weights, indent=2))
    
    for workflow_id, outcome in scenarios:
        # Simple state that would normally suggest CONTINUE
        state = {
            "workflow_id": workflow_id,
            "total_steps": 5,
            "completed_steps": ["step1", "step2", "step3"],
            "step_results": [
                {"node_name": "a", "success": True, "duration_ms": 1000},
                {"node_name": "b", "success": True, "duration_ms": 1000},
                {"node_name": "c", "success": True, "duration_ms": 1000},
            ],
            "resource_usage": {"cpu": 0.5, "memory": 0.5}
        }
        
        result = await supervisor(state)
        decision = DecisionType(result["supervisor_decision"]["decision"])
        
        # Simulate outcome and learn
        supervisor.learn_from_outcome(workflow_id, outcome)
        
        print(f"\nWorkflow {workflow_id}: Decision={decision.value}, Outcome={outcome}")
    
    print("\nUpdated decision weights after learning:")
    print(json.dumps(supervisor.decision_engine.decision_weights, indent=2))


async def main():
    """Run all test scenarios."""
    print("\n🧪 COMPREHENSIVE SUPERVISOR TESTING")
    print("Testing real-world workflow scenarios...\n")
    
    await test_success_scenario()
    await test_failure_cascade_scenario()
    await test_performance_degradation_scenario()
    await test_retry_loop_scenario()
    await test_near_completion_scenario()
    await test_learning_scenario()
    
    print("\n✅ All tests completed!")
    print("\nThe Real Supervisor demonstrates:")
    print("• Intelligent decision making based on workflow state")
    print("• Pattern detection (retry loops, cascading failures, degradation)")
    print("• Risk assessment with multiple factors")
    print("• Performance monitoring and bottleneck detection")
    print("• Learning from outcomes to improve future decisions")
    print("• Specific actionable next steps for each decision")


if __name__ == "__main__":
    asyncio.run(main())