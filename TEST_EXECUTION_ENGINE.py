#!/usr/bin/env python3
"""
Test script for AURA Execution Engine
Verifies that all components work together correctly
"""

import asyncio
import sys
import traceback
from datetime import datetime

# Test imports
print("=" * 60)
print("TESTING AURA EXECUTION ENGINE")
print("=" * 60)
print()

# Test 1: Import all modules
print("Test 1: Importing modules...")
try:
    from core.src.aura_intelligence.schemas.aura_execution import (
        AuraTask,
        AuraWorkflowState,
        TaskStatus,
        ExecutionPlan,
        ExecutionStep,
        ConsensusDecision,
        ObservationResult,
        TopologicalSignature,
        MemoryContext
    )
    print("✅ Schemas imported successfully")
except Exception as e:
    print(f"❌ Failed to import schemas: {e}")
    traceback.print_exc()
    sys.exit(1)

try:
    from core.src.aura_intelligence.orchestration.execution_engine import (
        UnifiedWorkflowExecutor,
        WorkflowMonitor
    )
    print("✅ Execution engine imported successfully")
except Exception as e:
    print(f"❌ Failed to import execution engine: {e}")
    traceback.print_exc()
    sys.exit(1)

try:
    from core.src.aura_intelligence.orchestration.aura_cognitive_workflow import (
        create_aura_workflow
    )
    print("✅ Workflow imported successfully")
except Exception as e:
    print(f"❌ Failed to import workflow: {e}")
    traceback.print_exc()
    sys.exit(1)

try:
    from core.src.aura_intelligence.tools.tool_registry import (
        ToolRegistry,
        ToolExecutor,
        ToolMetadata,
        ToolCategory
    )
    from core.src.aura_intelligence.tools.implementations.observation_tool import (
        SystemObservationTool,
        ObservationPlanner,
        PrometheusClient
    )
    print("✅ Tools imported successfully")
except Exception as e:
    print(f"❌ Failed to import tools: {e}")
    traceback.print_exc()
    sys.exit(1)

try:
    from core.src.aura_intelligence.consensus.real_consensus import (
        RealConsensusSystem,
        VotingStrategy,
        ConsensusProtocol
    )
    print("✅ Consensus system imported successfully")
except Exception as e:
    print(f"❌ Failed to import consensus: {e}")
    traceback.print_exc()
    sys.exit(1)

try:
    from core.src.aura_intelligence.agents.cognitive_agents import (
        PerceptionAgent,
        PlannerAgent,
        AnalystAgent
    )
    print("✅ Agents imported successfully")
except Exception as e:
    print(f"❌ Failed to import agents: {e}")
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 60)
print("Test 2: Creating components...")
print("=" * 60)

# Test 2: Create components
async def test_component_creation():
    """Test creating all components"""
    
    # Create agents
    print("\nCreating agents...")
    try:
        perception = PerceptionAgent()
        planner = PlannerAgent()
        analyst = AnalystAgent()
        agents = {
            "PerceptionAgent": perception,
            "PlannerAgent": planner,
            "AnalystAgent": analyst
        }
        print("✅ Agents created")
    except Exception as e:
        print(f"❌ Failed to create agents: {e}")
        return False
    
    # Create tools
    print("Creating tools...")
    try:
        tool_registry = ToolRegistry()
        obs_tool = SystemObservationTool()
        tool_registry.register(
            "SystemObservationTool",
            obs_tool,
            ToolMetadata(
                name="SystemObservationTool",
                category=ToolCategory.OBSERVATION,
                description="System observation with TDA"
            )
        )
        print("✅ Tools created and registered")
    except Exception as e:
        print(f"❌ Failed to create tools: {e}")
        return False
    
    # Create consensus
    print("Creating consensus system...")
    try:
        consensus = RealConsensusSystem()
        print("✅ Consensus system created")
    except Exception as e:
        print(f"❌ Failed to create consensus: {e}")
        return False
    
    # Create executor
    print("Creating executor...")
    try:
        executor = UnifiedWorkflowExecutor(
            memory=None,  # Will use default
            agents=agents,
            tools=tool_registry,
            consensus=consensus
        )
        print("✅ Executor created")
    except Exception as e:
        print(f"❌ Failed to create executor: {e}")
        traceback.print_exc()
        return False
    
    return executor


print("\n" + "=" * 60)
print("Test 3: Running simple task...")
print("=" * 60)

async def test_simple_task(executor):
    """Test running a simple task"""
    
    task = "Analyze system performance and detect anomalies"
    environment = {"target": "test-system", "duration": "5m"}
    
    print(f"\nTask: {task}")
    print(f"Environment: {environment}")
    print("\nExecuting...")
    
    try:
        start = datetime.utcnow()
        result = await executor.execute_task(task, environment)
        duration = (datetime.utcnow() - start).total_seconds()
        
        print(f"\n✅ Task completed in {duration:.2f} seconds")
        print(f"Status: {result.get('status')}")
        print(f"Task ID: {result.get('task_id')}")
        
        if result.get('execution_trace'):
            print("\nExecution trace (last 5):")
            for trace in result['execution_trace'][-5:]:
                print(f"  {trace}")
        
        if result.get('patterns'):
            print(f"\nPatterns found: {result.get('patterns_found', 0)}")
            for pattern in result['patterns'][:3]:
                print(f"  - {pattern.get('type')}: {pattern.get('description', 'N/A')}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Task execution failed: {e}")
        traceback.print_exc()
        return False


print("\n" + "=" * 60)
print("Test 4: Testing workflow nodes...")
print("=" * 60)

async def test_workflow_nodes():
    """Test individual workflow nodes"""
    
    from core.src.aura_intelligence.orchestration.aura_cognitive_workflow import (
        perception_node,
        planning_node,
        consensus_node,
        execution_node,
        analysis_consolidation_node
    )
    
    # Create minimal state
    executor = await test_component_creation()
    if not executor:
        return False
    
    state = {
        "task": AuraTask(
            objective="Test task",
            environment={"target": "test"}
        ).model_dump(),
        "executor_instance": executor,
        "plan": None,
        "consensus": None,
        "observations": [],
        "patterns": [],
        "execution_trace": [],
        "metrics": {}
    }
    
    print("\nTesting perception node...")
    try:
        result = await perception_node(state)
        print("✅ Perception node executed")
        state.update(result)
    except Exception as e:
        print(f"❌ Perception node failed: {e}")
        return False
    
    print("Testing planning node...")
    try:
        result = await planning_node(state)
        print("✅ Planning node executed")
        state.update(result)
    except Exception as e:
        print(f"❌ Planning node failed: {e}")
        return False
    
    print("Testing consensus node...")
    try:
        result = await consensus_node(state)
        print("✅ Consensus node executed")
        state.update(result)
    except Exception as e:
        print(f"❌ Consensus node failed: {e}")
        return False
    
    print("Testing execution node...")
    try:
        result = await execution_node(state)
        print("✅ Execution node executed")
        state.update(result)
    except Exception as e:
        print(f"❌ Execution node failed: {e}")
        return False
    
    print("Testing analysis node...")
    try:
        result = await analysis_consolidation_node(state)
        print("✅ Analysis node executed")
    except Exception as e:
        print(f"❌ Analysis node failed: {e}")
        return False
    
    return True


async def main():
    """Main test function"""
    
    # Test component creation
    executor = await test_component_creation()
    if not executor:
        print("\n❌ Component creation failed")
        return False
    
    # Test simple task
    print()
    success = await test_simple_task(executor)
    if not success:
        print("\n❌ Simple task test failed")
        return False
    
    # Test workflow nodes
    print()
    success = await test_workflow_nodes()
    if not success:
        print("\n❌ Workflow node test failed")
        return False
    
    # Test metrics
    print("\n" + "=" * 60)
    print("Test 5: Checking metrics...")
    print("=" * 60)
    
    metrics = executor.get_metrics()
    print(f"\nExecutor metrics:")
    print(f"  Tasks executed: {metrics['tasks_executed']}")
    print(f"  Tasks succeeded: {metrics['tasks_succeeded']}")
    print(f"  Tasks failed: {metrics['tasks_failed']}")
    
    if executor.consensus:
        consensus_metrics = executor.consensus.get_metrics()
        print(f"\nConsensus metrics:")
        print(f"  Total attempts: {consensus_metrics['total_consensus_attempts']}")
        print(f"  Successful: {consensus_metrics['successful_consensus']}")
    
    # Shutdown
    print("\n" + "=" * 60)
    print("Shutting down...")
    await executor.shutdown()
    print("✅ Shutdown complete")
    
    return True


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("AURA EXECUTION ENGINE TEST SUITE")
    print("=" * 60)
    
    try:
        success = asyncio.run(main())
        
        if success:
            print("\n" + "=" * 60)
            print("✅ ALL TESTS PASSED")
            print("The AURA Execution Engine is working correctly!")
            print("=" * 60)
        else:
            print("\n" + "=" * 60)
            print("❌ SOME TESTS FAILED")
            print("Please review the errors above")
            print("=" * 60)
            
    except KeyboardInterrupt:
        print("\n⚠️ Tests interrupted by user")
    except Exception as e:
        print(f"\n❌ Test suite failed: {e}")
        traceback.print_exc()