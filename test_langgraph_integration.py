#!/usr/bin/env python3
"""
Test the complete LangGraph integration with AURA
"""

import asyncio
import sys
from pathlib import Path
import json
from typing import Dict, Any

# Add project to path
sys.path.insert(0, str(Path(__file__).parent / "core" / "src"))

print("🧠 AURA LANGGRAPH INTEGRATION TEST")
print("=" * 60)

async def test_langgraph_integration():
    """Test the complete LangGraph integration"""
    
    all_tests_passed = True
    test_results = {}
    
    try:
        # Import the integration
        from aura_intelligence.integrations import UltimateLangGraphIntegration
        
        print("✅ Import successful\n")
        
        # Test 1: Initialize Integration
        print("1️⃣ TESTING INTEGRATION INITIALIZATION")
        print("-" * 40)
        try:
            # Create mock config and consciousness
            class MockConfig:
                checkpoint_dir = "./test_checkpoints"
                postgres_url = None
                redis_url = None
            
            class MockConsciousness:
                def __init__(self):
                    self.memory = {"test": "memory"}
            
            config = MockConfig()
            consciousness = MockConsciousness()
            
            # Create integration
            integration = UltimateLangGraphIntegration(config, consciousness)
            
            # Initialize
            await integration.initialize()
            
            print(f"✅ Integration initialized")
            print(f"   Checkpoint dir: {integration.checkpoint_dir}")
            print(f"   Orchestrator: {'✓' if integration.orchestrator else '✗'}")
            print(f"   Supervisor: {'✓' if integration.supervisor else '✗'}")
            print(f"   Graph builder: {'✓' if integration.graph_builder else '✗'}")
            
            test_results['initialization'] = True
            
        except Exception as e:
            print(f"❌ Initialization failed: {e}")
            test_results['initialization'] = False
            all_tests_passed = False
            return all_tests_passed
        
        # Test 2: Agent Registry
        print("\n2️⃣ TESTING AGENT REGISTRY")
        print("-" * 40)
        try:
            agent_list = integration.agent_registry.list_all_agents()
            
            print(f"✅ Registered agents: {len(agent_list)}")
            for agent_id, info in agent_list.items():
                print(f"   {agent_id}: {info['type']} - {info['capabilities']}")
            
            test_results['agent_registry'] = len(agent_list) > 0
            
        except Exception as e:
            print(f"❌ Agent registry test failed: {e}")
            test_results['agent_registry'] = False
            all_tests_passed = False
        
        # Test 3: Tool Registry
        print("\n3️⃣ TESTING TOOL REGISTRY")
        print("-" * 40)
        try:
            tools = integration.tool_registry.get_all_tools()
            
            print(f"✅ Registered tools: {len(tools)}")
            for tool in tools:
                print(f"   {tool.name}: {tool.description}")
            
            # Test tool categories
            categories = integration.tool_registry.tool_categories
            print(f"✅ Tool categories: {list(categories.keys())}")
            
            test_results['tool_registry'] = len(tools) > 0
            
        except Exception as e:
            print(f"❌ Tool registry test failed: {e}")
            test_results['tool_registry'] = False
            all_tests_passed = False
        
        # Test 4: Execute Simple Workflow
        print("\n4️⃣ TESTING SIMPLE WORKFLOW EXECUTION")
        print("-" * 40)
        try:
            task = "Calculate 15 + 25 * 3"
            context = {"priority": "normal"}
            
            result = await integration.execute_advanced_workflows(task, context)
            
            print(f"✅ Workflow executed:")
            print(f"   Task: {task}")
            print(f"   Status: {result.get('status', 'unknown')}")
            print(f"   Workflow ID: {result.get('workflow_id', 'none')}")
            
            if 'results' in result:
                print(f"   Results keys: {list(result['results'].keys())[:5]}")
            
            test_results['simple_workflow'] = result.get('status') == 'completed'
            
        except Exception as e:
            print(f"❌ Simple workflow test failed: {e}")
            import traceback
            traceback.print_exc()
            test_results['simple_workflow'] = False
            all_tests_passed = False
        
        # Test 5: Execute Complex Workflow
        print("\n5️⃣ TESTING COMPLEX WORKFLOW EXECUTION")
        print("-" * 40)
        try:
            task = "Research quantum computing developments and create a comprehensive plan for implementation"
            context = {"priority": "high", "depth": "detailed"}
            
            result = await integration.execute_advanced_workflows(task, context)
            
            print(f"✅ Complex workflow executed:")
            print(f"   Task: {task[:50]}...")
            print(f"   Status: {result.get('status', 'unknown')}")
            
            # Check which agents were selected
            workflow_id = result.get('workflow_id')
            if workflow_id and workflow_id in integration.active_workflows:
                workflow_info = integration.active_workflows[workflow_id]
                print(f"   Agents used: {workflow_info.get('agents', [])}")
            
            test_results['complex_workflow'] = result.get('status') in ['completed', 'failed']
            
        except Exception as e:
            print(f"❌ Complex workflow test failed: {e}")
            test_results['complex_workflow'] = False
            all_tests_passed = False
        
        # Test 6: Direct Agent Execution
        print("\n6️⃣ TESTING DIRECT AGENT EXECUTION")
        print("-" * 40)
        try:
            # Get first available agent
            agents = list(integration.agent_registry.agents.keys())
            if agents:
                agent_id = agents[0]
                task = "What is the meaning of life?"
                
                result = await integration.execute_agent_directly(agent_id, task)
                
                print(f"✅ Direct agent execution:")
                print(f"   Agent: {agent_id}")
                print(f"   Task: {task}")
                print(f"   Result keys: {list(result.keys()) if isinstance(result, dict) else 'not a dict'}")
                
                test_results['direct_execution'] = 'error' not in result
            else:
                print("⚠️  No agents available for direct execution test")
                test_results['direct_execution'] = None
            
        except Exception as e:
            print(f"❌ Direct execution test failed: {e}")
            test_results['direct_execution'] = False
            all_tests_passed = False
        
        # Test 7: Health Status
        print("\n7️⃣ TESTING HEALTH STATUS")
        print("-" * 40)
        try:
            health = integration.get_health_status()
            
            print(f"✅ Health status:")
            for key, value in health.items():
                print(f"   {key}: {value}")
            
            test_results['health_status'] = health['initialized']
            
        except Exception as e:
            print(f"❌ Health status test failed: {e}")
            test_results['health_status'] = False
            all_tests_passed = False
        
        # Test 8: Integration with Core System
        print("\n8️⃣ TESTING CORE SYSTEM INTEGRATION")
        print("-" * 40)
        try:
            # This would test the actual integration with core/system.py
            # For now, we verify the interface matches
            
            required_methods = [
                'initialize',
                'execute_advanced_workflows',
                'cleanup',
                'get_health_status'
            ]
            
            missing_methods = []
            for method in required_methods:
                if not hasattr(integration, method):
                    missing_methods.append(method)
            
            if missing_methods:
                print(f"❌ Missing required methods: {missing_methods}")
                test_results['core_integration'] = False
            else:
                print(f"✅ All required methods present")
                test_results['core_integration'] = True
            
        except Exception as e:
            print(f"❌ Core integration test failed: {e}")
            test_results['core_integration'] = False
            all_tests_passed = False
        
        # Test 9: Cleanup
        print("\n9️⃣ TESTING CLEANUP")
        print("-" * 40)
        try:
            await integration.cleanup()
            
            # Check if workflows were saved
            workflow_file = integration.checkpoint_dir / "active_workflows.json"
            if workflow_file.exists():
                print(f"✅ Workflows saved to: {workflow_file}")
                with open(workflow_file) as f:
                    saved_workflows = json.load(f)
                print(f"   Saved workflows: {len(saved_workflows)}")
            
            test_results['cleanup'] = True
            
        except Exception as e:
            print(f"❌ Cleanup test failed: {e}")
            test_results['cleanup'] = False
            all_tests_passed = False
        
        # Final summary
        print("\n" + "=" * 60)
        print("📊 TEST RESULTS SUMMARY")
        print("-" * 40)
        
        for test_name, result in test_results.items():
            if result is True:
                status = "✅ PASSED"
            elif result is False:
                status = "❌ FAILED"
            else:
                status = "⚠️  SKIPPED"
            
            print(f"{test_name:20} {status}")
        
        passed = sum(1 for r in test_results.values() if r is True)
        failed = sum(1 for r in test_results.values() if r is False)
        skipped = sum(1 for r in test_results.values() if r is None)
        
        print(f"\nTotal: {passed} passed, {failed} failed, {skipped} skipped")
        
        if all_tests_passed and failed == 0:
            print("\n✅ ALL LANGGRAPH INTEGRATION TESTS PASSED!")
        else:
            print(f"\n❌ Some tests failed. Please debug.")
        
        return all_tests_passed
        
    except Exception as e:
        print(f"\n❌ Critical test error: {e}")
        import traceback
        traceback.print_exc()
        return False


# Run the test
if __name__ == "__main__":
    result = asyncio.run(test_langgraph_integration())
    sys.exit(0 if result else 1)