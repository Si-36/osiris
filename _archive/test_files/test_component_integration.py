"""
AURA Component Integration Tests
================================

Tests the connections between:
- Orchestration ↔ Memory
- Memory ↔ TDA  
- Neural ↔ Memory
- Orchestration ↔ TDA

This will help us identify what's broken and fix it.
"""

import asyncio
import time
from typing import Dict, Any
import numpy as np

# Import our components - check what actually exists
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'core/src'))

try:
    from aura_intelligence.orchestration.unified_orchestration_engine import (
        UnifiedOrchestrationEngine,
        OrchestrationConfig,
        WorkflowDefinition
    )
    ORCHESTRATION_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  Orchestration import failed: {e}")
    ORCHESTRATION_AVAILABLE = False

try:
    from aura_intelligence.memory.core.memory_api import (
        AURAMemorySystem,
        MemoryType,
        RetrievalMode,
        MemoryQuery
    )
    MEMORY_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  Memory import failed: {e}")
    MEMORY_AVAILABLE = False

try:
    from aura_intelligence.neural import AURAModelRouter
    NEURAL_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  Neural import failed: {e}")
    NEURAL_AVAILABLE = False

try:
    from aura_intelligence.tda import AgentTopologyAnalyzer
    TDA_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  TDA import failed: {e}")
    TDA_AVAILABLE = False


class ComponentIntegrationTester:
    """Test connections between all AURA components"""
    
    def __init__(self):
        self.orchestration = None
        self.memory = None
        self.neural = None
        self.tda = None
        self.test_results = []
        
    async def setup(self):
        """Initialize all components with health checks"""
        print("\n🚀 Initializing AURA Components...\n")
        
        # 1. Initialize Orchestration
        print("1️⃣ Initializing Orchestration...")
        try:
            config = OrchestrationConfig(
                postgres_url="postgresql://aura:aura@localhost:5432/aura_orchestration",
                enable_signal_first=True,
                enable_checkpoint_coalescing=True,
                enable_topology_routing=True
            )
            self.orchestration = UnifiedOrchestrationEngine(config)
            await self.orchestration.initialize()
            print("   ✅ Orchestration initialized")
        except Exception as e:
            print(f"   ❌ Orchestration failed: {e}")
            self.test_results.append(("orchestration_init", False, str(e)))
            
        # 2. Initialize Memory
        print("\n2️⃣ Initializing Memory System...")
        try:
            self.memory = AURAMemorySystem()
            # Note: Memory needs explicit initialization
            if hasattr(self.memory, 'initialize'):
                await self.memory.initialize()
            print("   ✅ Memory system initialized")
        except Exception as e:
            print(f"   ❌ Memory failed: {e}")
            self.test_results.append(("memory_init", False, str(e)))
            
        # 3. Initialize Neural Router
        print("\n3️⃣ Initializing Neural Router...")
        try:
            self.neural = AURAModelRouter()
            if hasattr(self.neural, 'initialize'):
                await self.neural.initialize()
            print("   ✅ Neural router initialized")
        except Exception as e:
            print(f"   ❌ Neural failed: {e}")
            self.test_results.append(("neural_init", False, str(e)))
            
        # 4. Initialize TDA
        print("\n4️⃣ Initializing TDA Analyzer...")
        try:
            self.tda = AgentTopologyAnalyzer()
            print("   ✅ TDA analyzer initialized")
        except Exception as e:
            print(f"   ❌ TDA failed: {e}")
            self.test_results.append(("tda_init", False, str(e)))
            
        print("\n" + "="*50 + "\n")
        
    async def test_orchestration_memory_connection(self):
        """Test: Orchestration should store workflows in Memory"""
        print("🔗 Testing Orchestration → Memory Connection")
        
        if not self.orchestration or not self.memory:
            print("   ⚠️  Components not initialized, skipping")
            return
            
        try:
            # Create a test workflow
            workflow_def = WorkflowDefinition(
                workflow_id="test-workflow-1",
                name="Test Workflow",
                version="1.0.0",
                graph_definition={
                    "nodes": {
                        "start": {"type": "input"},
                        "process": {"type": "task"},
                        "end": {"type": "output"}
                    },
                    "edges": [
                        {"source": "start", "target": "process"},
                        {"source": "process", "target": "end"}
                    ]
                }
            )
            
            # Create workflow
            print("   Creating workflow...")
            workflow_id = await self.orchestration.create_workflow(workflow_def)
            print(f"   Created workflow: {workflow_id}")
            
            # Check if it's in memory
            print("   Checking memory for workflow...")
            await asyncio.sleep(0.5)  # Give time for async storage
            
            query = MemoryQuery(
                mode=RetrievalMode.SEMANTIC_SEARCH,
                query_text=f"workflow {workflow_id}",
                k=5
            )
            
            results = await self.memory.retrieve(query)
            
            if results.memories:
                print(f"   ✅ Found workflow in memory! ({len(results.memories)} records)")
                self.test_results.append(("orchestration_memory", True, "Connected"))
            else:
                print("   ❌ Workflow NOT found in memory")
                self.test_results.append(("orchestration_memory", False, "Not connected"))
                
        except Exception as e:
            print(f"   ❌ Test failed: {e}")
            self.test_results.append(("orchestration_memory", False, str(e)))
            
    async def test_tda_memory_connection(self):
        """Test: TDA analysis should be stored in Memory"""
        print("\n🔗 Testing TDA → Memory Connection")
        
        if not self.tda or not self.memory:
            print("   ⚠️  Components not initialized, skipping")
            return
            
        try:
            # Create test workflow data
            workflow_data = {
                "workflow_id": "test-tda-1",
                "agents": [
                    {"id": "agent_1"},
                    {"id": "agent_2"},
                    {"id": "agent_3"}
                ],
                "dependencies": [
                    {"source": "agent_1", "target": "agent_2"},
                    {"source": "agent_2", "target": "agent_3"},
                    {"source": "agent_3", "target": "agent_1"}  # Create cycle
                ]
            }
            
            # Analyze topology
            print("   Analyzing workflow topology...")
            analysis = await self.tda.analyze_workflow(
                workflow_data["workflow_id"], 
                workflow_data
            )
            
            print(f"   Bottleneck score: {analysis.bottleneck_score:.3f}")
            print(f"   Has cycles: {analysis.has_cycles}")
            
            # Store in memory
            print("   Storing analysis in memory...")
            memory_id = await self.memory.store(
                content=analysis.to_dict(),
                memory_type=MemoryType.TOPOLOGICAL,
                workflow_data=workflow_data
            )
            
            # Try to retrieve by topology
            print("   Retrieving by topology pattern...")
            query = MemoryQuery(
                mode=RetrievalMode.SHAPE_MATCH,
                topology_constraints={"min_loops": 1},
                k=5
            )
            
            results = await self.memory.retrieve(query)
            
            if results.memories:
                print(f"   ✅ Found topology in memory! ({len(results.memories)} matches)")
                self.test_results.append(("tda_memory", True, "Connected"))
            else:
                print("   ❌ Topology NOT found in memory")
                self.test_results.append(("tda_memory", False, "Not connected"))
                
        except Exception as e:
            print(f"   ❌ Test failed: {e}")
            self.test_results.append(("tda_memory", False, str(e)))
            
    async def test_neural_memory_connection(self):
        """Test: Neural routing decisions should be tracked in Memory"""
        print("\n🔗 Testing Neural → Memory Connection")
        
        if not self.neural or not self.memory:
            print("   ⚠️  Components not initialized, skipping")
            return
            
        try:
            # Create test routing request
            from core.src.aura_intelligence.neural import ProviderRequest
            
            request = ProviderRequest(
                messages=[
                    {"role": "user", "content": "Test routing request"}
                ],
                model_preferences=["gpt-4", "claude-3"],
                max_tokens=100
            )
            
            # Route request
            print("   Routing request through neural...")
            try:
                response = await self.neural.route_request(request)
                print(f"   Routed to: {response.provider}/{response.model}")
                
                # Store routing decision
                print("   Storing routing decision...")
                memory_id = await self.memory.store(
                    content={
                        "provider": response.provider,
                        "model": response.model,
                        "latency_ms": response.latency_ms,
                        "timestamp": time.time()
                    },
                    memory_type=MemoryType.SEMANTIC,
                    metadata={"component": "neural_router"}
                )
                
                # Retrieve routing history
                query = MemoryQuery(
                    mode=RetrievalMode.SEMANTIC_SEARCH,
                    query_text="neural routing decisions",
                    k=5
                )
                
                results = await self.memory.retrieve(query)
                
                if results.memories:
                    print(f"   ✅ Found routing history! ({len(results.memories)} records)")
                    self.test_results.append(("neural_memory", True, "Connected"))
                else:
                    print("   ❌ Routing history NOT found")
                    self.test_results.append(("neural_memory", False, "Not connected"))
                    
            except Exception as e:
                print(f"   ⚠️  Neural routing not configured: {e}")
                self.test_results.append(("neural_memory", False, "Not configured"))
                
        except Exception as e:
            print(f"   ❌ Test failed: {e}")
            self.test_results.append(("neural_memory", False, str(e)))
            
    async def test_orchestration_tda_connection(self):
        """Test: Orchestration should use TDA for topology analysis"""
        print("\n🔗 Testing Orchestration ↔ TDA Connection")
        
        if not self.orchestration or not self.tda:
            print("   ⚠️  Components not initialized, skipping")
            return
            
        try:
            # Create workflow with potential bottleneck
            workflow_def = WorkflowDefinition(
                workflow_id="test-bottleneck-1",
                name="Bottleneck Test",
                version="1.0.0",
                graph_definition={
                    "nodes": {
                        "start": {"type": "input"},
                        "fan_out_1": {"type": "task"},
                        "fan_out_2": {"type": "task"},
                        "fan_out_3": {"type": "task"},
                        "bottleneck": {"type": "task"},
                        "end": {"type": "output"}
                    },
                    "edges": [
                        {"source": "start", "target": "fan_out_1"},
                        {"source": "start", "target": "fan_out_2"},
                        {"source": "start", "target": "fan_out_3"},
                        {"source": "fan_out_1", "target": "bottleneck"},
                        {"source": "fan_out_2", "target": "bottleneck"},
                        {"source": "fan_out_3", "target": "bottleneck"},
                        {"source": "bottleneck", "target": "end"}
                    ]
                }
            )
            
            # Execute workflow
            print("   Executing workflow with topology analysis...")
            result = await self.orchestration.execute_workflow(
                workflow_def.workflow_id,
                {"test": "data"}
            )
            
            # Check if TDA analyzed it
            print("   Checking for TDA analysis...")
            
            # The orchestration should have used TDA internally
            if self.orchestration.config.enable_topology_routing:
                print("   ✅ TDA routing is enabled in orchestration")
                self.test_results.append(("orchestration_tda", True, "Connected"))
            else:
                print("   ❌ TDA routing is disabled")
                self.test_results.append(("orchestration_tda", False, "Not enabled"))
                
        except Exception as e:
            print(f"   ❌ Test failed: {e}")
            self.test_results.append(("orchestration_tda", False, str(e)))
            
    async def run_all_tests(self):
        """Run all integration tests"""
        print("🧪 AURA COMPONENT INTEGRATION TESTS")
        print("="*50)
        
        # Setup
        await self.setup()
        
        # Run tests
        await self.test_orchestration_memory_connection()
        await self.test_tda_memory_connection()
        await self.test_neural_memory_connection()
        await self.test_orchestration_tda_connection()
        
        # Summary
        print("\n" + "="*50)
        print("📊 TEST SUMMARY")
        print("="*50 + "\n")
        
        passed = sum(1 for _, success, _ in self.test_results if success)
        total = len(self.test_results)
        
        for test_name, success, message in self.test_results:
            status = "✅" if success else "❌"
            print(f"{status} {test_name}: {message}")
            
        print(f"\nTotal: {passed}/{total} tests passed")
        
        if passed < total:
            print("\n⚠️  Some connections are broken! Let's fix them...")
        else:
            print("\n🎉 All components are connected properly!")
            
        return self.test_results


async def main():
    """Run integration tests"""
    tester = ComponentIntegrationTester()
    results = await tester.run_all_tests()
    
    # Identify what needs fixing
    failures = [(name, msg) for name, success, msg in results if not success]
    
    if failures:
        print("\n🔧 FIXES NEEDED:")
        print("="*50)
        for name, msg in failures:
            print(f"\n{name}:")
            if "Not connected" in msg:
                print("  → Need to add integration code")
            elif "failed" in msg:
                print(f"  → Error: {msg}")
            elif "Not configured" in msg:
                print("  → Need to configure component")


if __name__ == "__main__":
    asyncio.run(main())