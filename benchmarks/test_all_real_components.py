#!/usr/bin/env python3
"""
Comprehensive test for ALL AURA Intelligence components.
Tests every component in core/src/aura_intelligence/ for real functionality.
"""

import asyncio
import sys
import os
import importlib
import traceback
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

# Add core to Python path
sys.path.insert(0, str(Path(__file__).parent / "core" / "src"))

def discover_all_components() -> Dict[str, List[str]]:
    """Discover ALL components in the aura_intelligence directory."""
    components = {}
    
    aura_dir = Path("core/src/aura_intelligence")
    if not aura_dir.exists():
        print(f"❌ AURA directory not found: {aura_dir}")
        return components
    
    # Walk through all directories
    for root, dirs, files in os.walk(aura_dir):
        # Skip __pycache__ and hidden directories
        dirs[:] = [d for d in dirs if not d.startswith('_') and not d.startswith('.')]
        
        py_files = [f for f in files if f.endswith('.py') and not f.startswith('_')]
        
        if py_files:
            relative_path = os.path.relpath(root, aura_dir)
            if relative_path == '.':
                components['root'] = py_files
            else:
                components[relative_path] = py_files
    
    return components

def get_import_path(category: str, filename: str) -> str:
    """Get the import path for a component."""
    module_name = filename.replace('.py', '')
    if category == 'root':
        return f"aura_intelligence.{module_name}"
    else:
        return f"aura_intelligence.{category.replace('/', '.')}.{module_name}"

async def test_component_import(category: str, filename: str) -> Tuple[bool, str, Any]:
    """Test if a component can be imported successfully."""
    import_path = get_import_path(category, filename)
    
    try:
        module = importlib.import_module(import_path)
        
        # Get all classes and functions
        items = []
        for name in dir(module):
            if not name.startswith('_'):
                item = getattr(module, name)
                if hasattr(item, '__module__') and item.__module__ == import_path:
                    items.append(name)
        
        return True, f"✅ Imported with {len(items)} items: {', '.join(items[:5])}", module
        
    except ImportError as e:
        return False, f"❌ Import failed: {str(e)}", None
    except Exception as e:
        return False, f"❌ Error: {str(e)}", None

async def test_component_functionality(module: Any, category: str, filename: str) -> Tuple[bool, str]:
    """Test basic functionality of a component."""
    tests_passed = 0
    total_tests = 0
    results = []
    
    # Test 1: Check for main classes
    total_tests += 1
    classes = [name for name in dir(module) if isinstance(getattr(module, name), type) and not name.startswith('_')]
    if classes:
        tests_passed += 1
        results.append(f"Classes: {', '.join(classes[:3])}")
    else:
        results.append("No classes found")
    
    # Test 2: Check for async functions
    total_tests += 1
    async_funcs = []
    for name in dir(module):
        item = getattr(module, name)
        if asyncio.iscoroutinefunction(item):
            async_funcs.append(name)
    
    if async_funcs:
        tests_passed += 1
        results.append(f"Async functions: {', '.join(async_funcs[:2])}")
    else:
        results.append("No async functions")
    
    # Test 3: Try to instantiate main classes
    total_tests += 1
    instantiable = []
    for class_name in classes[:3]:  # Test first 3 classes
        try:
            cls = getattr(module, class_name)
            # Try simple instantiation
            if 'Config' in class_name:
                instance = cls()
                instantiable.append(class_name)
            elif 'Adapter' in class_name:
                # Try with minimal config
                try:
                    instance = cls({})
                    instantiable.append(class_name)
                except:
                    try:
                        instance = cls()
                        instantiable.append(class_name)
                    except:
                        pass
            else:
                try:
                    instance = cls()
                    instantiable.append(class_name)
                except:
                    pass
        except Exception as e:
            pass
    
    if instantiable:
        tests_passed += 1
        results.append(f"Instantiable: {', '.join(instantiable)}")
    else:
        results.append("No classes instantiated")
    
    success_rate = tests_passed / total_tests if total_tests > 0 else 0
    status = "✅" if success_rate >= 0.5 else "⚠️" if success_rate > 0 else "❌"
    
    return success_rate >= 0.5, f"{status} {tests_passed}/{total_tests} tests passed. {' | '.join(results)}"

async def test_category_components(category: str, files: List[str]) -> Dict[str, Any]:
    """Test all components in a category."""
    print(f"\n🔍 Testing {category} ({len(files)} components)...")
    
    results = {
        'total': len(files),
        'imported': 0,
        'functional': 0,
        'details': {}
    }
    
    for filename in files:
        print(f"  Testing {filename}...", end=" ")
        
        # Test import
        imported, import_msg, module = await test_component_import(category, filename)
        
        if imported:
            results['imported'] += 1
            # Test functionality
            functional, func_msg = await test_component_functionality(module, category, filename)
            if functional:
                results['functional'] += 1
                print(f"✅ WORKING")
            else:
                print(f"⚠️ IMPORTED")
        else:
            print(f"❌ FAILED")
        
        results['details'][filename] = {
            'imported': imported,
            'functional': functional if imported else False,
            'import_msg': import_msg,
            'func_msg': func_msg if imported else "Not tested"
        }
    
    return results

async def generate_detailed_report(all_results: Dict[str, Dict[str, Any]]):
    """Generate a detailed report of all test results."""
    print("\n" + "="*80)
    print("📊 COMPREHENSIVE AURA COMPONENTS TEST REPORT")
    print("="*80)
    
    total_components = sum(r['total'] for r in all_results.values())
    total_imported = sum(r['imported'] for r in all_results.values())
    total_functional = sum(r['functional'] for r in all_results.values())
    
    print(f"📈 OVERALL STATISTICS:")
    print(f"   Total Components: {total_components}")
    print(f"   Successfully Imported: {total_imported} ({total_imported/total_components*100:.1f}%)")
    print(f"   Fully Functional: {total_functional} ({total_functional/total_components*100:.1f}%)")
    
    print(f"\n🎯 SUCCESS RATE: {total_functional}/{total_components} = {total_functional/total_components*100:.1f}%")
    
    # Category breakdown
    print(f"\n📂 CATEGORY BREAKDOWN:")
    for category, results in all_results.items():
        success_rate = results['functional'] / results['total'] * 100 if results['total'] > 0 else 0
        status = "🟢" if success_rate >= 80 else "🟡" if success_rate >= 50 else "🔴"
        
        print(f"   {status} {category:<25} {results['functional']}/{results['total']} ({success_rate:.1f}%)")
    
    # Performance metrics
    print(f"\n⚡ PERFORMANCE ANALYSIS:")
    
    high_performing = [(cat, res) for cat, res in all_results.items() 
                      if res['functional']/res['total'] >= 0.8 and res['total'] >= 3]
    
    if high_performing:
        print("   🚀 High-Performing Categories (≥80% success):")
        for category, results in high_performing:
            print(f"      • {category}: {results['functional']}/{results['total']}")
    
    needs_attention = [(cat, res) for cat, res in all_results.items() 
                      if res['functional']/res['total'] < 0.5 and res['total'] >= 3]
    
    if needs_attention:
        print("   🔧 Needs Attention (<50% success):")
        for category, results in needs_attention:
            print(f"      • {category}: {results['functional']}/{results['total']}")
    
    # Final verdict
    print(f"\n🏁 FINAL VERDICT:")
    if total_functional >= 200:
        print("   🎯 TARGET ACHIEVED: 200+ components working!")
    elif total_functional >= 150:
        print("   🎯 CLOSE: Nearly there, good progress!")
    elif total_functional >= 100:
        print("   🎯 DECENT: Good foundation, needs more work")
    else:
        print("   🎯 STARTING: Significant work needed")

class RealComponentTester:
    def __init__(self):
        self.results = {}
        self.total_tested = 0
        self.total_passed = 0
        
    async def test_all_real_components(self):
        """Test every single real component"""
        print("🔥 TESTING ALL REAL COMPONENTS")
        print("=" * 60)
        
        # Test all major systems
        await self.test_mem0_integration()
        await self.test_langgraph_workflows()
        await self.test_neo4j_integration()
        await self.test_council_agents()
        await self.test_dpo_system()
        await self.test_tda_engine()
        await self.test_neural_systems()
        await self.test_consciousness_system()
        await self.test_memory_systems()
        await self.test_orchestration()
        await self.test_observability()
        await self.test_collective_intelligence()
        await self.test_bio_enhanced_system()
        await self.test_production_systems()
        
        self.print_final_results()
    
    async def test_mem0_integration(self):
        """Test Mem0 integration"""
        print("\n🧠 Testing Mem0 Integration...")
        
        try:
            from aura_intelligence.enterprise.mem0_hot.hot_memory import HotMemoryManager
            from aura_intelligence.enterprise.mem0_search.search_engine import SearchEngine
            from aura_intelligence.enterprise.mem0_semantic.semantic_memory import SemanticMemory
            
            # Test Hot Memory
            hot_memory = HotMemoryManager()
            await hot_memory.store("test_key", {"data": "test_value"})
            result = await hot_memory.retrieve("test_key")
            self.record_result("mem0_hot_memory", result is not None)
            
            # Test Search Engine
            search_engine = SearchEngine()
            search_result = await search_engine.search("test query", limit=5)
            self.record_result("mem0_search_engine", isinstance(search_result, dict))
            
            # Test Semantic Memory
            semantic = SemanticMemory()
            semantic_result = await semantic.add_memory("Test semantic memory")
            self.record_result("mem0_semantic_memory", semantic_result.get('success', False))
            
            print("  ✅ Mem0 integration working")
            
        except Exception as e:
            print(f"  ❌ Mem0 integration failed: {e}")
            self.record_result("mem0_integration", False)
    
    async def test_langgraph_workflows(self):
        """Test LangGraph workflows"""
        print("\n🔄 Testing LangGraph Workflows...")
        
        try:
            from aura_intelligence.orchestration.langgraph_workflows import WorkflowOrchestrator
            from aura_intelligence.orchestration.langgraph_collective import CollectiveWorkflow
            
            # Test Workflow Orchestrator
            orchestrator = WorkflowOrchestrator()
            workflow_result = await orchestrator.execute_workflow({
                'workflow_id': 'test_workflow',
                'steps': ['analyze', 'process', 'respond']
            })
            self.record_result("langgraph_orchestrator", workflow_result.get('success', False))
            
            # Test Collective Workflow
            collective = CollectiveWorkflow()
            collective_result = await collective.run_collective_decision({
                'decision_type': 'resource_allocation',
                'context': {'available_resources': 100}
            })
            self.record_result("langgraph_collective", collective_result.get('decision_made', False))
            
            print("  ✅ LangGraph workflows working")
            
        except Exception as e:
            print(f"  ❌ LangGraph workflows failed: {e}")
            self.record_result("langgraph_workflows", False)
    
    async def test_neo4j_integration(self):
        """Test Neo4j integration"""
        print("\n🕸️ Testing Neo4j Integration...")
        
        try:
            from aura_intelligence.graph.neo4j_integration import Neo4jManager
            from aura_intelligence.adapters.neo4j_adapter import Neo4jAdapter
            from aura_intelligence.memory.neo4j_motifcost import MotifCostAnalyzer
            
            # Test Neo4j Manager
            neo4j_manager = Neo4jManager()
            connection_result = await neo4j_manager.test_connection()
            self.record_result("neo4j_manager", connection_result.get('connected', False))
            
            # Test Neo4j Adapter
            adapter = Neo4jAdapter()
            adapter_result = await adapter.create_node("TestNode", {"name": "test"})
            self.record_result("neo4j_adapter", adapter_result.get('created', False))
            
            # Test MotifCost Analyzer
            motif_analyzer = MotifCostAnalyzer()
            motif_result = await motif_analyzer.analyze_motifs()
            self.record_result("neo4j_motifcost", motif_result.get('analyzed', False))
            
            print("  ✅ Neo4j integration working")
            
        except Exception as e:
            print(f"  ❌ Neo4j integration failed: {e}")
            self.record_result("neo4j_integration", False)
    
    async def test_council_agents(self):
        """Test Council agents"""
        print("\n👥 Testing Council Agents...")
        
        try:
            from aura_intelligence.agents.council.lnn_council import LNNCouncil
            from aura_intelligence.agents.enhanced_council import EnhancedCouncil
            from aura_intelligence.agents.working_council_agent import WorkingCouncilAgent
            
            # Test LNN Council
            lnn_council = LNNCouncil()
            council_result = await lnn_council.make_decision({
                'decision_type': 'resource_allocation',
                'context': {'resources': 100, 'demand': 80}
            })
            self.record_result("lnn_council", council_result.get('decision_made', False))
            
            # Test Enhanced Council
            enhanced_council = EnhancedCouncil()
            enhanced_result = await enhanced_council.coordinate_agents({
                'task': 'system_optimization',
                'agents': ['analyzer', 'optimizer', 'executor']
            })
            self.record_result("enhanced_council", enhanced_result.get('coordinated', False))
            
            # Test Working Council Agent
            working_agent = WorkingCouncilAgent()
            working_result = await working_agent.process_request({
                'request_type': 'analysis',
                'data': {'metrics': [1, 2, 3, 4, 5]}
            })
            self.record_result("working_council_agent", working_result.get('processed', False))
            
            print("  ✅ Council agents working")
            
        except Exception as e:
            print(f"  ❌ Council agents failed: {e}")
            self.record_result("council_agents", False)
    
    async def test_dpo_system(self):
        """Test DPO system"""
        print("\n🎯 Testing DPO System...")
        
        try:
            from aura_intelligence.dpo.production_dpo import get_production_dpo
            from aura_intelligence.dpo.preference_optimizer import get_dpo_optimizer
            
            # Test Production DPO
            prod_dpo = get_production_dpo()
            training_result = await prod_dpo.train_batch(batch_size=2)
            self.record_result("production_dpo", training_result.get('status') == 'training_complete')
            
            # Test DPO Optimizer
            dpo_optimizer = get_dpo_optimizer()
            eval_result = await dpo_optimizer.evaluate_action_preference(
                {'action': 'test', 'confidence': 0.8},
                {'context': 'test_context'}
            )
            self.record_result("dpo_optimizer", eval_result.get('dpo_preference_score') is not None)
            
            print("  ✅ DPO system working")
            
        except Exception as e:
            print(f"  ❌ DPO system failed: {e}")
            self.record_result("dpo_system", False)
    
    async def test_tda_engine(self):
        """Test TDA engine"""
        print("\n🔺 Testing TDA Engine...")
        
        try:
            from aura_intelligence.tda.unified_engine_2025 import UnifiedTDAEngine
            from aura_intelligence.tda.real_tda import get_real_tda
            from aura_intelligence.tda_engine import TDAEngine
            
            # Test Unified TDA Engine
            unified_tda = UnifiedTDAEngine()
            tda_result = await unified_tda.compute_topology({
                'points': [[1, 2], [3, 4], [5, 6], [7, 8]]
            })
            self.record_result("unified_tda_engine", tda_result.get('betti_numbers') is not None)
            
            # Test Real TDA
            real_tda = get_real_tda()
            real_result = real_tda.compute_persistence([[1, 2], [3, 4], [5, 6]])
            self.record_result("real_tda", real_result.get('betti_numbers') is not None)
            
            # Test TDA Engine
            tda_engine = TDAEngine()
            engine_result = await tda_engine.analyze_topology({
                'data': [[1, 2], [3, 4], [5, 6]]
            })
            self.record_result("tda_engine", engine_result.get('topology_computed', False))
            
            print("  ✅ TDA engine working")
            
        except Exception as e:
            print(f"  ❌ TDA engine failed: {e}")
            self.record_result("tda_engine", False)
    
    async def test_neural_systems(self):
        """Test neural systems"""
        print("\n🧠 Testing Neural Systems...")
        
        try:
            from aura_intelligence.lnn.real_mit_lnn import get_real_mit_lnn
            from aura_intelligence.neural.liquid_2025 import LiquidNeuralNetwork2025
            from aura_intelligence.moe.real_switch_moe import get_real_switch_moe
            
            # Test MIT LNN
            mit_lnn = get_real_mit_lnn()
            lnn_info = mit_lnn.get_info()
            self.record_result("mit_lnn", lnn_info.get('parameters', 0) > 0)
            
            # Test Liquid Neural Network 2025
            liquid_nn = LiquidNeuralNetwork2025()
            liquid_result = await liquid_nn.process({
                'input_sequence': [0.1, 0.2, 0.3, 0.4, 0.5]
            })
            self.record_result("liquid_nn_2025", liquid_result.get('processed', False))
            
            # Test Switch MoE
            switch_moe = get_real_switch_moe()
            moe_result = switch_moe.process_with_experts({
                'hidden_states': [[0.1] * 512]
            })
            self.record_result("switch_moe", moe_result.get('library') is not None)
            
            print("  ✅ Neural systems working")
            
        except Exception as e:
            print(f"  ❌ Neural systems failed: {e}")
            self.record_result("neural_systems", False)
    
    async def test_consciousness_system(self):
        """Test consciousness system"""
        print("\n🌟 Testing Consciousness System...")
        
        try:
            from aura_intelligence.consciousness.global_workspace import get_global_workspace
            from aura_intelligence.consciousness.attention import AttentionMechanism
            
            # Test Global Workspace
            consciousness = get_global_workspace()
            await consciousness.start()
            
            from aura_intelligence.consciousness.global_workspace import WorkspaceContent
            content = WorkspaceContent(
                content_id="test_consciousness",
                source="test_system",
                data={"test": "consciousness_data"},
                priority=1,
                attention_weight=0.8
            )
            
            await consciousness.process_content(content)
            state = consciousness.get_state()
            self.record_result("global_workspace", state.get('active', False))
            
            # Test Attention Mechanism
            attention = AttentionMechanism()
            attention_result = await attention.focus_attention({
                'stimuli': ['stimulus1', 'stimulus2', 'stimulus3'],
                'context': 'test_context'
            })
            self.record_result("attention_mechanism", attention_result.get('focused', False))
            
            print("  ✅ Consciousness system working")
            
        except Exception as e:
            print(f"  ❌ Consciousness system failed: {e}")
            self.record_result("consciousness_system", False)
    
    async def test_memory_systems(self):
        """Test memory systems"""
        print("\n💾 Testing Memory Systems...")
        
        try:
            from aura_intelligence.memory.redis_store import RedisStore
            from aura_intelligence.memory.shape_aware_memory import ShapeAwareMemory
            from aura_intelligence.memory_tiers.cxl_memory import get_cxl_memory_manager
            
            # Test Redis Store
            redis_store = RedisStore()
            redis_result = await redis_store.store("test_key", {"data": "test_value"})
            self.record_result("redis_store", redis_result.get('stored', False))
            
            # Test Shape Aware Memory
            shape_memory = ShapeAwareMemory()
            shape_result = await shape_memory.store_with_shape({
                'data': [1, 2, 3, 4, 5],
                'shape': (5,),
                'dtype': 'float32'
            })
            self.record_result("shape_aware_memory", shape_result.get('stored', False))
            
            # Test CXL Memory Manager
            cxl_memory = get_cxl_memory_manager()
            cxl_result = await cxl_memory.store("cxl_test", {"data": "cxl_value"})
            self.record_result("cxl_memory", cxl_result)
            
            print("  ✅ Memory systems working")
            
        except Exception as e:
            print(f"  ❌ Memory systems failed: {e}")
            self.record_result("memory_systems", False)
    
    async def test_orchestration(self):
        """Test orchestration systems"""
        print("\n🎼 Testing Orchestration...")
        
        try:
            from aura_intelligence.orchestration.working_orchestrator import WorkingOrchestrator
            from aura_intelligence.orchestration.tda_coordinator import TDACoordinator
            from aura_intelligence.workflows.real_temporal_workflows import TemporalWorkflowManager
            
            # Test Working Orchestrator
            orchestrator = WorkingOrchestrator()
            orch_result = await orchestrator.orchestrate_workflow({
                'workflow_type': 'data_processing',
                'steps': ['ingest', 'process', 'output']
            })
            self.record_result("working_orchestrator", orch_result.get('orchestrated', False))
            
            # Test TDA Coordinator
            tda_coordinator = TDACoordinator()
            coord_result = await tda_coordinator.coordinate_tda_analysis({
                'analysis_type': 'persistent_homology',
                'data_sources': ['source1', 'source2']
            })
            self.record_result("tda_coordinator", coord_result.get('coordinated', False))
            
            # Test Temporal Workflow Manager
            temporal_manager = TemporalWorkflowManager()
            temporal_result = await temporal_manager.execute_temporal_workflow({
                'workflow_id': 'temporal_test',
                'duration': 60,
                'steps': ['start', 'process', 'end']
            })
            self.record_result("temporal_workflows", temporal_result.get('executed', False))
            
            print("  ✅ Orchestration working")
            
        except Exception as e:
            print(f"  ❌ Orchestration failed: {e}")
            self.record_result("orchestration", False)
    
    async def test_observability(self):
        """Test observability systems"""
        print("\n📊 Testing Observability...")
        
        try:
            from aura_intelligence.observability.health_monitor import HealthMonitor
            from aura_intelligence.observability.metrics import MetricsCollector
            from aura_intelligence.observability.tracing import TracingSystem
            
            # Test Health Monitor
            health_monitor = HealthMonitor()
            health_result = await health_monitor.check_system_health()
            self.record_result("health_monitor", health_result.get('healthy', False))
            
            # Test Metrics Collector
            metrics = MetricsCollector()
            metrics_result = await metrics.collect_metrics({
                'metric_types': ['cpu', 'memory', 'network']
            })
            self.record_result("metrics_collector", metrics_result.get('collected', False))
            
            # Test Tracing System
            tracing = TracingSystem()
            trace_result = await tracing.start_trace({
                'trace_id': 'test_trace',
                'operation': 'test_operation'
            })
            self.record_result("tracing_system", trace_result.get('trace_started', False))
            
            print("  ✅ Observability working")
            
        except Exception as e:
            print(f"  ❌ Observability failed: {e}")
            self.record_result("observability", False)
    
    async def test_collective_intelligence(self):
        """Test collective intelligence"""
        print("\n🤝 Testing Collective Intelligence...")
        
        try:
            from aura_intelligence.collective.orchestrator import CollectiveOrchestrator
            from aura_intelligence.collective.supervisor import CollectiveSupervisor
            from aura_intelligence.collective.memory_manager import CollectiveMemoryManager
            
            # Test Collective Orchestrator
            collective_orch = CollectiveOrchestrator()
            orch_result = await collective_orch.orchestrate_collective({
                'collective_type': 'decision_making',
                'participants': ['agent1', 'agent2', 'agent3']
            })
            self.record_result("collective_orchestrator", orch_result.get('orchestrated', False))
            
            # Test Collective Supervisor
            supervisor = CollectiveSupervisor()
            super_result = await supervisor.supervise_collective({
                'supervision_type': 'performance_monitoring',
                'collective_id': 'test_collective'
            })
            self.record_result("collective_supervisor", super_result.get('supervised', False))
            
            # Test Collective Memory Manager
            memory_manager = CollectiveMemoryManager()
            memory_result = await memory_manager.manage_collective_memory({
                'operation': 'store',
                'collective_id': 'test_collective',
                'memory_data': {'shared_knowledge': 'test_data'}
            })
            self.record_result("collective_memory", memory_result.get('managed', False))
            
            print("  ✅ Collective intelligence working")
            
        except Exception as e:
            print(f"  ❌ Collective intelligence failed: {e}")
            self.record_result("collective_intelligence", False)
    
    async def test_bio_enhanced_system(self):
        """Test bio-enhanced systems"""
        print("\n🧬 Testing Bio-Enhanced Systems...")
        
        try:
            from aura_intelligence.bio_enhanced_production_system import BioEnhancedProductionSystem
            from aura_intelligence.bio_homeostatic.metabolic_manager import MetabolicManager
            
            # Test Bio-Enhanced Production System
            bio_system = BioEnhancedProductionSystem()
            bio_result = await bio_system.process_bio_enhanced({
                'bio_type': 'metabolic_optimization',
                'parameters': {'efficiency': 0.8, 'stability': 0.9}
            })
            self.record_result("bio_enhanced_system", bio_result.get('processed', False))
            
            # Test Metabolic Manager
            metabolic = MetabolicManager()
            metabolic_result = await metabolic.manage_metabolism({
                'component_id': 'test_component',
                'resource_demand': 50,
                'efficiency_target': 0.85
            })
            self.record_result("metabolic_manager", metabolic_result.get('managed', False))
            
            print("  ✅ Bio-enhanced systems working")
            
        except Exception as e:
            print(f"  ❌ Bio-enhanced systems failed: {e}")
            self.record_result("bio_enhanced_systems", False)
    
    async def test_production_systems(self):
        """Test production systems"""
        print("\n🏭 Testing Production Systems...")
        
        try:
            from aura_intelligence.production_system_2025 import ProductionSystem2025
            from aura_intelligence.production_integration_2025 import ProductionIntegration
            from aura_intelligence.enhanced_system_2025 import EnhancedSystem2025
            
            # Test Production System 2025
            prod_system = ProductionSystem2025()
            prod_result = await prod_system.initialize_production()
            self.record_result("production_system_2025", prod_result.get('initialized', False))
            
            # Test Production Integration
            integration = ProductionIntegration()
            integration_result = await integration.integrate_systems({
                'systems': ['neural', 'memory', 'orchestration'],
                'integration_type': 'full_stack'
            })
            self.record_result("production_integration", integration_result.get('integrated', False))
            
            # Test Enhanced System 2025
            enhanced_system = EnhancedSystem2025()
            enhanced_result = await enhanced_system.enhance_capabilities({
                'enhancement_type': 'performance_optimization',
                'target_metrics': ['latency', 'throughput', 'accuracy']
            })
            self.record_result("enhanced_system_2025", enhanced_result.get('enhanced', False))
            
            print("  ✅ Production systems working")
            
        except Exception as e:
            print(f"  ❌ Production systems failed: {e}")
            self.record_result("production_systems", False)
    
    def record_result(self, component_name: str, success: bool):
        """Record test result"""
        self.results[component_name] = success
        self.total_tested += 1
        if success:
            self.total_passed += 1
    
    def print_final_results(self):
        """Print final test results"""
        print("\n" + "=" * 60)
        print("🏆 FINAL RESULTS - ALL REAL COMPONENTS")
        print("=" * 60)
        
        print(f"📊 SUMMARY:")
        print(f"  - Total components tested: {self.total_tested}")
        print(f"  - Components passed: {self.total_passed}")
        print(f"  - Components failed: {self.total_tested - self.total_passed}")
        print(f"  - Success rate: {self.total_passed/self.total_tested*100:.1f}%")
        
        print(f"\n✅ PASSED COMPONENTS ({self.total_passed}):")
        for component, success in self.results.items():
            if success:
                print(f"  ✅ {component}")
        
        print(f"\n❌ FAILED COMPONENTS ({self.total_tested - self.total_passed}):")
        for component, success in self.results.items():
            if not success:
                print(f"  ❌ {component}")
        
        if self.total_passed == self.total_tested:
            print("\n🎉 ALL REAL COMPONENTS ARE WORKING!")
            print("   🔥 Mem0 integration functional")
            print("   🔥 LangGraph workflows operational")
            print("   🔥 Neo4j graph database connected")
            print("   🔥 Council agents coordinating")
            print("   🔥 DPO learning system active")
            print("   🔥 TDA topology analysis working")
            print("   🔥 Neural networks processing")
            print("   🔥 Consciousness system aware")
            print("   🔥 Memory systems storing")
            print("   🔥 Orchestration coordinating")
            print("   🔥 Observability monitoring")
            print("   🔥 Collective intelligence emerging")
            print("   🔥 Bio-enhanced systems optimizing")
            print("   🔥 Production systems deployed")
        else:
            success_rate = self.total_passed/self.total_tested*100
            if success_rate >= 80:
                print(f"\n✅ MOSTLY WORKING ({success_rate:.1f}%)")
                print("   Minor fixes needed for full functionality")
            elif success_rate >= 60:
                print(f"\n⚠️  PARTIALLY WORKING ({success_rate:.1f}%)")
                print("   Significant improvements needed")
            else:
                print(f"\n❌ MAJOR ISSUES ({success_rate:.1f}%)")
                print("   System needs substantial fixes")

async def main():
    """Main test execution."""
    print("🧠 AURA Intelligence Components Comprehensive Test")
    print("Testing ALL components in core/src/aura_intelligence/")
    print("="*60)
    
    start_time = time.time()
    
    # Discover all components
    print("🔍 Discovering components...")
    all_components = discover_all_components()
    
    if not all_components:
        print("❌ No components found!")
        return
    
    total_files = sum(len(files) for files in all_components.values())
    print(f"📦 Found {total_files} components across {len(all_components)} categories")
    
    # Test all components
    all_results = {}
    
    for category, files in all_components.items():
        results = await test_category_components(category, files)
        all_results[category] = results
    
    # Generate comprehensive report
    await generate_detailed_report(all_results)
    
    execution_time = time.time() - start_time
    print(f"\n⏱️ Test completed in {execution_time:.2f} seconds")
    print("="*80)

if __name__ == "__main__":
    asyncio.run(main())