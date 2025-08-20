#!/usr/bin/env python3
"""
AURA System Status Test
Tests what actually works vs what's broken
"""

import asyncio
import traceback
import time
from typing import Dict, Any

def test_import(module_name: str, import_statement: str) -> Dict[str, Any]:
    """Test if a module can be imported"""
    try:
        exec(import_statement)
        return {"status": "✅ WORKS", "error": None}
    except Exception as e:
        return {"status": "❌ BROKEN", "error": str(e)}

def test_core_dependencies():
    """Test core research library dependencies"""
    print("🔍 TESTING CORE DEPENDENCIES")
    print("=" * 50)
    
    dependencies = {
        "PyTorch": "import torch",
        "MIT LNN (ncps)": "import ncps",
        "GUDHI TDA": "import gudhi", 
        "Ripser TDA": "import ripser",
        "Transformers": "import transformers",
        "Sentence Transformers": "import sentence_transformers",
        "TorchDiffeq": "import torchdiffeq",
        "Redis": "import redis",
        "Neo4j": "import neo4j"
    }
    
    results = {}
    for name, import_stmt in dependencies.items():
        result = test_import(name, import_stmt)
        print(f"{name}: {result['status']}")
        if result['error']:
            print(f"  └─ Error: {result['error']}")
        results[name] = result
    
    return results

def test_aura_components():
    """Test AURA component imports"""
    print("\n🧠 TESTING AURA COMPONENTS")
    print("=" * 50)
    
    components = {
        "LNN Core": "from core.src.aura_intelligence.lnn.core import LNNCore",
        "Switch MoE": "from core.src.aura_intelligence.moe.real_switch_moe import get_real_switch_moe",
        "Real TDA": "from core.src.aura_intelligence.tda.real_tda import get_real_tda",
        "Real Registry": "from core.src.aura_intelligence.components.real_registry import get_real_registry",
        "Real Components": "from core.src.aura_intelligence.components.real_components import create_real_component"
    }
    
    results = {}
    for name, import_stmt in components.items():
        result = test_import(name, import_stmt)
        print(f"{name}: {result['status']}")
        if result['error']:
            print(f"  └─ Error: {result['error']}")
        results[name] = result
    
    return results

async def test_component_processing():
    """Test actual component processing"""
    print("\n⚙️ TESTING COMPONENT PROCESSING")
    print("=" * 50)
    
    results = {}
    
    # Test LNN Core
    try:
        from core.src.aura_intelligence.lnn.core import LNNCore
        import torch
        
        lnn = LNNCore(input_size=10, output_size=5)
        test_data = torch.randn(1, 1, 10)
        output = lnn(test_data)
        
        print("MIT LNN Core: ✅ WORKS")
        print(f"  └─ Input shape: {test_data.shape}, Output shape: {output.shape}")
        results["LNN"] = {"status": "WORKS", "details": f"Output: {output.shape}"}
        
    except Exception as e:
        print(f"MIT LNN Core: ❌ BROKEN - {e}")
        results["LNN"] = {"status": "BROKEN", "error": str(e)}
    
    # Test Switch MoE
    try:
        from core.src.aura_intelligence.moe.real_switch_moe import get_real_switch_moe
        import torch
        
        moe = get_real_switch_moe(d_model=512, num_experts=4)
        test_data = torch.randn(1, 10, 512)
        output, aux_info = moe(test_data)
        
        print("Switch MoE: ✅ WORKS")
        print(f"  └─ Input: {test_data.shape}, Output: {output.shape}")
        print(f"  └─ Implementation: {aux_info.get('library', 'unknown')}")
        results["MoE"] = {"status": "WORKS", "details": aux_info}
        
    except Exception as e:
        print(f"Switch MoE: ❌ BROKEN - {e}")
        results["MoE"] = {"status": "BROKEN", "error": str(e)}
    
    # Test TDA
    try:
        from core.src.aura_intelligence.tda.real_tda import get_real_tda
        import numpy as np
        
        tda = get_real_tda()
        test_points = np.random.random((20, 2))
        result = tda.compute_persistence(test_points)
        
        print("TDA Engine: ✅ WORKS")
        print(f"  └─ Library: {result['library']}")
        print(f"  └─ Betti numbers: {result['betti_numbers']}")
        results["TDA"] = {"status": "WORKS", "details": result}
        
    except Exception as e:
        print(f"TDA Engine: ❌ BROKEN - {e}")
        results["TDA"] = {"status": "BROKEN", "error": str(e)}
    
    # Test Component Registry
    try:
        from core.src.aura_intelligence.components.real_registry import get_real_registry
        
        registry = get_real_registry()
        stats = registry.get_component_stats()
        
        print("Component Registry: ✅ WORKS")
        print(f"  └─ Total components: {stats['total_components']}")
        print(f"  └─ Active components: {stats['active_components']}")
        print(f"  └─ Health score: {stats['health_score']:.2f}")
        results["Registry"] = {"status": "WORKS", "details": stats}
        
    except Exception as e:
        print(f"Component Registry: ❌ BROKEN - {e}")
        results["Registry"] = {"status": "BROKEN", "error": str(e)}
    
    return results

async def test_individual_components():
    """Test individual components"""
    print("\n🔧 TESTING INDIVIDUAL COMPONENTS")
    print("=" * 50)
    
    test_data = {"values": [1.0, 2.0, 3.0, 4.0, 5.0]}
    results = {}
    
    try:
        from core.src.aura_intelligence.components.real_registry import get_real_registry
        registry = get_real_registry()
        
        # Test first few components
        component_ids = list(registry.components.keys())[:10]
        
        for component_id in component_ids:
            try:
                start_time = time.time()
                result = await registry.process_data(component_id, test_data)
                processing_time = time.time() - start_time
                
                print(f"{component_id}: ✅ WORKS ({processing_time:.3f}s)")
                results[component_id] = {"status": "WORKS", "result": result, "time": processing_time}
                
            except Exception as e:
                print(f"{component_id}: ❌ BROKEN - {e}")
                results[component_id] = {"status": "BROKEN", "error": str(e)}
    
    except Exception as e:
        print(f"Registry not available: {e}")
        results["registry_error"] = str(e)
    
    return results

def generate_status_report(dependency_results, component_results, processing_results, individual_results):
    """Generate comprehensive status report"""
    print("\n📊 SYSTEM STATUS REPORT")
    print("=" * 50)
    
    # Count working vs broken
    working_deps = sum(1 for r in dependency_results.values() if "WORKS" in r["status"])
    total_deps = len(dependency_results)
    
    working_components = sum(1 for r in component_results.values() if "WORKS" in r["status"])
    total_components = len(component_results)
    
    working_processing = sum(1 for r in processing_results.values() if "WORKS" in r.get("status", "BROKEN"))
    total_processing = len(processing_results)
    
    working_individual = sum(1 for r in individual_results.values() if isinstance(r, dict) and "WORKS" in r.get("status", "BROKEN"))
    total_individual = len(individual_results)
    
    print(f"Dependencies: {working_deps}/{total_deps} working ({working_deps/total_deps*100:.1f}%)")
    print(f"Components: {working_components}/{total_components} working ({working_components/total_components*100:.1f}%)")
    print(f"Processing: {working_processing}/{total_processing} working ({working_processing/total_processing*100:.1f}%)")
    print(f"Individual: {working_individual}/{total_individual} working ({working_individual/total_individual*100:.1f}%)")
    
    # Overall health score
    total_working = working_deps + working_components + working_processing + working_individual
    total_tests = total_deps + total_components + total_processing + total_individual
    health_score = total_working / total_tests * 100 if total_tests > 0 else 0
    
    print(f"\n🏥 OVERALL SYSTEM HEALTH: {health_score:.1f}%")
    
    if health_score >= 80:
        print("🟢 SYSTEM STATUS: HEALTHY")
    elif health_score >= 60:
        print("🟡 SYSTEM STATUS: NEEDS ATTENTION")
    else:
        print("🔴 SYSTEM STATUS: CRITICAL")
    
    # Recommendations
    print("\n💡 RECOMMENDATIONS:")
    if working_deps < total_deps:
        print("• Install missing research libraries (ncps, gudhi, transformers)")
    if working_components < total_components:
        print("• Fix component import issues")
    if working_processing < total_processing:
        print("• Fix component processing logic")
    if working_individual < total_individual / 2:
        print("• Reduce number of components to only working ones")

async def main():
    """Run comprehensive system test"""
    print("🚀 AURA INTELLIGENCE SYSTEM TEST")
    print("Testing what works vs what's broken...\n")
    
    # Run all tests
    dependency_results = test_core_dependencies()
    component_results = test_aura_components()
    processing_results = await test_component_processing()
    individual_results = await test_individual_components()
    
    # Generate report
    generate_status_report(
        dependency_results, 
        component_results, 
        processing_results, 
        individual_results
    )
    
    return {
        "dependencies": dependency_results,
        "components": component_results, 
        "processing": processing_results,
        "individual": individual_results
    }

if __name__ == "__main__":
    results = asyncio.run(main())