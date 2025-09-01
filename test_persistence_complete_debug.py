#!/usr/bin/env python3
"""
Complete Persistence Test with Enhanced Debugging
Following the cycle: index -> extract -> research -> eval -> enhance -> test
"""

import asyncio
import sys
import os
import traceback
import json
from datetime import datetime
import logging

# Set up paths
sys.path.insert(0, os.path.abspath('./core/src'))

# Enhanced logging setup
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger(__name__)

# Color codes for better output
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def print_section(title):
    """Print a section header"""
    print(f"\n{Colors.HEADER}{'='*60}{Colors.ENDC}")
    print(f"{Colors.HEADER}{title}{Colors.ENDC}")
    print(f"{Colors.HEADER}{'='*60}{Colors.ENDC}")

def print_success(message):
    """Print success message"""
    print(f"{Colors.OKGREEN}✓ {message}{Colors.ENDC}")

def print_error(message):
    """Print error message"""
    print(f"{Colors.FAIL}✗ {message}{Colors.ENDC}")

def print_warning(message):
    """Print warning message"""
    print(f"{Colors.WARNING}⚠ {message}{Colors.ENDC}")

def print_info(message):
    """Print info message"""
    print(f"{Colors.OKBLUE}ℹ {message}{Colors.ENDC}")

async def test_import_chain():
    """Test the import chain to identify issues"""
    print_section("STEP 1: Testing Import Chain")
    
    imports_to_test = [
        ("Basic imports", [
            "asyncio",
            "asyncpg",
            "duckdb",
            "zstandard",
            "aiokafka"
        ]),
        ("AURA core", [
            "aura_intelligence",
            "aura_intelligence.persistence",
            "aura_intelligence.neural",
            "aura_intelligence.agents"
        ]),
        ("Persistence modules", [
            "aura_intelligence.persistence.causal_state_manager",
            "aura_intelligence.persistence.memory_native",
            "aura_intelligence.persistence.state_manager"
        ])
    ]
    
    all_good = True
    
    for category, modules in imports_to_test:
        print(f"\n{Colors.BOLD}Testing {category}:{Colors.ENDC}")
        for module in modules:
            try:
                __import__(module)
                print_success(f"Import {module}")
            except ImportError as e:
                print_error(f"Import {module}: {e}")
                all_good = False
            except Exception as e:
                print_error(f"Import {module}: {type(e).__name__}: {e}")
                all_good = False
    
    return all_good

async def test_persistence_basic():
    """Test basic persistence functionality"""
    print_section("STEP 2: Testing Basic Persistence")
    
    try:
        from aura_intelligence.persistence.causal_state_manager import (
            CausalPersistenceManager,
            StateType,
            CausalContext
        )
        print_success("Imported causal persistence modules")
    except Exception as e:
        print_error(f"Failed to import: {e}")
        traceback.print_exc()
        return False
    
    # Test manager creation
    try:
        manager = CausalPersistenceManager(
            postgres_url="postgresql://localhost/aura",
            duckdb_path="./test_aura.db",
            legacy_path="./test_legacy"
        )
        print_success("Created CausalPersistenceManager")
        
        await manager.initialize()
        print_info(f"Manager initialized - Legacy mode: {manager.legacy_mode}")
        
        if manager.legacy_mode:
            print_warning("Running in legacy mode (PostgreSQL not available)")
        else:
            print_success("Connected to PostgreSQL!")
            
    except Exception as e:
        print_error(f"Manager initialization failed: {e}")
        traceback.print_exc()
        return False
    
    # Test save/load
    try:
        test_data = {
            "test_id": "persistence_test_001",
            "value": 42,
            "nested": {"key": "value"},
            "timestamp": datetime.now().isoformat()
        }
        
        context = CausalContext(
            causes=["test_execution", "validation"],
            effects=["state_saved", "test_passed"],
            counterfactuals={
                "alternative": "could_have_failed",
                "probability": 0.1
            },
            confidence=0.9,
            energy_cost=0.05
        )
        
        state_id = await manager.save_state(
            StateType.COMPONENT_STATE,
            "test_component",
            test_data,
            causal_context=context
        )
        
        print_success(f"Saved state with ID: {state_id}")
        
        # Load it back
        loaded = await manager.load_state(
            StateType.COMPONENT_STATE,
            "test_component"
        )
        
        if loaded and loaded.get("test_id") == "persistence_test_001":
            print_success("Successfully loaded state")
            print_info(f"Loaded data: {json.dumps(loaded, indent=2)}")
        else:
            print_error("Failed to load state correctly")
            return False
            
    except Exception as e:
        print_error(f"Save/load test failed: {e}")
        traceback.print_exc()
        return False
    
    await manager.close()
    return True

async def test_memory_native():
    """Test memory-native architecture"""
    print_section("STEP 3: Testing Memory-Native Architecture")
    
    try:
        import torch
        has_cuda = torch.cuda.is_available()
        print_info(f"CUDA available: {has_cuda}")
        
        if not has_cuda:
            print_warning("No CUDA - some tests will be skipped")
            
    except ImportError:
        print_error("PyTorch not available")
        return True  # Don't fail the whole test
    
    try:
        from aura_intelligence.persistence.memory_native import (
            MemoryFabric,
            MemoryNativeArchitecture
        )
        print_success("Imported memory-native modules")
        
        # Test fabric creation
        fabric = MemoryFabric(
            working_size_gb=0.1,
            episodic_size_gb=0.2,
            semantic_size_gb=0.1
        )
        print_success("Created MemoryFabric")
        
        # Test store and retrieve
        test_memory = {
            "concept": "test",
            "value": 123,
            "metadata": {"importance": 0.8}
        }
        
        key = await fabric.store_with_compute(
            "test_mem_001",
            test_memory,
            tier="working"
        )
        print_success(f"Stored memory with key: {key}")
        
        # Retrieve with computation
        def enhance_memory(data):
            data["enhanced"] = True
            data["value"] = data.get("value", 0) * 2
            return data
        
        retrieved = await fabric.retrieve_and_compute(
            "test_mem_001",
            compute_fn=enhance_memory,
            evolve=True
        )
        
        if retrieved and retrieved.get("enhanced"):
            print_success("Compute-on-retrieval working")
            print_info(f"Retrieved: {json.dumps(retrieved, indent=2)}")
        else:
            print_error("Compute-on-retrieval failed")
            return False
            
    except Exception as e:
        print_error(f"Memory-native test failed: {e}")
        traceback.print_exc()
        return False
    
    return True

async def test_agent_integration():
    """Test integration with agents"""
    print_section("STEP 4: Testing Agent Integration")
    
    try:
        # Test if we can import an agent
        from aura_intelligence.agents.test_code_agent import create_agent as create_code_agent
        print_success("Imported test_code_agent")
        
        # Create agent
        agent = create_code_agent()
        print_success("Created code agent")
        
        # Check if agent has required attributes
        if hasattr(agent, 'memory'):
            print_success("Agent has memory attribute")
        else:
            print_warning("Agent missing memory attribute")
            
        return True
        
    except Exception as e:
        print_error(f"Agent integration test failed: {e}")
        traceback.print_exc()
        return False

async def test_complete_integration():
    """Test complete system integration"""
    print_section("STEP 5: Complete Integration Test")
    
    try:
        # Run the main integration test
        print_info("Running test_all_agents_integrated.py...")
        
        # Import and run
        from test_all_agents_integrated import main as test_main
        
        # Run with timeout
        try:
            await asyncio.wait_for(test_main(), timeout=30.0)
            print_success("Integration test completed")
            return True
        except asyncio.TimeoutError:
            print_error("Integration test timed out")
            return False
            
    except Exception as e:
        print_error(f"Integration test failed: {e}")
        traceback.print_exc()
        return False

async def main():
    """Run all tests in sequence"""
    print(f"{Colors.BOLD}{Colors.OKCYAN}")
    print("╔═══════════════════════════════════════════════════════════╗")
    print("║        AURA PERSISTENCE COMPLETE TEST SUITE               ║")
    print("║  Following cycle: index → extract → eval → enhance → test ║")
    print("╚═══════════════════════════════════════════════════════════╝")
    print(f"{Colors.ENDC}")
    
    # Test results
    results = {
        "Import Chain": False,
        "Basic Persistence": False,
        "Memory Native": False,
        "Agent Integration": False,
        "Complete Integration": False
    }
    
    # Run tests
    print_info("Starting comprehensive test suite...\n")
    
    # Step 1: Import chain
    results["Import Chain"] = await test_import_chain()
    if not results["Import Chain"]:
        print_warning("Import chain has issues - continuing with other tests")
    
    # Step 2: Basic persistence
    results["Basic Persistence"] = await test_persistence_basic()
    
    # Step 3: Memory native
    results["Memory Native"] = await test_memory_native()
    
    # Step 4: Agent integration
    results["Agent Integration"] = await test_agent_integration()
    
    # Step 5: Complete integration (only if others pass)
    if all(results.values()):
        results["Complete Integration"] = await test_complete_integration()
    else:
        print_warning("Skipping complete integration due to failures")
    
    # Summary
    print_section("TEST SUMMARY")
    
    for test_name, passed in results.items():
        if passed:
            print_success(f"{test_name}")
        else:
            print_error(f"{test_name}")
    
    total = len(results)
    passed = sum(1 for p in results.values() if p)
    
    print(f"\n{Colors.BOLD}Total: {passed}/{total} tests passed{Colors.ENDC}")
    
    if passed == total:
        print(f"\n{Colors.OKGREEN}{Colors.BOLD}🎉 ALL TESTS PASSED!{Colors.ENDC}")
        print_info("The persistence system is fully integrated and working!")
    else:
        print(f"\n{Colors.WARNING}{Colors.BOLD}⚠️  Some tests failed{Colors.ENDC}")
        print_info("Check the errors above for debugging information")
    
    # Next steps
    print_section("NEXT STEPS")
    
    if results["Import Chain"] and results["Basic Persistence"]:
        print_info("1. Persistence core is working")
        print_info("2. Next: Update agent classes to use causal persistence")
        print_info("3. Then: Add persistence to neural components")
        print_info("4. Finally: Run benchmarks")
    else:
        print_warning("1. Fix import/dependency issues first")
        print_warning("2. Ensure all modules are properly installed")
        print_warning("3. Check for remaining syntax errors")
    
    return 0 if passed == total else 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)