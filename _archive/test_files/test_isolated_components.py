#!/usr/bin/env python3
"""
Test AURA components in isolation to find what actually works
Bypasses the main __init__.py files that have import cascades
"""

import os
import sys
import json
import importlib.util
import traceback
from pathlib import Path

# Add core to path
sys.path.insert(0, '/home/sina/projects/osiris-2/core/src')

def load_module_from_file(file_path: str, module_name: str):
    """Load a Python module directly from file path"""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None:
        raise ImportError(f"Could not load spec from {file_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def test_isolated_files():
    """Test individual AURA files that might work independently"""
    
    base_path = Path("/home/sina/projects/osiris-2/core/src/aura_intelligence")
    
    # Test files that might work in isolation (no complex imports)
    test_files = [
        # Config files (should be simple)
        ("config/base.py", "Config Base"),
        ("config/api.py", "API Config"),
        ("config/memory.py", "Memory Config"),
        
        # Utils (should be simple)
        ("utils/decorators.py", "Utils Decorators"),
        ("utils/logging.py", "Utils Logging"),
        
        # Simple resilience components
        ("resilience/metrics.py", "Resilience Metrics"),
        ("resilience/timeout.py", "Resilience Timeout"),
        
        # TDA algorithms (might work)
        ("tda/models.py", "TDA Models"),
        
        # Memory interfaces
        ("memory/storage_interface.py", "Memory Interface"),
        
        # Events schemas
        ("events/schemas.py", "Event Schemas"),
        
        # Core types
        ("core/types.py", "Core Types"),
        ("core/interfaces.py", "Core Interfaces"),
        
        # Schemas
        ("agents/schemas/base.py", "Agent Base Schema"),
        ("agents/schemas/enums.py", "Agent Enums"),
    ]
    
    working_files = []
    results = []
    
    print("🧪 Testing individual AURA files in isolation...")
    print(f"📁 Base path: {base_path}")
    
    for file_path, name in test_files:
        full_path = base_path / file_path
        
        if not full_path.exists():
            results.append({"name": name, "status": "NOT_FOUND", "path": str(full_path)})
            print(f"❓ {name}: File not found")
            continue
        
        try:
            # Try to compile first
            with open(full_path, 'r') as f:
                code = f.read()
            
            compile(code, str(full_path), 'exec')
            
            # If compilation works, try to load
            module = load_module_from_file(str(full_path), name.replace(" ", "_"))
            
            working_files.append(name)
            results.append({
                "name": name,
                "status": "SUCCESS", 
                "path": str(full_path),
                "classes": [item for item in dir(module) if not item.startswith('_') and item[0].isupper()],
                "functions": [item for item in dir(module) if not item.startswith('_') and item[0].islower()]
            })
            print(f"✅ {name}: SUCCESS")
            
        except SyntaxError as e:
            results.append({"name": name, "status": "SYNTAX_ERROR", "error": str(e), "path": str(full_path)})
            print(f"❌ {name}: Syntax Error - {e}")
            
        except Exception as e:
            results.append({"name": name, "status": "IMPORT_ERROR", "error": str(e), "path": str(full_path)})
            print(f"❌ {name}: Import Error - {str(e)[:80]}")
    
    # Save detailed results
    summary = {
        "total_tested": len(test_files),
        "working": len(working_files),
        "success_rate": len(working_files) / len(test_files) if test_files else 0,
        "working_files": working_files,
        "detailed_results": results
    }
    
    print(f"\n📊 Isolated Component Test Results:")
    print(f"   Total tested: {len(test_files)}")
    print(f"   Working: {len(working_files)}")
    print(f"   Success rate: {100*summary['success_rate']:.1f}%")
    print(f"   ✅ Working files: {working_files}")
    
    # Save to file
    with open('/home/sina/projects/osiris-2/isolated_component_results.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    return summary

if __name__ == "__main__":
    results = test_isolated_files()
    
    # Also test some simple functionality
    print(f"\n🔬 Testing working components functionality...")
    
    if results["working"]:
        print(f"✅ Found {len(results['working'])} working isolated components!")
        print("📝 You can now build APIs using these working components")
    else:
        print("❌ No components working in isolation - deeper syntax fixes needed")