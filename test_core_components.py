#!/usr/bin/env python3
"""
Test AURA core components - REAL testing, no fakes
"""

import sys
sys.path.insert(0, 'core/src')

print("🧪 TESTING AURA CORE COMPONENTS")
print("=" * 60)

# Test 1: Can we import core modules?
print("\n1️⃣ Testing Core Imports:")

test_modules = [
    ('config', 'aura_intelligence.core.config'),
    ('agents', 'aura_intelligence.core.agents'),
    ('memory', 'aura_intelligence.core.memory'),
    ('knowledge', 'aura_intelligence.core.knowledge'),
    ('supervisor', 'aura_intelligence.agents.supervisor')
]

working_modules = []

for name, module_path in test_modules:
    try:
        module = __import__(module_path, fromlist=[''])
        working_modules.append((name, module))
        print(f"✅ {name}: Imported successfully")
    except ImportError as e:
        print(f"❌ {name}: Import failed - {e}")
    except Exception as e:
        print(f"❌ {name}: Error - {type(e).__name__}: {e}")

print(f"\n📊 Success rate: {len(working_modules)}/{len(test_modules)}")

# Test 2: Can we instantiate components?
print("\n2️⃣ Testing Component Creation:")

if any(name == 'config' for name, _ in working_modules):
    try:
        from aura_intelligence.core.config import ConfigManager
        config_mgr = ConfigManager()
        print("✅ ConfigManager: Created successfully")
    except Exception as e:
        print(f"❌ ConfigManager: {type(e).__name__}: {e}")

if any(name == 'agents' for name, _ in working_modules):
    try:
        from aura_intelligence.core.agents import AgentOrchestrator
        orchestrator = AgentOrchestrator()
        print("✅ AgentOrchestrator: Created successfully")
    except Exception as e:
        print(f"❌ AgentOrchestrator: {type(e).__name__}: {e}")

if any(name == 'supervisor' for name, _ in working_modules):
    try:
        from aura_intelligence.agents.supervisor import EnhancedSupervisor
        supervisor = EnhancedSupervisor()
        print("✅ EnhancedSupervisor: Created successfully")
    except Exception as e:
        print(f"❌ EnhancedSupervisor: {type(e).__name__}: {e}")

# Test 3: Check what's still broken
print("\n3️⃣ Checking Core Folder Status:")

import os
import ast

core_files = []
for file in os.listdir('core/src/aura_intelligence/core'):
    if file.endswith('.py') and not file.endswith('.broken'):
        core_files.append(file)

working = 0
broken = []

for file in sorted(core_files):
    path = os.path.join('core/src/aura_intelligence/core', file)
    try:
        with open(path, 'r') as f:
            ast.parse(f.read())
        working += 1
    except SyntaxError as e:
        broken.append((file, e.lineno))

print(f"✅ Working: {working}/{len(core_files)}")
if broken:
    print(f"❌ Still broken: {', '.join([f'{f}:{l}' for f, l in broken])}")

# Test 4: Import chain analysis
print("\n4️⃣ Import Chain Status:")

critical_imports = {
    'httpx': 'HTTP client for external APIs',
    'neo4j': 'Graph database for knowledge storage',
    'redis': 'Cache and pub/sub',
    'kafka': 'Event streaming',
    'gudhi': 'TDA computations',
    'torch': 'Neural networks'
}

missing = []
for module, desc in critical_imports.items():
    try:
        __import__(module)
        print(f"✅ {module}: {desc}")
    except ImportError:
        missing.append(module)
        print(f"❌ {module}: {desc} - NOT INSTALLED")

# Summary
print("\n" + "=" * 60)
print("📋 SUMMARY:")
print(f"- Core modules working: {len(working_modules)}/{len(test_modules)}")
print(f"- Core files compiling: {working}/{len(core_files)}")
print(f"- Missing dependencies: {len(missing)}")

print("\n🔑 TO MAKE AURA FULLY WORK:")
if missing:
    print(f"1. Install dependencies: pip install {' '.join(missing)}")
if broken:
    print(f"2. Fix remaining syntax errors in: {', '.join([b[0] for b in broken])}")
print("3. Complete TDA implementation")
print("4. Connect all components")

print("\n✨ But the foundation is getting stronger!")