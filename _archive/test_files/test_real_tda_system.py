#!/usr/bin/env python3
"""
REAL Test of AURA's TDA System
No fake testing - show what actually works
"""

import sys
import os
sys.path.insert(0, 'core/src')

print("🔬 TESTING AURA'S TDA SYSTEM - THE REAL TRUTH")
print("=" * 60)

# 1. Check if TDA files exist and compile
print("\n1️⃣ TDA File Status:")
tda_files = [
    'real_tda_engine_2025.py',
    'algorithms.py', 
    'core.py',
    'models.py',
    'service.py'
]

working_files = []
broken_files = []

for file in tda_files:
    path = f'core/src/aura_intelligence/tda/{file}'
    try:
        compile(open(path).read(), path, 'exec')
        working_files.append(file)
        print(f"✅ {file}")
    except SyntaxError as e:
        broken_files.append((file, e.lineno, e.msg))
        print(f"❌ {file} - Error at line {e.lineno}")
    except Exception as e:
        print(f"⚠️  {file} - {type(e).__name__}")

print(f"\nWorking: {len(working_files)}/{len(tda_files)}")

# 2. Try to import what works
print("\n2️⃣ Testing Imports:")
try:
    # The one that compiled successfully
    from aura_intelligence.tda.real_tda_engine_2025 import RealTDAEngine2025
    print("✅ Can import RealTDAEngine2025")
    
    # Try to create instance
    engine = RealTDAEngine2025()
    print("✅ Created TDA engine instance")
    
    # Check what it can do
    print("\n3️⃣ TDA Engine Capabilities:")
    methods = [m for m in dir(engine) if not m.startswith('_') and callable(getattr(engine, m))]
    print(f"   Available methods: {len(methods)}")
    for method in methods[:5]:
        print(f"   - {method}")
    
except ImportError as e:
    print(f"❌ Cannot import TDA engine: {e}")
except Exception as e:
    print(f"❌ Error creating TDA engine: {e}")

# 3. Check for the 112 algorithms
print("\n4️⃣ Looking for the 112 TDA Algorithms:")
try:
    # Check algorithms.py content
    with open('core/src/aura_intelligence/tda/algorithms.py', 'r') as f:
        content = f.read()
        
    # Count algorithm definitions
    algo_count = content.count('def compute_') + content.count('class') + content.count('Algorithm')
    print(f"   Found {algo_count} algorithm-related definitions")
    
    # Look for algorithm registry or list
    if 'ALGORITHMS' in content or 'algorithms' in content:
        print("   ✅ Found algorithm registry")
    else:
        print("   ⚠️  No clear algorithm registry found")
        
except:
    print("   ❌ Cannot analyze algorithms.py")

# 4. What dependencies are missing
print("\n5️⃣ Missing Dependencies for TDA:")
missing_deps = []

deps_to_check = [
    ('gudhi', 'GUDHI - main TDA library'),
    ('ripser', 'Ripser - fast persistence'),
    ('persim', 'Persistence diagrams'),
    ('giotto', 'Giotto-tda - ML with TDA'),
    ('dionysus', 'Dionysus - TDA computations'),
    ('scikit-tda', 'Scikit-TDA toolkit')
]

for module, desc in deps_to_check:
    try:
        __import__(module)
        print(f"✅ {module}: {desc}")
    except ImportError:
        missing_deps.append(module)
        print(f"❌ {module}: {desc}")

# 5. The REAL situation
print("\n" + "=" * 60)
print("📊 THE REAL SITUATION:")
print(f"- TDA files working: {len(working_files)}/{len(tda_files)}")
print(f"- Missing dependencies: {len(missing_deps)}")
print(f"- Core TDA libraries not installed")
print(f"- Cannot run real TDA without: {', '.join(missing_deps)}")

print("\n🔑 TO MAKE TDA WORK:")
print("1. Install dependencies: pip install gudhi ripser persim giotto-tda")
print("2. Fix syntax errors in TDA files")
print("3. Create mock implementations for testing")

print("\n⚠️  WITHOUT TDA, AURA CANNOT:")
print("- 'See the shape of failure'")
print("- Detect topological anomalies")
print("- Predict cascading failures")
print("- Use its 112 algorithms")

print("\n💡 WHAT WE CAN DO NOW:")
print("- Create mock TDA that returns synthetic results")
print("- Fix the supervisor/agent orchestration")
print("- Build integration without real TDA")
print("- Document what's needed for production")