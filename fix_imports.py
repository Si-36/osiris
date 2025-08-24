#!/usr/bin/env python3
"""
Fix relative imports in AURA Intelligence modules
"""

import os
import re
from pathlib import Path

def fix_imports_in_file(filepath):
    """Fix relative imports in a single file"""
    
    # Read the file
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Patterns to replace
    replacements = [
        # Simple relative imports
        (r'from \.\.events import', 'from aura_intelligence.events import'),
        (r'from \.\.memory import', 'from aura_intelligence.memory import'),
        (r'from \.\.consensus import', 'from aura_intelligence.consensus import'),
        (r'from \.\.lnn import', 'from aura_intelligence.lnn import'),
        (r'from \.\.tda import', 'from aura_intelligence.tda import'),
        (r'from \.\.orchestration import', 'from aura_intelligence.orchestration import'),
        (r'from \.\.observability import', 'from aura_intelligence.observability import'),
        (r'from \.\.agents import', 'from aura_intelligence.agents import'),
        (r'from \.\.graph import', 'from aura_intelligence.graph import'),
        (r'from \.\.streaming import', 'from aura_intelligence.streaming import'),
        (r'from \.\.infrastructure import', 'from aura_intelligence.infrastructure import'),
        (r'from \.\.config import', 'from aura_intelligence.config import'),
        (r'from \.\.utils import', 'from aura_intelligence.utils import'),
        (r'from \.\.distributed import', 'from aura_intelligence.distributed import'),
        (r'from \.\.governance import', 'from aura_intelligence.governance import'),
        
        # Single dot relative imports
        (r'from \.events import', 'from aura_intelligence.events import'),
        (r'from \.memory import', 'from aura_intelligence.memory import'),
        (r'from \.consensus import', 'from aura_intelligence.consensus import'),
        (r'from \.lnn import', 'from aura_intelligence.lnn import'),
        (r'from \.tda import', 'from aura_intelligence.tda import'),
        (r'from \.orchestration import', 'from aura_intelligence.orchestration import'),
        (r'from \.observability import', 'from aura_intelligence.observability import'),
        (r'from \.agents import', 'from aura_intelligence.agents import'),
        (r'from \.graph import', 'from aura_intelligence.graph import'),
        (r'from \.streaming import', 'from aura_intelligence.streaming import'),
        (r'from \.infrastructure import', 'from aura_intelligence.infrastructure import'),
        (r'from \.config import', 'from aura_intelligence.config import'),
        (r'from \.utils import', 'from aura_intelligence.utils import'),
        (r'from \.distributed import', 'from aura_intelligence.distributed import'),
        (r'from \.governance import', 'from aura_intelligence.governance import'),
        
        # Three levels up
        (r'from \.\.\.', 'from aura_intelligence.'),
    ]
    
    # Apply replacements
    modified = False
    for pattern, replacement in replacements:
        if re.search(pattern, content):
            content = re.sub(pattern, replacement, content)
            modified = True
    
    # Write back if modified
    if modified:
        with open(filepath, 'w') as f:
            f.write(content)
        return True
    return False

def fix_all_imports():
    """Fix imports in all Python files"""
    base_path = Path('/workspace/core/src/aura_intelligence')
    fixed_files = []
    
    # Find all Python files
    for py_file in base_path.rglob('*.py'):
        if fix_imports_in_file(str(py_file)):
            fixed_files.append(str(py_file))
    
    return fixed_files

if __name__ == "__main__":
    print("🔧 Fixing relative imports in AURA Intelligence...")
    
    fixed = fix_all_imports()
    
    print(f"\n✅ Fixed {len(fixed)} files:")
    for f in sorted(fixed)[:10]:
        print(f"   - {f}")
    
    if len(fixed) > 10:
        print(f"   ... and {len(fixed) - 10} more files")
    
    print("\n📝 Next steps:")
    print("1. Test imports by running: python3 -m aura_intelligence")
    print("2. Check specific modules that were problematic")
    print("3. Update __init__.py files if needed")