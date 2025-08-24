#!/usr/bin/env python3
"""
Create missing __init__.py files for proper Python module structure
"""

import os
from pathlib import Path

# Key directories that need __init__.py
dirs_to_init = [
    '/workspace/core/src/aura_intelligence/distributed',
    '/workspace/core/src/aura_intelligence/persistence',
    '/workspace/core/src/aura_intelligence/components',
    '/workspace/core/src/aura_intelligence/models',
    '/workspace/core/src/aura_intelligence/moe',
    '/workspace/core/src/aura_intelligence/graph',
    '/workspace/core/src/aura_intelligence/routing',
    '/workspace/core/src/aura_intelligence/network',
    '/workspace/core/src/aura_intelligence/api',
]

for dir_path in dirs_to_init:
    init_file = Path(dir_path) / "__init__.py"
    if not init_file.exists():
        # Create a basic __init__.py
        module_name = Path(dir_path).name
        content = f'''"""
{module_name.title()} module for AURA Intelligence
"""

# This file makes the directory a Python package
'''
        
        init_file.write_text(content)
        print(f"✅ Created: {init_file}")
    else:
        print(f"⏭️  Already exists: {init_file}")

print("\n✨ All critical __init__.py files created!")