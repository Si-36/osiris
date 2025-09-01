#!/usr/bin/env python3
"""
Fix all syntax issues in observer.py
=====================================
"""

import re

def fix_observer():
    """Fix all syntax issues in observer.py"""
    filepath = "core/src/aura_intelligence/agents/v2/observer.py"
    
    print(f"🔧 Fixing {filepath}")
    
    try:
        with open(filepath, 'r') as f:
            content = f.read()
        
        # Fix pattern: incorrectly indented async methods
        # Pattern: 8 spaces + async def -> 4 spaces + async def
        content = re.sub(
            r'\n        (async def \w+)',  # 8 spaces before async def
            r'\n    \1',                   # Replace with 4 spaces
            content
        )
        
        # Fix pattern: remove pass statements before docstrings in methods
        content = re.sub(
            r'(async def [^:]+:)\n\s+pass\n\s+"""',
            r'\1\n        """',
            content
        )
        
        # Fix pattern: incorrectly indented def methods (non-async)
        content = re.sub(
            r'\n        (def \w+)',  # 8 spaces before def
            r'\n    \1',             # Replace with 4 spaces
            content
        )
        
        # Write back
        with open(filepath, 'w') as f:
            f.write(content)
        
        print(f"✅ Fixed {filepath}")
        
        # Test compilation
        try:
            with open(filepath, 'r') as f:
                code = f.read()
            compile(code, filepath, 'exec')
            print("✅ File now compiles successfully!")
        except SyntaxError as e:
            print(f"❌ Still has syntax error: {e}")
            print(f"   Line {e.lineno}: {e.text}")
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    fix_observer()