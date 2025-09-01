#!/usr/bin/env python3
"""
Fix the indentation in byzantine.py
====================================
All async methods have 8 spaces instead of 4
"""

import re

def fix_byzantine():
    """Fix the byzantine.py indentation issues"""
    filepath = "core/src/aura_intelligence/consensus/byzantine.py"
    
    print(f"🔧 Fixing {filepath}")
    
    try:
        with open(filepath, 'r') as f:
            content = f.read()
        
        # Fix methods with 8 spaces indentation to 4 spaces
        # This regex finds lines that start with exactly 8 spaces followed by async def
        content = re.sub(
            r'^        (async def \w+)',  # 8 spaces + async def
            r'    \1',                     # Replace with 4 spaces + async def
            content,
            flags=re.MULTILINE
        )
        
        # Also fix the pass statements that are misplaced
        # Pattern: method definition followed by pass on wrong indentation
        content = re.sub(
            r'(async def [^:]+:)\n            pass\n        ("""[^"]+""")',
            r'\1\n        \2',
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
    fix_byzantine()