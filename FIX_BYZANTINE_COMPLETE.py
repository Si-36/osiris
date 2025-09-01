#!/usr/bin/env python3
"""
Complete fix for byzantine.py
=============================
"""

import re

def fix_byzantine():
    """Fix all issues in byzantine.py"""
    filepath = "core/src/aura_intelligence/consensus/byzantine.py"
    
    print(f"🔧 Fixing {filepath}")
    
    try:
        with open(filepath, 'r') as f:
            lines = f.readlines()
        
        fixed_lines = []
        i = 0
        
        while i < len(lines):
            line = lines[i]
            
            # Skip standalone pass statements that are followed by docstrings
            if line.strip() == 'pass' and i + 1 < len(lines):
                next_line = lines[i + 1]
                if '"""' in next_line:
                    # Skip this pass statement
                    i += 1
                    continue
            
            # Fix methods that have wrong indentation after class methods
            if re.match(r'^\s{12}(async )?def \w+', line):
                # 12 spaces is too much, should be 4 for class methods
                line = re.sub(r'^(\s{12})', '    ', line)
            
            fixed_lines.append(line)
            i += 1
        
        # Join and write back
        content = ''.join(fixed_lines)
        
        # Additional fixes with regex
        # Fix the specific pattern where methods are nested inside other methods
        content = re.sub(
            r'(\n    async def [^:]+:.*?\n)        (async def)',
            r'\1    \2',
            content,
            flags=re.DOTALL
        )
        
        with open(filepath, 'w') as f:
            f.write(content)
        
        print(f"✅ Fixed {filepath}")
        
        # Test compilation
        try:
            compile(content, filepath, 'exec')
            print("✅ File now compiles successfully!")
        except SyntaxError as e:
            print(f"❌ Still has syntax error at line {e.lineno}: {e.msg}")
            if e.text:
                print(f"   {e.text.strip()}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    fix_byzantine()