#!/usr/bin/env python3
"""Fix all indentation issues in self_healing.py."""

import re

def fix_file():
    file_path = '/workspace/core/src/aura_intelligence/core/self_healing.py'
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Fix over-indented async/def statements
    # Pattern: too many spaces before async def or def
    content = re.sub(r'\n        (async def|def) (\w+)', r'\n    \1 \2', content)
    
    # Fix function signatures with extra spaces
    content = re.sub(r'(\w+)\(\s+self,\s+', r'\1(self, ', content)
    content = re.sub(r',\s+(\w+):\s+', r', \1: ', content)
    content = re.sub(r'\s+\)\s*->\s*', r') -> ', content)
    
    # Fix methods that should be at class level (4 spaces) not nested level (8 spaces)
    lines = content.split('\n')
    fixed_lines = []
    in_class = False
    
    for i, line in enumerate(lines):
        # Detect class definitions
        if line.strip().startswith('class ') and line.strip().endswith(':'):
            in_class = True
            fixed_lines.append(line)
            continue
        
        # If we're in a class and see a method with 8 spaces, fix to 4
        if in_class and line.startswith('        def ') or line.startswith('        async def '):
            # This is over-indented, should be 4 spaces not 8
            fixed_lines.append(line[4:])  # Remove 4 spaces
            continue
        
        # Reset in_class if we see a non-indented line
        if line and not line[0].isspace():
            in_class = False
        
        fixed_lines.append(line)
    
    content = '\n'.join(fixed_lines)
    
    with open(file_path, 'w') as f:
        f.write(content)
    
    print("✅ Fixed indentation in self_healing.py")

if __name__ == "__main__":
    fix_file()