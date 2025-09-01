#!/usr/bin/env python3
"""Aggressively fix all syntax issues in error_topology.py."""

import re

def fix_file():
    file_path = '/workspace/core/src/aura_intelligence/core/error_topology.py'
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Fix common patterns of split signatures
    # Pattern 1: parameter_name:\n    Type) ->
    content = re.sub(r'(\w+):\s*\n\s+(\w+[\w\[\], ]*)\)', r'\1: \2)', content)
    
    # Pattern 2: self, param:\n    Type) ->
    content = re.sub(r'(self,\s*\w+):\s*\n\s+(\w+)', r'\1: \2', content)
    
    # Pattern 3: Fix docstrings that are over-indented after function signatures
    lines = content.split('\n')
    fixed_lines = []
    
    for i, line in enumerate(lines):
        # Check if this is a function definition that ends with :
        if line.strip().endswith(':') and ') ->' in line:
            fixed_lines.append(line)
            
            # Check next line for over-indented docstring
            if i + 1 < len(lines) and '"""' in lines[i + 1]:
                next_line = lines[i + 1]
                # Calculate proper indentation (function indent + 4)
                func_indent = len(line) - len(line.lstrip())
                proper_indent = func_indent + 4
                
                # Count actual indent of docstring
                doc_indent = len(next_line) - len(next_line.lstrip())
                
                # If over-indented, fix it
                if doc_indent > proper_indent:
                    lines[i + 1] = ' ' * proper_indent + next_line.strip()
        else:
            fixed_lines.append(line)
    
    content = '\n'.join(fixed_lines)
    
    # Fix any remaining split signatures by looking for patterns
    # Pattern: word(\n    stuff) -> Type:
    content = re.sub(r'(\w+)\(\s*\n\s+([^)]+)\)\s*->\s*(\w+):', r'\1(\2) -> \3:', content)
    
    with open(file_path, 'w') as f:
        f.write(content)
    
    print("✅ Aggressively fixed error_topology.py")

if __name__ == "__main__":
    fix_file()