#!/usr/bin/env python3
"""Comprehensive fix for all indentation issues in unified_config.py."""

import re

def fix_file():
    file_path = '/workspace/core/src/aura_intelligence/core/unified_config.py'
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Fix patterns where class content is not indented
    lines = content.split('\n')
    fixed_lines = []
    i = 0
    in_class = False
    in_method = False
    method_indent = 0
    
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        
        # Detect class start
        if stripped.startswith('class ') and stripped.endswith(':'):
            in_class = True
            in_method = False
            fixed_lines.append(line)
            i += 1
            continue
        
        # Detect @dataclass
        if stripped == '@dataclass':
            in_class = False
            in_method = False
            fixed_lines.append(line)
            i += 1
            continue
        
        # If in a class
        if in_class:
            # Check for decorators that should be indented
            if stripped.startswith('@'):
                fixed_lines.append('    ' + stripped)
                i += 1
                continue
            
            # Check for methods
            if stripped.startswith('def '):
                in_method = True
                method_indent = 8  # Class method should be at 4, content at 8
                fixed_lines.append('    ' + stripped)
                i += 1
                continue
            
            # Check for docstrings in methods
            if in_method and stripped.startswith('"""'):
                fixed_lines.append(' ' * method_indent + stripped)
                i += 1
                continue
            
            # Check for method content
            if in_method and stripped:
                # If it's a new class or decorator, we're out of the method
                if stripped.startswith('class ') or stripped.startswith('@'):
                    in_class = False
                    in_method = False
                    fixed_lines.append(line)
                else:
                    fixed_lines.append(' ' * method_indent + stripped)
                i += 1
                continue
            
            # Empty line ends class
            if not stripped:
                in_class = False
                in_method = False
        
        fixed_lines.append(line)
        i += 1
    
    content = '\n'.join(fixed_lines)
    
    with open(file_path, 'w') as f:
        f.write(content)
    
    print("✅ Comprehensively fixed unified_config.py")

if __name__ == "__main__":
    fix_file()