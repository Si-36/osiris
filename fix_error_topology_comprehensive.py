#!/usr/bin/env python3
"""Comprehensively fix all syntax issues in error_topology.py."""

import re

def fix_error_topology():
    """Fix all syntax issues in error_topology.py."""
    
    file_path = '/workspace/core/src/aura_intelligence/core/error_topology.py'
    
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    fixed_lines = []
    i = 0
    
    while i < len(lines):
        line = lines[i]
        
        # Check for pattern: ) -> Type:        """
        # This is a malformed function definition where docstring is on same line
        if ') -> ' in line and '"""' in line and ':        """' in line:
            # Split the line properly
            parts = line.split(':        """')
            if len(parts) == 2:
                # Fix it to be on separate lines
                fixed_lines.append(parts[0] + ':\n')
                # Get proper indentation for docstring
                indent = len(line) - len(line.lstrip())
                func_indent = ' ' * (indent + 4)
                fixed_lines.append(func_indent + '"""' + parts[1])
                i += 1
                continue
        
        # Check for pattern where ) -> Type: is followed by pass
        if i < len(lines) - 1 and ') -> ' in line and line.strip().endswith(':'):
            next_line = lines[i + 1]
            # Check if next line has just 'pass'
            if next_line.strip() == 'pass':
                # Check if line after that is docstring
                if i < len(lines) - 2 and '"""' in lines[i + 2]:
                    fixed_lines.append(line)  # Add function definition
                    # Skip the 'pass' line
                    i += 2
                    continue
        
        fixed_lines.append(line)
        i += 1
    
    # Write back
    with open(file_path, 'w') as f:
        f.writelines(fixed_lines)
    
    print(f"✅ Fixed {file_path}")
    
    # Now also fix any remaining pass before docstring issues
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Remove standalone pass lines that appear right before docstrings
    lines = content.split('\n')
    fixed_lines = []
    i = 0
    
    while i < len(lines):
        line = lines[i]
        
        # Skip lines that are just whitespace + pass if followed by docstring
        if line.strip() == 'pass':
            if i < len(lines) - 1 and '"""' in lines[i + 1]:
                # Skip this pass line
                i += 1
                continue
        
        fixed_lines.append(line)
        i += 1
    
    content = '\n'.join(fixed_lines)
    
    with open(file_path, 'w') as f:
        f.write(content)
    
    print("✅ Removed standalone pass statements before docstrings")

if __name__ == "__main__":
    fix_error_topology()