#!/usr/bin/env python3
"""Fix all empty except blocks in Python files."""

import os
import re
from pathlib import Path

def fix_empty_except_blocks(content):
    """Fix empty except blocks by adding pass statements."""
    
    # Pattern to find except blocks followed by non-indented code
    # This handles cases like:
    # except Something:
    # next_line
    
    lines = content.split('\n')
    fixed_lines = []
    
    for i, line in enumerate(lines):
        fixed_lines.append(line)
        
        # Check if this is an except line
        if line.strip().startswith('except') and line.strip().endswith(':'):
            # Check next line
            if i + 1 < len(lines):
                next_line = lines[i + 1]
                # Get indentation of except line
                except_indent = len(line) - len(line.lstrip())
                
                # Check if next line has same or less indentation (means empty block)
                if next_line.strip() and (len(next_line) - len(next_line.lstrip())) <= except_indent:
                    # Insert pass with proper indentation
                    fixed_lines.append(' ' * (except_indent + 4) + 'pass')
    
    return '\n'.join(fixed_lines)

def fix_file(file_path):
    """Fix empty except blocks in a file."""
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        fixed_content = fix_empty_except_blocks(content)
        
        if fixed_content != content:
            with open(file_path, 'w') as f:
                f.write(fixed_content)
            return True
        return False
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return False

def main():
    # Fix unified_interfaces.py specifically
    file_path = Path('/workspace/core/src/aura_intelligence/core/unified_interfaces.py')
    
    if fix_file(file_path):
        print(f"✅ Fixed empty except blocks in {file_path.name}")
    else:
        print(f"ℹ️ No empty except blocks found in {file_path.name}")

if __name__ == "__main__":
    main()