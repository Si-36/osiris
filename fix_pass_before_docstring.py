#!/usr/bin/env python3
"""Fix all instances of 'pass' appearing before docstrings in Python files."""

import re
import os

def fix_pass_before_docstring(file_path):
    """Fix pass statements that appear before docstrings."""
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    original_content = content
    
    # Pattern 1: pass followed by docstring on next line
    pattern1 = r'(\s+)pass\n(\s+)(""".*?""")'
    content = re.sub(pattern1, r'\2\3', content, flags=re.DOTALL)
    
    # Pattern 2: pass followed by docstring with different indentation
    pattern2 = r'(\):)\s*\n\s+pass\s*\n(\s+)(""")'
    content = re.sub(pattern2, r'\1\n\2\3', content)
    
    # Pattern 3: Fix specific case where pass is on same indentation as docstring
    lines = content.split('\n')
    fixed_lines = []
    i = 0
    while i < len(lines):
        line = lines[i]
        
        # Check if current line is just 'pass' and next line starts with """
        if i < len(lines) - 1:
            current_stripped = line.strip()
            next_line = lines[i + 1] if i + 1 < len(lines) else ""
            
            if current_stripped == 'pass' and '"""' in next_line:
                # Skip the pass line
                i += 1
                continue
        
        fixed_lines.append(line)
        i += 1
    
    content = '\n'.join(fixed_lines)
    
    if content != original_content:
        with open(file_path, 'w') as f:
            f.write(content)
        return True
    return False

def main():
    file_path = '/workspace/core/src/aura_intelligence/core/error_topology.py'
    
    print(f"🔧 Fixing pass before docstring issues in {file_path}")
    
    if fix_pass_before_docstring(file_path):
        print("✅ Fixed pass before docstring issues")
    else:
        print("ℹ️ No changes needed")

if __name__ == "__main__":
    main()