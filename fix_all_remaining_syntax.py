#!/usr/bin/env python3
"""Comprehensive fix for all remaining syntax issues."""

import re
from pathlib import Path

def fix_pass_statements(content):
    """Remove unnecessary pass statements."""
    lines = content.split('\n')
    fixed_lines = []
    
    for i, line in enumerate(lines):
        # Skip pass lines that are followed by actual code
        if line.strip() == 'pass':
            # Check if next line has actual code (not a comment or empty)
            if i + 1 < len(lines):
                next_line = lines[i + 1]
                if next_line.strip() and not next_line.strip().startswith('#'):
                    # Skip this pass statement
                    continue
        
        fixed_lines.append(line)
    
    return '\n'.join(fixed_lines)

def fix_file(file_path):
    """Fix all syntax issues in a file."""
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    original = content
    
    # Fix pass statements
    content = fix_pass_statements(content)
    
    # Fix specific patterns
    # Remove pass followed by if/return/try/etc
    content = re.sub(r'\n(\s+)pass\n(\s+)(if |return |try:|except:|else:|elif |for |while |with |def |class |import |from |\w+ =)', 
                     r'\n\2\3', content)
    
    if content != original:
        with open(file_path, 'w') as f:
            f.write(content)
        return True
    return False

def main():
    core_dir = Path('/workspace/core/src/aura_intelligence/core')
    
    print("🔧 Fixing all remaining syntax issues in core files...")
    
    # Process all Python files
    for py_file in sorted(core_dir.glob('*.py')):
        if fix_file(py_file):
            print(f"✅ Fixed {py_file.name}")
        else:
            print(f"ℹ️  No changes needed in {py_file.name}")

if __name__ == "__main__":
    main()