#!/usr/bin/env python3
"""Fix ALL indentation issues in core files."""

import os
import re
from pathlib import Path

def fix_file(file_path):
    """Fix common indentation patterns in a Python file."""
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    original = content
    
    # Pattern 1: Remove pass before docstrings
    content = re.sub(r'(\):)\s*\n\s+pass\s*\n(\s+)(""")', r'\1\n\2\3', content)
    
    # Pattern 2: Fix docstrings that are over-indented after function signatures
    # This handles cases where ) -> Type: is followed by wrongly indented docstring
    lines = content.split('\n')
    fixed_lines = []
    
    for i, line in enumerate(lines):
        # Check if this is a function/method definition ending
        if ') -> ' in line and line.strip().endswith(':'):
            fixed_lines.append(line)
            
            # Check next lines for pass and/or docstring
            if i + 1 < len(lines):
                next_line = lines[i + 1]
                
                # Skip pass lines
                if next_line.strip() == 'pass':
                    if i + 2 < len(lines) and '"""' in lines[i + 2]:
                        # Skip the pass, move to docstring
                        continue
                
                # Check for over-indented docstring
                if i + 1 < len(lines) and '"""' in lines[i + 1]:
                    # Get the proper indentation (function indent + 4)
                    func_indent = len(line) - len(line.lstrip())
                    proper_doc_indent = func_indent + 4
                    
                    # Fix the docstring line indentation
                    doc_line = lines[i + 1]
                    doc_content = doc_line.strip()
                    lines[i + 1] = ' ' * proper_doc_indent + doc_content
        
        elif line.strip() == 'pass' and i > 0:
            # Check if previous line is a function def and next is docstring
            prev_line = lines[i - 1] if i > 0 else ""
            next_line = lines[i + 1] if i + 1 < len(lines) else ""
            
            if '):' in prev_line and '"""' in next_line:
                # Skip this pass line
                continue
        
        fixed_lines.append(line)
    
    content = '\n'.join(fixed_lines)
    
    # Pattern 3: Fix module-level functions that are indented
    content = re.sub(r'\n    def (create_\w+|analyze_\w+)\(', r'\ndef \1(', content)
    
    if content != original:
        with open(file_path, 'w') as f:
            f.write(content)
        return True
    return False

def main():
    core_dir = Path('/workspace/core/src/aura_intelligence/core')
    
    files_to_fix = [
        'self_healing.py',
        'error_topology.py',
        'exceptions.py',
        'types.py',
        'interfaces.py',
        # Add more as needed
    ]
    
    print("🔧 Fixing indentation issues in core files...")
    
    for file_name in files_to_fix:
        file_path = core_dir / file_name
        if file_path.exists():
            if fix_file(file_path):
                print(f"✅ Fixed {file_name}")
            else:
                print(f"ℹ️ No changes needed in {file_name}")
        else:
            print(f"⚠️ {file_name} not found")

if __name__ == "__main__":
    main()