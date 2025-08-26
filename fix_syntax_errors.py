#!/usr/bin/env python3
"""
Fix common syntax errors in the AURA codebase
"""

import os
import re
from pathlib import Path

def fix_indentation_errors(file_path):
    """Fix common indentation errors in Python files"""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    original_content = content
    
    # Fix pattern: function definition followed by docstring with wrong indentation
    pattern = r'(def [^:]+:)\s*\n\s*"""([^"]*?)"""'
    matches = re.findall(pattern, content, re.MULTILINE | re.DOTALL)
    
    for match in matches:
        func_def, docstring = match
        # Replace with properly indented docstring
        old_text = f'{func_def}\n        """{docstring}"""'
        new_text = f'{func_def}\n            """{docstring}"""\n            pass'
        content = content.replace(old_text, new_text)
    
    # Fix pattern: class __init__ followed by unindented docstring
    pattern = r'(def __init__\([^)]*\):\s*)\n\s*"""([^"]*?)"""'
    matches = re.findall(pattern, content, re.MULTILINE | re.DOTALL)
    
    for match in matches:
        init_def, docstring = match
        old_text = f'{init_def}\n        """{docstring}"""'
        new_text = f'{init_def}\n            """{docstring}"""\n            pass'
        content = content.replace(old_text, new_text)
    
    # Fix standalone docstrings without proper indentation
    content = re.sub(
        r'(\n\s*def [^:]+:)\s*\n\s*"""TODO: Implement this method"""',
        r'\1\n            """TODO: Implement this method"""\n            pass',
        content
    )
    
    # Write back if changed
    if content != original_content:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"Fixed syntax errors in {file_path}")
        return True
    
    return False

def find_and_fix_python_files(directory):
    """Find and fix syntax errors in all Python files"""
    fixed_count = 0
    
    for root, dirs, files in os.walk(directory):
        # Skip certain directories
        skip_dirs = {'.git', '__pycache__', '.pytest_cache', 'node_modules', 'archive'}
        dirs[:] = [d for d in dirs if d not in skip_dirs]
        
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                if fix_indentation_errors(file_path):
                    fixed_count += 1
    
    return fixed_count

if __name__ == "__main__":
    # Fix syntax errors in the core directory
    core_dir = "core/src/aura_intelligence"
    
    print(f"🔧 Scanning {core_dir} for syntax errors...")
    fixed = find_and_fix_python_files(core_dir)
    print(f"✅ Fixed syntax errors in {fixed} files")