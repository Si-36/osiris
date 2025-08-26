#!/usr/bin/env python3

import os
import re
import ast

def fix_critical_indentation_issues(file_path):
    """Fix critical indentation issues in Python files."""
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return False
    
    original_content = content
    
    # Fix except blocks with wrong indentation
    content = re.sub(
        r'(\s+except [^:]+:\s*\n)\s{0,3}(\s*pass)', 
        r'\1                    \2',
        content, flags=re.MULTILINE
    )
    
    # Fix try blocks with wrong indentation  
    content = re.sub(
        r'(\s+try:\s*\n)\s{0,7}([^\s])',
        r'\1        \2',
        content, flags=re.MULTILINE
    )
    
    # Fix function bodies with wrong indentation after def
    lines = content.split('\n')
    fixed_lines = []
    
    for i, line in enumerate(lines):
        # Skip empty lines
        if not line.strip():
            fixed_lines.append(line)
            continue
            
        # Check if previous line is a function or method definition
        if i > 0:
            prev_line = lines[i-1]
            if (re.match(r'\s+(def |async def )', prev_line) and 
                prev_line.endswith(':') and 
                line.strip() and
                not line.startswith('        ')):
                
                # Fix indentation to 8 spaces for method body
                line = '        ' + line.strip()
        
        # Fix except statements that don't have proper indentation for their body
        if (re.match(r'\s+except [^:]+:\s*$', line) and 
            i + 1 < len(lines) and 
            lines[i + 1].strip() and
            not lines[i + 1].startswith('                    ')):
            
            # The next line should be indented properly
            next_line = lines[i + 1]
            if next_line.strip():
                lines[i + 1] = '                    ' + next_line.strip()
        
        fixed_lines.append(line)
    
    content = '\n'.join(fixed_lines)
    
    # Only write if content changed
    if content != original_content:
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            return True
        except Exception as e:
            print(f"Error writing {file_path}: {e}")
            return False
    
    return False

def main():
    # Focus on the core files that are causing import issues
    critical_files = [
        '/home/sina/projects/osiris-2/core/src/aura_intelligence/core/unified_interfaces.py',
        '/home/sina/projects/osiris-2/core/src/aura_intelligence/core/unified_system.py',
        '/home/sina/projects/osiris-2/core/src/aura_intelligence/core/types.py',
        '/home/sina/projects/osiris-2/core/src/aura_intelligence/core/interfaces.py',
        '/home/sina/projects/osiris-2/core/src/aura_intelligence/core/exceptions.py',
        '/home/sina/projects/osiris-2/core/src/aura_intelligence/__init__.py',
        '/home/sina/projects/osiris-2/core/src/aura_intelligence/core/__init__.py',
    ]
    
    fixed_count = 0
    
    for file_path in critical_files:
        if os.path.exists(file_path) and fix_critical_indentation_issues(file_path):
            print(f"Fixed: {file_path}")
            fixed_count += 1
    
    print(f"Fixed critical indentation in {fixed_count} files")

if __name__ == "__main__":
    main()