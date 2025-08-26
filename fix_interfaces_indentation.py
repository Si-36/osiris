#!/usr/bin/env python3
"""
Fix all indentation issues in interfaces.py
"""
import re

def fix_indentation():
    file_path = '/workspace/core/src/aura_intelligence/core/interfaces.py'
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Fix patterns where @abstractmethod is followed by extra-indented method
    pattern = r'(\s*)@abstractmethod\n\s{8}(async def|def)'
    content = re.sub(pattern, r'\1@abstractmethod\n\1\2', content)
    
    # Fix methods that are incorrectly indented without decorators
    lines = content.split('\n')
    fixed_lines = []
    in_class = False
    class_indent = 0
    
    for i, line in enumerate(lines):
        if line.strip().startswith('class '):
            in_class = True
            class_indent = len(line) - len(line.lstrip())
            fixed_lines.append(line)
        elif in_class and line.strip() and not line.startswith(' '):
            # End of class
            in_class = False
            fixed_lines.append(line)
        elif in_class and line.strip().startswith('async def ') or line.strip().startswith('def '):
            # Method definition - should be indented 4 spaces from class
            method_line = line.lstrip()
            correct_indent = ' ' * (class_indent + 4)
            fixed_lines.append(correct_indent + method_line)
        else:
            fixed_lines.append(line)
    
    content = '\n'.join(fixed_lines)
    
    with open(file_path, 'w') as f:
        f.write(content)
    
    print(f"Fixed indentation in {file_path}")

if __name__ == "__main__":
    fix_indentation()