#!/usr/bin/env python3
"""
Fix all indentation issues in tracing.py
"""

import re

# Read the file
with open('core/src/aura_intelligence/observability/tracing.py', 'r') as f:
    lines = f.readlines()

# Track indentation level
fixed_lines = []
in_mock_block = False
in_mock_class = False
current_class_indent = 0
in_method = False
method_indent = 0

i = 0
while i < len(lines):
    line = lines[i]
    
    # Check if we're entering the mock block
    if 'if not OPENTELEMETRY_AVAILABLE:' in line:
        in_mock_block = True
        fixed_lines.append(line)
        i += 1
        continue
    
    # Inside mock block
    if in_mock_block:
        # Check for class definitions
        if line.strip().startswith('class Mock'):
            # This should be indented 4 spaces from the if
            fixed_lines.append('    ' + line.strip() + '\n')
            in_mock_class = True
            current_class_indent = 4
            i += 1
            continue
        
        # Check if we're inside a mock class
        if in_mock_class:
            stripped = line.strip()
            
            # Empty lines
            if not stripped:
                fixed_lines.append('\n')
                i += 1
                continue
            
            # Docstrings
            if stripped.startswith('"""'):
                if in_method:
                    fixed_lines.append(' ' * (method_indent + 4) + stripped + '\n')
                else:
                    fixed_lines.append(' ' * (current_class_indent + 4) + stripped + '\n')
                i += 1
                continue
            
            # Method definitions
            if stripped.startswith('def '):
                fixed_lines.append(' ' * (current_class_indent + 4) + stripped + '\n')
                in_method = True
                method_indent = current_class_indent + 4
                i += 1
                continue
            
            # Inside method
            if in_method:
                # Check if this line starts a new method or class
                if stripped.startswith('def ') and not line.startswith(' ' * (method_indent)):
                    in_method = False
                    fixed_lines.append(' ' * (current_class_indent + 4) + stripped + '\n')
                    in_method = True
                    method_indent = current_class_indent + 4
                elif stripped.startswith('class '):
                    in_method = False
                    in_mock_class = True
                    fixed_lines.append('    ' + stripped + '\n')
                    current_class_indent = 4
                else:
                    # Regular method content
                    if stripped.startswith('return ') or stripped.startswith('if ') or \
                       stripped.startswith('elif ') or stripped.startswith('else:') or \
                       stripped.startswith('import ') or stripped.startswith('from ') or \
                       stripped.startswith('#'):
                        fixed_lines.append(' ' * (method_indent + 4) + stripped + '\n')
                    elif stripped.startswith('}') or stripped.startswith(']'):
                        # Closing brackets
                        fixed_lines.append(' ' * (method_indent + 4) + stripped + '\n')
                    elif 'import ' in stripped and stripped.index('import') == 0:
                        fixed_lines.append(' ' * (method_indent + 4) + stripped + '\n')
                    else:
                        # Check if it's a continuation of previous line
                        if i > 0 and (lines[i-1].strip().endswith(',') or 
                                     lines[i-1].strip().endswith('(') or
                                     lines[i-1].strip().endswith('{')):
                            # This is likely part of a multi-line statement
                            fixed_lines.append(' ' * (method_indent + 8) + stripped + '\n')
                        else:
                            fixed_lines.append(' ' * (method_indent + 4) + stripped + '\n')
                i += 1
                continue
    
    # Not in mock block, keep original line
    fixed_lines.append(line)
    i += 1

# Write the fixed file
with open('core/src/aura_intelligence/observability/tracing.py', 'w') as f:
    f.writelines(fixed_lines)

print("Fixed indentation in tracing.py")