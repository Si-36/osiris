#!/usr/bin/env python3
"""Fix indentation issues in timeout.py"""

import re

# Read the file
with open('core/src/aura_intelligence/resilience/timeout.py', 'r') as f:
    lines = f.readlines()

# Track current class/function context
in_class = False
class_level = 0
in_function = False
function_level = 0

fixed_lines = []
i = 0

while i < len(lines):
    line = lines[i]
    stripped = line.strip()
    
    # Skip empty lines and comments at the start
    if not stripped or stripped.startswith('#'):
        fixed_lines.append(line)
        i += 1
        continue
    
    # Detect class definition
    if re.match(r'^class\s+\w+.*:$', line):
        in_class = True
        class_level = 0
        fixed_lines.append(line)
        i += 1
        continue
    
    # If we're inside a class and the line is not indented, it needs fixing
    if in_class and not line.startswith((' ', '\t')):
        # Check if it's a method definition
        if re.match(r'^(async\s+)?def\s+\w+.*:$', stripped) or \
           re.match(r'^@\w+', stripped) or \
           stripped.startswith('"""') or \
           re.match(r'^\w+:', stripped):  # attribute definition
            # This should be indented inside the class
            fixed_lines.append('    ' + line)
            i += 1
            continue
    
    # Check for unindented lines that should be part of a method
    if in_class and re.match(r'^[a-zA-Z]', line) and not re.match(r'^class\s+', line):
        # Look at previous non-empty line to determine context
        j = len(fixed_lines) - 1
        while j >= 0 and not fixed_lines[j].strip():
            j -= 1
        
        if j >= 0:
            prev_line = fixed_lines[j]
            # If previous line was a method/function definition or indented code
            if 'def ' in prev_line or prev_line.startswith('    '):
                # This line should be indented
                if 'self.' in stripped or 'return ' in stripped or '=' in stripped:
                    fixed_lines.append('        ' + line)  # method body
                else:
                    fixed_lines.append('    ' + line)  # class attribute
                i += 1
                continue
    
    # Default: keep the line as is
    fixed_lines.append(line)
    i += 1

# Write the fixed file
with open('core/src/aura_intelligence/resilience/timeout.py', 'w') as f:
    f.writelines(fixed_lines)

print("✅ Fixed indentation in timeout.py")