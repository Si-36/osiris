#!/usr/bin/env python3
"""
Fix indentation issues in tracing.py
"""

import re

# Read the broken file
with open('/workspace/core/src/aura_intelligence/observability/tracing.py', 'r') as f:
    content = f.read()

# Fix pattern 1: Unindented docstrings after function definitions
# This regex finds function definitions followed by unindented docstrings
pattern = r'(def process\(self, data: Dict\[str, Any\]\) -> Dict\[str, Any\]:\n)(\s*)(""".*?""")'

def fix_docstring(match):
    func_def = match.group(1)
    spaces = match.group(2)
    docstring = match.group(3)
    # Add proper indentation (4 spaces for methods inside classes)
    return func_def + '            ' + docstring

# Fix all occurrences
fixed_content = re.sub(pattern, fix_docstring, content, flags=re.MULTILINE)

# Fix pattern 2: Fix the rest of the method body indentation
# Look for lines that should be indented but aren't after the docstring
lines = fixed_content.split('\n')
fixed_lines = []
in_process_method = False
need_indent = False

for i, line in enumerate(lines):
    if 'def process(self, data: Dict[str, Any])' in line:
        in_process_method = True
        need_indent = False
        fixed_lines.append(line)
    elif in_process_method and '"""' in line and line.strip().startswith('"""'):
        # This is the docstring, should be indented
        if not line.startswith('            '):
            fixed_lines.append('            ' + line.strip())
        else:
            fixed_lines.append(line)
        need_indent = True
    elif in_process_method and need_indent:
        # Check if this is the next method/class definition
        if line.strip() and not line.startswith(' ') and ('def ' in line or 'class ' in line):
            in_process_method = False
            need_indent = False
            fixed_lines.append(line)
        elif line.strip() == '':
            fixed_lines.append(line)
        elif not line.startswith('            ') and line.strip():
            # This line needs indentation
            fixed_lines.append('            ' + line.strip())
        else:
            fixed_lines.append(line)
    else:
        fixed_lines.append(line)

# Join back
fixed_content = '\n'.join(fixed_lines)

# Write the fixed file
with open('/workspace/core/src/aura_intelligence/observability/tracing.py', 'w') as f:
    f.write(fixed_content)

print("Fixed indentation issues in tracing.py")

# Test if we can import it now
try:
    import importlib.util
    spec = importlib.util.spec_from_file_location("tracing", "/workspace/core/src/aura_intelligence/observability/tracing.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    print("✅ File can now be imported successfully!")
except Exception as e:
    print(f"❌ Still has issues: {e}")