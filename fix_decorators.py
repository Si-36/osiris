#!/usr/bin/env python3
"""
Fix decorators.py indentation issues
"""

# Read the file
with open('core/src/aura_intelligence/utils/decorators.py', 'r') as f:
    content = f.read()

# Fix specific known issues
# 1. sync_wrapper body not indented
content = content.replace(
    """        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
        if not self.can_attempt_call():""",
    """        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            if not self.can_attempt_call():"""
)

# 2. Fix the try block
content = content.replace(
    """            
        try:
            result = func(*args, **kwargs)
        self.call_succeeded()
        return result
        except self.config.expected_exception as e:""",
    """            
            try:
                result = func(*args, **kwargs)
                self.call_succeeded()
                return result
            except self.config.expected_exception as e:"""
)

# 3. Fix other unindented lines in sync_wrapper
lines = content.split('\n')
fixed_lines = []
in_sync_wrapper = False
wrapper_indent = 0

for i, line in enumerate(lines):
    if 'def sync_wrapper(' in line:
        in_sync_wrapper = True
        wrapper_indent = len(line) - len(line.lstrip()) + 4
        fixed_lines.append(line)
    elif in_sync_wrapper and line.strip() and not line.strip().startswith('@'):
        # Check if this line is part of sync_wrapper
        current_indent = len(line) - len(line.lstrip())
        if current_indent < wrapper_indent and line.strip() not in ['', '}', ']', ')']:
            # Under-indented, fix it
            fixed_lines.append(' ' * wrapper_indent + line.strip())
        else:
            fixed_lines.append(line)
        
        # Check if we're out of sync_wrapper
        if 'def async_wrapper(' in line or ('def ' in line and current_indent <= wrapper_indent - 4):
            in_sync_wrapper = False
    else:
        fixed_lines.append(line)

# Write back
with open('core/src/aura_intelligence/utils/decorators.py', 'w') as f:
    f.write('\n'.join(fixed_lines))

# Test
import ast
try:
    with open('core/src/aura_intelligence/utils/decorators.py', 'r') as f:
        ast.parse(f.read())
    print("✅ decorators.py fixed!")
except SyntaxError as e:
    print(f"❌ Still has error at line {e.lineno}: {e.msg}")