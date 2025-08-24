#!/usr/bin/env python3
"""
Final fix for tracing.py - handle both method and class indentation
"""

# Read the broken file
with open('/workspace/core/src/aura_intelligence/observability/tracing.py.broken', 'r') as f:
    content = f.read()

# Fix the most common pattern: unindented docstrings after method definitions
import re

def fix_method_docstring(content):
    """Fix docstrings that lost their indentation after method definitions"""
    lines = content.split('\n')
    fixed_lines = []
    
    i = 0
    while i < len(lines):
        line = lines[i]
        
        # Look for the specific pattern of method definition
        if re.match(r'\s+def process\(self, data: Dict\[str, Any\]\) -> Dict\[str, Any\]:', line):
            fixed_lines.append(line)
            i += 1
            
            # The next line should be the docstring
            if i < len(lines):
                next_line = lines[i]
                # Check if it's an unindented docstring
                if next_line.strip() == '"""REAL processing implementation"""':
                    # Get the indentation from the def line
                    indent = len(line) - len(line.lstrip())
                    # Add 4 more spaces for the docstring
                    fixed_lines.append(' ' * (indent + 4) + '"""REAL processing implementation"""')
                    i += 1
                    
                    # Now fix the following lines until we hit another def or class
                    while i < len(lines):
                        current = lines[i]
                        
                        # Stop if we hit another method or class at the same or less indentation
                        if current.strip() and not current.startswith(' ' * (indent + 4)):
                            if 'def ' in current or 'class ' in current:
                                break
                        
                        # Empty line - keep as is
                        if not current.strip():
                            fixed_lines.append(current)
                        # Line that needs indentation
                        elif current.strip() and not current.startswith(' ' * (indent + 4)):
                            # Special handling for if statements
                            if current.strip().startswith('if '):
                                fixed_lines.append(' ' * (indent + 4) + current.strip())
                                # Next line after if should be indented more
                                i += 1
                                if i < len(lines) and lines[i].strip():
                                    fixed_lines.append(' ' * (indent + 8) + lines[i].strip())
                                else:
                                    continue
                            else:
                                fixed_lines.append(' ' * (indent + 4) + current.strip())
                        else:
                            fixed_lines.append(current)
                        
                        i += 1
                else:
                    fixed_lines.append(next_line)
                    i += 1
        else:
            fixed_lines.append(line)
            i += 1
    
    return '\n'.join(fixed_lines)

# Apply the fix
fixed_content = fix_method_docstring(content)

# Write the fixed file
with open('/workspace/core/src/aura_intelligence/observability/tracing.py', 'w') as f:
    f.write(fixed_content)

print("Applied final fix to tracing.py")

# Now test if it can be imported
try:
    import sys
    sys.path.insert(0, '/workspace/core/src')
    
    # First try to import just the module
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "tracing", 
        "/workspace/core/src/aura_intelligence/observability/tracing.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    print("✅ Successfully imported tracing.py directly!")
    
    # Now try the full import chain
    import aura_intelligence
    print("✅ Successfully imported aura_intelligence package!")
    
except SyntaxError as e:
    print(f"❌ Syntax error: {e}")
    print(f"   Line {e.lineno}: {e.text}")
except Exception as e:
    print(f"❌ Import error: {e}")