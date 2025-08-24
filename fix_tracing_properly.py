#!/usr/bin/env python3
"""
Properly fix ALL indentation issues in tracing.py
"""

# Read the original broken file
with open('/workspace/core/src/aura_intelligence/observability/tracing.py.broken', 'r') as f:
    lines = f.readlines()

fixed_lines = []
i = 0

while i < len(lines):
    line = lines[i]
    
    # Check if this is a process method definition
    if 'def process(self, data: Dict[str, Any]) -> Dict[str, Any]:' in line:
        fixed_lines.append(line)
        i += 1
        
        # Next line should be the docstring
        if i < len(lines) and '"""' in lines[i]:
            # Fix docstring indentation
            fixed_lines.append('            """REAL processing implementation"""\n')
            i += 1
            
            # Now fix the method body
            while i < len(lines):
                current_line = lines[i]
                
                # Check if we've reached the next method or class
                if current_line.strip() and not current_line.startswith(' '):
                    if 'def ' in current_line or 'class ' in current_line:
                        break
                
                # Skip the already processed docstring lines
                if '"""REAL processing implementation"""' in current_line:
                    i += 1
                    continue
                
                # Handle import statements
                if current_line.strip().startswith('import '):
                    fixed_lines.append('            ' + current_line.strip() + '\n')
                    i += 1
                    continue
                
                # Handle empty lines
                if current_line.strip() == '':
                    fixed_lines.append(current_line)
                    i += 1
                    continue
                
                # Handle if statements
                if current_line.strip().startswith('if '):
                    fixed_lines.append('            ' + current_line.strip() + '\n')
                    i += 1
                    # The next line should be indented more
                    if i < len(lines) and lines[i].strip() and not lines[i].strip().startswith('elif') and not lines[i].strip().startswith('else'):
                        fixed_lines.append('                ' + lines[i].strip() + '\n')
                        i += 1
                    continue
                
                # Handle return statements that are part of if blocks
                if 'return {' in current_line and not current_line.startswith('            '):
                    # Check if previous line was an if
                    if i > 0 and 'if ' in lines[i-1]:
                        fixed_lines.append('                ' + current_line.strip() + '\n')
                    else:
                        fixed_lines.append('            ' + current_line.strip() + '\n')
                    i += 1
                    continue
                
                # Handle regular statements
                if current_line.strip() and not current_line.startswith('        '):
                    fixed_lines.append('            ' + current_line.strip() + '\n')
                else:
                    fixed_lines.append(current_line)
                
                i += 1
        else:
            i += 1
    else:
        fixed_lines.append(line)
        i += 1

# Write the fixed content
with open('/workspace/core/src/aura_intelligence/observability/tracing.py', 'w') as f:
    f.writelines(fixed_lines)

print("Fixed tracing.py with proper indentation")

# Test import
try:
    import sys
    sys.path.insert(0, '/workspace/core/src')
    import aura_intelligence.observability.tracing
    print("✅ Successfully imported tracing module!")
except Exception as e:
    print(f"❌ Import error: {e}")
    
    # Try direct module import
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location("tracing", "/workspace/core/src/aura_intelligence/observability/tracing.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        print("✅ Direct module import successful!")
    except Exception as e2:
        print(f"❌ Direct import also failed: {e2}")