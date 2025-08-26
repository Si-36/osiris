#!/usr/bin/env python3
"""Final comprehensive fix for error_topology.py."""

def fix_file():
    file_path = '/workspace/core/src/aura_intelligence/core/error_topology.py'
    
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    fixed_lines = []
    i = 0
    
    while i < len(lines):
        line = lines[i]
        
        # Skip standalone pass lines that shouldn't be there
        if line.strip() == 'pass':
            # Check if it's in a function that already has content
            if i < len(lines) - 1:
                next_line = lines[i + 1]
                # If next line is not empty and has code content, skip the pass
                if next_line.strip() and not next_line.strip().startswith('#'):
                    i += 1
                    continue
        
        # Check for malformed function signatures with docstring on same line
        if ') -> ' in line and ':' in line:
            # Check if there's content after the colon
            parts = line.split(':')
            if len(parts) >= 2:
                after_colon = ':'.join(parts[1:]).strip()
                if after_colon and after_colon != '':
                    # There's content after colon, split it
                    fixed_lines.append(parts[0] + ':\n')
                    # Add proper indentation for next content
                    indent = len(line) - len(line.lstrip())
                    if '"""' in after_colon:
                        # It's a docstring, indent it properly
                        fixed_lines.append(' ' * (indent + 4) + after_colon.lstrip() + '\n')
                    else:
                        fixed_lines.append(' ' * (indent + 4) + after_colon.lstrip() + '\n')
                    i += 1
                    continue
        
        fixed_lines.append(line)
        i += 1
    
    # Write back
    with open(file_path, 'w') as f:
        f.writelines(fixed_lines)
    
    print(f"✅ Applied comprehensive fixes to {file_path}")
    
    # Second pass: remove any remaining issues
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Fix specific patterns
    import re
    
    # Pattern: function with pass followed by actual code
    content = re.sub(r'(\n\s+)pass\n(\s+)if ', r'\1\2if ', content)
    content = re.sub(r'(\n\s+)pass\n(\s+)return ', r'\1\2return ', content)
    content = re.sub(r'(\n\s+)pass\n(\s+)try:', r'\1\2try:', content)
    content = re.sub(r'(\n\s+)pass\n(\s+)G = ', r'\1\2G = ', content)
    content = re.sub(r'(\n\s+)pass\n(\s+)# ', r'\1\2# ', content)
    
    with open(file_path, 'w') as f:
        f.write(content)
    
    print("✅ Removed unnecessary pass statements")

if __name__ == "__main__":
    fix_file()