#!/usr/bin/env python3
"""Fix class body indentation in unified_config.py."""

def fix_file():
    file_path = '/workspace/core/src/aura_intelligence/core/unified_config.py'
    
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    fixed_lines = []
    in_class = False
    
    for i, line in enumerate(lines):
        # Detect class definition
        if line.strip().startswith('class ') and line.strip().endswith(':'):
            in_class = True
            fixed_lines.append(line)
            continue
        
        # If we're in a class and the line is not indented
        if in_class and line.strip() and not line.startswith('    ') and not line.startswith('\t'):
            # Check if it's a new class or module-level code
            if line.strip().startswith('class ') or line.strip().startswith('def ') or line.strip().startswith('#'):
                in_class = False
                fixed_lines.append(line)
            else:
                # This should be indented as part of the class
                fixed_lines.append('    ' + line)
        else:
            # Reset in_class if we see an empty line after class content
            if not line.strip() and in_class:
                in_class = False
            fixed_lines.append(line)
    
    # Write back
    with open(file_path, 'w') as f:
        f.writelines(fixed_lines)
    
    print("✅ Fixed class indentation in unified_config.py")

if __name__ == "__main__":
    fix_file()