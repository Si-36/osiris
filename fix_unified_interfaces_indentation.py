#!/usr/bin/env python3
"""Fix indentation issues in unified_interfaces.py."""

def fix_file():
    file_path = '/workspace/core/src/aura_intelligence/core/unified_interfaces.py'
    
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    fixed_lines = []
    
    for i, line in enumerate(lines):
        # Fix over-indented async def and def (8 spaces -> 4 spaces)
        if line.startswith('        async def ') or line.startswith('        def '):
            # Check if previous line is a decorator
            if i > 0 and '@' in lines[i-1]:
                # This is likely a decorated method that's over-indented
                fixed_lines.append(line[4:])  # Remove 4 spaces
            else:
                fixed_lines.append(line)
        else:
            fixed_lines.append(line)
    
    # Write back
    with open(file_path, 'w') as f:
        f.writelines(fixed_lines)
    
    print("✅ Fixed indentation in unified_interfaces.py")

if __name__ == "__main__":
    fix_file()